"""Eval packs: self-contained folders of questions.

    packs/<name>/
      pack.toml        label, grader, system prompt, needs, …   (see packs/README.md)
      *.yaml | *.csv   the questions (all files, unless pack.toml lists `tests`)
      system.txt       system prompt (or whatever pack.toml `system` names)
      grader.py        optional: graders this pack brings with it (see graders/__init__.py)
      images/, …       assets referenced by tests

Add a folder and it shows up in the TUI; delete or rename it and it's gone. Every pack has a
fingerprint of its content, stored with results, so results from an older version of the
questions are flagged instead of being compared as if nothing changed.
"""
import csv
import dataclasses
import hashlib
import json
import os
import re
import tomllib

import yaml

from . import graders
from . import workspace

# `needs` entries tuieval knows; any other entry names a Python module the pack's grading needs
# (e.g. "pandas"), and the pack is skipped where that module isn't installed.
CAPABILITIES = ("vision", "tools", "long_context")


class PackError(Exception):
    pass


@dataclasses.dataclass
class Pack:
    name: str
    path: str
    label: str
    group: str
    description: str
    grader: str
    system: str            # system prompt text ("" if none)
    needs: list            # e.g. ["vision"], ["tools"], ["long_context"], ["pandas"]
    tools: list            # OpenAI-style tool definitions sent with every request, if any
    tests: list
    fingerprint: str
    gate: dict             # release criteria (see verdict.py)
    screen: int            # tests in a screening run
    certify_repeat: int    # repeats in a certification run (0 = models.toml default)

    @property
    def modules(self):
        """Python modules the pack needs (the `needs` entries that aren't capabilities)."""
        return [n for n in self.needs if n not in CAPABILITIES]

    def select(self, tier):
        """The tests a tier runs: smoke = first 3; screen = `screen` tests spread evenly over
        categories (one variant per group); certify = all."""
        if tier == "smoke":
            return self.tests[:3]
        if tier != "screen" or self.screen >= len(self.tests):
            return list(self.tests)
        by_cat, seen_groups = {}, set()
        for t in self.tests:
            key = t.get("group") or t["id"]
            if key in seen_groups:
                continue
            seen_groups.add(key)
            by_cat.setdefault(t.get("category", ""), []).append(t)
        # Alternate categories that can fail critically with the rest, so even a small screen
        # probes the disqualifying behaviours (traps, hard limits) that drop weak models early.
        crit = [list(v) for v in by_cat.values() if v[0].get("critical_trial")]
        other = [list(v) for v in by_cat.values() if not v[0].get("critical_trial")]
        queues = [q for pair in zip(crit, other) for q in pair] + crit[len(other):] + other[len(crit):]
        picked = []
        while len(picked) < self.screen and any(queues):
            for q in queues:
                if q and len(picked) < self.screen:
                    picked.append(q.pop(0))
        order = {t["id"]: i for i, t in enumerate(self.tests)}
        return sorted(picked, key=lambda t: order[t["id"]])

    def asset(self, rel):
        return os.path.join(self.path, rel)


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")[:60]


def _read_csv(path):
    """CSV questions: columns id, input, expected, tolerance, expected_text, category, … (any extra
    column becomes a test field). Empty cells are dropped; numbers are converted."""
    tests = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            t = {}
            for k, v in row.items():
                if k is None or v is None or v.strip() == "":
                    continue
                v = v.strip()
                if k in ("tolerance",) or (k == "expected" and re.fullmatch(r"-?\d+(\.\d+)?", v)):
                    v = float(v) if "." in v else int(v)
                t[k.strip()] = v
            tests.append(t)
    return tests


def _read_tests(path):
    if path.endswith(".csv"):
        return _read_csv(path)
    with open(path) as f:
        data = yaml.safe_load(f) or []
    if not isinstance(data, list):
        raise PackError(f"{path}: expected a list of tests")
    return data


# Fields that never reach the model or the grader's decision: editing them doesn't make
# results outdated.
NOT_FINGERPRINTED = {"reference", "wrong", "description", "category", "critical", "critical_trial", "group",
                     "captured_from", "captured_answer", "difficulty"}
DIFFICULTIES = ("easy", "medium", "hard")


def _fingerprint(system, tools, grader, tests, images):
    """Hash of what the model sees and what decides pass/fail."""
    h = hashlib.sha256()
    h.update(json.dumps({"system": system, "tools": tools, "grader": grader,
                         "tests": [{k: v for k, v in t.items() if k not in NOT_FINGERPRINTED} for t in tests]},
                        sort_keys=True, default=str).encode())
    for path in sorted(images):
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:12]


def load_pack(path):
    name = os.path.basename(path.rstrip("/"))
    if os.path.isfile(os.path.join(path, "grader.py")):
        graders.load_file(os.path.join(path, "grader.py"))
    meta_path = os.path.join(path, "pack.toml")
    with open(meta_path, "rb") as f:
        meta = tomllib.load(f)
    images = []

    system = ""
    if meta.get("system"):
        sys_path = os.path.join(path, meta["system"])
        with open(sys_path) as f:
            system = f.read().strip()

    tools = []
    if meta.get("tools"):
        tools_path = os.path.join(path, meta["tools"])
        with open(tools_path) as f:
            tools = yaml.safe_load(f) or []

    special = {meta.get("system"), meta.get("tools")}
    # tests.* first (hand-written, and what a smoke run picks), then any other files by name
    test_files = meta.get("tests") or sorted(
        (f for f in os.listdir(path)
         if f.endswith((".yaml", ".yml", ".csv")) and not f.startswith("_") and f not in special),
        key=lambda f: (not f.startswith("tests"), f))
    tests, seen = [], set()
    for tf in test_files:
        tpath = os.path.join(path, tf)
        for i, t in enumerate(_read_tests(tpath)):
            if not isinstance(t, dict) or "input" not in t:
                raise PackError(f"{tpath}: test #{i + 1} has no `input`")
            t = dict(t)
            t.setdefault("description", str(t.get("id") or t["input"])[:80])
            t.setdefault("id", _slug(t["description"]) or f"{tf}-{i + 1}")
            if t["id"] in seen:
                raise PackError(f"{tpath}: duplicate test id {t['id']!r}")
            seen.add(t["id"])
            t.setdefault("grader", meta.get("grader", "answer"))
            t.setdefault("category", "")
            # Your label only: never sent to the model (engine builds requests from input/image/system/tools).
            t.setdefault("difficulty", meta.get("difficulty", "unrated"))
            if t["difficulty"] not in DIFFICULTIES + ("unrated",):
                raise PackError(f"{tpath}: test {t['id']!r} has difficulty {t['difficulty']!r}; use easy, medium or hard")
            t["critical_trial"] = graders.is_critical(t)  # counts toward the critical-trial total
            if t.get("image"):
                img = os.path.join(path, t["image"])
                if not os.path.isfile(img):
                    raise PackError(f"{tpath}: image not found: {t['image']}")
                images.append(img)
            if t.get("system_file"):
                with open(os.path.join(path, t["system_file"])) as f:
                    t["system"] = f.read().strip()
            tests.append(t)
    if not tests:
        raise PackError(f"{path}: no tests")

    needs = meta.get("needs", [])
    if isinstance(needs, str):
        needs = [needs]
    if tools and "tools" not in needs:
        needs = list(needs) + ["tools"]
    gate = {"min_accuracy": 0.8, **meta.get("gate", {})}
    return Pack(name=name, path=path, label=meta.get("label", name), group=meta.get("group", "Other"),
                description=meta.get("description", ""), grader=meta.get("grader", "answer"),
                system=system, needs=list(needs), tools=tools, tests=tests,
                fingerprint=_fingerprint(system, tools, meta.get("grader", "answer"), tests, images),
                gate=gate, screen=int(meta.get("screen", 10)),
                certify_repeat=int(meta.get("certify", {}).get("repeat", 0)))


def load_packs(packs_dir=None, errors=None):
    """All packs, ordered by pack.toml `order` then name. packs_dir defaults to the workspace's
    packs/; a missing folder means no packs. The workspace's own graders/ are loaded first.

    A broken pack raises PackError, or, when an `errors` list is given, is skipped and its
    message appended there (so one typo doesn't take every other pack down with it)."""
    packs_dir = packs_dir or workspace.path("packs")
    try:
        graders.load_dir(os.path.join(os.path.dirname(os.path.abspath(packs_dir)), "graders"))
    except Exception as e:  # noqa: BLE001 - a broken grader file must not hide every pack
        if errors is None:
            raise
        errors.append(str(e))
    packs = []
    for entry in sorted(os.listdir(packs_dir)) if os.path.isdir(packs_dir) else []:
        path = os.path.join(packs_dir, entry)
        if os.path.isfile(os.path.join(path, "pack.toml")) and not entry.startswith(("_", ".")):
            try:
                packs.append(load_pack(path))
            except (PackError, OSError, ValueError, tomllib.TOMLDecodeError, yaml.YAMLError) as e:
                if errors is None:
                    raise
                errors.append(f"pack '{entry}' skipped: {e}")
    order = {}
    for p in packs:
        with open(os.path.join(p.path, "pack.toml"), "rb") as f:
            order[p.name] = tomllib.load(f).get("order", 100)
    return {p.name: p for p in sorted(packs, key=lambda p: (order[p.name], p.name))}
