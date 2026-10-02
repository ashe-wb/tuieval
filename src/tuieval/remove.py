"""Remove models from the workspace: their models.toml entry, results (screen and smoke), tuning
profiles and hidden mark. Nothing is deleted: everything moves to removed/<label>-<time>/, with the
models.toml block saved there as models.toml-entry.txt, so a removal can be undone by hand.
"""
import difflib
import os
import re
import shutil
import time
import tomllib

from . import engine as engine_mod
from . import profiles

HEADER = re.compile(r"^\s*\[")
MODELS_HEADER = re.compile(r"^\s*\[\[models\]\]\s*(#.*)?$")


def model_block(text, label):
    """(start, end) line range of label's [[models]] block in models.toml text, with the comment lines
    directly above it, or None. The entries' order in the file is their order in the parsed TOML."""
    entries = tomllib.loads(text).get("models", [])
    labels = [m.get("label") or engine_mod.slug(m.get("model", "")) for m in entries]
    if label not in labels:
        return None
    lines = text.splitlines()
    headers = [i for i, l in enumerate(lines) if MODELS_HEADER.match(l)]
    start = headers[labels.index(label)]
    end = next((i for i in range(start + 1, len(lines)) if HEADER.match(lines[i])), len(lines))
    # comment and blank lines at the end introduce the next section
    while end > start + 1 and (not lines[end - 1].strip() or lines[end - 1].lstrip().startswith("#")):
        end -= 1
    while start > 0 and lines[start - 1].lstrip().startswith("#"):
        start -= 1
    return start, end


def without_block(text, label):
    """models.toml text without label's block (and one blank line around it), or None."""
    span = model_block(text, label)
    if span is None:
        return None
    lines = text.splitlines()
    start, end = span
    while end < len(lines) and not lines[end].strip():
        end += 1
    if end == len(lines):
        while start > 0 and not lines[start - 1].strip():
            start -= 1
    return "\n".join(lines[:start] + lines[end:]).rstrip("\n") + "\n"


class Plan:
    def __init__(self, label):
        self.label = label
        self.moves = []          # (source path, path under the archive folder)
        self.models_text = None  # (old, new) models.toml text
        self.hidden = False

    def describe(self, root):
        rel = lambda p: os.path.relpath(p, root) if p.startswith(root) else p  # noqa: E731
        out = [f"  move {rel(src)}/" if os.path.isdir(src) else f"  move {rel(src)}" for src, _ in self.moves]
        if self.hidden:
            out.append("  drop it from results/hidden.txt")
        return out


def plan(eng, label):
    """What removing label does. Raises ValueError for an unknown label."""
    if not any(m["label"] == label for m in eng.cfg["models"]):
        raise ValueError(f"unknown model {label!r}; see tuieval list")
    p = Plan(label)
    for sub in ("", "smoke"):
        d = os.path.join(eng.results_dir, sub, label)
        if os.path.isdir(d):
            p.moves.append((d, os.path.join("results", sub, label)))
    for machine in sorted(os.listdir(eng.tuning_dir)) if os.path.isdir(eng.tuning_dir) else []:
        f = profiles.path(machine, label, eng.tuning_dir)
        if os.path.isfile(f):
            p.moves.append((f, os.path.join("tuning", machine, label + ".toml")))
    with open(eng.models_path) as f:
        old = f.read()
    new = without_block(old, label)
    if new is not None:
        p.models_text = (old, new)
    p.hidden = label in eng.hidden()
    return p


def models_diff(plans, models_path):
    """The models.toml diff of removing every planned model at once."""
    old = new = None
    for p in plans:
        if p.models_text:
            old = old or p.models_text[0]
            new = without_block(new or old, p.label)
    if old is None:
        return "", None
    return "\n".join(difflib.unified_diff(old.splitlines(), new.splitlines(), models_path,
                                          models_path + " (new)", lineterm="", n=1)), new


def apply(eng, plans):
    """Archive and remove the planned models. Returns the archive folders."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    folders = []
    with open(eng.models_path) as f:
        text = f.read()
    for p in plans:
        dest = os.path.join(eng.root, "removed", f"{p.label}-{stamp}")
        os.makedirs(dest, exist_ok=True)
        for src, rel in p.moves:
            target = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.move(src, target)
        if p.models_text:
            span = model_block(text, p.label)
            with open(os.path.join(dest, "models.toml-entry.txt"), "w") as f:
                f.write("\n".join(text.splitlines()[span[0]:span[1]]) + "\n")
            text = without_block(text, p.label)
        if p.hidden:
            eng.set_hidden(p.label, False)
        folders.append(dest)
    if any(p.models_text for p in plans):
        shutil.copy2(eng.models_path, os.path.join(folders[0], "models.toml.before"))
        tmp = eng.models_path + ".tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, eng.models_path)
    return folders
