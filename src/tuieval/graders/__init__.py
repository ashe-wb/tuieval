"""Grader registry.

A grader is a function registered by name:

    from tuieval.graders import grader, result

    @grader("my_grader")
    def my_grader(answer, test, meta):
        ok = ...
        return result(ok, "why", score=None)   # score defaults to 1.0/0.0

- answer: the model's final answer text (reasoning already separated out)
- test:   the test dict from the pack (all its fields)
- meta:   {"finish": "stop"|"length"|…, "tool_calls": [{"name", "arguments"}]}

A failure that should disqualify a model from production (breaking a hard rule, inventing data,
an unwanted action) returns result(False, why, severity="critical"). A test marked
`critical: true` makes *any* of its failures critical. is_critical(test) says which tests can
produce critical failures at all; verdict.py counts those "critical trials" to bound the
critical failure rate. A grader whose every test can fail critically (e.g. any tool call can be
an unwanted action) registers with critical=True.

`template` is the skeleton `tuieval capture` and `tuieval new-pack` write for a new test of
this grader (the grader-specific fields, with TODO where you fill in).

Packs choose a grader with `grader = "<name>"` in pack.toml (a test may override it). The
built-in graders live in this folder. Your own load from the workspace's graders/ folder (one
.py file each) and from a pack's own grader.py, so a pack can bring its grading with it.
"""
import importlib
import importlib.util
import os
import pkgutil
import re
import sys

REGISTRY = {}
CRITICAL = set()     # graders whose every test can fail critically
TEMPLATES = {}       # grader -> fields of a new test's skeleton
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_LOADED = {}         # file path -> modification time it was loaded at


class GraderError(ValueError):
    pass


def grader(name, critical=False, template=None):
    def register(fn):
        REGISTRY[name] = fn
        if critical:
            CRITICAL.add(name)
        else:
            CRITICAL.discard(name)
        if template is not None:
            TEMPLATES[name] = template
        return fn
    return register


def load_file(path):
    """Import one grader file (a workspace's or a pack's). Loaded again only when it changes."""
    path = os.path.abspath(path)
    mtime = os.path.getmtime(path)
    if _LOADED.get(path) == mtime:
        return
    name = "tuieval_user_grader_" + re.sub(r"\W", "_", path)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    except Exception as e:
        raise GraderError(f"grader file {path} failed to load: {e!r}") from e
    _LOADED[path] = mtime


def load_dir(folder):
    """Import every .py file in a folder (not starting with _), if the folder exists."""
    if not os.path.isdir(folder):
        return
    for f in sorted(os.listdir(folder)):
        if f.endswith(".py") and not f.startswith("_"):
            load_file(os.path.join(folder, f))


def result(ok, reason, score=None, severity=None):
    out = {"pass": bool(ok), "score": float(score if score is not None else ok), "reason": reason}
    if severity and not ok:
        out["severity"] = severity
    return out


def is_critical(test):
    """Tests whose failures can be critical: marked critical, 'data isn't there' traps, and every
    test of a grader registered with critical=True (e.g. tool_call: any wrong or unwanted action)."""
    if "critical" in test:
        return bool(test["critical"])
    if str(test.get("expected")) == "NOT_AVAILABLE":
        return True
    return test.get("grader", "answer") in CRITICAL


def strip_thinking(text):
    """Safety net for servers that leave <think> blocks in the answer."""
    text = THINK_RE.sub("", text or "")
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def grade(test, answer, meta=None):
    """Grade one answer. Common failures (truncated, empty) are handled here for every grader."""
    meta = meta or {}
    name = test.get("grader", "answer")
    fn = REGISTRY.get(name)
    if fn is None:
        return result(False, f"unknown grader {name!r}; known: {', '.join(sorted(REGISTRY))}")
    if meta.get("finish") == "length":
        return result(False, "TRUNCATED: hit max_tokens before finishing")
    text = strip_thinking(answer)
    if not text and not meta.get("tool_calls"):
        return result(False, "Empty answer")
    try:
        out = fn(text, test, meta)
    except Exception as e:  # a grader bug must not stop a run
        return result(False, f"grader error: {e!r}")
    if not out["pass"] and test.get("critical") and "severity" not in out:
        out["severity"] = "critical"
    return out


for _mod in pkgutil.iter_modules([os.path.dirname(__file__)]):
    if _mod.name.startswith("_"):
        continue
    if _mod.name in globals():  # e.g. a file named grade.py would shadow grade()
        raise ImportError(f"graders/{_mod.name}.py clashes with graders.{_mod.name}; rename the file")
    importlib.import_module(f"{__name__}.{_mod.name}")
