"""Agent packs: coding tasks a command-line agent works on, graded by tests.

models.toml says how to start each agent (any program; {root} is the workspace):

    [agents.my-agent]
    cmd = ["python3", "{root}/agents/my_agent.py"]

    [defaults]
    agent = "my-agent"      # for agent packs that don't name one (pack.toml agent = "...")

tuieval serves the model as for any pack, then for each task:

  1. copies the task's repo/ to a scratch folder, a git repository, so the agent's changes show as a diff
  2. runs the agent's command there; every line it prints shows live in the run screen
  3. stops it at the task's time limit (task.toml minutes); the agent keeps to max_turns itself
  4. grades the folder with pytest: the task's original visible tests (editing or deleting them gains
     the agent nothing, and is recorded) plus its hidden tests, which the agent never sees

The agent learns the rest from its environment:

  TUIEVAL_BASE_URL     OpenAI-compatible URL of the model (…/v1)
  TUIEVAL_MODEL_ID     the model name to send
  TUIEVAL_API_KEY      the API key for hosted models ("" for local ones)
  TUIEVAL_PROMPT_FILE  the task's instruction
  TUIEVAL_SAMPLING     the model's sampling settings as JSON (temperature, max_tokens, reasoning_effort, …)
  TUIEVAL_CONTEXT      the model's context size (tokens), if known
  TUIEVAL_MAX_TURNS    the task's turn limit
  TUIEVAL_STATS_FILE   where it may write its stats as JSON when it finishes (all optional):
                         agent ("name version"), turns, input_tokens, output_tokens, tool_calls,
                         tool_errors (failed edits, bad tool calls; not commands that exit nonzero),
                         ran_tests, stop ("done", "turn limit" or "error"), final_message, and
                         server_error: the model's server failed, not the model (the task isn't counted
                         and is asked again)

PATH starts with the pack's Python (pack.toml python, else tuieval's), so `python` runs the tasks'
code and tests. The agent's exit code decides nothing; the tests do.

The agent runs as you, with your permissions, in the scratch folder: there is no sandbox.
"""
import filecmp
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET

PROMPT_TAIL = "\n\nYou're working in a project in the current directory. Run Python with `python`.\n"
GRADE_TIMEOUT_S = 300
DIFF_KEEP = 20_000      # characters of the diff kept in the result (the whole diff is in the log)
HIDDEN_RE = re.compile(r"\.?_hidden\b")   # junit names of the hidden tests (a collection error starts with ".")
STATS_KEYS = ("agent", "turns", "input_tokens", "output_tokens", "tool_calls", "tool_errors", "ran_tests", "stop",
              "final_message", "server_error")


class AgentError(Exception):
    pass


class ServerGone(Exception):
    """The model's server stopped while the agent worked: not the agent's fault."""


def agent_for(cfg, pack):
    """(name, [agents.<name>] table or None) for an agent pack."""
    name = pack.agent or cfg.get("defaults", {}).get("agent", "")
    return name, cfg.get("agents", {}).get(name) if name else None


def identity(cfg, pack):
    """What a result's settings record about the agent: its name and command (a different command
    makes earlier results outdated; a new version of the same agent doesn't, it's shown per answer)."""
    name, a = agent_for(cfg, pack)
    return {"name": name, "cmd": list((a or {}).get("cmd", []))}


def python_for(pack, root):
    """The Python an agent pack's tasks and tests run with."""
    if not pack.python:
        return sys.executable
    p = os.path.expanduser(pack.python)
    return p if os.path.isabs(p) else os.path.join(root, p)


_PYTEST = {}


def problem(cfg, pack, root):
    """Why an agent pack can't run here (a note for a skipped job), or None."""
    name, a = agent_for(cfg, pack)
    if not name:
        return 'no agent: add [agents.<name>] to models.toml and [defaults] agent = "<name>"'
    if not a or not a.get("cmd"):
        return f"agent {name!r} isn't defined: add [agents.{name}] with a cmd to models.toml"
    py = python_for(pack, root)
    if py not in _PYTEST:
        try:
            _PYTEST[py] = subprocess.run([py, "-c", "import pytest"], capture_output=True, timeout=60).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            _PYTEST[py] = False
    if not _PYTEST[py]:
        return f"pytest isn't installed in {py} (pack.toml python), which grades the tasks"
    return None


def env_for(python, extra):
    """The environment an agent and its tests run in: the pack's Python first on PATH."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "VIRTUAL_ENV", "CONDA_PREFIX"))}
    env["PATH"] = os.path.dirname(python) + os.pathsep + env.get("PATH", "/usr/bin:/bin")
    venv = os.path.dirname(os.path.dirname(python))
    if os.path.isfile(os.path.join(venv, "pyvenv.cfg")):
        env["VIRTUAL_ENV"] = venv
    env["MPLBACKEND"] = "Agg"
    env.update(extra)
    return env


def _git(work, *args):
    try:
        return subprocess.run(["git", "-c", "user.name=tuieval", "-c", "user.email=tuieval@localhost",
                               "-c", "commit.gpgsign=false", *args], cwd=work, capture_output=True, text=True,
                              timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None


def prepare(test, parent):
    """Copy a task's repo/ into parent/work as a fresh git repository; returns the folder."""
    work = os.path.join(parent, "work")
    shutil.copytree(os.path.join(test["task_dir"], "repo"), work,
                    ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".git"))
    if _git(work, "init", "-q") is not None:
        with open(os.path.join(work, ".git", "info", "exclude"), "a") as f:
            f.write("__pycache__/\n.pytest_cache/\n")
        _git(work, "add", "-A")
        _git(work, "commit", "-qm", "task")
    return work


def diff(work):
    """The agent's changes as a patch (empty without git)."""
    if not os.path.isdir(os.path.join(work, ".git")):
        return ""
    _git(work, "add", "-A")
    out = _git(work, "diff", "--cached", "--stat", "-p")
    return out.stdout if out and out.returncode == 0 else ""


def _kill(proc):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            continue


def run(cmd, work, env, limit_s, on_line, check, alive, log):
    """Run the agent in work until it exits or limit_s passes. Every output line goes to on_line and
    the log file. check() may raise (a cancel): the agent is stopped first. Returns (stop, seconds,
    exit code); stop is "exited" or "time limit". Raises ServerGone when alive() turns False."""
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, cwd=work, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, start_new_session=True)
    lines = queue.Queue()

    def read():
        for raw in iter(proc.stdout.readline, b""):
            lines.put(raw.decode("utf-8", "replace"))
        lines.put(None)
    threading.Thread(target=read, daemon=True).start()
    stop = "exited"
    try:
        while True:
            check()
            if not alive():
                raise ServerGone("the model's server stopped while the agent worked")
            if time.monotonic() - t0 > limit_s:
                stop = "time limit"
                break
            try:
                line = lines.get(timeout=0.5)
            except queue.Empty:
                continue
            if line is None:
                break
            log.write(line)
            on_line(line)
    finally:
        if proc.poll() is None:
            _kill(proc)
        else:   # exited: its own children may still run
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    return stop, round(time.monotonic() - t0, 1), proc.wait()


def read_stats(path):
    """The stats the agent wrote (only the known keys), or {}."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return {k: data[k] for k in STATS_KEYS if k in data} if isinstance(data, dict) else {}


def tampered(test, work):
    """Visible test files the agent changed or removed."""
    src = os.path.join(test["task_dir"], "repo", test["visible_tests"])
    out = []
    for dirpath, _, files in os.walk(src):
        for fn in sorted(files):
            if fn.endswith(".py"):
                a = os.path.join(dirpath, fn)
                rel = os.path.relpath(a, os.path.join(test["task_dir"], "repo"))
                b = os.path.join(work, rel)
                if not os.path.isfile(b) or not filecmp.cmp(a, b, shallow=False):
                    out.append(rel)
    return sorted(out)


def _cases(xml_path):
    """{test name: "pass" | "fail" | "skip"} from a pytest junit file."""
    try:
        root = ET.parse(xml_path).getroot()
    except (OSError, ET.ParseError):
        return {}
    out = {}
    for case in root.iter("testcase"):
        name = f'{case.get("classname")}.{case.get("name")}'
        bad = case.find("failure") is not None or case.find("error") is not None
        out[name] = "skip" if case.find("skipped") is not None else "fail" if bad else "pass"
    return out


def grade(test, work, python):
    """Run the task's original visible tests and its hidden tests on a copy of work.
    Returns {pass, score, visible, hidden, failed, output}; score is the share of hidden tests passed."""
    with tempfile.TemporaryDirectory(prefix="tuieval-grade-") as tmp:
        g = os.path.join(tmp, "work")
        shutil.copytree(work, g, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
        vis_src = os.path.join(test["task_dir"], "repo", test["visible_tests"])
        vis = os.path.join(g, test["visible_tests"])
        has_visible = os.path.isdir(vis_src)
        if os.path.isdir(vis):
            shutil.rmtree(vis)
        if has_visible:
            shutil.copytree(vis_src, vis)
        shutil.copytree(os.path.join(test["task_dir"], "hidden"), os.path.join(g, "_hidden"))
        xml = os.path.join(tmp, "junit.xml")
        cmd = [python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
               "--continue-on-collection-errors", f"--junitxml={xml}",
               *([test["visible_tests"]] if has_visible else []), "_hidden"]
        try:
            p = subprocess.run(cmd, cwd=g, env=env_for(python, {}), capture_output=True, text=True,
                               timeout=GRADE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return {"pass": False, "score": 0.0, "visible": "0/0", "hidden": "0/0", "failed": [],
                    "output": f"the tests ran over {GRADE_TIMEOUT_S} s"}
        cases = _cases(xml)
    hidden = {k: v for k, v in cases.items() if HIDDEN_RE.match(k)}
    visible = {k: v for k, v in cases.items() if not HIDDEN_RE.match(k)}
    failed = [k for k, v in cases.items() if v == "fail"]
    passed_hidden = sum(v == "pass" for v in hidden.values())

    def frac(c):
        return f'{sum(v == "pass" for v in c.values())}/{len(c)}'
    # a module that no longer imports shows as a collection error: a nonzero exit, never a pass
    ok = bool(hidden) and not failed and p.returncode == 0
    return {"pass": ok, "score": round(passed_hidden / len(hidden), 4) if hidden else 0.0,
            "visible": frac(visible), "hidden": frac(hidden), "failed": failed,
            "output": (p.stdout[-6000:] + p.stderr[-2000:]).strip()}


def reason(g, stop, edited, exit_code, stats):
    """One line about a graded task, for the answer's reason."""
    parts = [f"hidden tests {g['hidden']}, visible {g['visible']}"]
    if g["failed"]:
        names = [re.sub(r"^\.?(_hidden|tests)\.", "", f) for f in g["failed"]]
        parts.append("failed: " + ", ".join(names[:6]) + (f" and {len(names) - 6} more" if len(names) > 6 else ""))
    elif not g["pass"] and g["output"]:
        parts.append(g["output"].splitlines()[-1][:200])
    if stop == "time limit":
        parts.append("stopped at the time limit")
    elif stats.get("stop") == "turn limit":
        parts.append("stopped at the turn limit")
    elif stats.get("stop") == "error":
        parts.append("the agent stopped with an error")
    elif exit_code not in (0, None) and not stats:
        parts.append(f"the agent exited with code {exit_code}")
    if edited:
        parts.append("edited the tests: " + ", ".join(edited))
    return "; ".join(parts)


def check_task(test, python):
    """Selftest errors for one task: the untouched repo must fail, solution/ must pass, each
    wrong/<name>/ must fail. (A task without solution/ is only checked to fail untouched.)"""
    errors = []
    with tempfile.TemporaryDirectory(prefix="tuieval-selftest-") as tmp:
        def variant(name, overlay_dir=None):
            work = os.path.join(tmp, name)
            shutil.copytree(os.path.join(test["task_dir"], "repo"), work,
                            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".git"))
            if overlay_dir:
                overlay(overlay_dir, work)
            return grade(test, work, python)
        if variant("base")["pass"]:
            errors.append(f"{test['id']}: the untouched repo already passes its tests")
        sol = os.path.join(test["task_dir"], "solution")
        if os.path.isdir(sol):
            g = variant("solution", sol)
            if not g["pass"]:
                errors.append(f"{test['id']}: solution/ fails: {g['failed'][:4] or g['output'][-300:]}")
        wrong = os.path.join(test["task_dir"], "wrong")
        for w in sorted(os.listdir(wrong)) if os.path.isdir(wrong) else []:
            if os.path.isdir(os.path.join(wrong, w)) and variant(f"wrong-{w}", os.path.join(wrong, w))["pass"]:
                errors.append(f"{test['id']}: wrong/{w} passes, so the tests can't catch that mistake")
    return errors


def overlay(src, dst):
    """Copy src's files over dst; a .delete file in src lists paths (one per line) to remove from dst."""
    for dirpath, dirnames, files in os.walk(src):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in files:
            if fn == ".delete":
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), src)
            os.makedirs(os.path.dirname(os.path.join(dst, rel)), exist_ok=True)
            shutil.copy2(os.path.join(dirpath, fn), os.path.join(dst, rel))
    listing = os.path.join(src, ".delete")
    if os.path.isfile(listing):
        with open(listing) as f:
            for line in f.read().split():
                target = os.path.join(dst, line)
                if os.path.isdir(target):
                    shutil.rmtree(target)
                elif os.path.exists(target):
                    os.remove(target)
