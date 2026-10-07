"""tuieval doctor: checks the setup and says what to fix (servers, models, packs, API keys).

    tuieval doctor

Each line is ✓ (fine), ! (worth knowing) or ✗ (a run would fail), with the fix. Exits 1 if any ✗.
"""
import argparse
import os
import platform
import sys

from . import engine
from . import machines
from . import workspace

OK, WARN, BAD = "✓", "!", "✗"
COLOR = {OK: "\033[32m", WARN: "\033[33m", BAD: "\033[31m"}


class Report:
    def __init__(self):
        self.bad = 0
        self.color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")

    def line(self, mark, text, fix=""):
        self.bad += mark == BAD
        m = f"{COLOR[mark]}{mark}\033[0m" if self.color else mark
        print(f"  {m} {text}" + (f"\n      → {fix}" if fix else ""))

    def head(self, text):
        print(f"\n{text}")


def check(models_path=None):
    """Print the report; returns the number of problems that would make a run fail."""
    r = Report()
    r.head("tuieval")
    py = platform.python_version()
    r.line(OK if sys.version_info >= (3, 11) else BAD, f"Python {py}", "" if sys.version_info >= (3, 11)
           else "tuieval needs Python 3.11 or newer")
    try:
        e = engine.Engine(models_path=models_path)
    except Exception as ex:   # a broken models.toml: nothing else can be checked
        r.line(BAD, f"models.toml in {workspace.root()}: {ex}", "fix the file (tuieval init makes a fresh one elsewhere)")
        return r.bad
    r.line(OK, f"workspace {e.root}")

    r.head("Machine")
    mc = e.machine()
    r.line(OK, f"{mc.id}: {mc.summary}")
    if mc.discrete:
        r.line(OK, f"models are sized to {mc.gpu_limit_gb:.0f} GB of VRAM, keeping {e.headroom_gb(mc):g} GB free "
                   "(memory_headroom_gb, gpu_memory_gb under [machines.<id>] change it)")
        new = machines.gpu_id(machines.discrete_gpus(), mc.ram_gb)
        if mc.id != new and not os.environ.get("EVALS_MACHINE"):
            r.line(WARN, f"keeping the id {mc.id} this machine had before its GPU was detected, so earlier results "
                         "and tuning still match", f"to switch: EVALS_MACHINE={new} (results under the old id "
                                                   "then count as another machine)")
    elif platform.system() == "Linux" and any(os.path.exists(p) for p in ("/dev/nvidia0", "/dev/kfd")):
        r.line(WARN, "this machine seems to have an Nvidia or AMD GPU, but nvidia-smi / amd-smi / rocm-smi "
                     "isn't installed, so models are sized against system RAM",
               "install the vendor's tools, or set gpu_memory_gb under [machines.<id>] in models.toml")

    r.head("Packs")
    for err in e.pack_errors:
        r.line(BAD, err, "fix the pack, then: tuieval selftest")
    if e.packs:
        tests = sum(len(p.tests) for p in e.packs.values())
        r.line(OK, f"{len(e.packs)} pack(s), {tests} tests  (tuieval selftest checks their answers)")
    elif not e.pack_errors:
        r.line(BAD, "no packs yet", "tuieval new-pack my-first-pack   (a pack of example questions to edit)")

    r.head("Servers")
    found = engine.detect_servers(engine.detect_ports(e.cfg))
    used = {m["server"] for m in e.cfg["models"]}
    answering = {}
    for name, server in e.cfg["servers"].items():
        if server.get("any_model") or server.get("api_key_env"):
            key = server.get("api_key_env")
            if key and os.environ.get(key):
                r.line(OK, f"{name}: {key} is set")
            elif key:
                r.line(WARN, f"{name}: {key} isn't set (needed only for {name} models)",
                       f"export {key}=… in the shell you start tuieval from")
            continue
        if server.get("cmd"):
            missing = engine.missing_program([str(a) for a in server["cmd"]],
                                             engine.expand(server["cwd"]) if server.get("cwd") else None)
            if missing:
                r.line(BAD if name in used else WARN,
                       f"{name}: {missing} isn't installed" + ("" if name in used else " (no model uses it yet)"),
                       engine.install_hint(missing))
            else:
                r.line(OK, f"{name}: {server['cmd'][0]} is installed; tuieval starts it for each model")
            continue
        url = server.get("url", "").rstrip("/").removesuffix("/v1")
        ids = engine.probe_models(url) if url else None
        answering[name] = ids
        if ids is not None:
            r.line(OK, f"{name}: {url} answers, serving {len(ids)} model(s)" + (f": {', '.join(ids[:5])}" if ids else ""))
        else:
            elsewhere = [f for f in found if f["url"] != url]
            fix = (f"start the server at {url}, or "
                   + ("run `tuieval add` to add the models of the one running at " + ", ".join(
                       f"{f['url']}" + (f" ({f['usual']})" if f["usual"] else "") for f in elsewhere)
                      if elsewhere else engine.url_fix(e.cfg, name)))
            r.line(BAD if name in used else WARN,
                   f"{name}: nothing answers at {url}" + ("" if name in used else " (no model uses it yet)"), fix)
    known = {s.get("url", "").rstrip("/").removesuffix("/v1") for s in e.cfg["servers"].values()}
    for f in found:
        if f["url"] not in known:
            r.line(WARN, f"found a server at {f['url']}" + (f" ({f['usual']}'s usual port)" if f["usual"] else "")
                   + f" serving {len(f['models'])} model(s), not in models.toml",
                   "add its models with: tuieval add")

    r.head("Models")
    if not e.cfg["models"]:
        r.line(BAD, "no models yet", "tuieval add <GGUF path or model id>, or press a in the TUI")
    for m in e.cfg["models"]:
        server = e.cfg["servers"][m["server"]]
        label = f"{m['label']} ({m['server']})"
        if server.get("model_is_path"):
            path = engine.expand(m["model"])
            r.line(OK if os.path.exists(path) else BAD, f"{label}: {path}",
                   "" if os.path.exists(path) else "the file is gone; fix `model` in models.toml or tuieval remove it")
        elif m["server"] in answering:
            ids = answering[m["server"]]
            if ids is None:
                r.line(BAD, f"{label}: its server isn't running", "see Servers above")
            elif engine.serves(ids, m["served_name"]):
                r.line(OK, f"{label}: served")
            else:
                r.line(BAD, f"{label}: the server doesn't list {m['served_name']!r}",
                       f"load it in the server, or use one of: {', '.join(ids[:8]) or 'none listed'}")
        else:
            r.line(OK, label)

    print("\n" + ("Ready: open the TUI with `tuieval`, or run `tuieval run --tier smoke`." if not r.bad
                  else f"{r.bad} problem(s) to fix before a run (✗ above)."))
    return r.bad


def cmd_doctor(argv):
    p = argparse.ArgumentParser(prog="tuieval doctor", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default=None, help="default: the workspace's models.toml")
    a = p.parse_args(argv)
    sys.exit(1 if check(a.models) else 0)
