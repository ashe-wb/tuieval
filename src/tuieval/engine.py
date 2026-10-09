"""The eval engine behind the TUI (tui.py) and the CLI (run_evals.py).

  models.toml  which models exist and how to serve them      (in the workspace, see workspace.py)
  packs/       the questions (see packs.py)                   (in the workspace)
  graders/     how answers are scored (see graders/__init__.py; built in, plus the workspace's own)

A run is a queue of (model, pack) jobs:

  for each model:  start its server (or use an already-running URL) -> wait until it serves the
                   expected name -> measure load time and memory
    for each pack:   send every test (x repeats) straight to the server, streaming;
                     grade each answer as it arrives; append it to a .partial.jsonl

Every request is a fresh single-turn conversation (system + one user message; no earlier question
or answer). Repeats go round by round, each round in its own fixed shuffled order, so a test never
follows itself or always the same neighbour. A server's `request` table adds fields to every request
(llama: cache_prompt = false, so no prompt state carries over); each answer records the messages
sent and any prompt tokens the server reports reusing, as proof.
                     -> results/<label>/<pack>.json when the pack is complete
  stop the server

Each result stores the pack fingerprint and the model's effective settings (sampling plus the
output-affecting part of how it is served), so results from changed questions or settings are
reported as outdated rather than silently compared. Speed-only server flags come from the tuned
profile for this model on this machine (tuning/, see tune.py) and are recorded, not fingerprinted. An
interrupted run resumes from its .partial.jsonl, request by request.

Events are delivered as on_event(kind, **data), from the engine thread.
"""
import base64
import contextlib
import dataclasses
import fcntl
import hashlib
import importlib.util
import json
import mimetypes
import os
import random
import re
import shutil
import signal
import socket
import statistics
import subprocess
import threading
import time
import tomllib
import urllib.request

from . import client
from . import graders
from . import machines
from . import packs as packs_mod
from . import profiles
from . import workspace
LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
RESULT_FORMAT = "evals/1"
TIERS = ("smoke", "screen", "certify")
DEFAULT_SECONDS_PER_REQUEST = 60


# ------------------------------------------------------------------ models
class ConfigError(Exception):
    pass


def slug(model):
    """Default label: the file or repo name, lowercased (…/Foo-Q4_K_M.gguf -> foo-q4_k_m)."""
    name = os.path.basename(model.rstrip("/"))
    if name.lower().endswith(".gguf"):
        name = name[:-5]
    return re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-.")


BUILTIN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "builtin.toml")


def builtin_config():
    with open(BUILTIN_PATH, "rb") as f:
        return tomllib.load(f)


def load_config(path):
    """Read models.toml, fill in the built-ins it leaves out (builtin.toml) and each model's defaults.
    [defaults] are filled key by key; [sampling] only when models.toml has none (sampling is part of
    every result's fingerprint, so a workspace's own keys are never added to); servers by name."""
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    builtin = builtin_config()
    cfg["defaults"] = {**builtin["defaults"], **cfg.get("defaults", {})}
    cfg.setdefault("sampling", builtin["sampling"])
    own = cfg.get("servers", {})
    cfg["servers"] = {**builtin["servers"], **own}
    cfg["builtin_servers"] = sorted(set(builtin["servers"]) - set(own))   # not defined in models.toml
    cfg.setdefault("models", [])
    seen = set()
    for m in cfg["models"]:
        server = cfg["servers"].get(m.get("server"))
        if server is None:
            raise ConfigError(f"{path}: model {m.get('model')!r} has unknown server {m.get('server')!r}; "
                              f"servers are {sorted(cfg['servers'])}")
        m.setdefault("label", slug(m["model"]))
        if not LABEL_RE.match(m["label"]):
            raise ConfigError(f"{path}: label {m['label']!r} must be lowercase letters, digits, '.', '_' or '-'")
        if m["label"] in seen:
            raise ConfigError(f"{path}: label {m['label']!r} is used twice; labels name the results folders")
        seen.add(m["label"])
        # llama serves under --alias {served_name}, so any name works; other servers use the model id.
        m.setdefault("served_name", m["label"] if server.get("model_is_path") else m["model"])
        m["vision"] = bool(m.get("vision") or m.get("mmproj"))
        m.setdefault("tags", [])
    return cfg


def fit_check(server):
    """Whether tuieval sizes the context for this server's models from GGUF headers (machines.fit,
    which assumes llama.cpp's memory use). By default only servers whose command takes {ctx}, i.e.
    where tuieval chooses the context; others size their own memory, and their context is the
    model's max_context. models.toml `fit_check = true|false` on a server overrides it."""
    if "fit_check" in server:
        return bool(server["fit_check"])
    return any("{ctx}" in str(a) for a in server.get("cmd", []))


def effective_sampling(cfg, m):
    """The request settings this model runs with: [sampling], then the model's own overrides."""
    s = dict(cfg["sampling"])
    s.update(m.get("sampling") or {})
    if "thinking" in m:
        s["enable_thinking"] = bool(m["thinking"])
    return s


# Flags that only set how much context a server holds. Context size doesn't change the answer to a
# prompt that fits (a pack that needs more than a model holds is skipped), so changing it keeps
# earlier results current and interrupted packs resumable. The context used is still recorded
# (serving identity and run.ctx).
CONTEXT_FLAGS = {"--max-context", "--max-context-tokens", "--ctx-size", "-c", "--context-length", "--max-model-len"}


def strip_context(args):
    """A command line without its context-size flags (and their values)."""
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
        elif a in CONTEXT_FLAGS:
            skip = True
        elif not (isinstance(a, str) and a.split("=", 1)[0] in CONTEXT_FLAGS):
            out.append(a)
    return out


def comparable(settings):
    """Settings (sampling + serving identity) as matched between runs: context size left out."""
    serving = settings.get("serving") if isinstance(settings, dict) else None
    if not isinstance(serving, dict):
        return settings
    serving = {k: strip_context(v) if k in ("cmd", "server_args", "perf") and isinstance(v, list) else v
               for k, v in serving.items()}
    return {**settings, "serving": serving}


def _fp(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def settings_fingerprint(settings):
    return _fp(comparable(settings))


def stored_settings(data):
    """The settings a result file ran with, in the shape Engine.settings() returns."""
    if "serving" not in data:
        return data["settings"]
    return {"sampling": data["settings"], "serving": data["serving"]}


def settings_match(data, settings):
    """Does a stored result match these settings? Matched on what the file records, so results
    fingerprinted before context size was left out still match. Results written before serving
    settings were fingerprinted (no "serving" key) are matched on sampling alone."""
    if "serving" not in data:
        return data["settings_fingerprint"] in (settings_fingerprint(settings["sampling"]),
                                                _fp(settings["sampling"]))
    return settings_fingerprint(stored_settings(data)) == settings_fingerprint(settings)


def header_matches(head, pack_fingerprint, settings):
    """Does an unfinished run's .partial.jsonl header match this pack and these settings? Headers
    written before context size was left out carry a fingerprint over the full settings."""
    if not head or head.get("pack_fingerprint") != pack_fingerprint:
        return False
    return head.get("settings_fingerprint") in (settings_fingerprint(settings), _fp(settings))


def _short(v):
    return " ".join(map(str, v)) if isinstance(v, list) else str(v)


def describe_change(before, now):
    """What differs between two settings dicts, for people: 'max_tokens 20480 → 16384; server_args
    --flash on → (none)'. before may be None (not recorded)."""
    if not isinstance(before, dict):
        return "settings changed since it started"
    before, now = comparable(before), comparable(now)
    if "sampling" not in before:   # a result from before serving settings were recorded
        before = {"sampling": before, "serving": now.get("serving")}
    parts = []
    for part in ("sampling", "serving"):
        b, n = before.get(part) or {}, now.get(part) or {}
        for k in sorted(set(b) | set(n)):
            if b.get(k) != n.get(k):
                parts.append(f"{k} {_short(b.get(k)) if k in b else '(none)'} → {_short(n.get(k)) if k in n else '(none)'}")
    return "settings changed: " + "; ".join(parts) if parts else "settings changed"


@dataclasses.dataclass
class Serving:
    """How a model is served on this machine."""
    machine: machines.Machine
    ctx: int | None            # context size (-c); None when the server decides
    kv_type: str | None
    fits: bool
    fit_note: str              # "" when not checked (not a local GGUF, or the file is missing)
    perf_args: list            # speed-only flags (fixed perf + tuned or default knobs)
    perf_source: str           # tuned | seeded | untuned | outdated: <why> | n/a
    identity: dict             # output-affecting settings, part of the results fingerprint
    profile: dict | None = None
    mtp_layers: int = 0
    need_gb: float | None = None   # estimated GPU memory at ctx (from the fit check); None when unknown


def infer_server(cfg, model):
    is_file = model.lower().endswith(".gguf")
    choices = [n for n, s in cfg["servers"].items()
               if bool(s.get("model_is_path")) == is_file and not s.get("any_model")]
    own = [n for n in choices if n not in cfg.get("builtin_servers", ())]   # the workspace's own come first
    choices = own or choices
    return choices[0] if len(choices) == 1 else None


def remote_label(server_name, server, model_id):
    """Label for a model named at run time: openrouter:qwen/qwen3-32b -> or-qwen-qwen3-32b."""
    name = re.sub(r"[^a-z0-9._-]+", "-", model_id.lower()).strip("-.")
    return f"{server.get('label_prefix', server_name)}-{name}"


def fetch_catalogue(server):
    """{model id: metadata} from a server's /v1/models (OpenRouter lists every model it serves)."""
    url = server["url"].rstrip("/").removesuffix("/v1") + "/v1/models"
    with urllib.request.urlopen(url, timeout=20) as r:
        return {x["id"]: x for x in json.load(r).get("data", []) if x.get("id")}


def fetch_endpoints(server, model_id):
    """Every provider endpoint serving a model on OpenRouter (tag, quantization, parameters, uptime…)."""
    url = server["url"].rstrip("/").removesuffix("/v1") + f"/v1/models/{model_id}/endpoints"
    with urllib.request.urlopen(url, timeout=20) as r:
        return (json.load(r).get("data") or {}).get("endpoints") or []


# Quantization ranks for choosing an endpoint: the closest to the released weights first. A provider
# that doesn't say ("unknown") ranks below fp8, except the model maker's own endpoint.
PRECISION = {"fp32": 5, "bf16": 5, "fp16": 5, "fp8": 3, "int8": 3, "unknown": 2}


def choose_endpoint(endpoints, model_id, sampling, needs_tools, previous=None):
    """The one endpoint every answer of a model comes from. OpenRouter otherwise spreads requests over
    providers running different quantizations and implementations, so one model's answers would be a
    mix of several models. Keeps the endpoint earlier results used while it is still offered."""
    ok = [e for e in endpoints if e.get("status", 0) == 0 and e.get("tag")] or [e for e in endpoints if e.get("tag")]
    if not ok:
        return None
    long_enough = [e for e in ok if (e.get("max_completion_tokens") or 10**9) >= sampling.get("max_tokens", 0)]
    ok = long_enough or ok
    up = [e for e in ok if (e.get("uptime_last_30m") or 100) >= 90]
    ok = up or ok
    for e in ok:
        if e["tag"] == previous:
            return e
    author = model_id.split("/")[0].lower()
    wanted = [k for k in ("temperature", "top_p", "top_k", "min_p") if k in sampling]

    def rank(e):
        params = e.get("supported_parameters") or []
        q = (e.get("quantization") or "unknown").lower()
        precision = PRECISION.get(q, 1) + (2 if q == "unknown" and e["tag"].split("/")[0] == author else 0)
        price = sum(float((e.get("pricing") or {}).get(k) or 0) for k in ("prompt", "completion"))
        return (("tools" in params) or not needs_tools, precision, sum(k in params for k in wanted),
                "reasoning" in params or not sampling.get("enable_thinking"),
                not e.get("supports_implicit_caching"), e.get("uptime_last_30m") or 0, -price)
    return sorted(sorted(ok, key=lambda e: e["tag"]), key=rank, reverse=True)[0]


def add_model(models_path, model, server=None, label=None, vision=False, mmproj=None, thinking=None, tags=None):
    """Append a [[models]] block to models.toml. Returns (label, warnings). Raises ConfigError."""
    cfg = load_config(models_path)
    home = os.path.expanduser("~") + os.sep
    fix = lambda p: "~/" + p[len(home):] if p and p.startswith(home) else p  # noqa: E731
    model, mmproj = fix((model or "").strip()), fix((mmproj or "").strip()) or None
    if not model:
        raise ConfigError("model is empty")
    server = server or infer_server(cfg, model)
    if not server:
        raise ConfigError(f"can't tell which server runs {model!r}; choose one of {sorted(cfg['servers'])}")
    if server not in cfg["servers"]:
        raise ConfigError(f"unknown server {server!r}; servers are {sorted(cfg['servers'])}")
    label = (label or "").strip() or slug(model) + ("-nothink" if thinking is False else "")
    if not LABEL_RE.match(label):
        raise ConfigError(f"label {label!r} must be lowercase letters, digits, '.', '_' or '-'")
    if label in {m["label"] for m in cfg["models"]}:
        raise ConfigError(f"label {label!r} already exists; choose another")
    warnings = []
    if cfg["servers"][server].get("model_is_path") and not os.path.exists(os.path.expanduser(model)):
        warnings.append(f"{model} doesn't exist (yet); runs will fail until it does")
    if mmproj and not os.path.isfile(os.path.expanduser(mmproj)):
        warnings.append(f"{mmproj} doesn't exist (yet)")
    block = [f"label = {json.dumps(label)}", f"server = {json.dumps(server)}", f"model = {json.dumps(model)}"]
    if vision or mmproj:
        block.append("vision = true")
    if mmproj:
        block.append(f"mmproj = {json.dumps(mmproj)}")
    if thinking is False:
        block.append("thinking = false")
    tags = [t for t in (tags or []) if t]
    if tags:
        block.append("tags = [" + ", ".join(json.dumps(t) for t in tags) + "]")
    with open(models_path) as f:
        original = f.read()
    with open(models_path, "w") as f:
        f.write(original.rstrip("\n") + "\n\n[[models]]\n" + "\n".join(block) + "\n")
    try:
        load_config(models_path)
    except BaseException:
        with open(models_path, "w") as f:
            f.write(original)
        raise
    return label, warnings


def scan_models(cfg, dirs=None):
    """GGUF files under the model folders that aren't in models.toml yet, with a guessed mmproj.

    Returns [{"model": "~/…gguf", "mmproj": "~/…" or None, "label": suggested}]. Split-file
    shards other than the first, and mmproj files themselves, are skipped.
    """
    dirs = dirs or cfg["defaults"].get("model_dirs", ["~/models"])
    known = {os.path.realpath(os.path.expanduser(m["model"])) for m in cfg["models"]}
    labels = {m["label"] for m in cfg["models"]}
    home = os.path.expanduser("~") + os.sep
    found = []
    for d in dirs:
        d = os.path.expanduser(d)
        for root, _, files in os.walk(d):
            ggufs = sorted(f for f in files if f.lower().endswith(".gguf"))
            projectors = [f for f in ggufs if "mmproj" in f.lower()]
            for f in ggufs:
                if f in projectors or re.search(r"-0000[2-9]-of-|-000[1-9]\d-of-", f):
                    continue
                path = os.path.join(root, f)
                if os.path.realpath(path) in known:
                    continue
                short = lambda p: "~/" + p[len(home):] if p.startswith(home) else p  # noqa: E731
                label = slug(f)
                found.append({"model": short(path), "label": label if label not in labels else "",
                              "mmproj": short(os.path.join(root, projectors[0])) if projectors else None})
    return found


# ------------------------------------------------------------------ presets
def load_presets(path):
    """presets.toml: named selections, e.g.

        [nightly]
        packs = ["coding", "support-bot"]
        tags = ["moe"]            # models with any of these tags …
        models = ["foo", "bar"]   # … and/or these labels (omit both for every model)
        repeat = 3
        tests = { coding = ["parse-dates"] }   # optional: only these tests of a pack
    """
    if not os.path.isfile(path):
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def resolve_preset(cfg, preset, pack_names):
    """(labels, packs, repeat, tests) for a preset; unknown names are dropped. tests: {pack: [test
    ids]} for packs where only some tests run."""
    labels = [m["label"] for m in cfg["models"]]
    if preset.get("models") or preset.get("tags"):
        want, tags = set(preset.get("models", [])), set(preset.get("tags", []))
        labels = [m["label"] for m in cfg["models"] if m["label"] in want or tags & set(m["tags"])]
    packs = [p for p in preset.get("packs", pack_names) if p in pack_names]
    tests = {p: list(ids) for p, ids in (preset.get("tests") or {}).items() if p in packs and ids}
    return labels, packs, preset.get("repeat"), tests


def save_preset(path, name, labels, packs, repeat, tests=None):
    """Add or replace one preset, keeping the rest of the file as written."""
    tests = {p: ids for p, ids in (tests or {}).items() if p in packs and ids}
    block = (f"[{name}]\nmodels = [{', '.join(json.dumps(l) for l in labels)}]\n"
             f"packs = [{', '.join(json.dumps(p) for p in packs)}]\n" + (f"repeat = {int(repeat)}\n" if repeat else "")
             + ("tests = { " + ", ".join(f"{json.dumps(p)} = [{', '.join(json.dumps(i) for i in ids)}]"
                                         for p, ids in tests.items()) + " }\n" if tests else ""))
    text = open(path).read() if os.path.isfile(path) else \
        "# Saved selections for the TUI (p) and tuieval run --preset <name>. See engine.load_presets.\n"
    pattern = re.compile(rf"^\[{re.escape(name)}\]\n(?:(?!\[).*\n?)*", re.MULTILINE)
    text = pattern.sub(block + "\n", text) if pattern.search(text) else text.rstrip("\n") + "\n\n" + block
    with open(path, "w") as f:
        f.write(text)
    load_presets(path)  # raises if the file became invalid


# ------------------------------------------------------------------ jobs
@dataclasses.dataclass
class Job:
    model: dict
    pack: packs_mod.Pack
    repeat: int
    tests: list             # the tests this tier runs (or the ones picked)
    out_path: str
    sampling: dict
    tier: str = "certify"
    settings: dict = dataclasses.field(default_factory=dict)  # sampling + serving identity (fingerprinted)
    merge: bool = True      # keep matching earlier results and only run what's missing
    redo: bool = False      # rerun the picked tests, keeping the pack's other answers
    fresh: bool = False     # retest: an unfinished run's matching answers are set aside too, not resumed
    status: str = "waiting"  # waiting, loading, running, done, failed, skipped
    note: str = ""
    discards: list = dataclasses.field(default_factory=list)  # [(what, answers, why)] set aside on start
    done: int = 0
    passed: int = 0
    failed: int = 0
    started: float = 0.0
    finished: float = 0.0
    resumed: int = 0        # answers already done when this sitting started
    earlier: list = dataclasses.field(default_factory=list)  # earlier sittings (see Engine._sittings)

    @property
    def label(self):
        return self.model["label"]

    @property
    def count(self):
        return len(self.tests)

    @property
    def total(self):
        return self.count * self.repeat

    @property
    def key(self):
        return f"{self.label}/{self.pack.name}"


def port_in_use(port):
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def expand(arg, **values):
    arg = arg.format(**values)
    return os.path.expanduser(arg) if arg.startswith("~") else arg


ENV_PREFIX = "env:"   # a server arg "env:NAME=value" sets an environment variable instead


def split_env(args):
    """(args, ["NAME=value", …]): pull "env:NAME=value" items out of a server argument list."""
    env = [a[len(ENV_PREFIX):] for a in args if a.startswith(ENV_PREFIX)]
    return [a for a in args if not a.startswith(ENV_PREFIX)], env


def read_log_facts(path, start, patterns):
    """{name: value} from a server's startup log (written since `start`), using the server's
    log_facts regexes; the last match of each wins (e.g. a cache size the server chose itself)."""
    try:
        with open(path, errors="replace") as f:
            f.seek(start)
            text = f.read()
    except OSError:
        return {}
    facts = {}
    for name, pattern in patterns.items():
        found = re.findall(pattern, text)
        if found:
            v = found[-1]
            facts[name] = int(v) if v.isdigit() else float(v) if re.fullmatch(r"\d+\.\d+", v) else v
    return facts


def read_result(path):
    """A result file (evals format), or None."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if data.get("format") != RESULT_FORMAT:
        return None
    data["results"] = [r for r in data["results"] if not client.server_error_row(r)]
    return data


def server_error(log, start):
    """Why a server stopped while loading: the last error line it wrote since start, when it
    names one (some servers refuse a context that does not fit with one)."""
    try:
        with open(log, "rb") as f:
            f.seek(start)
            lines = f.read().decode(errors="replace").splitlines()
    except OSError:
        return None
    # The last block of error lines, read from its start: servers often follow the specific error
    # ("this GGUF stores tensors … cannot load") with a generic one ("model failed to load"), and
    # the specific one is what the user needs.
    block = []
    for line in reversed(lines):
        if line.startswith("error: "):
            block.append(line)
        elif block:
            break
    if not block:
        return None
    line = block[-1]
    if line.startswith("error: runtime bootstrap failed"):
        return line.split("]: ", 1)[-1]
    return line.removeprefix("error: ")

def missing_program(cmd, cwd=None):
    """The program a server command runs, if it isn't installed (not on PATH, or no such file); else None."""
    exe = next((a for a in cmd if a != "env" and not re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", a)), None)
    if not exe:
        return None
    if "/" in exe:
        return None if os.path.isfile(os.path.join(cwd or "", expand(exe))) else exe
    return None if shutil.which(exe) else exe


def install_hint(program):
    """How to get a server program, for the ones tuieval's templates use."""
    if os.path.basename(program) == "llama-server":
        return ("install llama.cpp (macOS: brew install llama.cpp; others: a release from "
                "https://github.com/ggml-org/llama.cpp/releases) so llama-server is on PATH")
    return "install it, or fix the server's cmd in models.toml"


def probe_models(base_url, timeout=2.0, headers=None):
    """The model ids an OpenAI-compatible server lists at base_url, or None when nothing answers."""
    req = urllib.request.Request(base_url.rstrip("/").removesuffix("/v1") + "/v1/models", headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return [str(x.get("id", "")) for x in json.load(r).get("data", []) if isinstance(x, dict)]
    except (OSError, ValueError, AttributeError):
        return None


# Where local OpenAI-compatible servers usually listen. models.toml [defaults] detect_ports
# replaces the list (e.g. a server on a custom port, or [] to never look).
DETECT_PORTS = {1234: "LM Studio", 11434: "Ollama", 8080: "llama-server", 8000: "vLLM"}


SERVER_NAMES = {1234: "lmstudio", 11434: "ollama", 8080: "llamacpp", 8000: "vllm"}   # for new [servers.*] entries


def detect_ports(cfg=None):
    """Ports to look for running servers on: $TUIEVAL_DETECT_PORTS (comma-separated, empty = none),
    else models.toml [defaults] detect_ports, else the usual ones."""
    env = os.environ.get("TUIEVAL_DETECT_PORTS")
    if env is not None:
        return [int(p) for p in env.replace(" ", "").split(",") if p]
    return [int(p) for p in ((cfg or {}).get("defaults") or {}).get("detect_ports", list(DETECT_PORTS))]


def _norm_url(url):
    return (url or "").rstrip("/").removesuffix("/v1").replace("localhost", "127.0.0.1")


def server_for_url(cfg, url):
    """The models.toml server already pointing at url (a server tuieval doesn't start), or None."""
    return next((n for n, s in cfg["servers"].items() if not s.get("cmd") and _norm_url(s.get("url")) == _norm_url(url)),
                None)


def ensure_server(models_path, url, port=None):
    """The name of a models.toml server for an already-running server at url, adding a
    [servers.<name>] block when there's none (named after the app that usually uses the port)."""
    cfg = load_config(models_path)
    name = server_for_url(cfg, url)
    if name:
        return name
    base = SERVER_NAMES.get(port, f"local-{port}" if port else "local-server")
    name = base if base not in cfg["servers"] else f"{base}-{port}"
    with open(models_path) as f:
        original = f.read()
    with open(models_path, "w") as f:
        f.write(original.rstrip("\n") + f'\n\n[servers.{name}]\nurl = "{_norm_url(url)}"\n')
    try:
        load_config(models_path)
    except BaseException:
        with open(models_path, "w") as f:
            f.write(original)
        raise
    return name


def url_fix(cfg, name):
    """How to point a server at another URL in models.toml (a built-in one has no section there yet)."""
    if name in cfg.get("builtin_servers", ()):
        return f'add [servers.{name}] with url = "http://127.0.0.1:<port>" to models.toml'
    return f"fix `url` under [servers.{name}] in models.toml"


def find_running(cfg, model):
    """Running servers (detect_servers) that list this model id, best match first."""
    found = detect_servers(detect_ports(cfg))
    exact = [f for f in found if model in f["models"]]
    return exact or [f for f in found if serves(f["models"], model)]


def detect_servers(ports, timeout=1.0):
    """[{url, port, usual, models}] for each local port with an OpenAI-compatible server answering.
    usual: the app that normally uses that port (a hint, not a check)."""
    found = []
    for port in ports:
        url = f"http://127.0.0.1:{port}"
        ids = probe_models(url, timeout)
        if ids is not None:
            found.append({"url": url, "port": port, "usual": DETECT_PORTS.get(port), "models": ids})
    return found


def serves(ids, name):
    """Does a server listing these model ids serve this model (names match loosely: aliases, paths)?"""
    want = name.lower()
    return any(want == i.lower() or want in i.lower() or (i and i.lower() in want) for i in ids)


def pack_min_context(pack):
    """Rough context a pack needs: its longest prompt (~4 chars/token) plus room to answer."""
    return max(len(t["input"]) for t in pack.tests) // 4 + 4096


class Cancelled(Exception):
    pass


class ModelFailed(Exception):
    pass


class PackFailed(Exception):
    """The server kept failing on one pack while still running: move on to the model's next pack."""


def holder_text(holder):
    """One line about another tuieval window holding the machine (Engine.machine_lock)."""
    since = f", started {time.strftime('%H:%M', time.localtime(holder['started']))}" if holder.get("started") else ""
    pid = f" (pid {holder['pid']})" if holder.get("pid") else ""
    return (f"another tuieval window is running {holder.get('what', 'an eval')}{pid}{since}. "
            "One eval or tune per machine: this starts when it ends")


class Lane:
    """One model being run: its own request stream, stall reason and skip switch, so models run
    side by side (parallel_models) stop independently. Runs and tunes in one lane use the engine's
    main lane."""
    def __init__(self, label=None):
        self.label = label
        self.stream = None
        self.stall = None         # set by the stall watcher: why this model's server stalled
        self.skip = threading.Event()
        self.procs = []
        self.port = None
        self.seen = set()         # other models served during this model's current load or request


class Engine:
    def __init__(self, on_event=None, root=None, models_path=None, packs_dir=None,
                 results_dir=None, log_dir=None):
        self.root = root = root or workspace.root()
        self.models_path = models_path or os.path.join(root, "models.toml")
        self.packs_dir = packs_dir or os.path.join(root, "packs")
        self.results_dir = results_dir or os.path.join(root, "results")
        self.log_dir = log_dir or os.path.join(root, "logs")
        self.on_event = on_event or (lambda kind, **data: None)
        self.cfg = load_config(self.models_path)
        self.pack_errors = []
        self.packs = packs_mod.load_packs(self.packs_dir, self.pack_errors)
        self._cancel = threading.Event()
        self._main_lane = Lane()
        self._here = threading.local()   # .lane: the lane this thread runs
        self._lanes = {}          # label -> Lane of each model being served now
        self._lanes_lock = threading.Lock()
        self._procs = []
        self._stop_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._run = None          # the run going: {"jobs", "pending", "open", "machine", "cond"} (add_jobs)
        self._header_cache = {}
        self._serving = {}
        self.tuning_dir = os.path.join(root, "tuning")
        self._remote = {}         # label -> entry for models named at run time (openrouter:<id>)
        self._endpoints = {}      # label -> the provider endpoint a hosted model is pinned to
        self._catalogues = {}
        self._add_remote_models()

    # ---- the lane this thread runs (see Lane)
    def _lane(self):
        return getattr(self._here, "lane", None) or self._main_lane

    def _all_lanes(self):
        with self._lanes_lock:
            return [self._main_lane, *self._lanes.values()]

    @property
    def _stream(self):
        return self._lane().stream

    @_stream.setter
    def _stream(self, value):
        self._lane().stream = value

    @property
    def _stall(self):
        return self._lane().stall

    @_stall.setter
    def _stall(self, value):
        self._lane().stall = value

    @property
    def _skip(self):
        return self._lane().skip

    def reload(self):
        self._serving = {}
        self.cfg = load_config(self.models_path)
        self._add_remote_models()
        self.pack_errors = []
        self.packs = packs_mod.load_packs(self.packs_dir, self.pack_errors)

    def emit(self, kind, **data):
        try:
            self.on_event(kind, **data)
        except Exception:  # a UI error must never kill a run
            pass

    def model(self, label):
        return next(m for m in self.cfg["models"] if m["label"] == label)

    # ---- models named at run time: <server>:<model id> for a server with any_model = true
    def resolve(self, spec):
        """A model label, or e.g. openrouter:qwen/qwen3-32b for any model that server serves (no
        models.toml entry needed). Returns the label. Raises ConfigError."""
        spec = spec.strip()
        if any(m["label"] == spec for m in self.cfg["models"]):
            return spec
        name, _, model_id = spec.partition(":")
        server = self.cfg["servers"].get(name)
        if not model_id or not server or not server.get("any_model"):
            any_servers = [n for n, s in self.cfg["servers"].items() if s.get("any_model")]
            raise ConfigError(f"unknown model {spec!r}; see tuieval list"
                              + "".join(f", or name any {n} model as {n}:<model id>" for n in any_servers))
        model_id = model_id.strip()
        catalogue = self.catalogue(name)
        meta = catalogue.get(model_id)
        if meta is None:
            import difflib
            close = difflib.get_close_matches(model_id, list(catalogue), n=5, cutoff=0.5)
            raise ConfigError(f"{name} has no model {model_id!r}"
                              + (f"; did you mean {', '.join(close)}?" if close else ""))
        arch, params = meta.get("architecture") or {}, meta.get("supported_parameters") or []
        ctxs = [c for c in (meta.get("context_length"), (meta.get("top_provider") or {}).get("context_length")) if c]
        entry = {"label": remote_label(name, server, model_id), "server": name, "model": model_id,
                 "served_name": model_id, "vision": "image" in (arch.get("input_modalities") or []),
                 "tags": [name], "remote": True}
        if params and "tools" not in params:
            entry["tools"] = False
        if ctxs:
            entry["max_context"] = min(ctxs)
        self._remote[entry["label"]] = entry
        self._add_remote_models()
        return entry["label"]

    def catalogue(self, name):
        """{model id: metadata} for a server with any_model = true (read once per session)."""
        if name not in self._catalogues:
            try:
                self._catalogues[name] = fetch_catalogue(self.cfg["servers"][name])
            except (OSError, ValueError) as e:
                raise ConfigError(f"couldn't read {name}'s model list ({e})") from None
        return self._catalogues[name]

    def has_results(self):
        """Has any model run anything yet (smoke runs included)?"""
        for base in (self.results_dir, os.path.join(self.results_dir, "smoke")):
            for label in os.listdir(base) if os.path.isdir(base) else []:
                d = os.path.join(base, label)
                if os.path.isdir(d) and any(f.endswith(".json") for f in os.listdir(d)):
                    return True
        return False

    def _add_remote_models(self):
        """Add the models named at run time this session, and ones that have results from an
        earlier session (their result files record the model), to cfg["models"]."""
        known = {m["label"] for m in self.cfg["models"]}
        for base in (self.results_dir, os.path.join(self.results_dir, "smoke")):
            for label in sorted(os.listdir(base)) if os.path.isdir(base) else []:
                if label in known or label in self._remote or not LABEL_RE.match(label):
                    continue
                d = os.path.join(base, label)
                files = sorted(f for f in os.listdir(d) if f.endswith(".json")) if os.path.isdir(d) else []
                data = read_result(os.path.join(d, files[0])) if files else None
                m = (data or {}).get("model") or {}
                if m.get("label") == label and self.cfg["servers"].get(m.get("server"), {}).get("any_model"):
                    self._remote[label] = dict({k: v for k, v in m.items() if v is not None}, remote=True)
        for label, entry in self._remote.items():
            if label not in known:
                m = dict(entry)
                m.setdefault("tags", [])
                m["vision"] = bool(m.get("vision"))
                self.cfg["models"].append(m)
                known.add(label)

    # ---- models hidden from the Setup list (results kept; one label per line, synced with results/)
    def hidden_path(self):
        return os.path.join(self.results_dir, "hidden.txt")

    def hidden(self):
        try:
            with open(self.hidden_path()) as f:
                return {line.strip() for line in f if line.strip() and not line.startswith("#")}
        except FileNotFoundError:
            return set()

    def set_hidden(self, label, hide):
        labels = self.hidden() - {label} | ({label} if hide else set())
        os.makedirs(self.results_dir, exist_ok=True)
        tmp = self.hidden_path() + ".tmp"
        with open(tmp, "w") as f:
            f.write("# Models hidden from the TUI's model list (x toggles). Their results are kept.\n")
            f.writelines(f"{l}\n" for l in sorted(labels))
        os.replace(tmp, self.hidden_path())

    def endpoint(self, m):
        """The provider endpoint a hosted model's answers all come from (servers with
        pin_endpoint = true), chosen once per session; None if there's none to use."""
        server = self.cfg["servers"][m["server"]]
        if not server.get("pin_endpoint"):
            return None
        if m["label"] not in self._endpoints and m.get("model"):
            previous, previous_quant = None, None
            for tier in ("certify", "smoke"):
                d = os.path.dirname(self.result_path(m["label"], "x", tier))
                for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
                    data = self._result(os.path.join(d, f)) if f.endswith(".json") else None
                    before = (data or {}).get("serving") or {}
                    if not previous and before.get("endpoint"):
                        previous, previous_quant = before["endpoint"], before.get("quantization")
            try:
                eps = fetch_endpoints(server, m["model"])
                needs_tools = m.get("tools") is not False and any("tools" in p.needs for p in self.packs.values())
                ep = choose_endpoint(eps, m["model"], effective_sampling(self.cfg, m), needs_tools, previous)
                ep = ep and {"tag": ep["tag"], "provider": ep.get("provider_name"),
                             "quantization": ep.get("quantization") or "unknown"}
            except (OSError, ValueError):
                # offline: assume the endpoint earlier results used (the run itself will need the network)
                ep = {"tag": previous, "provider": None, "quantization": previous_quant} if previous else None
            self._endpoints[m["label"]] = ep
        return self._endpoints.get(m["label"])

    def request_headers(self, m):
        """Extra HTTP headers for a model's requests: the server's `headers`, and its API key."""
        server = self.cfg["servers"][m["server"]]
        headers = dict(server.get("headers", {}))
        if server.get("api_key_env") and os.environ.get(server["api_key_env"]):
            headers["Authorization"] = "Bearer " + os.environ[server["api_key_env"]]
        return headers

    # ---- results status
    def result_path(self, label, pack, tier="certify"):
        return os.path.join(self.results_dir, *(["smoke"] if tier == "smoke" else []), label, f"{pack}.json")

    def _result(self, path):
        """Cached read of a result file (re-read when it changes)."""
        if not os.path.isfile(path):
            return None
        mtime = os.path.getmtime(path)
        cached = self._header_cache.get(path)
        if not cached or cached[0] != mtime:
            cached = (mtime, read_result(path))
            self._header_cache[path] = cached
        return cached[1]

    def records(self, label, pack_name):
        """Stored result rows for a model and pack, or [] (smoke runs excluded)."""
        data = self._result(self.result_path(label, pack_name))
        return data["results"] if data else []

    def certify_repeat(self, pack):
        return pack.certify_repeat or self.cfg["defaults"]["repeat"]

    def default_repeat(self, pack, tier):
        """Repeats when none is chosen: smoke and screen ask each question once."""
        return 1 if tier in ("smoke", "screen") else self.certify_repeat(pack)

    def result_status(self, label, pack_name, tier="certify"):
        """'certified', 'screened', 'partial', 'outdated: <why>', or None."""
        path = self.result_path(label, pack_name, tier)
        if os.path.isfile(path + ".partial.jsonl") and not os.path.isfile(path):
            return "partial"
        if not os.path.isfile(path):
            return None
        data = self._result(path)
        if data is None:
            return "outdated: unreadable or old format"
        pack, m = self.packs.get(pack_name), self.model(label)
        if pack and data["pack"]["fingerprint"] != pack.fingerprint:
            return "outdated: questions changed"
        if not settings_match(data, self.settings(m)):
            before, now = data.get("serving") or {}, self.settings(m)["serving"]
            ran_on = before.get("machine")
            if "endpoint" in now and before.get("endpoint") != now["endpoint"]:
                return ("outdated: ran before providers were pinned (answers could come from any of them)"
                        if not before.get("endpoint")
                        else f"outdated: ran on provider {before['endpoint']}, now pinned to {now['endpoint']}")
            if "quantization" in now and before.get("quantization") != now["quantization"]:
                return (f"outdated: {now['endpoint']} now declares quantization {now['quantization']}, "
                        f"these answers ran on {before.get('quantization')}")
            if ran_on and ran_on != self.machine().id:
                return f"outdated: ran on {ran_on}, and this server's answers depend on the machine"
            if "perf" in before and {**before, "perf": None} == {**now, "perf": None} \
                    and data["settings"] == self.settings(m)["sampling"]:
                return "outdated: memory settings retuned, and this server's answers depend on them"
            return "outdated: " + describe_change(stored_settings(data), self.settings(m))
        if pack is None:
            return "screened"
        keys = {(r["test"], r["repeat"]) for r in data["results"]}
        need = {(t["id"], r) for t in pack.tests for r in range(self.certify_repeat(pack))}
        return "certified" if need <= keys else "screened"

    def _partial_lines(self, path):
        """(header, answers) of an unfinished run's .partial.jsonl, cached like _result;
        (None, []) if there is none."""
        if not os.path.isfile(path):
            return None, []
        mtime = os.path.getmtime(path)
        cached = self._header_cache.get(path)
        if not cached or cached[0] != mtime:
            try:
                with open(path) as f:
                    lines = [json.loads(l) for l in f if l.strip()]
            except (OSError, ValueError):
                lines = []
            cached = (mtime, (lines[0] if lines else None,
                              [r for r in lines[1:] if not client.server_error_row(r)]))
            self._header_cache[path] = cached
        return cached[1]

    def _partial_rows(self, path):
        return self._partial_lines(path)[1]

    def answer_times(self):
        """{label: {pack: [seconds per answer]}} from every stored answer, finished or partial,
        smoke included. Answers timed on this machine are used when a model has any."""
        here = self.machine().id
        out = {}
        for tier in ("certify", "smoke"):
            base = os.path.dirname(os.path.dirname(self.result_path("x", "x", tier)))
            for label in sorted(os.listdir(base)) if os.path.isdir(base) else []:
                for name in self.packs:
                    path = self.result_path(label, name, tier)
                    rows = ((self._result(path) or {}).get("results") or []) + self._partial_rows(path + ".partial.jsonl")
                    out.setdefault(label, {}).setdefault(name, []).extend(
                        (r.get("machine"), r["total_s"]) for r in rows if r.get("total_s"))
        for label, per_pack in out.items():
            local = any(m == here for rows in per_pack.values() for m, _ in rows)
            for name, rows in list(per_pack.items()):
                times = [t for m, t in rows if not local or m in (here, None)]
                if times:
                    per_pack[name] = times
                else:
                    del per_pack[name]
        return {l: p for l, p in out.items() if p}

    def seconds_per_answer(self, jobs, waiting_only=True):
        """({job key: expected seconds per answer}, borrowed, guessed, thin) for the waiting jobs.
        A pack the model has answered uses its own median time there. Otherwise the median of
        other models' times on that pack, scaled by how much slower or faster this model is than
        them on packs both have answered. The sets say which estimates are rough (see estimate_seconds)."""
        times = self.answer_times()
        med = {l: {p: statistics.median(ts) for p, ts in pp.items()} for l, pp in times.items()}
        spr_of, borrowed, guessed, thin = {}, set(), set(), set()
        for j in jobs:
            if j.status in ("done", "skipped", "failed") or (waiting_only and j.status != "waiting"):
                continue
            own = med.get(j.label, {})
            ref = {p: statistics.median(v[p] for l, v in med.items() if p in v and l != j.label)
                   for p in self.packs if any(p in v for l, v in med.items() if l != j.label)}
            if j.pack.name in own:
                spr = own[j.pack.name]
                if len(times[j.label][j.pack.name]) < 5:
                    thin.add(j.key)
            elif j.pack.name in ref:
                shared = [own[p] / ref[p] for p in own if p in ref and ref[p] > 0]
                factor = statistics.median(shared) if shared else 1.0
                spr = ref[j.pack.name] * factor
                borrowed.add((j.key, round(factor, 1) if shared else None, len(shared)))
            else:
                spr = statistics.median(own.values()) if own else DEFAULT_SECONDS_PER_REQUEST
                guessed.add((j.key, bool(own)))
            spr_of[j.key] = spr
        return spr_of, borrowed, guessed, thin

    def estimate_seconds(self, jobs):
        """(seconds, notes): expected run time for the waiting jobs, estimated per pack
        (seconds_per_answer). notes lists what made the estimate rough ([] = solid)."""
        spr_of, borrowed, guessed, thin = self.seconds_per_answer(jobs)
        total = 0.0
        for j in jobs:
            if j.key in spr_of:
                done = len(self._done_keys(j)) if j.merge else 0
                total += max(0, j.total - done) * spr_of[j.key]
        notes = []
        if borrowed:
            factors = sorted({f for _, f, _ in borrowed if f is not None})
            basis = {n for _, f, n in borrowed if f is not None}
            scale = (f", scaled ×{factors[0]} from the model's time on {basis.pop()} other pack(s)"
                     if len(factors) == 1 and len(basis) == 1 else
                     f", scaled ×{factors[0]} for this model" if len(factors) == 1
                     else f", scaled ×{factors[0]}–×{factors[-1]} per model" if factors else "")
            unscaled = any(f is None for _, f, _ in borrowed)
            notes.append(f"{len(borrowed)} pack(s) from other models' times{scale}"
                         + (" (unscaled: model not measured yet)" if unscaled and factors else
                            " (model not measured yet)" if unscaled else ""))
        if thin:
            notes.append(f"{len(thin)} pack(s) measured on under 5 answers")
        if guessed:
            notes.append(f"{len(guessed)} pack(s) never run by any model: "
                         + ("the model's time on its other packs" if all(o for _, o in guessed)
                            else f"{DEFAULT_SECONDS_PER_REQUEST} s per answer") + " assumed")
        return total, notes

    def _done_keys(self, job):
        """(test, repeat) pairs already answered for this job's selection (final file + partial)."""
        wanted = {(t["id"], r) for t in job.tests for r in range(job.repeat)}
        keys = set()
        data = self._result(job.out_path)
        if data and data["pack"]["fingerprint"] == job.pack.fingerprint and \
                settings_match(data, job.settings):
            keys |= {(r["test"], r["repeat"]) for r in data["results"]}
        partial = job.out_path + ".partial.jsonl"
        if os.path.isfile(partial):
            with open(partial) as f:
                lines = [json.loads(l) for l in f if l.strip()]
            if lines and self._header_ok(lines[0], job):
                keys |= {(r["test"], r["repeat"]) for r in lines[1:] if not client.server_error_row(r)}
        if job.fresh:   # a retest asks these again
            keys = {k for k in keys if k[0] not in self._redo_ids(job)}
        return keys & wanted

    def _stored(self, job):
        """(kept, new, earlier) already on disk for a job, read only: rows of its finished result
        file that stay, rows of its unfinished run, and the sittings they took."""
        kept, earlier = [], []
        if job.merge:
            data = self._result(job.out_path)
            if data and data["pack"]["fingerprint"] == job.pack.fingerprint and settings_match(data, job.settings):
                kept = self._kept(job, data["results"])
                if kept:
                    earlier = self._file_sittings(data["run"])
        head, new = self._partial_lines(job.out_path + ".partial.jsonl")
        if not self._header_ok(head, job):
            return kept, [], earlier
        saved = self._saved_sittings(job)
        return kept, new, earlier + saved + self._rebuild_sittings(new, saved + earlier)

    @staticmethod
    def _kept(job, rows):
        """A finished file's rows that stay: all of them, less the picked tests' when redoing them."""
        if not job.redo:
            return rows
        redo = {t["id"] for t in job.tests}
        return [r for r in rows if r["test"] not in redo]

    def _fill_progress(self, job):
        """Set a job's done/passed/failed and earlier sittings from what's stored."""
        kept, new, job.earlier = self._stored(job)
        wanted = {(t["id"], r) for t in job.tests for r in range(job.repeat)}
        rows = {(r["test"], r["repeat"]): r for r in kept + new if (r["test"], r["repeat"]) in wanted}
        job.done = job.resumed = len(rows)
        job.passed = sum(bool(r["pass"]) for r in rows.values())
        job.failed = job.done - job.passed

    def _partial_header(self, job):
        return {"pack_fingerprint": job.pack.fingerprint, "settings_fingerprint": settings_fingerprint(job.settings),
                "settings": job.settings}

    def _header_ok(self, head, job):
        return header_matches(head, job.pack.fingerprint, job.settings)

    def _discards(self, job):
        """[(what, answers, why)]: earlier answers a start of this job would set aside because the
        questions or settings changed (a finished result file, an unfinished run, or both)."""
        out = []
        if job.merge:
            data = self._result(job.out_path)
            if data and data["results"]:
                if data["pack"]["fingerprint"] != job.pack.fingerprint:
                    out.append(("finished", len(data["results"]), "questions changed"))
                elif not settings_match(data, job.settings):
                    out.append(("finished", len(data["results"]), describe_change(stored_settings(data), job.settings)))
        head, rows = self._partial_lines(job.out_path + ".partial.jsonl")
        if head is not None and rows and not self._header_ok(head, job):
            why = ("questions changed" if head.get("pack_fingerprint") != job.pack.fingerprint
                   else describe_change(head.get("settings"), job.settings))
            out.append(("unfinished", len(rows), why))
        return out

    def set_aside(self, job):
        """(finished, unfinished): earlier answers a retest of this job asks again, so they move to
        history/ and stop counting: in its result file and its unfinished run, under the job's
        questions and settings (only the picked tests' when it redoes those)."""
        redo = self._redo_ids(job)
        data = self._result(job.out_path)
        finished = 0
        if data and data["pack"]["fingerprint"] == job.pack.fingerprint and settings_match(data, job.settings):
            finished = sum(r["test"] in redo for r in data["results"])
        head, rows = self._partial_lines(job.out_path + ".partial.jsonl")
        unfinished = sum(r["test"] in redo and not client.server_error_row(r) for r in rows) \
            if self._header_ok(head, job) else 0
        return finished, unfinished

    def superseded(self, path):
        """Is this result file about to be replaced? True when it no longer matches the current
        questions or settings and an unfinished run of the same pack does."""
        label, name = os.path.basename(os.path.dirname(path)), os.path.basename(path)[:-len(".json")]
        head, _ = self._partial_lines(path + ".partial.jsonl")
        data, pack = self._result(path), self.packs.get(name)
        if head is None or data is None or pack is None:
            return False
        try:
            settings = self.settings(self.model(label))
        except Exception:   # a model no longer in models.toml: nothing is replacing its results
            return False
        current = data["pack"]["fingerprint"] == pack.fingerprint and settings_match(data, settings)
        return not current and header_matches(head, pack.fingerprint, settings)

    # ---- planning
    def plan(self, labels, pack_names, repeat=None, tier="screen", force=False, tests=None):
        """Jobs for models x packs. tier: smoke (3 tests), screen (a spread sample), certify (all
        tests). repeat: times each question is asked; None = the tier default (1 for smoke and
        screen, the pack's certification repeats for certify). tests: {pack: [test ids]} runs just
        those tests of a pack instead of the tier's sample (unknown ids raise PackError). Earlier
        matching results are kept and only missing answers are run, unless force; force with
        picked tests reruns those and keeps the pack's other answers. force is True, False, or the
        keys ("label/pack") of the jobs to retest, the others resuming."""
        if tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}")
        jobs = []
        for label in labels:
            m = self.model(label)
            for name in pack_names:
                pack = self.packs[name]
                reps = repeat or self.default_repeat(pack, tier)
                picked = (tests or {}).get(name)
                retest = force if isinstance(force, bool) else f"{label}/{name}" in force
                job = Job(m, pack, reps, pack.pick(picked) if picked else pack.select(tier),
                          self.result_path(label, name, tier), effective_sampling(self.cfg, m), tier=tier,
                          merge=not retest or bool(picked), redo=retest and bool(picked), fresh=retest,
                          settings=self.settings(m))
                sv = self.serving(m)
                status = self.result_status(label, name, tier)
                job.discards = self._discards(job)
                missing = [n for n in pack.modules if not importlib.util.find_spec(n)]
                if "vision" in pack.needs and not m["vision"]:
                    job.status, job.note = "skipped", "no vision"
                elif "tools" in pack.needs and m.get("tools") is False:
                    job.status, job.note = "skipped", "no tool calling"
                elif not sv.fits:
                    job.status, job.note = "skipped", sv.fit_note
                elif (sv.ctx or 10**9) < pack_min_context(pack):
                    job.status, job.note = "skipped", \
                        f"needs ~{pack_min_context(pack) // 1024}k context, {sv.ctx // 1024}k fits on {sv.machine.id}"
                elif missing:
                    job.status, job.note = "skipped", f"{', '.join(missing)} not installed in tuieval's Python"
                elif status and status.startswith("outdated"):
                    job.merge, job.redo, job.note = False, False, status.split(": ", 1)[-1] + ", rerunning"
                    if not retest:   # a rerun under the new settings already under way resumes
                        self._fill_progress(job)
                        if job.done:
                            job.note += f" ({job.done} of {job.total} done)"
                elif not retest:
                    self._fill_progress(job)
                    if job.done >= job.total:
                        job.status, job.note = "done", "done earlier"
                    elif job.done:
                        job.note = f"{job.done} of {job.total} done earlier, running the rest"
                if job.status != "waiting":
                    job.discards = []
                for what, n, why in job.discards:
                    if what == "unfinished":   # (a finished file's reason is already the note)
                        job.note = "; ".join(filter(None, [job.note, f"{n} unfinished answers set aside ({why})"]))
                jobs.append(job)
        return jobs

    # ---- serving on this machine
    def machine(self):
        """This machine, with its models.toml [machines.<id>] gpu_memory_gb applied. A Linux machine
        that recorded itself before its GPU was detected keeps that id, so its results and tuning
        still match (EVALS_MACHINE renames it)."""
        mc = machines.detect()
        if mc.discrete and not os.environ.get("EVALS_MACHINE"):
            old = machines.machine_id(mc.chip, mc.ram_gb)
            if os.path.isdir(os.path.join(self.tuning_dir, old)) and \
                    not os.path.isdir(os.path.join(self.tuning_dir, mc.id)):
                mc = dataclasses.replace(mc, id=old)
        override = self.cfg.get("machines", {}).get(mc.id, {}).get("gpu_memory_gb")
        if override:
            mc = dataclasses.replace(mc, gpu_limit_gb=float(override))
        return mc

    def machines(self):
        """This machine first, then every other machine recorded in tuning/ (synced)."""
        here = self.machine()
        try:
            profiles.save_machine(here, self.tuning_dir)
        except OSError:
            pass
        return [here] + [mc for mc in profiles.known_machines(self.tuning_dir) if mc.id != here.id]

    def machine_settings(self, machine_id=None):
        """models.toml [machines.<id>] (optional): memory_headroom_gb, gpu_memory_gb, name."""
        return self.cfg.get("machines", {}).get(machine_id or self.machine().id, {})

    def headroom_gb(self, mc):
        """GPU memory kept free: memory_headroom_gb, else 4 GB on a Mac (macOS and other apps share
        it) and 1.5 GB in a discrete GPU's VRAM."""
        return float(self.machine_settings(mc.id).get("memory_headroom_gb", 1.5 if mc.discrete else 4.0))

    def parallel_models(self):
        """How many models a run serves at a time on this machine: models.toml [machines.<id>]
        parallel_models (default 1)."""
        try:
            return max(1, int(self.machine_settings().get("parallel_models", 1)))
        except (TypeError, ValueError):
            return 1

    def memory_need_gb(self, m):
        """GPU memory a model takes while served here: 0 for one we don't start (a hosted API or a
        server already running), the fit check's estimate for a local GGUF, else None (unknown)."""
        if not self.cfg["servers"][m["server"]].get("cmd"):
            return 0.0
        return self.serving(m).need_gb

    def memory_available_gb(self):
        """GPU memory models may take here (what the fit check sizes contexts against)."""
        mc = self.machine()
        return mc.gpu_limit_gb - self.headroom_gb(mc)

    def runs_alone(self, m):
        """A server whose before_start step may stop other servers never runs next to another model."""
        return bool(self.cfg["servers"][m["server"]].get("before_start"))

    def gpu_residency_gb(self, machine=None):
        """GPU memory this machine's driver keeps resident without churning (see machines.py)."""
        mc = machine or self.machine()
        return machines.gpu_residency_gb(mc, self.machine_settings(mc.id).get("gpu_residency_gb"))

    def knob_values(self, machine=None):
        """Placeholders for commands and tune options: {p} P-cores, {p_minus_2}, {all} cores, and
        {gpu_safe_gb} = GPU residency limit minus 2 GB (also _minus_1, _minus_2, _plus_1)."""
        mc = machine or self.machine()
        safe = int(self.gpu_residency_gb(mc) - 2)
        return {"p": mc.p_cores, "p_minus_2": max(1, mc.p_cores - 2), "all": mc.p_cores + mc.e_cores,
                "gpu_safe_gb": safe, "gpu_safe_gb_minus_1": safe - 1, "gpu_safe_gb_minus_2": safe - 2,
                "gpu_safe_gb_plus_1": safe + 1}

    def default_knob_args(self, server, machine=None):
        """The first option of every tune knob: what an untuned model runs with."""
        vals = self.knob_values(machine)
        return [a.format(**vals) for opts in server.get("tune", {}).values() if opts for a in opts[0]]

    def serving(self, m, machine=None):
        """Serving for a model on this machine: context from the fit check, speed flags from its
        tuned profile (or the defaults), and the output-affecting identity."""
        mc = machine or self.machine()
        key = (m["label"], mc.id)
        if key in self._serving:
            return self._serving[key]
        server = self.cfg["servers"][m["server"]]
        kv = m.get("kv_type") or server.get("kv_type")
        want = min(x for x in (server.get("max_ctx"), m.get("ctx"), m.get("max_context"), 10**9) if x)
        want = None if want == 10**9 else want
        ctx, fits, note, mtp, size, need = want, True, "", 0, None, None
        path = expand(m["model"])
        if server.get("model_is_path") and os.path.isfile(path):
            try:
                info = machines.read_gguf(path)
                mtp, size = info["mtp_layers"], info["bytes"]
                if fit_check(server):
                    mmproj = expand(m.get("mmproj", ""))
                    extra = os.path.getsize(mmproj) if mmproj and os.path.isfile(mmproj) else 0
                    f = machines.fit(path, mc, kv_type=kv or "f16", headroom_gb=self.headroom_gb(mc), extra_bytes=extra,
                                     want_ctx=want)
                    ctx, fits, note, need = f.max_ctx, f.fits, f.note, f.need_gb
            except (OSError, ValueError) as e:
                note = f"couldn't read the GGUF header ({e}); context not checked"
        profile, perf, source = None, [], "n/a"
        if server.get("cmd") and (server.get("perf") or server.get("tune")):
            perf = list(server.get("perf", []))
            profile = profiles.load(mc.id, m["label"], self.tuning_dir)
            if profile:
                perf += list(profile["args"])
                why = profiles.stale(profile, self.server_version(m), size)
                source = f"outdated: {why}" if why else \
                    ("seeded" if profile.get("meta", {}).get("method") == "seeded" else "tuned")
            else:
                perf += self.default_knob_args(server, mc)
                source = "untuned"
        flag = server.get("ctx_flag")      # e.g. --max-context-tokens, when the server takes one
        if flag and ctx is None:
            args = list(m.get("server_args", [])) + perf
            if flag in args[:-1]:
                try:
                    ctx = int(args[args.index(flag) + 1])
                except ValueError:
                    pass
        identity = self._identity(m, server, kv, size)
        if server.get("outputs_depend_on_machine"):
            # Answers depend on the machine and on its memory settings (e.g. which experts stay
            # resident), so both are part of what a result is valid for.
            identity["machine"] = mc.id
            identity["perf"] = perf
        sv = Serving(mc, ctx, kv, fits, note, perf, source, identity, profile, mtp, need)
        self._serving[key] = sv
        return sv

    def _identity(self, m, server, kv, size):
        """Output-affecting serving settings. Paths, port, alias and context size are left out so
        results carry over between machines and folders (context only has to fit each pack)."""
        if not server.get("cmd"):
            ident = {"server": m["server"], "url": server.get("url")}
            if server.get("pin_endpoint"):
                ep = self.endpoint(m) or {}
                # the provider's declared quantization ("unknown" when it doesn't say) is part of what a
                # result is valid for: if the provider changes it, earlier answers show as outdated
                ident["endpoint"], ident["quantization"] = ep.get("tag"), ep.get("quantization")
            return ident
        values = {"model": os.path.basename(m["model"]), "served_name": "<name>", "port": "<port>", "ctx": "<ctx>",
                  "kv_type": kv or "", "mmproj": os.path.basename(m.get("mmproj", ""))}
        cmd = [a.format(**values) for a in server["cmd"]]
        if m.get("mmproj"):
            cmd += [a.format(**values) for a in server.get("vision_args", [])]
        return {"server": m["server"], "cmd": cmd, "server_args": list(m.get("server_args", [])),
                "model_bytes": size}

    def settings(self, m):
        """What a result's settings fingerprint covers: request sampling plus serving identity."""
        return {"sampling": effective_sampling(self.cfg, m), "serving": self.serving(m).identity}

    def server_command(self, m, perf_args=None, port=None):
        """(cmd or None, cwd, base_url) for a model. cmd is None for an already-running server.
        perf_args replaces the speed flags (the tuner uses this); port overrides the server's."""
        server = self.cfg["servers"][m["server"]]
        if not server.get("cmd"):
            return None, None, server["url"].rstrip("/").removesuffix("/v1")
        sv = self.serving(m)
        port = port or server["port"]
        values = {"model": expand(m["model"]), "served_name": m["served_name"], "port": port,
                  "mmproj": expand(m.get("mmproj", "")), "ctx": sv.ctx or "", "kv_type": sv.kv_type or "",
                  "root": self.root, **self.knob_values(sv.machine)}
        cmd = [expand(a, **values) for a in server["cmd"]]
        cmd += [expand(a, **values) for a in (sv.perf_args if perf_args is None else perf_args)]
        if m.get("mmproj"):
            cmd += [expand(a, **values) for a in server.get("vision_args", [])]
        cmd += [expand(a, **values) for a in m.get("server_args", [])]
        cmd, env = split_env(cmd)
        env = [f"{k}={expand(str(v), **values)}" for k, v in server.get("env", {}).items()] + env
        if env:   # env(1) keeps the command one copy-pasteable line and one process group
            cmd = ["env", *env, *cmd]
        cwd = expand(server["cwd"]) if server.get("cwd") else None
        return cmd, cwd, f"http://127.0.0.1:{port}"

    def server_version(self, m):
        """The server software's version (e.g. the llama.cpp build), or None."""
        vc = self.cfg["servers"][m["server"]].get("version_cmd")
        if not vc:
            return None
        key = ("version", m["server"])
        if key not in self._serving:
            try:
                out = subprocess.run([expand(a) for a in vc], capture_output=True, text=True, timeout=15)
                lines = (out.stdout + out.stderr).strip().splitlines()
                self._serving[key] = next((l.strip() for l in lines if re.search(r"\d", l)), None)
            except (OSError, subprocess.TimeoutExpired):
                self._serving[key] = None
        return self._serving[key]

    # ---- control
    def cancel(self):
        self._cancel.set()
        self._abort(self._all_lanes(), list(self._procs))

    def skip_model(self, label=None):
        """Skip a model being run (label), or every one being run now."""
        with self._lanes_lock:
            lanes = [self._lanes[label]] if label in self._lanes else list(self._lanes.values()) or [self._main_lane]
        for lane in lanes:
            lane.skip.set()
        self._abort(lanes, [p for lane in lanes for p in lane.procs] if self._lanes else list(self._procs))

    def _abort(self, lanes, procs):
        for lane in lanes:
            if lane.stream:
                lane.stream.close()
        for p in procs:
            self._stop(p, grace=10)

    def _check(self):
        if self._cancel.is_set():
            raise Cancelled()
        if self._skip.is_set():
            raise ModelFailed("skipped by user")
        if self._stall:
            raise ModelFailed(self._stall)

    SERVER_RETRY_WAITS_S = (5, 10, 20, 30, 60, 60)   # then that pack fails (about 3 minutes)
    PACKS_FAILED_STOP = 2   # packs failing that way in a row: the server is broken, stop the model

    STALL_WINDOW_S = 60        # judge the kernel share over this long
    STALL_KERNEL_SHARE = 0.70  # a healthy GPU server spends ~35% in the kernel; stalled ones 70-95%
    STALL_MIN_BUSY = 0.5       # and the server must be using at least half a core

    def _watch_stall(self, proc, m, stop, info):
        """While a server serves, sample its process group's own vs kernel CPU time and the
        system's GPU allocation. If the kernel share stays above STALL_KERNEL_SHARE for a full
        STALL_WINDOW_S, the GPU driver is churning memory: record why and abort the request."""
        mc = self.machine()
        limit = self.gpu_residency_gb(mc)
        lane = self._lane()
        samples = []   # (time, user, kernel)

        def run():
            while not stop.wait(5):
                cpu = machines.tree_cpu_seconds(proc.pid)
                alloc = machines.gpu_allocated_gb()
                if alloc is not None:
                    info["gpu_peak_gb"] = round(max(info.get("gpu_peak_gb") or 0, alloc), 2)
                if cpu is None:
                    continue
                now = time.time()
                samples.append((now, *cpu))
                # keep the newest sample that is at least a full window old, so the span is >= the window
                while len(samples) > 2 and now - samples[1][0] >= self.STALL_WINDOW_S:
                    samples.pop(0)
                t0, u0, k0 = samples[0]
                du, dk, dt = cpu[0] - u0, cpu[1] - k0, now - t0
                if dt < self.STALL_WINDOW_S or du + dk <= 0:
                    continue
                share = dk / (du + dk)
                info["kernel_share_max"] = max(info.get("kernel_share_max") or 0, round(share, 2))
                if share > self.STALL_KERNEL_SHARE and (du + dk) / dt > self.STALL_MIN_BUSY and not lane.stall:
                    over = f"{alloc:.1f} GB of GPU memory allocated; " if alloc else ""
                    lane.stall = (f"{m['label']} stalled in the GPU driver on {mc.id}: {100 * share:.0f}% of its "
                                   f"CPU time went to the kernel for {self.STALL_WINDOW_S}s while the GPU waited "
                                   f"({over}this machine keeps ~{limit:.0f} GB resident). "
                                   "Lower its GPU memory or retune it.")
                    self.emit("server_stall", label=m["label"], message=lane.stall)
                    if lane.stream:
                        lane.stream.close()
        threading.Thread(target=run, daemon=True).start()

    # ---- processes
    def _start(self, cmd, log_path, cwd=None):
        out = open(log_path, "ab")
        p = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.DEVNULL, stdout=out,
                             stderr=subprocess.STDOUT, start_new_session=True)
        self._procs.append(p)
        self._lane().procs.append(p)
        return p

    def _stop(self, p, grace=30):
        """Stop a process group. Safe to call twice and from two threads (cancel races shutdown)."""
        if p is None:
            return
        with self._stop_lock:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                    p.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                    p.wait()
                except (ProcessLookupError, PermissionError):  # already exiting (macOS says EPERM)
                    p.wait()
            if p in self._procs:
                self._procs.remove(p)
            for lane in self._all_lanes():
                if p in lane.procs:
                    lane.procs.remove(p)

    def _tail(self, path, label, stop):
        def run():
            with open(path, errors="replace") as f:
                f.seek(0, os.SEEK_END)
                while not stop.is_set():
                    line = f.readline()
                    if line:
                        self.emit("server_log", label=label, line=line.rstrip("\n"))
                    else:
                        time.sleep(0.2)
        threading.Thread(target=run, daemon=True).start()

    VRAM_EVERY_S = 6   # how often to read a discrete GPU's VRAM while a server runs

    def _watch_memory(self, proc, label, stop, peak, info=None):
        """Peak resident memory of the server's process group (MB). On Apple Silicon this includes
        the model weights mapped for the GPU. With discrete GPUs, also the peak VRAM it holds
        (info["vram_peak_gb"]): its own processes' on Nvidia, the machine's on AMD."""
        discrete = info is not None and self.machine().discrete
        last_vram = [0.0]

        def run():
            while not stop.is_set():
                if discrete and time.time() - last_vram[0] >= self.VRAM_EVERY_S:
                    last_vram[0] = time.time()
                    used = machines.vram_used_gb(machines.tree_pids(proc.pid))
                    if used:
                        info["vram_peak_gb"] = round(max(info.get("vram_peak_gb") or 0, used), 2)
                try:
                    out = subprocess.run(["ps", "-axo", "pgid=,rss="], capture_output=True, text=True).stdout
                    rss = sum(int(r) for g, r in (l.split() for l in out.splitlines() if l.strip())
                              if int(g) == proc.pid) / 1024
                    if rss > peak[0]:
                        peak[0] = rss
                        self.emit("resource", label=label, rss_mb=rss)
                except (OSError, ValueError):
                    pass
                stop.wait(2)
        threading.Thread(target=run, daemon=True).start()

    # ---- one eval or tune per machine, across terminals
    @property
    def machine_lock_path(self):
        return os.path.join(self.log_dir, "machine.lock")

    def machine_holder(self):
        """What holds the machine lock right now ({pid, what, started}), or None if it's free."""
        try:
            with open(self.machine_lock_path) as f:
                try:
                    fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    return None              # nobody holds it
                except BlockingIOError:
                    return json.loads(f.read() or "{}") or {"what": "another eval"}
        except (OSError, ValueError):
            return None

    @contextlib.contextmanager
    def machine_lock(self, what):
        """Hold the machine for a run or tune: a second tuieval window (TUI or CLI) waits here until
        the first one finishes, instead of starting servers next to it. flock is local to this Mac,
        so the synced project folder never blocks another machine; it's released if the process dies.
        Emits machine_busy (once) while waiting and machine_free when it gets the lock."""
        os.makedirs(self.log_dir, exist_ok=True)
        f = open(self.machine_lock_path, "a+")
        try:
            waited = False
            while True:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if not waited:
                        waited = True
                        try:
                            f.seek(0)
                            holder = json.loads(f.read() or "{}")
                        except ValueError:
                            holder = {}
                        self.emit("machine_busy", holder=holder or {"what": "another eval"})
                    if self._cancel.wait(1.0):
                        raise Cancelled()
            if waited:
                self.emit("machine_free")
            f.seek(0)
            f.truncate()
            f.write(json.dumps({"pid": os.getpid(), "what": what, "started": time.time()}))
            f.flush()
            yield
        finally:
            try:
                f.truncate(0)
            except OSError:
                pass
            f.close()                        # closing releases the lock

    def needs_machine(self, jobs):
        """Only local servers compete for this Mac; a hosted-only run (OpenRouter) doesn't wait."""
        return any(self.cfg["servers"][j.model["server"]].get("cmd") for j in jobs if j.status == "waiting")

    # ---- running
    def run(self, jobs, parallel=None):
        """Run the jobs (blocking). Jobs of the same model share one server start. parallel: how many
        models are served at a time (default: parallel_models for this machine). add_jobs() can add
        more while it runs; they run after the ones before them."""
        self._cancel.clear()
        self.emit("queue_started", jobs=jobs)
        models = sorted({j.label for j in jobs if j.status == "waiting"})
        machine = bool(jobs) and self.needs_machine(jobs)
        lock = (self.machine_lock(f"{jobs[0].tier} run: {', '.join(models[:3])}"
                                  + (f" +{len(models) - 3}" if len(models) > 3 else ""))
                if machine else contextlib.nullcontext())
        order = []
        for j in jobs:
            if j.label not in order:
                order.append(j.label)
        groups = [(label, [j for j in jobs if j.label == label and j.status == "waiting"]) for label in order]
        with self._run_lock:
            self._run = {"jobs": jobs, "pending": [g for g in groups if g[1]], "open": True, "machine": machine,
                         "cond": threading.Condition(self._run_lock)}
        try:
            with lock:
                self._run_lanes(max(1, int(parallel or self.parallel_models())))
            if self._cancel.is_set():
                raise Cancelled()
        except Cancelled:
            with self._run_lock:
                self._run["open"] = False
            for j in jobs:
                if j.status in ("waiting", "loading", "running"):
                    j.status, j.note = "skipped", "cancelled (progress kept)" if j.done else "cancelled"
                    self.emit("job_done", job=j)
        finally:
            with self._run_lock:
                self._run["open"] = False
            changes = []
            if any(j.status == "done" and j.started and j.tier != "smoke" for j in jobs):
                try:
                    from . import verdict
                    changes = verdict.record_history(self, sorted({j.label for j in jobs}))
                except Exception as e:  # history must never break a run
                    self.emit("proxy_error", message=f"could not record verdict history: {e!r}")
            if changes:
                self.emit("verdicts", changes=changes)
            self.emit("queue_done", jobs=jobs, cancelled=self._cancel.is_set())

    def add_jobs(self, new):
        """Add planned jobs to the run going on, after its other jobs. Returns (added, why not): a
        pack already waiting or running in it isn't added twice, and nothing is added once the run is
        finishing or cancelled, or when it holds no machine lock (hosted only) and these need one.
        Runs on the caller's thread, so the caller shows what was added (no event)."""
        with self._run_lock:
            r = self._run
            if not r or not r["open"] or self._cancel.is_set():
                return [], "the run is finishing"
            if not r["machine"] and self.needs_machine(new):
                return [], "the run going uses only hosted models, and these need this machine"
            busy = {j.key for j in r["jobs"] if j.status in ("waiting", "loading", "running")}
            added = [j for j in new if j.status == "waiting" and j.key not in busy]
            r["jobs"].extend(added)
            for j in added:   # a model still waiting takes them on; otherwise it comes again at the end
                group = next((g for g in r["pending"] if g[0] == j.label), None)
                if group:
                    group[1].append(j)
                else:
                    r["pending"].append((j.label, [j]))
            r["cond"].notify_all()
        return added, None if added else "everything picked is already in the run"

    def _run_lanes(self, parallel):
        """Run each model's jobs in its own lane, up to `parallel` at a time, in order: the next
        model starts once a lane is free, nothing running stops it (runs_alone), and its memory fits
        next to the ones running (memory_need_gb; an unknown need is left to the setting). Lanes
        wait for jobs added meanwhile (add_jobs) until every model is done."""
        pending, running, told = self._run["pending"], {}, set()
        cond = self._run["cond"]

        def blocked(label):
            """Why the next model can't start now (None: it can)."""
            if not running:
                return None
            if len(running) >= parallel:
                return ""
            m = self.model(label)
            if self.runs_alone(m) or any(self.runs_alone(self.model(l)) for l in running):
                alone = label if self.runs_alone(m) else next(l for l in running if self.runs_alone(self.model(l)))
                return (f"{alone}'s server runs a before_start step that may stop other servers, so it runs "
                        "on its own")
            need = self.memory_need_gb(m)
            used = sum(n for n in running.values() if n)
            room = self.memory_available_gb()
            if need and used + need > room:
                return (f"needs ~{need:.0f} GB; {', '.join(running)} take ~{used:.0f} of {room:.0f} GB, so it "
                        "starts when there's room")
            return None

        def lane():
            while True:
                mine, note = None, None
                with cond:   # events go out after the lock is released: the UI may be waiting for it (add_jobs)
                    if self._cancel.is_set() or not pending and not running:
                        self._run["open"] = False   # nothing more can be added
                        cond.notify_all()
                        return
                    # in order, except that a model added again while it runs waits for itself to end
                    # without holding up the ones behind it
                    nxt = next((i for i, g in enumerate(pending) if g[0] not in running), None)
                    label = pending[nxt][0] if nxt is not None else None
                    why = blocked(label) if label else ""   # none can start: a model still runs, more may come
                    if why is None:
                        label, mine = pending.pop(nxt)
                        running[label] = self.memory_need_gb(self.model(label))
                    elif why and label not in told:
                        told.add(label)
                        note = f"{label} waits: {why}"
                    else:
                        cond.wait(1.0)
                if note:
                    self.emit("model_waiting", label=label, message=note)
                if mine is None:
                    continue
                try:
                    self._run_lane(label, mine)
                finally:
                    with cond:
                        running.pop(label, None)
                        cond.notify_all()

        if parallel == 1:
            lane()
            return
        threads = [threading.Thread(target=lane, name=f"lane-{i + 1}") for i in range(parallel)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def _run_lane(self, label, mine):
        """One model's jobs, in this thread's own lane. Other models served meanwhile are noted on
        each answer (ran_alongside), since they share the machine's GPU and memory bandwidth."""
        lane = Lane(label)
        m = mine[0].model
        with self._lanes_lock:
            for other in self._lanes.values():
                other.seen.add(label)
            lane.seen = set(self._lanes)
            if self.cfg["servers"][m["server"]].get("cmd"):
                lane.port = self._lane_port(m)
            self._lanes[label] = lane
        self._here.lane = lane
        try:
            try:
                self._run_model(mine)
            except (Cancelled, ModelFailed):
                raise
            except Exception as e:  # never let one model's surprise kill the whole queue
                raise ModelFailed(f"unexpected error: {e!r}") from e
        except ModelFailed as e:
            for j in mine:
                if j.status in ("waiting", "loading", "running"):
                    j.status = "skipped" if "by user" in str(e) else "failed"
                    j.note, j.finished = str(e).splitlines()[0], time.time()
                    self.emit("job_done", job=j)
            self.emit("model_failed", label=label, message=str(e))
        except Cancelled:
            pass   # run() marks what's left
        finally:
            self._here.lane = None
            with self._lanes_lock:
                self._lanes.pop(label, None)

    def _lane_port(self, m):
        """The server's port, or a free one when another model being served has it (call with
        _lanes_lock held)."""
        port = self.cfg["servers"][m["server"]].get("port")
        if port is None or not any(l.port == port for l in self._lanes.values()):
            return port
        taken = {l.port for l in self._lanes.values()}
        while True:
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                free = s.getsockname()[1]
            if free not in taken:
                return free

    @contextlib.contextmanager
    def serve(self, m, perf_args=None, port=None, log_name=None):
        """Start a model's server (or use the running one), wait until it serves the expected name,
        yield (base_url, info), and stop it afterwards. info: load_s, command, alive(), peak_mb()."""
        defaults = self.cfg["defaults"]
        server = self.cfg["servers"][m["server"]]
        cmd, cwd, base_url = self.server_command(m, perf_args=perf_args, port=port)
        port = port or server.get("port")
        srv, stop, peak = None, threading.Event(), [0.0]
        info = {"load_s": None, "command": cmd, "alive": lambda: srv is None or srv.poll() is None,
                "peak_mb": lambda: round(peak[0]) or None, "facts": {}}
        log, log_start = None, 0
        try:
            if cmd is not None:
                model_path = expand(m["model"])
                if server.get("model_is_path") and not os.path.exists(model_path):
                    raise ModelFailed(f"model not found: {model_path}")
                if m.get("mmproj") and not os.path.isfile(expand(m["mmproj"])):
                    raise ModelFailed(f"mmproj file not found: {expand(m['mmproj'])}")
                if cwd and not os.path.isdir(cwd):
                    raise ModelFailed(f"server directory not found: {cwd}")
                missing = missing_program(cmd, cwd)
                if missing:
                    raise ModelFailed(f"{missing} isn't installed: {install_hint(missing)}")
                if port_in_use(port):
                    raise ModelFailed(f"port {port} is already in use; stop the server running there")
                os.makedirs(os.path.join(self.log_dir, "server"), exist_ok=True)
                log = os.path.join(self.log_dir, "server", f"{log_name or m['label']}.log")
                open(log, "ab").close()
                log_start = os.path.getsize(log)
                self.emit("model_loading", label=m["label"], command=cmd, log=log)
                self._tail(log, m["label"], stop)
                if server.get("before_start"):
                    # e.g. free memory another server of the same machine holds
                    pre = [expand(a) for a in server["before_start"]]
                    with open(log, "ab") as out:
                        done = subprocess.run(pre, cwd=cwd, stdin=subprocess.DEVNULL, stdout=out,
                                              stderr=subprocess.STDOUT, timeout=300)
                    if done.returncode:
                        raise ModelFailed(f"{' '.join(pre)} failed with code {done.returncode} "
                                          f"(see logs/server/{log_name or m['label']}.log)")
                t0 = time.time()
                srv = self._start(cmd, log, cwd=cwd)
                self._watch_memory(srv, m["label"], stop, peak, info)
            else:
                key_env = server.get("api_key_env")
                if key_env and not os.environ.get(key_env):
                    raise ModelFailed(f"{key_env} is not set; export it in the shell you start tuieval from")
                where = f"using the server at {base_url}"
                if server.get("pin_endpoint"):
                    ep = self.endpoint(m)
                    if not ep or not ep.get("tag"):
                        raise ModelFailed(f"couldn't read {m['model']}'s provider endpoints to pin one; "
                                          "not running on a mix of providers")
                    where = f"{base_url}, every answer from provider endpoint {ep['tag']} ({ep['quantization']})"
                self.emit("model_loading", label=m["label"], command=[f"({where})"], log="")
                t0 = time.time()
            deadline = time.time() + defaults["ready_timeout_s"]
            if srv is None and not server.get("health") and probe_models(base_url, 5, self.request_headers(m)) is None:
                # a server we don't start: if nothing answers now, waiting won't help
                time.sleep(3)
                if probe_models(base_url, 5, self.request_headers(m)) is None:
                    raise ModelFailed(
                        f"nothing is answering at {base_url}. Start that server, or {url_fix(self.cfg, m['server'])} "
                        "(common local ports: LM Studio 1234, Ollama 11434, llama-server 8080, vLLM 8000); "
                        "`tuieval doctor` checks them all")
            while True:
                self._check()
                if srv is not None and srv.poll() is not None:
                    reason = server_error(log, log_start) if log else None
                    raise ModelFailed(f"server exited with code {srv.returncode} while loading"
                                      + (f": {reason}" if reason else
                                         f" (see logs/server/{log_name or m['label']}.log)"))
                try:
                    if server.get("health"):  # servers without /v1/models
                        with urllib.request.urlopen(base_url + server["health"], timeout=5) as r:
                            ids = [str(json.load(r).get("model", ""))]
                    else:
                        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5) as r:
                            ids = [x.get("id", "") for x in json.load(r).get("data", [])]
                    break
                except (OSError, ValueError, AttributeError):
                    pass
                if time.time() > deadline:
                    raise ModelFailed(f"server not ready after {defaults['ready_timeout_s']}s")
                time.sleep(0.5)
            info["load_s"] = time.time() - t0 if srv is not None else None
            # A health-checked server we started ourselves serves exactly the model we gave it.
            if not (server.get("health") and srv is not None) and not serves(ids, m["served_name"]):
                raise ModelFailed(f"server reports {ids}, expected {m['served_name']!r}; "
                                  "not running to avoid mislabelled results")
            if log and server.get("log_facts"):
                info["facts"] = read_log_facts(log, log_start, server["log_facts"])
                if info["facts"]:
                    self.emit("server_facts", label=m["label"], facts=info["facts"])
            for name, pattern in server.get("require_facts", {}).items():
                got = info["facts"].get(name)
                if got is None or not re.search(pattern, str(got)):
                    raise ModelFailed(f"{m['label']}: its startup log shows {name} = {got!r}, not {pattern!r} "
                                      f"(require_facts in [servers.{m['server']}]); not running, so no answers "
                                      "are recorded under the wrong conditions")
            self.emit("model_ready", label=m["label"], ids=ids, load_s=info["load_s"])
            self._stall = None
            if srv is not None and server.get("stall_guard"):
                self._watch_stall(srv, m, stop, info)
            yield base_url, info
        finally:
            stop.set()
            self._stop(srv)
            if srv is not None:
                deadline = time.time() + 60
                while port_in_use(port) and time.time() < deadline:
                    time.sleep(0.5)

    def _run_model(self, jobs):
        m = jobs[0].model
        sv = self.serving(m)
        local = bool(self.cfg["servers"][m["server"]].get("cmd"))
        serving_info = {"machine": sv.machine.id, "machine_summary": sv.machine.summary, "ctx": sv.ctx,
                        "kv_type": sv.kv_type, "perf_args": sv.perf_args if local else None,
                        "perf_source": sv.perf_source, "server_version": self.server_version(m) if local else None}
        ep = self.endpoint(m)
        if ep:
            serving_info.update(endpoint=ep["tag"], provider=ep.get("provider"), quantization=ep.get("quantization"))
        for j in jobs:
            j.status = "loading"
            self.emit("job_update", job=j)
        lane = self._lane()
        with self.serve(m, port=lane.port) as (base_url, info):
            with self._lanes_lock:
                loaded_alongside = sorted(lane.seen)   # served while this one loaded: its load time is shared

            def run_info():
                return {"load_s": info["load_s"], "peak_rss_mb": info["peak_mb"](),
                        "server_facts": info["facts"] or None, "gpu_peak_gb": info.get("gpu_peak_gb"),
                        "vram_peak_gb": info.get("vram_peak_gb"),
                        "kernel_share_max": info.get("kernel_share_max"),
                        **({"loaded_alongside": loaded_alongside} if loaded_alongside else {}), **serving_info}
            failed_in_row = 0
            for j in jobs:
                self._check()
                try:
                    self._run_job(j, base_url, run_info, alive=info["alive"])
                except BaseException as ex:
                    if j.status == "running":   # stopped part way: keep this sitting's time for next time
                        try:
                            self._save_sitting(j, run_info())
                        except Exception:
                            pass
                    if not isinstance(ex, PackFailed):
                        raise
                    j.status, j.note, j.finished = "failed", str(ex), time.time()
                    self.emit("job_done", job=j)
                    failed_in_row += 1
                    if failed_in_row >= self.PACKS_FAILED_STOP:
                        raise ModelFailed(f"the server kept failing on {failed_in_row} packs in a row, so "
                                          f"{m['label']} was stopped. Last: {ex}") from ex
                    continue
                failed_in_row = 0

    def _messages(self, job, test):
        system = test.get("system") or job.pack.system
        msgs = [{"role": "system", "content": system}] if system else []
        if test.get("image"):
            path = job.pack.asset(test["image"])
            mime = mimetypes.guess_type(path)[0] or "image/png"
            with open(path, "rb") as f:
                url = f"data:{mime};base64," + base64.b64encode(f.read()).decode()
            msgs.append({"role": "user", "content": [{"type": "text", "text": test["input"]},
                                                     {"type": "image_url", "image_url": {"url": url}}]})
        else:
            msgs.append({"role": "user", "content": test["input"]})
        return msgs

    def _body(self, job, test, rep=0):
        s = job.sampling
        body = {"model": job.model["served_name"], "messages": self._messages(job, test),
                "temperature": s["temperature"], "top_p": s["top_p"], "max_tokens": s["max_tokens"]}
        for k in ("top_k", "min_p", "presence_penalty", "repeat_penalty", "seed", "reasoning_effort"):
            if k in s:
                body[k] = s[k]
        if "seed" in s:  # the same seed would make every repeat an identical copy
            body["seed"] = s["seed"] + rep
        body.update(self.cfg["servers"][job.model["server"]].get("request", {}))
        ep = self.endpoint(job.model)
        if ep and ep.get("tag"):   # one provider endpoint for every answer, no fallback to another
            body["provider"] = {**body.get("provider", {}), "order": [ep["tag"]], "allow_fallbacks": False}
        if self.cfg["servers"][job.model["server"]].get("thinking_param") == "reasoning":
            body["reasoning"] = {"enabled": bool(s.get("enable_thinking", True))}   # OpenRouter's switch
        else:
            body["chat_template_kwargs"] = {"enable_thinking": bool(s.get("enable_thinking", True))}
        tools = test.get("tools") or job.pack.tools
        if tools:
            body["tools"] = tools
        return body

    def _log_request(self, job, test, res, meta):
        """Write the full exchange (prompt, reasoning, answer) to logs/live/; returns the file name."""
        d = os.path.join(self.log_dir, "live")
        os.makedirs(d, exist_ok=True)
        stamp = time.strftime("%H%M%S")
        name = f"{time.strftime('%Y%m%d')}_{stamp}_{job.label}_{job.pack.name}_{test['id'][:40]}.txt"
        with open(os.path.join(d, name), "w") as f:
            f.write(f"MODEL: {job.model['served_name']}\nPROMPT: {test['input']}\n\n=== REASONING ===\n"
                    f"{res['reasoning']}\n\n=== ANSWER ===\n{res['answer']}\n"
                    + (f"\n=== TOOL CALLS ===\n{json.dumps(res['tool_calls'])}\n" if res["tool_calls"] else "")
                    + f"\n=== {meta.get('completion_tokens')} tokens, {res['total_s']:.1f}s, "
                    f"finish={res['finish']} ===\n")
        return name

    @staticmethod
    def round_order(job, rep):
        """The tests in the order round `rep` asks them: the pack's order first, then a fixed
        shuffle per round (the same on every run, so resuming and reruns ask in the same order)."""
        tests = list(job.tests)
        if rep:
            random.Random(f"{job.pack.name}:{rep}").shuffle(tests)
        return tests

    # ---- sittings: a pack run over several sessions (stopped and resumed) keeps its total time
    SITTING_KEYS = ("load_s", "peak_rss_mb", "gpu_peak_gb", "vram_peak_gb", "kernel_share_max", "machine",
                    "loaded_alongside")

    def _sitting(self, job, info):
        now = time.time()
        return {"started": job.started, "finished": now, "wall_s": round(now - job.started, 1),
                "answers": job.done - job.resumed, **{k: info.get(k) for k in self.SITTING_KEYS}}

    def _sittings_path(self, job):
        return job.out_path + ".sittings"   # not *.json: result globs must not pick it up

    def _saved_sittings(self, job):
        try:
            with open(self._sittings_path(job)) as f:
                return json.load(f)
        except (OSError, ValueError):
            return []

    def _save_sitting(self, job, info):
        path = self._sittings_path(job)
        saved = self._saved_sittings(job)
        rebuilt = [s for s in job.earlier if s.get("rebuilt") and s not in saved]   # keep them on disk too
        sittings = saved + rebuilt + [self._sitting(job, info)]
        with open(path + ".tmp", "w") as f:
            json.dump(sittings, f)
        os.replace(path + ".tmp", path)

    SITTING_GAP_S = 300   # answers further apart than this belong to different sittings

    @classmethod
    def _rebuild_sittings(cls, rows, covered=()):
        """Sittings for saved answers that no sitting covers (stopped by older code, or a crash),
        from each answer's log name (the time it finished) and its duration. Model load time
        isn't known, so these slightly undercount."""
        spans = []
        for r in rows:
            try:
                end = time.mktime(time.strptime(r["log"][:15], "%Y%m%d_%H%M%S"))
            except (KeyError, TypeError, ValueError):
                continue
            if not any(s["started"] - 1 <= end <= s["finished"] + 1 for s in covered):
                spans.append((end - (r.get("total_s") or 0), end, r.get("machine")))
        sittings = []
        for start, end, machine in sorted(spans):
            s = sittings[-1] if sittings else None
            if s and start - s["finished"] <= cls.SITTING_GAP_S:
                s["finished"], s["answers"] = max(s["finished"], end), s["answers"] + 1
            else:
                sittings.append({"started": start, "finished": end, "answers": 1, "machine": machine,
                                 "rebuilt": True, **{k: None for k in cls.SITTING_KEYS if k != "machine"}})
        for s in sittings:
            s["wall_s"] = round(s["finished"] - s["started"], 1)
        return sittings

    @staticmethod
    def _file_sittings(run):
        """The sittings behind a finished result file (older files: one, from its run info)."""
        if run.get("sittings_log"):
            return run["sittings_log"]
        if not run.get("started"):
            return []
        return [{"started": run["started"], "finished": run.get("finished"), "wall_s": run.get("wall_s"),
                 "answers": None, **{k: run.get(k) for k in Engine.SITTING_KEYS}}]

    @staticmethod
    def sittings_summary(sittings):
        """Run info totals over every sitting: first start, summed time, peaks."""
        def peak(k):
            vals = [s[k] for s in sittings if s.get(k) is not None]
            return max(vals) if vals else None
        out = {"started": min(s["started"] for s in sittings),
               "wall_s": round(sum(s.get("wall_s") or 0 for s in sittings), 1),
               "sittings": len(sittings), "sittings_log": sittings}
        for k in ("peak_rss_mb", "gpu_peak_gb", "vram_peak_gb", "kernel_share_max"):
            if peak(k) is not None:
                out[k] = peak(k)
        along = sorted({l for s in sittings for l in s.get("loaded_alongside") or []})
        if along:
            out["loaded_alongside"] = along
        return out

    def _run_job(self, job, base_url, run_info, alive=lambda: True):
        partial = job.out_path + ".partial.jsonl"
        os.makedirs(os.path.dirname(job.out_path), exist_ok=True)
        kept = []          # earlier results for this pack that stay in the file
        job.earlier = []
        if job.merge:
            data = self._result(job.out_path)
            if data and data["pack"]["fingerprint"] == job.pack.fingerprint and \
                    settings_match(data, job.settings):
                kept = self._kept(job, data["results"])
                if kept:
                    job.earlier = self._file_sittings(data["run"])
        new = []
        if os.path.isfile(partial):
            with open(partial) as f:
                lines = [json.loads(l) for l in f if l.strip()]
            if lines and self._header_ok(lines[0], job):
                new = [r for r in lines[1:] if not client.server_error_row(r)]
                if job.fresh:   # a retest asks again what the unfinished run already answered
                    stay = [r for r in new if r["test"] not in self._redo_ids(job)]
                    if len(stay) < len(new):
                        self._partial_to_history(job, keep=bool(stay))
                        new = stay
                        if stay:
                            with open(partial, "w") as f:
                                f.write("".join(json.dumps(l) + "\n" for l in [lines[0]] + stay))
            else:   # other questions or settings: kept in history/, not deleted
                self._partial_to_history(job)
        if not os.path.isfile(partial):
            with contextlib.suppress(OSError):   # a fresh start: sittings of a discarded partial don't count
                os.remove(self._sittings_path(job))
            with open(partial, "w") as f:
                f.write(json.dumps(self._partial_header(job)) + "\n")
        saved = self._saved_sittings(job)
        job.earlier += saved + self._rebuild_sittings(new, saved + job.earlier)
        wanted = {(t["id"], r) for t in job.tests for r in range(job.repeat)}
        done_keys = {(r["test"], r["repeat"]) for r in kept + new} & wanted
        job.status, job.started = "running", time.time()
        mine = [r for r in kept + new if (r["test"], r["repeat"]) in wanted]
        job.done, job.passed = len(done_keys), sum(r["pass"] for r in mine)
        job.failed = len(mine) - job.passed
        job.resumed = job.done
        self.emit("job_started", job=job, resumed=len(done_keys),
                  earlier_s=sum(s.get("wall_s") or 0 for s in job.earlier), sittings=len(job.earlier))
        timeout = self.cfg["defaults"]["request_timeout_ms"] / 1000
        headers = self.request_headers(job.model)
        # a hosted API has no /tokenize to split reasoning from answer tokens (it reports the split)
        tokenize_url = None if self.cfg["servers"][job.model["server"]].get("api_key_env") else base_url
        warned_reuse = False
        for rep in range(job.repeat):
            for test in self.round_order(job, rep):
                if (test["id"], rep) in done_keys:
                    continue
                self._check()
                lane = self._lane()
                with self._lanes_lock:
                    lane.seen = set(self._lanes) - {job.label}
                self.emit("request_started", job=job, test=test["description"], test_id=test["id"],
                          has_image=bool(test.get("image")), repeat=rep, difficulty=test.get("difficulty", "unrated"))
                waits = list(self.SERVER_RETRY_WAITS_S)
                while True:
                    self._stream = client.Stream()
                    try:
                        body = self._body(job, test, rep)
                        res = client.stream_chat(base_url, body,
                                                 lambda kind, text: self.emit("delta", stream=kind, text=text, label=job.label),
                                                 timeout=timeout, stream=self._stream, headers=headers)
                    except client.Cancelled:
                        self._check()
                        raise Cancelled()
                    finally:
                        self._stream = None
                    self._check()
                    if res["error"] and not alive():
                        raise ModelFailed(f"the server stopped: {res['error']}")
                    if client.is_unavailable(res["error"]):   # never an answer; retrying won't help
                        raise ModelFailed(f"the model isn't available there (not counted against it): "
                                          f"{res['error'][:300]}")
                    if not client.is_server_error(res["error"]):
                        break
                    # the server failed, not the model: never recorded as an answer
                    if not waits:
                        raise PackFailed(
                            f"server kept failing for {sum(self.SERVER_RETRY_WAITS_S) // 60} min on {test['id']}; "
                            f"answers so far kept, the rest run next time. Last error: {res['error'][:300]}")
                    wait = waits.pop(0)
                    self.emit("server_retry", label=job.label, message=(
                        f"{job.label}: server error on {test['id']}, not counted; retrying in {wait}s "
                        f"({len(waits)} tries left): {res['error'][:200]}"))
                    for _ in range(wait):
                        self._check()
                        if not alive():
                            raise ModelFailed(f"the server stopped: {res['error']}")
                        self._cancel.wait(1)
                meta = client.metrics(res, tokenize_url)
                ep = self.endpoint(job.model)
                if not res["error"] and ep and ep.get("provider") and meta.get("provider") \
                        and meta["provider"].lower() != ep["provider"].lower():
                    res["error"] = (f"server returned an answer from {meta['provider']}, not the pinned "
                                    f"{ep['provider']} ({ep['tag']}); not counted")
                if res["error"]:
                    g = {"pass": False, "score": 0.0, "reason": res["error"]}
                else:
                    g = graders.grade(test, res["answer"], {"finish": res["finish"], "tool_calls": res["tool_calls"]})
                log_name = self._log_request(job, test, res, meta)
                rec = {"test": test["id"], "description": test["description"], "category": test.get("category", ""),
                       "difficulty": test.get("difficulty", "unrated"),
                       "group": test.get("group"), "critical_test": bool(test.get("critical_trial")),
                       "repeat": rep, "pass": g["pass"], "score": g["score"], "reason": g["reason"],
                       "severity": g.get("severity"), "answer": res["answer"], "tool_calls": res["tool_calls"],
                       "finish": res["finish"], "reasoning_chars": len(res["reasoning"]),
                       "log": log_name,   # logs/live/<log>: the reasoning, which the result doesn't keep
                       **({"checks": g["checks"]} if "checks" in g else {}),
                       "machine": self.machine().id,
                       **({"quantization": ep.get("quantization")} if ep else {}),
                       # other models served while this answer ran: they shared the machine, so its speed did too
                       **({"ran_alongside": sorted(lane.seen)} if lane.seen else {}),
                       "sent": "+".join(msg["role"] for msg in body["messages"]),  # proof: single turn
                       **({"seed": body["seed"]} if "seed" in body else {}), **meta}
                # The same long answer twice for one question, with sampling on, means a cached response
                # rather than a fresh one. Flagged (and shown in the run), not failed.
                fp = hashlib.sha256((res["reasoning"] + "\x00" + res["answer"]).encode()).hexdigest()[:16]
                if not res["error"] and len(res["reasoning"]) + len(res["answer"]) >= 200 \
                        and job.sampling.get("temperature", 0) > 0:
                    rec["answer_sha"] = fp
                    same = next((r["repeat"] for r in kept + new if r["test"] == test["id"]
                                 and r.get("answer_sha") == fp), None)
                    if same is not None:
                        rec["identical_to_repeat"] = same
                        self.emit("identical_answer", label=job.label, pack=job.pack.label, test=test["id"],
                                  repeat=rep, same=same)
                if meta.get("cached_tokens") and not warned_reuse:
                    warned_reuse = True
                    self.emit("prompt_reused", label=job.label, pack=job.pack.label, tokens=meta["cached_tokens"],
                              server=job.model["server"],
                              hosted=not self.cfg["servers"][job.model["server"]].get("cmd"))
                with open(partial, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                new.append(rec)
                job.done += 1
                job.passed += bool(g["pass"])
                job.failed += not g["pass"]
                self.emit("request_done", job=job, record=rec)
        job.status, job.finished = "done", time.time()
        merged = {(r["test"], r["repeat"]): r for r in kept}
        merged.update({(r["test"], r["repeat"]): r for r in new})
        info = run_info()
        self._write_result(job, list(merged.values()),
                           {**info, **self.sittings_summary(job.earlier + [self._sitting(job, info)])})
        os.remove(partial)
        with contextlib.suppress(OSError):
            os.remove(self._sittings_path(job))
        job.note = f"{job.passed}/{job.done} passed"
        self.emit("job_done", job=job)

    def _partial_to_history(self, job, keep=False):
        """Keep an unfinished run that is set aside in history/; keep: copy it (some of its answers stay)."""
        partial = job.out_path + ".partial.jsonl"
        hist = os.path.join(os.path.dirname(job.out_path), "history")
        os.makedirs(hist, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(os.path.getmtime(partial)))
        (shutil.copy2 if keep else os.replace)(partial, os.path.join(hist, f"{job.pack.name}-{stamp}.partial.jsonl"))

    @staticmethod
    def _redo_ids(job):
        """The tests whose earlier answers a retest replaces: the picked ones, or all of them."""
        return {t["id"] for t in job.tests} if job.redo else {t["id"] for t in job.pack.tests}

    def _archive(self, job, keep=False):
        """Keep a result file that is about to be replaced (rerun or outdated) in history/; keep:
        copy it (some of its answers stay in the new file). Smoke results too: nothing is deleted."""
        if not os.path.isfile(job.out_path):
            return
        hist = os.path.join(os.path.dirname(job.out_path), "history")
        os.makedirs(hist, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(os.path.getmtime(job.out_path)))
        (shutil.copy2 if keep else os.replace)(job.out_path, os.path.join(hist, f"{job.pack.name}-{stamp}.json"))

    def _write_result(self, job, records, run_info):
        if not job.merge or job.redo:
            self._archive(job, keep=job.redo)
        m = job.model
        data = {
            "format": RESULT_FORMAT,
            "model": {**{k: m.get(k) for k in ("label", "served_name", "server", "model", "tags", "vision", "mmproj")},
                      **{k: m[k] for k in ("tools", "max_context", "thinking") if k in m}},
            "pack": {"name": job.pack.name, "label": job.pack.label, "group": job.pack.group,
                     "fingerprint": job.pack.fingerprint, "tests": len(job.pack.tests)},
            "settings": job.sampling,
            "serving": job.settings.get("serving"),
            "settings_fingerprint": settings_fingerprint(job.settings),
            "run": {"tier": job.tier, "repeat": job.repeat, "started": job.started, "finished": job.finished,
                    "wall_s": round(job.finished - job.started, 1), **run_info},
            "results": sorted(records, key=lambda r: (r["repeat"], r["test"])),
        }
        tmp = job.out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, job.out_path)


def regrade(path, packs_dir=None):
    """Re-score a result file's stored answers with the current graders (e.g. after fixing one).
    Returns (before, after) pass counts."""
    with open(path) as f:
        data = json.load(f)
    pack = packs_mod.load_packs(packs_dir).get(data["pack"]["name"])
    if pack is None:
        raise ConfigError(f"pack {data['pack']['name']!r} not found")
    tests = {t["id"]: t for t in pack.tests}
    before = sum(r["pass"] for r in data["results"])
    for r in data["results"]:
        t = tests.get(r["test"])
        if t is None:
            continue
        if r["reason"].startswith(("server returned", "connection error")):
            continue
        g = graders.grade(t, r["answer"], {"finish": r["finish"], "tool_calls": r.get("tool_calls", [])})
        r.update({"pass": g["pass"], "score": g["score"], "reason": g["reason"], "severity": g.get("severity"),
                  "critical_test": bool(t.get("critical_trial")), "group": t.get("group")})
        r.pop("checks", None)
        if "checks" in g:
            r["checks"] = g["checks"]
    with open(path, "w") as f:
        json.dump(data, f)
    return before, sum(r["pass"] for r in data["results"])
