"""Export a model's serving settings to the pi coding agent, so pi serves it exactly as the evals did:
the same model file, output-affecting flags and context, plus the speed flags tuned on this machine.

  a section in llama-router's presets (~/models/presets.ini; keys that equal the [*] section are
  left out) and an entry under pi's `llama` provider modelOverrides. Models on llama servers only.

Every file is backed up next to itself (<file>.bak-tuieval-<time>) before it changes, and the diff
is shown first. Paths come from models.toml [export.pi]:

  [export.pi]
  presets = "~/models/presets.ini"            # llama-router --models-preset
  pi_models = "~/.pi/agent/models.json"
  llama_provider = "llama"                     # pi's provider name for the router
  servers = ["llama"]                          # models.toml servers that llama-router can serve
"""
import configparser
import difflib
import json
import os
import re
import shutil
import time

from . import engine as engine_mod

DEFAULTS = {"presets": "~/models/presets.ini", "pi_models": "~/.pi/agent/models.json",
            "llama_provider": "llama", "servers": ["llama"]}
# llama.cpp short flags -> the long names presets use as keys
SHORT = {"-m": "model", "-c": "ctx-size", "-t": "threads", "-tb": "threads-batch", "-ub": "ubatch-size",
         "-b": "batch-size", "-fa": "flash-attn", "-ngl": "n-gpu-layers", "-np": "parallel",
         "-ctk": "cache-type-k", "-ctv": "cache-type-v", "-a": "alias"}
ROUTER_OWNED = {"alias", "host", "port"}   # the router sets these per worker


class ExportError(Exception):
    pass


def settings(eng):
    return {**DEFAULTS, **eng.cfg.get("export", {}).get("pi", {})}


def default_id(m):
    return os.path.basename(engine_mod.expand(m["model"])).removesuffix(".gguf")


def flags_to_keys(args):
    """A llama.cpp command's flags -> [(preset key, value)]: flags without a value become "true"."""
    out, i = [], 0
    while i < len(args):
        a = args[i]
        if not a.startswith("-"):
            raise ExportError(f"unexpected argument {a!r} in the server command")
        key = SHORT.get(a) or (a[2:] if a.startswith("--") else None)
        if key is None:
            raise ExportError(f"don't know the preset key for {a}; add it to export.SHORT")
        if i + 1 < len(args) and not args[i + 1].startswith("-"):
            out.append((key, args[i + 1]))
            i += 2
        else:
            out.append((key, "true"))
            i += 1
    return out


def _ini_sections(text):
    """{section: {key: value}} of a presets file (comments ignored)."""
    p = configparser.ConfigParser(interpolation=None, strict=False, inline_comment_prefixes=("#", ";"))
    p.optionxform = str
    p.read_string("[__top__]\n" + text)
    return {s: dict(p[s]) for s in p.sections() if s != "__top__"}


def set_ini_section(text, name, lines):
    """text with section [name] replaced by lines (or appended at the end); everything else is kept as
    written. Comments and blank lines at the end of the old section introduce the next one, so they stay."""
    src = text.splitlines()
    header = re.compile(r"^\s*\[(.+?)\]\s*$")
    block = [f"[{name}]"] + lines
    start = next((i for i, l in enumerate(src) if (h := header.match(l)) and h.group(1) == name), None)
    if start is None:
        while src and not src[-1].strip():
            src.pop()
        return "\n".join(src + [""] + block) + "\n"
    end = next((i for i in range(start + 1, len(src)) if header.match(src[i])), len(src))
    keep = end
    while keep > start + 1 and (not src[keep - 1].strip() or src[keep - 1].lstrip().startswith(("#", ";"))):
        keep -= 1
    while keep < end and not src[keep].strip():
        keep += 1
    out = src[:start] + block + ([""] if keep < len(src) else []) + src[keep:]
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out) + "\n"


# request sampling (models.toml [sampling] and a model's overrides) -> llama.cpp server defaults, so
# pi gets the evals' sampling without sending it
SAMPLING_KEYS = {"temperature": "temp", "top_p": "top-p", "top_k": "top-k", "min_p": "min-p",
                 "presence_penalty": "presence-penalty", "frequency_penalty": "frequency-penalty",
                 "repetition_penalty": "repeat-penalty"}


class Change:
    """One file's new text, with notes for the reader."""
    def __init__(self, path, old, new):
        self.path, self.old, self.new = path, old, new

    def diff(self):
        lines = difflib.unified_diff(self.old.splitlines(), self.new.splitlines(), self.path, self.path + " (new)",
                                     lineterm="", n=2)
        # never echo credentials from the file around a change
        return "\n".join(re.sub(r'("(?:api_?key|token|secret)[^"]*"\s*:\s*)"[^"]*"', r'\1"<redacted>"', l, flags=re.I)
                         for l in lines)


def _read(path):
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return ""


def plan(eng, label, model_id=None, name=None):
    """(changes, notes) that export one model to pi. Nothing is written."""
    m = eng.model(label)
    s = settings(eng)
    if m["server"] not in s["servers"]:
        raise ExportError(f"{label} runs on the {m['server']} server; pi export writes llama-router presets, "
                          "for llama servers ([export.pi] servers lists them)")
    model_id = model_id or m.get("pi_id") or default_id(m)
    name = name or m.get("pi_name") or model_id
    sv = eng.serving(m)
    notes = []
    sampling = engine_mod.effective_sampling(eng.cfg, m)
    vision = bool(m.get("vision") or m.get("mmproj"))
    pi_path = engine_mod.expand(s["pi_models"])
    pi_old = _read(pi_path)
    if not pi_old:
        raise ExportError(f"pi's model file {pi_path} not found ([export.pi] pi_models)")
    pi = json.loads(pi_old)
    providers = pi.setdefault("providers", {})
    changes = []
    if sv.perf_source == "untuned":
        notes.append(f"{label} isn't tuned on this machine: exporting the untuned defaults (tuieval tune {label})")
    elif sv.perf_source.startswith("outdated"):
        notes.append(f"{label}'s tuning is {sv.perf_source}; consider tuieval tune {label}")
    cmd, _, _ = eng.server_command(m)
    if cmd[0] == "env":
        raise ExportError("the server command sets environment variables, which presets can't hold")
    first = next(i for i, a in enumerate(cmd) if a.startswith("-"))
    flags = {}
    for k, v in flags_to_keys(cmd[first:]):
        if k not in ROUTER_OWNED:
            flags[k] = v   # a repeated flag: the last wins, as on the command line
    for k, key in SAMPLING_KEYS.items():
        if sampling.get(k) is not None:
            flags[key] = str(sampling[k])
    path = engine_mod.expand(s["presets"])
    old = _read(path)
    if not old:
        raise ExportError(f"presets file {path} not found ([export.pi] presets)")
    sections = _ini_sections(old)
    common = sections.get("*", {})
    order = ["model", "mmproj"] + [k for k in flags if k not in ("model", "mmproj")]
    lines = [f"{k:<16} = {flags[k]}" for k in order if k in flags and common.get(k) != flags[k]]
    tuned = {k for opts in eng.cfg["servers"][m["server"]].get("tune", {}).values() for o in opts
             for k, _ in flags_to_keys(o)}
    for k in common:
        if k in tuned and k not in flags:
            notes.append(f"presets [*] sets {k} = {common[k]}, which {label}'s tuned flags leave off; "
                         "the router will still apply it")
    broken = [sec for sec, kv in sections.items() if sec != model_id
              and any(kv.get(k) and not os.path.exists(os.path.expanduser(kv[k])) for k in ("model", "mmproj"))]
    if broken:
        notes.append(f"{path}: {len(broken)} other section(s) name files that don't exist, so pi can't load "
                     f"them: {', '.join(broken)}")
    changes.append(Change(path, old, set_ini_section(old, model_id, lines)))
    prov = providers.setdefault(s["llama_provider"], {})
    entry = prov.setdefault("modelOverrides", {}).setdefault(model_id, {})
    entry.update({"name": name, "contextWindow": sv.ctx,
                  "maxTokens": entry.get("maxTokens", sampling.get("max_tokens")),
                  "reasoning": bool(sampling.get("enable_thinking", True)),
                  "input": ["text", "image"] if vision else ["text"]})
    notes.append("restart llama-router to load the new presets (llama-router restart)")
    if sampling.get("reasoning_effort"):
        notes.append(f"the evals ran reasoning_effort {sampling['reasoning_effort']}: pick that thinking level in pi")
    pi_new = json.dumps(pi, indent=2, ensure_ascii=False) + "\n"
    if pi_new != pi_old:
        changes.append(Change(pi_path, pi_old, pi_new))
    return [c for c in changes if c.new != c.old], notes


def apply(changes):
    """Write the changes, backing each file up first. Returns the backup paths."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backups = []
    for c in changes:
        if _read(c.path) != c.old:
            raise ExportError(f"{c.path} changed since the export was planned; run it again")
    for c in changes:
        b = f"{c.path}.bak-tuieval-{stamp}"
        shutil.copy2(c.path, b)
        backups.append(b)
        tmp = c.path + ".tuieval-tmp"
        with open(tmp, "w") as f:
            f.write(c.new)
        shutil.copymode(c.path, tmp)
        os.replace(tmp, c.path)
    return backups
