"""Tuned speed settings per model per machine: tuning/<machine-id>/<label>.toml (synced).

A profile holds the chosen option for each speed knob in models.toml [servers.<name>.tune], the
resolved flags, the measured speeds, and what it was tuned against (llama.cpp version, model
file). Only speed-only flags live here; nothing in a profile changes answers.
"""
import datetime
import json
import os
import tomllib

from . import workspace


def path(machine_id, label, root=None):
    return os.path.join(root or workspace.path("tuning"), machine_id, f"{label}.toml")


def load(machine_id, label, root=None):
    p = path(machine_id, label, root)
    if not os.path.isfile(p):
        return None
    with open(p, "rb") as f:
        return tomllib.load(f)


def _toml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    return json.dumps(str(v))


def save(machine_id, label, profile, root=None):
    """Write a profile: {"choices": {knob: index}, "args": [...], "measured": {...}, "meta": {...}}."""
    p = path(machine_id, label, root)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    lines = [f"# Tuned speed settings for {label} on {machine_id}. Written by tuieval tune; safe to edit.",
             f"args = {_toml_value(profile['args'])}", ""]
    for section in ("choices", "measured", "meta"):
        lines.append(f"[{section}]")
        for k, v in profile.get(section, {}).items():
            if v is not None:
                lines.append(f"{k} = {_toml_value(v)}")
        lines.append("")
    with open(p, "w") as f:
        f.write("\n".join(lines))
    return p


def stale(profile, server_version, model_bytes):
    """Why a profile should be re-tuned, or None."""
    meta = (profile or {}).get("meta", {})
    if meta.get("model_bytes") and model_bytes and meta["model_bytes"] != model_bytes:
        return "the model file changed"
    if meta.get("server_version") and server_version and meta["server_version"] != server_version:
        return f"the server changed ({meta['server_version']} -> {server_version})"
    return None


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------- known machines
def save_machine(machine, root=None):
    """Record a machine's facts (tuning/<id>/machine.toml) so the others can check fit for it."""
    import dataclasses
    root = root or workspace.path("tuning")
    p = os.path.join(root, machine.id, "machine.toml")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    lines = [f"# {machine.summary}. Written automatically when evals runs on this machine."]
    lines += [f"{k} = {_toml_value(v)}" for k, v in dataclasses.asdict(machine).items() if v is not None]
    text = "\n".join(lines) + "\n"
    if not os.path.isfile(p) or open(p).read() != text:
        with open(p, "w") as f:
            f.write(text)
    return p


def known_machines(root=None):
    """[Machine] for every machine that has recorded itself in tuning/."""
    from . import machines
    root = root or workspace.path("tuning")
    out = []
    for mid in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        p = os.path.join(root, mid, "machine.toml")
        if os.path.isfile(p):
            with open(p, "rb") as f:
                d = tomllib.load(f)
            try:
                out.append(machines.Machine(**{**{"gpu_cores": None}, **d}))
            except TypeError:
                pass
    return out
