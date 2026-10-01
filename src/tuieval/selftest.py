"""Check the tests themselves, so a wrong expected answer can't silently fail good models.

    tuieval selftest              # every pack
    tuieval selftest my-pack      # one pack

For every test:
  - its `reference` (a correct model output) must pass the grader with full score
  - every entry in `wrong` (known-bad outputs) must fail
  - tool tests: the expected tool must exist in the pack's tools, with declared, required arguments
  - no TODO placeholders left (from tuieval capture)
Per pack: tool schemas must have string keys and valid `required` lists, no two tests may send
the same prompt (which would make results ambiguous), and the gate must be reachable: enough
critical trials to show the critical rate is below max_critical_rate, and enough trials that a
flawless certification clears min_accuracy (verdict.py uses the 95% lower bound).

Isolation (every pack): each request is one fresh single-turn conversation (optional system message
plus exactly one user message, never an earlier answer), every server's `request` fields are sent,
a fixed seed differs per repeat, and each repeat round asks the tests in a different order.

Tests without a reference are listed as warnings: they still run, they just aren't self-checked.
"""
import collections
import hashlib
import json

from . import graders
from . import packs as packs_mod
from . import verdict

DEFAULT_REPEAT = 3  # models.toml [defaults] repeat, used when a pack sets no certify repeat


def _tool_names(pack):
    out = {}
    for tool in pack.tools:
        fn = tool.get("function", {})
        out[fn.get("name")] = fn.get("parameters", {})
    return out


def check_gate(pack, default_repeat=DEFAULT_REPEAT):
    """Errors for gates a certification run could never pass."""
    errors = []
    repeat = pack.certify_repeat or default_repeat
    rate = pack.gate.get("max_critical_rate")
    crit = sum(1 for t in pack.tests if t.get("critical_trial")) * repeat
    if rate and crit < verdict.trials_needed(rate):
        errors.append(f"gate unreachable: {crit} critical trials at certification, but max_critical_rate = {rate} "
                      f"needs {verdict.trials_needed(rate)}; add critical tests, repeats, or raise the rate")
    trials = len(pack.tests) * repeat
    lo = verdict.wilson(trials, trials)[0]
    if lo < pack.gate.get("min_accuracy", 0.8):
        errors.append(f"gate unreachable: even {trials}/{trials} correct gives a 95% lower bound of {100 * lo:.1f}%, "
                      f"under min_accuracy {pack.gate['min_accuracy']}; add tests or repeats")
    return errors


def check_pack(pack):
    """(errors, warnings) for one pack."""
    errors, warnings = check_gate(pack), []
    tools = _tool_names(pack)
    for name, params in tools.items():
        props = params.get("properties", {})
        bad = [k for k in props if not isinstance(k, str)]
        if bad:
            errors.append(f"tool {name}: non-string parameter names {bad} (quote YAML keys like \"on\")")
        missing = [r for r in params.get("required", []) if r not in props]
        if missing:
            errors.append(f"tool {name}: required {missing} not declared in properties")
    seen = collections.defaultdict(list)
    for t in pack.tests:
        img = ""
        if t.get("image"):
            with open(pack.asset(t["image"]), "rb") as f:
                img = hashlib.sha1(f.read()).hexdigest()
        seen[(" ".join(str(t["input"]).split()), img)].append(t["id"])
        tid = t["id"]
        if "TODO" in json.dumps({k: v for k, v in t.items() if k not in ("input", "reference", "wrong")}):
            errors.append(f"{tid}: still has TODO placeholders")
            continue
        if t.get("grader") == "tool_call" and t.get("expect_tool"):
            exp = t["expect_tool"]
            if exp["name"] not in tools:
                errors.append(f"{tid}: expects tool {exp['name']!r}, which the pack doesn't define")
            else:
                props = tools[exp["name"]].get("properties", {})
                undeclared = [a for a in exp.get("arguments", {}) if a not in props]
                if undeclared:
                    errors.append(f"{tid}: expected arguments {undeclared} aren't declared by {exp['name']}")
            meta = {"finish": "stop", "tool_calls": [{"name": exp["name"], "arguments": exp.get("arguments", {})}]}
            r = graders.grade(t, "", meta)
            if not r["pass"]:
                errors.append(f"{tid}: the expected tool call itself fails: {r['reason'][:100]}")
        ref = t.get("reference")
        if t.get("difficulty", "unrated") == "unrated":
            warnings.append(f"{tid}: no difficulty label (easy, medium or hard)")
        if ref is None:
            if not (t.get("grader") == "tool_call" and t.get("expect_tool")):
                warnings.append(f"{tid}: no reference answer")
        else:
            r = graders.grade(t, ref, {"finish": "stop", "tool_calls": t.get("reference_tool_calls", [])})
            if not r["pass"] or r["score"] < 1.0:
                errors.append(f"{tid}: reference answer fails: {r['reason'][:140]}")
        for i, w in enumerate(t.get("wrong", [])):
            r = graders.grade(t, w, {"finish": "stop", "tool_calls": []})
            if r["pass"]:
                errors.append(f"{tid}: wrong[{i}] passes the grader, so the test can't catch that mistake")
    for (_, _), ids in seen.items():
        if len(ids) > 1:
            errors.append(f"tests {', '.join(ids)} send the same prompt")
    return errors, warnings


def check_isolation(pack, eng, repeats=3):
    """Errors if a request for this pack could carry anything from an earlier one."""
    from . import engine
    errors = []
    job = None
    for server_name, server in eng.cfg["servers"].items():
        model = {"label": "selftest", "served_name": "selftest", "server": server_name}
        job = engine.Job(model, pack, repeats, pack.tests, "", dict(eng.cfg["sampling"], seed=7))
        if server.get("pin_endpoint"):
            eng._endpoints["selftest"] = {"tag": "selftest/bf16", "provider": "Selftest", "quantization": "bf16"}
        bodies = [eng._body(job, pack.tests[0], r) for r in range(repeats)]
        eng._endpoints.pop("selftest", None)
        if server.get("pin_endpoint") and any(b.get("provider", {}).get("order") != ["selftest/bf16"]
                                              or b["provider"].get("allow_fallbacks") is not False for b in bodies):
            errors.append(f"[servers.{server_name}] requests aren't pinned to one provider endpoint without fallback")
        if len({b["seed"] for b in bodies}) != repeats:
            errors.append("repeats share a seed, so they would be identical copies")
        for k, v in server.get("request", {}).items():
            if bodies[0].get(k) != v:
                errors.append(f"[servers.{server_name}] request field {k} = {v!r} is not sent")
    for test in pack.tests:  # messages don't depend on the server or the repeat
        roles = [m["role"] for m in eng._messages(job, test)]
        if roles not in (["user"], ["system", "user"]):
            errors.append(f"{test['id']}: request sends {'+'.join(roles)}, not one fresh single-turn conversation")
    orders = [[t["id"] for t in eng.round_order(job, r)] for r in range(repeats)]
    if any(sorted(o) != sorted(orders[0]) for o in orders):
        errors.append("a repeat round skips or duplicates tests")
    if len(pack.tests) > 2 and len({tuple(o) for o in orders}) != repeats:
        errors.append("repeat rounds ask the tests in the same order")
    return errors


def run(pack_names=None, packs_dir=None, verbose=True):
    """Check packs; returns True if there are no errors."""
    errors_load = []
    packs = packs_mod.load_packs(packs_dir, errors_load)
    ok = not errors_load
    for e in errors_load:
        print("ERROR", e)
    try:
        from . import engine
        eng = engine.Engine(packs_dir=packs_dir)
    except Exception as ex:  # a broken models.toml shouldn't hide the pack checks
        print("ERROR isolation checks skipped: models.toml:", ex)
        eng, ok = None, False
    if not packs and not errors_load:
        print("No packs yet. Create one with: tuieval new-pack <name>   (see docs/writing-packs.md)")
    for name, pack in packs.items():
        if pack_names and name not in pack_names:
            continue
        errors, warnings = check_pack(pack)
        if eng:
            errors += check_isolation(pack, eng)
        refs = sum(1 for t in pack.tests if t.get("reference") is not None or
                   (t.get("grader") == "tool_call" and t.get("expect_tool")))
        status = "ok" if not errors else f"{len(errors)} ERRORS"
        print(f"{name:14s} {len(pack.tests):4d} tests  {refs:4d} self-checked  "
              f"{sum(len(t.get('wrong', [])) for t in pack.tests):4d} wrong answers  {status}")
        if verbose:
            for e in errors[:20]:
                print("   ERROR", e)
            if len(errors) > 20:
                print(f"   … and {len(errors) - 20} more")
            if warnings:
                print(f"   {len(warnings)} warning(s), e.g. {warnings[0]}")
        ok = ok and not errors
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run(sys.argv[1:] or None) else 1)
