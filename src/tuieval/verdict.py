"""Production readiness: turn a model's results on a pack into PASS / FAIL / INCONCLUSIVE.

Gates live in each pack's pack.toml:

    [gate]
    min_accuracy = 0.95        # the 95% lower bound of the pass rate must reach this to PASS
    max_critical_rate = 0.01   # prove critical failures are rarer than this (rule of three)
    max_p90_s = 120            # 90th-percentile seconds per answer
    max_truncation = 0.01      # share of answers cut off by max_tokens
    min_consistency = 0.95     # share of variant groups where every variant passes

Rules, in order:
  FAIL          any critical failure (breaking a hard rule, inventing data, an unwanted
                action); a budget exceeded; or even the optimistic bound of accuracy is too low.
  INCONCLUSIVE  not a full certification run yet; accuracy could still go either way; or too few
                critical trials to show the critical rate is below max_critical_rate.
  PASS          everything above holds on a full certification run.

A screening run can say FAIL or "promising", never PASS.

These verdicts are about answer quality and travel between machines (results carry over when the
output-affecting serving settings match). Speed is judged per machine by latency(): measured on
that machine when it has run the pack, otherwise projected from each answer's token counts and
the machine's tuned speeds (tuning/<machine>/<label>.toml).
"""
import dataclasses
import datetime
import json
import math
import os
import statistics

ORDER = {"FAIL": 0, "INCONCLUSIVE": 1, "NO DATA": 2, "PASS": 3}


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def trials_needed(max_rate):
    """Failure-free trials needed to show a rate is below max_rate at 95% confidence (rule of three)."""
    return math.ceil(3 / max_rate)


@dataclasses.dataclass
class Verdict:
    status: str                 # PASS, FAIL, INCONCLUSIVE, NO DATA
    reasons: list
    evidence: dict

    @property
    def summary(self):
        return f"{self.status}: " + "; ".join(self.reasons[:3]) if self.reasons else self.status


def consistency(records):
    """Share of variant groups (2+ variants run) where every variant passed every repeat."""
    groups = {}
    for r in records:
        if r.get("group"):
            groups.setdefault(r["group"], set()).add(r["test"])
    full = [g for g, tests in groups.items() if len(tests) >= 2]
    if not full:
        return None, 0
    ok = sum(all(r["pass"] for r in records if r.get("group") == g) for g in full)
    return ok / len(full), len(full)


def evaluate(pack, records, repeat_default=3, speed_gate=False):
    """Quality verdict for one model on one pack. `records` are result rows from
    results/<label>/<pack>.json. The p90 time limit is judged per machine by latency() unless
    speed_gate is set."""
    gate = pack.gate
    if not records:
        return Verdict("NO DATA", ["not run yet"], {})
    n = len(records)
    passed = sum(r["pass"] for r in records)
    lo, hi = wilson(passed, n)
    crit = [r for r in records if r.get("severity") == "critical"]
    crit_tests = {t["id"] for t in pack.tests if t.get("critical_trial")}
    crit_trials = sum(r["test"] in crit_tests for r in records)
    times = sorted(r["total_s"] for r in records if r.get("total_s") is not None)
    p90 = statistics.quantiles(times, n=10)[-1] if len(times) > 1 else (times[0] if times else None)
    trunc = sum(r.get("finish") == "length" for r in records) / n
    cons, groups = consistency(records)
    want_repeat = pack.certify_repeat or repeat_default
    run_tests = {r["test"] for r in records}
    repeats = min(sum(r["test"] == t for r in records) for t in run_tests) if run_tests else 0
    certified = run_tests >= {t["id"] for t in pack.tests} and repeats >= want_repeat
    ev = {"trials": n, "passed": passed, "accuracy": passed / n, "acc_lo": lo, "acc_hi": hi,
          "critical_failures": len(crit), "critical_trials": crit_trials, "p90_s": p90,
          "truncation": trunc, "consistency": cons, "groups": groups, "certified": certified,
          "tests_run": len(run_tests), "tests_total": len(pack.tests), "repeats": repeats,
          "want_repeat": want_repeat, "critical_examples": [f"{r['test']}: {r['reason'][:120]}" for r in crit[:5]]}

    fails, pending = [], []
    if crit:
        fails.append(f"{len(crit)} critical failure{'s' if len(crit) > 1 else ''} (e.g. {crit[0]['test']}: "
                     f"{crit[0]['reason'][:80]})")
    if "max_truncation" in gate and trunc > gate["max_truncation"]:
        fails.append(f"{100 * trunc:.1f}% of answers cut off by max_tokens (limit {100 * gate['max_truncation']:.0f}%)")
    if speed_gate and "max_p90_s" in gate and p90 is not None and p90 > gate["max_p90_s"] and len(times) >= 10:
        fails.append(f"p90 {p90:.0f}s per answer (limit {gate['max_p90_s']}s)")
    min_acc = gate.get("min_accuracy", 0.8)
    if hi < min_acc:
        fails.append(f"accuracy {100 * passed / n:.1f}% (95% CI {100 * lo:.0f}-{100 * hi:.0f}%) is below "
                     f"the {100 * min_acc:.0f}% bar")
    if "min_consistency" in gate and cons is not None and certified and cons < gate["min_consistency"]:
        fails.append(f"only {100 * cons:.0f}% of {groups} rephrased cases answered consistently "
                     f"(needs {100 * gate['min_consistency']:.0f}%)")
    if fails:
        return Verdict("FAIL", fails, ev)

    if not certified:
        outlook = "promising" if passed / n >= min_acc else "uncertain"
        if len(run_tests) < len(pack.tests):
            pending.append(f"screened only ({len(run_tests)}/{len(pack.tests)} tests); {outlook}, run Certify to decide")
        else:
            pending.append(f"all tests run but only {repeats} repeat(s) of the {want_repeat} certification needs; "
                           f"{outlook}, run Certify to finish")
    elif lo < min_acc:
        pending.append(f"accuracy {100 * passed / n:.1f}% but the 95% lower bound {100 * lo:.1f}% is under "
                       f"{100 * min_acc:.0f}%: more tests or repeats needed")
    max_rate = gate.get("max_critical_rate")
    if max_rate and crit_tests:
        need = trials_needed(max_rate)
        if crit_trials < need:
            pending.append(f"{crit_trials} critical trials without failure; {need} are needed to show the critical "
                           f"failure rate is below {100 * max_rate:g}%")
    if pending:
        return Verdict("INCONCLUSIVE", pending, ev)
    reasons = [f"accuracy {100 * passed / n:.1f}% (95% CI {100 * lo:.0f}-{100 * hi:.0f}%) over {n} answers"]
    if crit_tests:
        reasons.append(f"0 critical failures in {crit_trials} critical trials")
    return Verdict("PASS", reasons, ev)


# ---------------------------------------------------------------- latency per machine
LAT_ORDER = {"DOESN'T FIT": 0, "TOO SLOW": 1, "NO DATA": 2, "OK": 3}
MIN_TIMED = 10   # measured answers needed before measured beats projected


@dataclasses.dataclass
class Latency:
    machine: str
    status: str          # OK, TOO SLOW, NO DATA, DOESN'T FIT
    p90_s: float | None
    kind: str            # measured, projected, or ""
    note: str = ""
    limit: float | None = None

    @property
    def summary(self):
        if self.status == "DOESN'T FIT":
            return self.note or "doesn't fit"
        if self.status == "NO DATA":
            return "no data" + (f": {self.note}" if self.note else "")
        lim = f" (limit {self.limit:g}s)" if self.limit else ""
        return f"{self.status} p90 {self.p90_s:.0f}s {self.kind}{lim}"


def projected_seconds(r, speeds):
    """An answer's time on a machine with these speeds (pp_tps, tg_tps), from its token counts."""
    if not r.get("completion_tokens") or not speeds.get("tg_tps") or not speeds.get("pp_tps"):
        return None
    return (r.get("prompt_tokens") or 0) / speeds["pp_tps"] + r["completion_tokens"] / speeds["tg_tps"]


def p90_of(times):
    times = sorted(times)
    return statistics.quantiles(times, n=10)[-1] if len(times) > 1 else (times[0] if times else None)


def latency(pack, records, machine_id, speeds=None, fit_note=None):
    """Latency verdict for one pack on one machine. fit_note set = the model doesn't fit there."""
    limit = pack.gate.get("max_p90_s")
    if fit_note:
        return Latency(machine_id, "DOESN'T FIT", None, "", fit_note, limit)
    measured = [r["total_s"] for r in records if r.get("machine") == machine_id and r.get("total_s") is not None]
    kind, times = "measured", measured
    if len(measured) < min(MIN_TIMED, len(records)):   # too few answers timed here: project instead
        proj = [t for t in (projected_seconds(r, speeds or {}) for r in records) if t is not None]
        if len(proj) > len(measured):
            kind, times = "projected", proj
        elif not measured:
            why = "not run here and not tuned here" if not speeds else "no token counts to project from"
            return Latency(machine_id, "NO DATA", None, "", why if records else "no results", limit)
    p90 = p90_of(times)
    status = "TOO SLOW" if limit and p90 > limit else "OK"
    return Latency(machine_id, status, p90, kind, "", limit)


def machine_speeds(engine, m, machine):
    """(speeds or None, fit_note or None) for a model on a machine, from its tuned profile."""
    sv = engine.serving(m, machine)
    speeds = (sv.profile or {}).get("measured") or None
    if speeds and not (speeds.get("pp_tps") and speeds.get("tg_tps")):
        speeds = None
    return speeds, (None if sv.fits else sv.fit_note)


def latency_table(engine, labels=None, table=None):
    """{label: {use_case: {machine_id: Latency}}}: the worst pack with a time limit per use case
    (use cases whose packs have no limit report their slowest pack)."""
    table = table if table is not None else readiness(engine, labels)
    out = {}
    known = engine.machines()
    for label, groups in table.items():
        m = engine.model(label)
        per_machine = {mc.id: machine_speeds(engine, m, mc) for mc in known}
        out[label] = {}
        for group, (_, packs) in groups.items():
            row = {}
            for mc in known:
                speeds, fit_note = per_machine[mc.id]
                lats = [latency(engine.packs[name], engine.records(label, name), mc.id, speeds, fit_note)
                        for name, pv in packs.items() if pv.status != "NO DATA"]
                limited = [x for x in lats if x.limit] or lats
                if limited:
                    row[mc.id] = min(limited, key=lambda x: (LAT_ORDER[x.status], -(x.p90_s or 0)))
            out[label][group] = row
    return out


def readiness(engine, labels=None):
    """{label: {use_case: (verdict, {pack: verdict})}} for every model with results.
    A use case is a pack group (pack.toml `group`), e.g. Coding = the coding and coding-advanced packs."""
    out = {}
    for m in engine.cfg["models"]:
        if labels and m["label"] not in labels:
            continue
        by_group = {}
        for name, pack in engine.packs.items():
            recs = engine.records(m["label"], name)
            if engine.result_status(m["label"], name) not in ("certified", "screened"):
                recs = []  # missing or outdated results don't count
            by_group.setdefault(pack.group, {})[name] = evaluate(pack, recs, engine.certify_repeat(pack))
        groups = {g: (combine(v), v) for g, v in by_group.items()
                  if any(x.status != "NO DATA" for x in v.values())}
        if groups:
            out[m["label"]] = groups
    return out


def _plain_fail(v):
    """A FAIL's main reason in a few plain words."""
    ev = v.evidence
    if ev.get("critical_failures"):
        n = ev["critical_failures"]
        return f"{n} critical failure{'s' * (n > 1)}"
    text = (v.reasons or ["failed"])[0]
    if "cut off" in text:
        return "too many answers cut off by max_tokens"
    if "accuracy" in text and ev.get("trials"):
        return f"{100 * ev['accuracy']:.0f}% right, under the bar"
    return text.split(" (")[0]


def _plain_todo(v):
    """What it takes to decide an INCONCLUSIVE verdict, in plain words (None: not an action)."""
    ev = v.evidence
    if not ev:   # a use case with packs not run yet
        return ("run " + v.reasons[0].replace(" not run yet", "")) if v.reasons else None
    if not ev.get("certified"):
        if ev.get("tests_run", 0) < ev.get("tests_total", 0):
            promising = any("promising" in r for r in v.reasons)
            return "run Certify" + (" (the sample screened so far looks promising)" if promising
                                    else " (only a sample has been screened)")
        return f"run Certify to finish ({ev['want_repeat'] - ev['repeats']} more round(s) of every question)"
    if any("critical trials" in r for r in v.reasons):
        return "needs more answers to rule out rare critical failures: more tests that can fail critically, or more repeats"
    return "needs more tests or repeats to be sure"


def plain_summary(table):
    """[(label, sentence, [(use case, what to do, [packs])])] per model: what it's ready for, what it
    isn't, and what's undecided with the step that decides it."""
    out = []
    for label, groups in table.items():
        ready = [g for g, (v, _) in groups.items() if v.status == "PASS"]
        not_ready = []
        todo = []
        for g, (v, packs) in groups.items():
            if v.status == "FAIL":
                bad = next((pv for pv in packs.values() if pv.status == "FAIL"), v)
                not_ready.append(f"{g} ({_plain_fail(bad)})")
            elif v.status == "INCONCLUSIVE":
                pending = {n: pv for n, pv in packs.items() if pv.status in ("INCONCLUSIVE", "NO DATA")}
                first = next((pv for pv in pending.values() if pv.status == "INCONCLUSIVE"), v)
                todo.append((g, _plain_todo(first) or "; ".join(v.reasons), list(pending)))
        parts = []
        if ready:
            parts.append("ready for " + ", ".join(ready))
        if not_ready:
            parts.append("not ready for " + ", ".join(not_ready))
        if todo:
            parts.append("not decided yet for " + ", ".join(g for g, _, _ in todo))
        out.append((label, "; ".join(parts) or "no verdicts yet", todo))
    return out


def combine(verdicts):
    """A use case (all packs in a group) is as good as its worst pack."""
    if not verdicts:
        return Verdict("NO DATA", ["no packs"], {})
    ran = {k: v for k, v in verdicts.items() if v.status != "NO DATA"}
    missing = [k for k, v in verdicts.items() if v.status == "NO DATA"]
    worst = min(ran.values(), key=lambda v: ORDER[v.status]) if ran else list(verdicts.values())[0]
    if missing and worst.status == "PASS":
        return Verdict("INCONCLUSIVE", [f"{', '.join(missing)} not run yet"], {})
    reasons = [f"{name}: {v.summary}" for name, v in verdicts.items() if v.status != "PASS"] or \
              [f"all {len(verdicts)} pack(s) pass"]
    return Verdict(worst.status, reasons, {})


# ---------------------------------------------------------------- history
# results/verdicts.jsonl is append-only: one line each time a model's verdict for a use case
# changes (status, the packs behind it, their question versions, or certification coverage).
# It keeps the record of what passed when, even after results are rerun or become outdated.

def history_path(engine):
    return os.path.join(engine.results_dir, "verdicts.jsonl")


def load_history(engine):
    path = history_path(engine)
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _snapshot(v, packs, engine):
    out = {}
    for name, pv in packs.items():
        if pv.status == "NO DATA":
            continue
        ev = pv.evidence
        out[name] = {"status": pv.status, "reasons": pv.reasons[:2], "fingerprint": engine.packs[name].fingerprint,
                     "certified": ev.get("certified"), "accuracy": round(ev.get("accuracy", 0), 4),
                     "acc_lo": round(ev.get("acc_lo", 0), 4), "acc_hi": round(ev.get("acc_hi", 0), 4),
                     "critical_failures": ev.get("critical_failures"), "critical_trials": ev.get("critical_trials")}
    return out


def _key(entry):
    return (entry["status"], tuple(sorted((n, p["status"], p["fingerprint"], p["certified"])
                                          for n, p in entry["packs"].items())),
            tuple(sorted(entry.get("latency", {}).items())))


def latest(history):
    """{(model, use_case): last entry}"""
    out = {}
    for e in history:
        out[(e["model"], e["use_case"])] = e
    return out


def record_history(engine, labels=None):
    """Append an entry for every (model, use case) whose verdict changed since its last entry.
    Returns the new entries (each with 'previous', the prior status or None)."""
    table = readiness(engine, labels)
    last = latest(load_history(engine))
    now = datetime.datetime.now().isoformat(timespec="seconds")
    new = []
    for label, groups in table.items():
        m = engine.model(label)
        from .engine import settings_fingerprint, _fp  # local: engine imports this module
        sfp = settings_fingerprint(engine.settings(m))
        same_settings = (sfp, _fp(engine.settings(m)))   # entries fingerprinted with the context size too
        try:
            lat = latency_table(engine, [label], {label: groups}).get(label, {})
        except Exception:  # latency must never block recording the quality verdict
            lat = {}
        for group, (v, packs) in groups.items():
            entry = {"time": now, "model": label, "use_case": group, "status": v.status, "reasons": v.reasons[:3],
                     "packs": _snapshot(v, packs, engine), "settings_fingerprint": sfp,
                     "latency": {mid: x.status for mid, x in lat.get(group, {}).items() if x.status != "NO DATA"}}
            prev = last.get((label, group))
            if prev and _key(prev) == _key(entry) and prev.get("settings_fingerprint") in same_settings:
                continue
            entry["previous"] = prev["status"] if prev else None
            new.append(entry)
    if new:
        os.makedirs(os.path.dirname(history_path(engine)), exist_ok=True)
        with open(history_path(engine), "a") as f:
            for e in new:
                f.write(json.dumps(e) + "\n")
    return new


def is_current(entry, engine):
    """True if an entry was based on today's questions for every pack it covers."""
    return all(engine.packs.get(n) is not None and engine.packs[n].fingerprint == p["fingerprint"]
               for n, p in entry["packs"].items())
