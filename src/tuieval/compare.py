"""Scorecards from result files: accuracy, speed and token use per model.

Usage:
    tuieval compare                          # everything in results/<model>/<pack>.json
    tuieval compare --speed                  # speed & tokens table (TTFT, tok/s, tokens, memory)
    tuieval compare --machine m3max-64gb     # speed on another machine (measured there, or projected)
    tuieval compare --pairwise --failures    # is each difference real? which tests failed?
    tuieval compare results/some-model/*.json    # specific files
"""
import collections
import glob
import json
import os
import random
import re
import statistics
import sys

from . import client
from . import graders
from . import workspace

BOOTSTRAP = 2000
REPEAT_RE = re.compile(r"(.{1,8}?)\1{49,}", re.DOTALL)


# ------------------------------------------------------------------ loading
def repetitive(text):
    """True if the tail is one short pattern repeated 50+ times (a degenerate loop).
    Patterns without a letter or digit are ignored, so '# ------' rulers don't count."""
    return any(re.search(r"[A-Za-z0-9]", m.group(1)) for m in REPEAT_RE.finditer((text or "")[-400:]))


def _load_native(data):
    m, pack, run = data["model"], data["pack"], data["run"]
    info = {"label": m["label"], "served": m.get("served_name"), "tags": m.get("tags") or [],
            "settings": data.get("settings", {}), "load_s": run.get("load_s"),
            "peak_rss_mb": run.get("peak_rss_mb"), "pack": pack["name"], "pack_fp": pack["fingerprint"],
            # hosted (OpenRouter) models: the provider endpoint and its declared quantization
            "endpoint": run.get("endpoint"), "quantization": run.get("quantization")}
    rows = []
    for r in data["results"]:
        rows.append({
            "model": m["label"], "suite": pack["name"], "test": f"{pack['name']}: {r['test']}",
            "name": f"{pack['name']}: {r['description']}",
            "ok": bool(r["pass"]), "score": float(r["score"]), "reason": r["reason"],
            "error": r["reason"].startswith(("server returned", "connection error")),
            "truncated": r.get("finish") == "length", "output": r.get("answer") or "",
            "tokens": r.get("completion_tokens") or 0, "reasoning_tokens": r.get("reasoning_tokens"),
            # a reused (cached) prompt makes time-to-first-token and prompt speed look faster than they are
            "answer_tokens": r.get("answer_tokens"), "ttft": None if r.get("cached_tokens") else r.get("ttft_s"),
            "latency": (r.get("total_s") or 0) * 1000, "gen_tps": r.get("gen_tps"),
            "prompt_tps": None if r.get("cached_tokens") else r.get("prompt_tps"),
            "severity": r.get("severity"), "group": r.get("group"), "repeat": r.get("repeat", 0),
            "difficulty": r.get("difficulty", "unrated"),
            "machine": r.get("machine") or run.get("machine"), "prompt_tokens": r.get("prompt_tokens"),
        })
    return rows, info


def load(paths):
    """(rows, infos): one row per answer; one info per (model, result file)."""
    rows, infos = [], []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        if data.get("format") != "evals/1":
            continue
        data["results"] = [r for r in data["results"] if not client.server_error_row(r)]
        r, i = _load_native(data)
        for row, rec in zip(r, data["results"]):
            row.update(raw=rec, path=path, pack_fp=data["pack"]["fingerprint"])
        rows += r
        infos.append(i)
    return rows, infos


def default_paths(results_dir=None):
    """results/<label>/<pack>.json, excluding smoke runs and the archive."""
    results_dir = results_dir or workspace.path("results")
    return sorted(p for p in glob.glob(os.path.join(results_dir, "*", "*.json"))
                  if os.path.basename(os.path.dirname(p)) not in ("smoke", "archive"))


PARTIAL_SUFFIX = ".json.partial.jsonl"


def load_partials(results_dir=None):
    """Rows for answers of packs still running (or stopped before the end): results/<label>/
    <pack>.json.partial.jsonl, marked partial. They never count toward scores or verdicts."""
    results_dir = results_dir or workspace.path("results")
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "*" + PARTIAL_SUFFIX))):
        label = os.path.basename(os.path.dirname(path))
        if label in ("smoke", "archive"):
            continue
        try:
            with open(path) as f:
                lines = [json.loads(l) for l in f if l.strip()]
        except (OSError, ValueError):
            continue  # being written right now; the next refresh gets it
        if len(lines) < 2:
            continue
        head, recs = lines[0], [r for r in lines[1:] if not client.server_error_row(r)]
        data = {"model": {"label": label}, "pack": {"name": os.path.basename(path)[:-len(PARTIAL_SUFFIX)],
                                                    "fingerprint": head.get("pack_fingerprint")},
                "run": {}, "results": recs}
        r, _ = _load_native(data)
        for row, rec in zip(r, recs):
            row.update(raw=rec, path=path, pack_fp=head.get("pack_fingerprint"), partial=True)
        rows += r
    return rows


# ------------------------------------------------------------------ stats
def p90(values):
    values = [v for v in values if v is not None]
    return statistics.quantiles(values, n=10)[-1] if len(values) > 1 else (values[0] if values else None)


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def per_test_means(rows):
    by_test = collections.defaultdict(list)
    for r in rows:
        by_test[r["test"]].append(r["ok"])
    return {t: sum(v) / len(v) for t, v in by_test.items()}


def bootstrap_ci(values, rng):
    """95% CI of the mean, resampling tests (each test's repeats stay together)."""
    if not values:
        return 0.0, 0.0
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(BOOTSTRAP))
    return means[int(0.025 * BOOTSTRAP)], means[int(0.975 * BOOTSTRAP) - 1]


def _fmt(v, spec, dash="-"):
    return dash if v is None else format(v, spec)


# ------------------------------------------------------------------ reports
def settings_notes(infos):
    """Human-readable warnings: settings that differ between models, packs with mixed versions."""
    notes = []
    per_model = {}
    for i in infos:
        per_model.setdefault(i["label"], i["settings"])
    keys = sorted({k for s in per_model.values() for k in s})
    differing = [k for k in keys if len({json.dumps(s.get(k)) for s in per_model.values()}) > 1]
    if differing:
        detail = "; ".join(f"{k}: " + ", ".join(f"{m}={per_model[m].get(k)}" for m in sorted(per_model))
                           for k in differing)
        notes.append(f"Settings differ between models ({detail}). Compare those models knowing that.")
    fps = collections.defaultdict(set)
    for i in infos:
        if i.get("pack"):
            fps[i["pack"]].add(i["pack_fp"])
    for pack, v in sorted(fps.items()):
        if len(v) > 1:
            notes.append(f"Pack '{pack}': results come from {len(v)} different versions of its questions; "
                         "rerun the outdated ones before comparing on it.")
    return notes


def scorecard(rows, rng=None):
    """(header, table): accuracy per pack and overall."""
    rng = rng or random.Random(0)
    models = sorted({r["model"] for r in rows})
    suites = sorted({r["suite"] for r in rows})
    extra = dict(graders.COLUMNS)   # columns your graders add (graders.scorecard_column)
    header = ["model"] + suites + ["overall", "95% CI", "mean score", "all repeats", *extra, "errors", "trunc", "rep"]
    table = []
    for m in models:
        mine = [r for r in rows if r["model"] == m]
        line = [m]
        for s in suites:
            sub = [r for r in mine if r["suite"] == s]
            line.append(f"{sum(r['ok'] for r in sub)}/{len(sub)}" if sub else "-")
        lo, hi = bootstrap_ci(list(per_test_means(mine).values()), rng)
        line += [f"{100 * sum(r['ok'] for r in mine) / len(mine):.1f}%", f"{100 * lo:.0f}-{100 * hi:.0f}%"]
        # partial credit (e.g. the code grader's share of checks passed), and tests passed on every repeat
        line.append(f"{statistics.fmean(r['score'] for r in mine):.2f}")
        tmeans = per_test_means(mine)
        line.append(f"{sum(v == 1 for v in tmeans.values())}/{len(tmeans)}")
        for title, fn in extra.items():
            try:
                cell = fn(mine)
            except Exception:  # a broken column must not hide the scorecard
                cell = "error"
            line.append("-" if cell is None else str(cell))
        line += [str(sum(r["error"] for r in mine)), str(sum(r["truncated"] for r in mine)),
                 str(sum(repetitive(r["output"]) for r in mine))]
        table.append(line)
    return header, table


def timing(r, machine=None, speeds=None):
    """(seconds, ttft, gen_tps, prompt_tps, kind) for one answer. With a machine: measured if the
    answer was run there, else projected from its token counts and that machine's tuned speeds
    (speeds = {"pp_tps", "tg_tps"}), else unknown."""
    if machine is None or r.get("machine") == machine:
        return r["latency"] / 1000, r["ttft"], r["gen_tps"], r["prompt_tps"], "measured"
    if speeds and speeds.get("pp_tps") and speeds.get("tg_tps") and r.get("tokens"):
        ttft = (r.get("prompt_tokens") or 0) / speeds["pp_tps"]
        return ttft + r["tokens"] / speeds["tg_tps"], ttft, speeds["tg_tps"], speeds["pp_tps"], "projected"
    return None, None, None, None, None


def speed(rows, infos, machine=None, speeds=None):
    """(header, table, frontier): speed and token use per model.

    machine: show times on that machine (measured there, or projected with speeds[model]).
    frontier = models no other model beats on both accuracy and median time per answer."""
    speeds = speeds or {}
    models = sorted({r["model"] for r in rows})
    by_model = {m: [r for r in rows if r["model"] == m] for m in models}
    tim = {id(r): timing(r, machine, speeds.get(r["model"])) for r in rows}
    acc = {m: sum(r["ok"] for r in v) / len(v) for m, v in by_model.items()}
    med_time = {m: median(tim[id(r)][0] for r in v) or 0 for m, v in by_model.items()}
    frontier = {m for m in models if not any(
        acc[o] >= acc[m] and med_time[o] <= med_time[m] and (acc[o] > acc[m] or med_time[o] < med_time[m])
        for o in models if o != m)}
    load = {}
    for i in infos:
        cur = load.setdefault(i["label"], {"load_s": None, "peak_rss_mb": None, "tags": i["tags"],
                                           "think": i["settings"].get("enable_thinking"), "served": set()})
        if i.get("endpoint"):
            q = i.get("quantization")
            cur["served"].add(f"{'undeclared' if q in (None, 'unknown') else q} via {i['endpoint']}")
        for k in ("load_s", "peak_rss_mb"):
            if i.get(k) is not None:
                cur[k] = max(cur[k] or 0, i[k])
    header = ["", "model", "quantization", "tags", "think", "timing", "accuracy", "med s/answer", "p90 s", "TTFT s", "gen tok/s",
              "prompt tok/s", "med tokens", "reasoning", "answer", "tokens per ✓", "load s", "peak mem GB"]
    table = []
    for m in sorted(models, key=lambda m: (-acc[m], med_time[m])):
        v, info = by_model[m], load.get(m, {})
        passes = sum(r["ok"] for r in v)
        t = [tim[id(r)] for r in v]
        kinds = {x[4] for x in t if x[4]}
        where = sorted({r["machine"] for r in v if r.get("machine")})
        kind = ("mixed" if len(kinds) > 1 else next(iter(kinds))) if kinds else "no data"
        here = machine is None or machine in where   # load time and memory only where it ran
        if machine is None:
            kind = "measured on " + (", ".join(where) or "?")
        table.append([
            "★" if m in frontier else "", m, "; ".join(sorted(info.get("served") or [])) or "-",
            ",".join(info.get("tags") or []),
            {True: "on", False: "off"}.get(info.get("think"), "-"), kind,
            f"{100 * acc[m]:.1f}%", _fmt(med_time[m] or None, ".1f"), _fmt(p90(x[0] for x in t), ".1f"),
            _fmt(median(x[1] for x in t), ".2f"), _fmt(median(x[2] for x in t), ".1f"),
            _fmt(median(x[3] for x in t), ".0f"), _fmt(median(r["tokens"] for r in v), ".0f"),
            _fmt(median(r["reasoning_tokens"] for r in v), ".0f"), _fmt(median(r["answer_tokens"] for r in v), ".0f"),
            f"{sum(r['tokens'] for r in v) / passes:,.0f}" if passes else "-",
            _fmt(info.get("load_s") if here else None, ".0f"),
            _fmt(info["peak_rss_mb"] / 1024 if info.get("peak_rss_mb") and here else None, ".1f"),
        ])
    return header, table, frontier


def per_question(rows):
    """[{test, name, suite, difficulty, cells: {model: cell}, spread}] with one cell per model that
    answered: n answers, passes, median tokens / seconds / tok/s over repeats, truncated count,
    partial (some answers from a pack still running), rows (the answers, for the detail view).
    spread = {"tokens": .., "secs": ..}: largest / smallest model median (1 if one model)."""
    tests = {}
    for r in rows:
        t = tests.setdefault(r["test"], {"test": r["test"], "name": r.get("name", r["test"]), "suite": r["suite"],
                                         "difficulty": r.get("difficulty", "unrated"), "cells": {}})
        c = t["cells"].setdefault(r["model"], {"rows": []})
        c["rows"].append(r)
    out = []
    for t in tests.values():
        for c in t["cells"].values():
            v = sorted(c["rows"], key=lambda r: r.get("repeat", 0))
            c.update(rows=v, n=len(v), passed=sum(r["ok"] for r in v), truncated=sum(r["truncated"] for r in v),
                     partial=any(r.get("partial") for r in v),
                     tokens=median(r["tokens"] or None for r in v), secs=median(r["latency"] / 1000 or None for r in v),
                     tps=median(r["gen_tps"] for r in v))
        t["spread"] = _spread(t["cells"])
        out.append(t)
    return out


def _spread(cells):
    out = {}
    for k in ("tokens", "secs"):
        vals = [c[k] for c in cells.values() if c[k]]
        out[k] = max(vals) / min(vals) if len(vals) > 1 else 1.0
    return out


def with_models(test, models):
    """A per_question entry narrowed to some models (spread recomputed over them)."""
    cells = {m: c for m, c in test["cells"].items() if m in models}
    return dict(test, cells=cells, spread=_spread(cells))


def compare_models(tests, models):
    """Per model over per_question entries that all `models` answered: passes, token and time
    totals/medians (totals add each question's median over repeats, so repeats don't inflate them), questions where it used the fewest tokens / was fastest (among passing
    answers for tokens; ties count for each), answers cut off."""
    out = []
    for m in models:
        cs = [t["cells"][m] for t in tests]
        passed = sum(c["passed"] == c["n"] for c in cs)
        total = sum((c["tokens"] or 0) for c in cs)
        fewest = fastest = 0
        for t in tests:
            ok = [c["tokens"] for c in t["cells"].values() if c["passed"] and c["tokens"]]
            c = t["cells"][m]
            if c["passed"] and c["tokens"] and c["tokens"] == min(ok):
                fewest += 1
            secs = [x["secs"] for x in t["cells"].values() if x["secs"]]
            if c["secs"] and c["secs"] == min(secs):
                fastest += 1
        rs = [r for c in cs for r in c["rows"]]
        out.append({"model": m, "n": len(cs), "passed": passed, "total_tokens": total,
                    "avg_pp_tps": _rate([(r.get("prompt_tokens"), r["prompt_tps"]) for r in rs]),
                    "avg_gen_tps": _rate([(r["tokens"], r["gen_tps"]) for r in rs]),
                    "med_tokens": median(c["tokens"] for c in cs), "med_secs": median(c["secs"] for c in cs),
                    "total_secs": sum(c["secs"] or 0 for c in cs) or None,
                    "tokens_per_pass": total / passed if passed else None,
                    "fewest": fewest, "fastest": fastest, "cut": sum(c["truncated"] for c in cs)})
    return out


def _rate(pairs):
    """Average tok/s over answers given (tokens, tok/s) pairs: all their tokens over all their time,
    so long prompts and answers weigh as much as they took (a plain mean would over-count short ones)."""
    pairs = [(n, tps) for n, tps in pairs if n and tps]
    secs = sum(n / tps for n, tps in pairs)
    return sum(n for n, _ in pairs) / secs if secs else None


def head_to_head(tests, a, b):
    """Two models on the questions both answered: who passed what, who used fewer tokens."""
    h = {"only_a": 0, "only_b": 0, "both": 0, "neither": 0, "fewer_a": 0, "fewer_b": 0}
    ratios = []
    for t in tests:
        ca, cb = t["cells"][a], t["cells"][b]
        pa, pb = ca["passed"] == ca["n"], cb["passed"] == cb["n"]
        h[{(True, False): "only_a", (False, True): "only_b", (True, True): "both", (False, False): "neither"}[pa, pb]] += 1
        if ca["tokens"] and cb["tokens"]:
            h["fewer_a"] += ca["tokens"] < cb["tokens"]
            h["fewer_b"] += cb["tokens"] < ca["tokens"]
            ratios.append(ca["tokens"] / cb["tokens"])
    h["ratio"] = median(ratios)
    return h


def separating(rows):
    """Tests where models disagree: ([(test, {model: [ok, ...]})], total tests)."""
    by_test = collections.defaultdict(dict)
    for r in rows:
        by_test[r["test"]].setdefault(r["model"], []).append(r["ok"])
    split = [t for t, d in by_test.items() if len({round(sum(v) / len(v), 2) for v in d.values()}) > 1]
    return [(t, by_test[t]) for t in sorted(split)], len(by_test)


def pairwise(rows, rng=None):
    """[(a, b, diff, lo, hi, verdict, n_tests)] for every pair of models, over shared tests."""
    rng = rng or random.Random(0)
    models = sorted({r["model"] for r in rows})
    means = {m: per_test_means([r for r in rows if r["model"] == m]) for m in models}
    out = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            shared = sorted(set(means[a]) & set(means[b]))
            if not shared:
                continue
            diffs = [means[a][t] - means[b][t] for t in shared]
            lo, hi = bootstrap_ci(diffs, rng)
            out.append((a, b, statistics.fmean(diffs), lo, hi, "clear" if lo > 0 or hi < 0 else "noise", len(shared)))
    return out


def failures(rows):
    return [(r["model"], r.get("name", r["test"]), ("TRUNCATED " if r["truncated"] and not r["reason"].startswith("TRUNCATED")
                                                    else "") + r["reason"], r.get("difficulty", "unrated"))
            for r in rows if not r["ok"]]


LEVELS = ("easy", "medium", "hard")


def by_difficulty(rows):
    """(header, table): pass rate per difficulty level, per model and pack."""
    header = ["model", "pack"] + list(LEVELS)
    table = []
    for m in sorted({r["model"] for r in rows}):
        for s in sorted({r["suite"] for r in rows if r["model"] == m}):
            sub = [r for r in rows if r["model"] == m and r["suite"] == s]
            cells = []
            for lvl in LEVELS:
                x = [r["ok"] for r in sub if r.get("difficulty") == lvl]
                cells.append(f"{sum(x)}/{len(x)} ({100 * sum(x) / len(x):.0f}%)" if x else "-")
            table.append([m, s] + cells)
    return header, table


def items(rows):
    """Test-quality flags from results across models: [(test, flags, {model: pass rate})].

    no signal   every model passed every time (too easy to separate anyone)
    easier/harder than labelled   results contradict the test's difficulty label
    suspicious  every model failed every time (check the test before trusting it)
    inverted    the least accurate model passed more often than the most accurate one
    flaky       the same model both passed and failed it across repeats
    """
    models = sorted({r["model"] for r in rows})
    acc = {m: statistics.fmean(r["ok"] for r in rows if r["model"] == m) for m in models}
    best, worst = max(models, key=acc.get), min(models, key=acc.get)
    by_test = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by_test[r["test"]][r["model"]].append(r["ok"])
    level = {r["test"]: r.get("difficulty", "unrated") for r in rows}
    out = []
    for test, per in sorted(by_test.items()):
        rates = {m: sum(v) / len(v) for m, v in per.items()}
        flags = []
        mean = statistics.fmean(rates.values())
        if len(rates) >= 2 and level[test] == "hard" and mean == 1:
            flags.append("easier than labelled")
        if len(rates) >= 2 and level[test] == "easy" and mean <= 0.5:
            flags.append("harder than labelled")
        if len(rates) >= 2 and all(x == 1 for x in rates.values()):
            flags.append("no signal")
        if len(rates) >= 2 and all(x == 0 for x in rates.values()):
            flags.append("suspicious")
        if best != worst and best in rates and worst in rates and rates[worst] > rates[best]:
            flags.append("inverted")
        if any(0 < x < 1 and len(per[m]) > 1 for m, x in rates.items()):
            flags.append("flaky")
        if flags:
            out.append((test, flags, rates))
    return out


def print_table(header, table):
    widths = [max(len(str(x)) for x in col) for col in zip(header, *table)]
    fmt = "  ".join("{:<%d}" % w for w in widths)
    print(fmt.format(*header).rstrip())
    for line in table:
        print(fmt.format(*line).rstrip())


def scan_logs(log_dir):
    """Truncation/repetition counts from saved request logs (reasoning included)."""
    counts = collections.defaultdict(collections.Counter)
    for path in glob.glob(os.path.join(log_dir, "*.txt")):
        with open(path, errors="replace") as f:
            text = f.read()
        m = re.search(r"^MODEL: (.*)$", text, re.MULTILINE)
        model = m.group(1) if m else "?"
        counts[model]["logs"] += 1
        counts[model]["repetitive"] += repetitive(text.rsplit("\n\n=== ", 1)[0])
        counts[model]["finish=length"] += "finish=length ===" in text
    print(f"\nRequest logs in {log_dir} (reasoning included):")
    print_table(["served model", "logs", "finish=length", "repetitive tail"],
                [[m, c["logs"], c["finish=length"], c["repetitive"]] for m, c in sorted(counts.items())])


def main(argv):
    flags = {a for a in argv if a.startswith("--")}
    log_dir = argv[argv.index("--logs") + 1] if "--logs" in argv else None
    machine_arg = argv[argv.index("--machine") + 1] if "--machine" in argv else None
    paths = [a for a in argv if not a.startswith("--") and a not in (log_dir, machine_arg)] or default_paths()
    if not paths:
        sys.exit(__doc__ + "\n(no results found in results/<model>/<pack>.json: run some evals first)")
    rows, infos = load(paths)
    rng = random.Random(0)
    from . import packs as packs_mod
    packs_mod.load_packs(errors=[])   # loads the workspace's graders, and their Scorecard columns
    for note in settings_notes(infos):
        print("NOTE:", note)
    print_table(*scorecard(rows, rng))
    print("\nmean score = average score, with partial credit; all repeats = tests passed on every repeat.")
    print("trunc = hit max_tokens; rep = output ends in a degenerate repeated pattern.")
    header, table = by_difficulty(rows)
    if any(r.get("difficulty") in LEVELS for r in rows):
        print("\nPass rate by difficulty:")
        print_table(header, table)
    machine = argv[argv.index("--machine") + 1] if "--machine" in argv else None
    if "--speed" in flags or machine:
        speeds = {}
        if machine:
            from . import engine
            e = engine.Engine()
            mc = next((x for x in e.machines() if x.id == machine), None)
            if mc is None:
                sys.exit(f"unknown machine {machine!r}; known: {', '.join(x.id for x in e.machines())}")
            for m in e.cfg["models"]:
                sv = e.serving(m, mc)
                speeds[m["label"]] = (sv.profile or {}).get("measured")
        header, table, _ = speed(rows, infos, machine, speeds)
        print(f"\nSpeed and tokens{' on ' + machine if machine else ''} "
              "(★ = no other model is both more accurate and faster per answer):")
        print_table(header, table)
    split, n_tests = separating(rows)
    print(f"\n{len(split)} of {n_tests} tests separate the models:")
    for t, d in split:
        print(f"  {t}: " + ", ".join(f"{m} {sum(v)}/{len(v)}" for m, v in sorted(d.items())))
    if "--pairwise" in flags:
        print("\nPaired differences in pass rate (row minus column model, 95% bootstrap CI over shared tests):")
        for a, b, diff, lo, hi, verdict, n in pairwise(rows, rng):
            print(f"  {a} - {b}: {100 * diff:+.1f} pts [{100 * lo:+.1f}, {100 * hi:+.1f}]  {verdict}  ({n} tests)")
    if "--failures" in flags:
        print("\nFailures:")
        for model, test, reason, level in failures(rows):
            print(f"  [{model}] ({level}) {test}: {reason[:150]}")
    if log_dir:
        scan_logs(log_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
