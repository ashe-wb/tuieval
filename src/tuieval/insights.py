"""Insights: what the results say to do next, as a chart per question with plain action lines under it
(Results → Insights, `tuieval insights`).

  1 vs reference    each model against one reference model, paired on the questions both answered:
                    accuracy points with a 95% CI, better / tie / worse
  2 use cases       accuracy per model and use case (pack group) as a heatmap; the best marked
  3 vs base         a fine-tune against its base model (models.toml `base`): per pack, and the question
                    categories it gained or lost most
  4 why they fail   each model's answers split into right, wrong, cut off, no answer / bad format, code
                    crashed or hung, and critical failures
  5 accuracy / cost accuracy against median tokens (or seconds) per question; the models no other beats
                    on both
  6 questions       questions every model fails (check them), every model passes (no signal), and the
                    ones that split models most
  7 next runs       the Certify runs that would settle the open verdicts of the strongest models, with
                    their time and command

Every panel returns {"title", "lines", "actions"}: lines are Rich markup (the chart), actions plain
sentences. Server and connection errors are never answers here either.
"""
import collections
import math
import statistics

from rich.markup import escape as rich_escape

from . import compare

MIN_SHARE = 0.5      # like the PTA index: models with fewer answers than this share of the most-answered are left out
HALF = 15            # half width of a diverging bar
COVERED = 0.8        # a use case counts as covered when a model answered this share of its questions
MAX_ACTIONS = 6


def _esc(s):
    return rich_escape(str(s))


def prepare(rows, picked=None, ref=None):
    """(rows without errors, the models to show): picked ones, else every model with enough answers;
    the reference is always shown when it has answers."""
    rows = [r for r in rows if not r.get("error")]
    counts = collections.Counter(r["model"] for r in rows)
    if picked:
        models = sorted(m for m in picked if m in counts)
    else:
        most = max(counts.values(), default=0)
        models = sorted(m for m, n in counts.items() if n >= MIN_SHARE * most)
    if ref in counts and ref not in models:
        models.append(ref)
    return [r for r in rows if r["model"] in models], models


def default_reference(rows, models, configured=None):
    """models.toml [defaults] reference when it has answers, else the most accurate model shown."""
    if configured in models:
        return configured
    acc = {m: statistics.fmean(r["ok"] for r in rows if r["model"] == m) for m in models
           if any(r["model"] == m for r in rows)}
    return max(acc, key=lambda m: (acc[m], m), default=None)


def _pts(x):
    return f"{100 * x:+.1f}"


def diverging(diff, lo, hi, scale, color):
    """A bar from the centre line to diff, the CI around it as a thin line: -scale .. +scale."""
    out = []
    for i in range(2 * HALF + 1):
        v = (i - HALF) / HALF * scale
        step = scale / HALF / 2
        filled = (0 < v <= diff + step) if diff > 0 else (diff - step <= v < 0) if diff < 0 else False
        if i == HALF:
            out.append(f"[{color}]█[/]" if filled else "│")
        elif filled:
            out.append(f"[{color}]█[/]")
        elif lo - step <= v <= hi + step:
            out.append("[#808080]─[/]")
        else:
            out.append(" ")
    return "".join(out)


VERDICT_COLOR = {"better": "green", "worse": "red", "tie": "yellow"}


def _vs_lines(results, name_of=lambda x: x["model"]):
    if not results:
        return []
    scale = max(0.05, *(max(abs(x["lo"]), abs(x["hi"]), abs(x["diff"])) for x in results))
    w = max(len(name_of(x)) for x in results)
    lines = [" " * (w + 2) + f"[dim]{_pts(-scale)} pts" + " " * (2 * HALF - 16) + f"{_pts(scale)} pts[/dim]"]
    for x in results:
        color = VERDICT_COLOR[x["verdict"]]
        lines.append(f"{_esc(name_of(x)):<{w}}  {diverging(x['diff'], x['lo'], x['hi'], scale, color)}  "
                     f"[{color}]{_pts(x['diff'])} pts[/] [dim]({_pts(x['lo'])} to {_pts(x['hi'])}, "
                     f"{x['n']} q)[/dim] [{color}]{x['verdict'].upper()}[/]")
    return lines


# ---------------------------------------------------------------- 1 vs reference
def vs_reference(rows, models, ref):
    title = "Who's worth keeping: accuracy against the reference"
    if not ref:
        return {"title": title, "lines": ["[dim]No reference model with answers.[/dim]"], "actions": []}
    res = sorted((x for x in compare.vs_reference(rows, ref) if x["pack"] == "all" and x["model"] in models),
                 key=lambda x: -x["diff"])
    acc = statistics.fmean(r["ok"] for r in rows if r["model"] == ref) if any(r["model"] == ref for r in rows) else 0
    lines = [f"[dim]Reference: [b]{_esc(ref)}[/b] ({compare.pct(acc)} right). Paired on the questions both "
             "answered; the bar is the difference in points, the thin line its 95% CI.[/dim]"] + _vs_lines(res)
    actions = []
    for x in res:
        if x["verdict"] == "better":
            actions.append(f"{x['model']} beats {ref} by {100 * x['diff']:.1f} pts on {x['n']} shared questions.")
    worse = [x for x in res if x["verdict"] == "worse"]
    if worse:
        actions.append(f"Behind {ref} for certain: " + ", ".join(f"{x['model']} ({_pts(x['diff'])})" for x in worse[:5])
                       + (f" and {len(worse) - 5} more" if len(worse) > 5 else "") + ": candidates to drop.")
    same = [x for x in res if x["verdict"] == "tie" and x["lo"] == x["hi"] == 0]
    if same:
        actions.append("Same result as " + ref + " on every shared question: " + ", ".join(x["model"] for x in same)
                       + " (harder questions would separate them).")
    ties = [x for x in res if x["verdict"] == "tie" and x not in same]
    if ties:
        actions.append("Too close to call against " + ref + ": " + ", ".join(
            f"{x['model']} ({_pts(x['diff'])}, CI {_pts(x['lo'])} to {_pts(x['hi'])})" for x in ties[:4])
            + ": more questions or a Certify run would decide.")
    return {"title": title, "lines": lines, "actions": actions}


# ---------------------------------------------------------------- 2 use cases
HEAT = ((0.9, "bold green"), (0.75, "bold yellow"), (0.5, "bold #ff8c00"), (0.0, "bold red"))
CELL = 12


def _heat(acc):
    return next(style for floor, style in HEAT if acc >= floor)


def use_cases(rows, models, packs):
    """packs: {name: Pack} (for groups and question counts)."""
    title = "Best model per use case"
    groups = []
    for name, p in packs.items():
        if p.group not in groups:
            groups.append(p.group)
    groups = [g for g in groups if any(packs.get(r["suite"]) and packs[r["suite"]].group == g for r in rows)]
    if not groups:
        return {"title": title, "lines": ["[dim]No results yet.[/dim]"], "actions": []}
    size = {g: sum(len(p.tests) for p in packs.values() if p.group == g) for g in groups}
    per = collections.defaultdict(lambda: collections.defaultdict(list))     # (model, group) -> test -> oks
    for r in rows:
        if r["suite"] in packs:
            per[r["model"], packs[r["suite"]].group][r["test"]].append(r["ok"])
    cell = {}
    for m in models:
        for g in groups:
            tests = per.get((m, g))
            if tests:
                cell[m, g] = (statistics.fmean(statistics.fmean(v) for v in tests.values()), len(tests) / max(1, size[g]))
    best = {}
    for g in groups:
        full = {m: a for (m, gg), (a, cov) in cell.items() if gg == g and cov >= COVERED}
        if full:
            top = max(full.values())
            best[g] = (sorted((m for m, a in full.items() if a == top)), sorted(full.items(), key=lambda x: -x[1]))
    w = max(len(m) for m in models)
    short = {g: (g if len(g) <= CELL - 1 else f"{groups.index(g) + 1} {g[:CELL - 4]}…") for g in groups}
    lines = [" " * (w + 2) + "".join(f"[b]{short[g]:<{CELL}}[/b]" for g in groups)]
    for m in models:
        cells = []
        for g in groups:
            if (m, g) not in cell:
                cells.append(f"[#808080]{'-':^{CELL - 1}}[/] ")
                continue
            a, cov = cell[m, g]
            star = "★" if g in best and m in best[g][0] else " "
            text = f"██ {100 * a:>3.0f}%{'◔' if cov < COVERED else ' '}{star}"
            cells.append(f"[{_heat(a)}]{text:<{CELL - 1}}[/] ")
        lines.append(f"{_esc(m):<{w}}  " + "".join(cells))
    lines.append("[dim]★ best (among models that answered most of it) · ◔ answered under "
                 f"{100 * COVERED:.0f}% of its questions ·[/dim] " + " ".join(f"[{st}]██[/] [dim]{lbl}[/dim]" for (_, st), lbl in
                                                                zip(HEAT, ("90%+", "75%+", "50%+", "under 50%"))))
    if any(short[g] != g for g in groups):
        lines.append("[dim]" + " · ".join(f"{i + 1} {g}" for i, g in enumerate(groups) if short[g] != g) + "[/dim]")
    actions = []
    for g in groups:
        if g not in best:
            actions.append(f"{g}: no model has answered most of its questions yet.")
            continue
        tops, ranked = best[g]
        top = ranked[0][1]
        names = " or ".join(tops[:2]) + (f" (and {len(tops) - 2} more tied)" if len(tops) > 2 else "")
        line = f"{g}: use {names} ({compare.pct(top)})"
        rest = [x for x in ranked if x[0] not in tops]
        if rest:
            gap = top - rest[0][1]
            line += f"; next {rest[0][0]} ({compare.pct(rest[0][1])}, {100 * gap:.0f} pts behind)"
            if gap < 0.05:
                line += ": not a clear winner"
        if top < 0.5:
            line += ". No model gets half of it right"
        actions.append(line + ".")
    return {"title": title, "lines": lines, "actions": actions}


# ---------------------------------------------------------------- 3 vs base
def vs_base(rows, models, bases, packs):
    """bases: {model: its base model's label} (models.toml `base`)."""
    title = "Fine-tunes against their base model"
    pairs = [(m, bases[m]) for m in models if bases.get(m) and any(r["model"] == bases[m] for r in rows)]
    if not pairs:
        missing = [m for m in models if bases.get(m)]
        why = (f"The base models of {', '.join(missing)} have no answers to compare with."
               if missing else 'Add base = "<label of its base model>" to a fine-tune in models.toml to see what it '
                               "gained or lost against its base.")
        return {"title": title, "lines": [f"[dim]{_esc(why)}[/dim]"], "actions": []}
    lines, actions = [], []
    label = lambda p: packs[p].label if p in packs else p
    for m, base in pairs:
        two = [r for r in rows if r["model"] in (m, base)]
        res = [x for x in compare.vs_reference(two, base) if x["model"] == m]
        overall = next((x for x in res if x["pack"] == "all"), None)
        if not overall:
            continue
        color = VERDICT_COLOR[overall["verdict"]]
        lines.append(f"[b]{_esc(m)}[/b] vs [b]{_esc(base)}[/b]: [{color}]{_pts(overall['diff'])} pts "
                     f"{overall['verdict'].upper()}[/] [dim]({overall['n']} shared questions)[/dim]")
        lines += ["  " + line for line in _vs_lines([x for x in res if x["pack"] != "all"], lambda x: label(x["pack"]))]
        cats = categories(two, m, base)
        lost = [c for c in cats if c[1] <= -0.34][:3]
        won = [c for c in sorted(cats, key=lambda c: -c[1]) if c[1] >= 0.34][:3]
        if lost:
            lines.append("  [red]lost on:[/] " + " · ".join(f"{_esc(c)} {_pts(d)} pts ({n} q)" for c, d, n in lost))
        if won:
            lines.append("  [green]gained on:[/] " + " · ".join(f"{_esc(c)} {_pts(d)} pts ({n} q)" for c, d, n in won))
        lines.append("")
        worse = [x for x in res if x["pack"] != "all" and x["verdict"] == "worse"]
        better = [x for x in res if x["pack"] != "all" and x["verdict"] == "better"]
        if worse:
            actions.append(f"{m} is worse than its base on " + ", ".join(f"{label(x['pack'])} ({_pts(x['diff'])})"
                                                                         for x in worse)
                           + (f"; it lost most on {', '.join(c for c, _, _ in lost)}: gaps in its fine-tuning data"
                              if lost else "") + ".")
        if better:
            actions.append(f"{m} beats its base on " + ", ".join(f"{label(x['pack'])} ({_pts(x['diff'])})" for x in better)
                           + ".")
        if not worse and not better:
            actions.append(f"{m}: no clear difference from {base} on any pack yet"
                           + (f" (losses on {', '.join(c for c, _, _ in lost)} are worth a look)" if lost else "") + ".")
    return {"title": title, "lines": lines[:-1] if lines and not lines[-1] else lines, "actions": actions}


def categories(rows, m, base, min_q=2):
    """[(category, mean pass-rate difference m - base, questions)] over categories with at least min_q
    questions both answered, worst first. A question's category is its `category`, else its pack."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    cat = {}
    for r in rows:
        by[r["model"]][r["test"]].append(r["ok"])
        cat[r["test"]] = (r.get("raw") or {}).get("category") or r["suite"]
    per = collections.defaultdict(list)
    for t in set(by[m]) & set(by[base]):
        per[cat[t]].append(statistics.fmean(by[m][t]) - statistics.fmean(by[base][t]))
    return sorted(((c, statistics.fmean(v), len(v)) for c, v in per.items() if len(v) >= min_q), key=lambda c: c[1])


# ---------------------------------------------------------------- 4 why they fail
KINDS = (("right", "green", "█"), ("wrong", "red", "█"), ("cut off", "yellow", "█"),
         ("no answer / format", "magenta", "█"), ("code crashed or hung", "cyan", "█"), ("critical", "#ff5faf", "▓"))
FORMAT = ("No 'ANSWER:' line", "Empty answer", "No code", "no tool call")
CRASHED = ("Timed out after", "Harness crashed", "grader error")


def kind(r):
    if r["ok"]:
        return "right"
    if r.get("severity") == "critical":
        return "critical"
    if r["truncated"] or r["reason"].startswith("TRUNCATED"):
        return "cut off"
    if r["reason"].startswith(FORMAT):
        return "no answer / format"
    if r["reason"].startswith(CRASHED):
        return "code crashed or hung"
    return "wrong"


def _split(counts, total, width):
    """Blocks per kind, largest remainders first, so they add up to width."""
    raw = {k: width * counts.get(k, 0) / total for k, *_ in KINDS}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: -(raw[k] - out[k]))[:width - sum(out.values())]:
        out[k] += 1
    for k in out:       # a kind that happened shows at least one block
        if counts.get(k) and not out[k]:
            big = max(out, key=out.get)
            out[big], out[k] = out[big] - 1, 1
    return out


def failure_mix(rows, models, width=36):
    title = "Why models fail"
    counts = {m: collections.Counter(kind(r) for r in rows if r["model"] == m) for m in models}
    counts = {m: c for m, c in counts.items() if c}
    if not counts:
        return {"title": title, "lines": ["[dim]No answers yet.[/dim]"], "actions": []}
    w = max(len(m) for m in counts)
    lines = [" " * (w + 2) + "  ".join(f"[{c}]{ch}[/] {k}" for k, c, ch in KINDS)]
    actions = []
    for m in sorted(counts, key=lambda m: -counts[m]["right"] / sum(counts[m].values())):
        c = counts[m]
        total = sum(c.values())
        blocks = _split(c, total, width)
        bar = "".join(f"[{color}]{ch * blocks[k]}[/]" for k, color, ch in KINDS)
        fails = total - c["right"]
        detail = ", ".join(f"{c[k]} {k}" for k, *_ in KINDS[1:] if c[k])
        lines.append(f"{_esc(m):<{w}}  {bar}  {compare.pct(c['right'] / total)} right [dim]({total} answers"
                     + (f"; {detail}" if detail else "") + ")[/dim]")
        if fails >= 3:
            top, n = max(((k, c[k]) for k, *_ in KINDS[1:-1]), key=lambda x: x[1])
            if n / fails >= 0.4 and top != "wrong":
                actions.append(f"{m}: {n} of its {fails} failures are {top}: " + {
                    "cut off": "it thinks too long for max_tokens, rather than reasoning wrong.",
                    "no answer / format": "it doesn't follow the answer format (check its chat template and the "
                                          "pack's instructions).",
                    "code crashed or hung": "its code crashes or never finishes."}[top])
    crit = sorted(((m, c["critical"]) for m, c in counts.items() if c["critical"]), key=lambda x: -x[1])
    if crit:
        actions.insert(0, "Critical failures: " + ", ".join(f"{m} ({n})" for m, n in crit[:5])
                       + (f" and {len(crit) - 5} more" if len(crit) > 5 else "")
                       + ": not safe where those matter, whatever the accuracy (Failures lists them).")
    return {"title": title, "lines": lines, "actions": actions}


# ---------------------------------------------------------------- 5 accuracy / cost
MARKS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def cost_points(rows, models, by="tokens"):
    """({model: (accuracy, median cost per question)}, questions, shared): over the questions every model
    answered when there are at least 5 of them, else each model over its own."""
    tests = compare.per_question(rows)
    shared = [t for t in tests if all(m in t["cells"] for m in models)]
    use = shared if len(shared) >= 5 else tests
    out = {}
    for m in models:
        cells = [t["cells"][m] for t in use if m in t["cells"]]
        answers = sum(c["n"] for c in cells)
        cost = compare.median(c[by] for c in cells)
        if answers and cost:
            out[m] = (sum(c["passed"] for c in cells) / answers, cost)
    return out, len(use), use is shared


def front(points):
    """The models no other model beats on both accuracy and cost."""
    return {m for m, (a, c) in points.items()
            if not any(a2 >= a and c2 <= c and (a2 > a or c2 < c) for m2, (a2, c2) in points.items() if m2 != m)}


def accuracy_cost(rows, models, by="tokens", width=56, height=12, hosted=()):
    """hosted: models served by an API elsewhere; with local ones too, the local trade-offs get a line."""
    unit = "tokens" if by == "tokens" else "seconds"
    title = f"Accuracy against {unit} per question"
    pts, n, shared = cost_points(rows, models, "tokens" if by == "tokens" else "secs")
    if len(pts) < 2:
        return {"title": title, "lines": ["[dim]Needs two models with answers.[/dim]"], "actions": []}
    best = front(pts)
    order = sorted(pts, key=lambda m: (-pts[m][0], pts[m][1]))
    mark = {m: MARKS[i % len(MARKS)] for i, m in enumerate(order)}
    accs, costs = [a for a, _ in pts.values()], [c for _, c in pts.values()]
    a_lo = math.floor(min(accs) * 10) / 10
    a_lo = min(a_lo, 0.9)
    c_lo, c_hi = math.log(min(costs)), math.log(max(costs))
    grid = [[" "] * width for _ in range(height)]
    for m, (a, c) in pts.items():
        x = 0 if c_hi == c_lo else round((math.log(c) - c_lo) / (c_hi - c_lo) * (width - 1))
        y = round((1 - (a - a_lo) / (1 - a_lo)) * (height - 1)) if a_lo < 1 else 0
        free = next((xx for xx in (x, x + 1, x - 1) if 0 <= xx < width and grid[y][xx] == " "), None)
        if free is None:
            grid[y][x] = "[b]*[/b]"
        else:
            grid[y][free] = f"[bold green]{mark[m]}[/]" if m in best else mark[m]
    fmt = (lambda v: f"{v:,.0f}") if by == "tokens" else (lambda v: f"{v:.0f}s" if v >= 10 else f"{v:.1f}s")
    lines = []
    for i, row in enumerate(grid):
        label = "100%" if i == 0 else f"{100 * a_lo:.0f}%" if i == height - 1 else ""
        lines.append(f"[dim]{label:>5} │[/dim]" + "".join(row))
    lo_s, hi_s = fmt(min(costs)), fmt(max(costs))
    lines.append("[dim]      └" + "─" * width + "[/dim]")
    lines.append(f"[dim]       {lo_s}{' ' * max(1, width - len(lo_s) - len(hi_s))}{hi_s}  median {unit} "
                 "per question (log scale) →[/dim]")
    lines.append(f"[dim]Over {n} questions {'every model answered' if shared else 'each model answered (too few in common)'}. "
                 "[bold green]Green[/] = no other model is both more accurate and cheaper. * = several models.[/dim]")
    for m in order:
        a, c = pts[m]
        tag = f"[bold green]{mark[m]}[/]" if m in best else mark[m]
        lines.append(f"  {tag} {_esc(m)}  [dim]{compare.pct(a)} · {fmt(c)}{' · hosted' if m in hosted else ''}[/dim]")
    beaten = collections.defaultdict(list)     # front model -> the models it beats on both
    for m in order:
        if m in best:
            continue
        a, c = pts[m]
        better = [(m2, c2) for m2, (a2, c2) in pts.items() if m2 in best and a2 >= a and c2 <= c]
        if better:
            m2, c2 = min(better, key=lambda x: x[1])
            beaten[m2].append(f"{m} ({100 * c2 / c:.0f}% of its {unit}"
                              + (", same accuracy)" if pts[m2][0] == a else ")"))
    actions = ["Best trade-offs: " + ", ".join(m for m in order if m in best) + "."]
    local = {m: p for m, p in pts.items() if m not in hosted}
    if hosted and len(local) >= 2 and len(local) < len(pts):
        lf = front(local)
        actions.append("Among local models: " + ", ".join(m for m in order if m in lf) + ".")
    for m2, items in beaten.items():
        actions.append(f"{m2} is at least as accurate with fewer {unit} than " + ", ".join(items[:4])
                       + (f" and {len(items) - 4} more" if len(items) > 4 else "") + ".")
    return {"title": title, "lines": lines, "actions": actions}


# ---------------------------------------------------------------- 6 question health
def question_health(rows, models, min_models=3, width=40):
    title = "Question health"
    need = min(min_models, len(models))
    tests = [t for t in compare.per_question(rows) if len(t["cells"]) >= max(2, need)]
    if not tests:
        return {"title": title, "lines": [f"[dim]Needs questions answered by at least {max(2, need)} models.[/dim]"],
                "actions": []}
    rate = lambda t: statistics.fmean(c["passed"] / c["n"] for c in t["cells"].values())
    none = [t for t in tests if all(c["passed"] == 0 for c in t["cells"].values())]
    every = [t for t in tests if all(c["passed"] == c["n"] for c in t["cells"].values())]
    split = sorted((t for t in tests if t not in none and t not in every),
                   key=lambda t: (abs(rate(t) - 0.5), -len(t["cells"])))
    n = len(tests)
    blocks = _split({"wrong": len(none), "right": len(every), "cut off": len(split)}, n, width)
    bar = f"[red]{'█' * blocks['wrong']}[/][yellow]{'█' * blocks['cut off']}[/][#808080]{'█' * blocks['right']}[/]"
    lines = [f"{bar}  [dim]{n} questions answered by {max(2, need)}+ models[/dim]",
             f"[yellow]█[/] {len(split)} split models  [red]█[/] {len(none)} every model fails  "
             f"[#808080]█[/] {len(every)} every model passes"]
    desc = lambda t: (t["cells"][next(iter(t["cells"]))]["rows"][0].get("raw") or {}).get("description") or t["name"]
    for name, items, color in (("Split models most", split, "yellow"), ("Every model fails", none, "red"),
                               ("Every model passes", every, "#808080")):
        if items:
            lines.append(f"[{color}]{name}:[/]")
            for t in items[:5]:
                lines.append(f"  [dim]{_esc(t['suite'])}[/dim] {_esc(desc(t)[:70])}"
                             + (f" [dim]({100 * rate(t):.0f}% of answers right)[/dim]" if items is split else ""))
            if len(items) > 5:
                lines.append(f"  [dim]… and {len(items) - 5} more[/dim]")
    actions = []
    if none:
        actions.append(f"{len(none)} question{'s' * (len(none) > 1)} no model passes: check their expected answers "
                       "(or keep them as stretch goals).")
    if every:
        secs = statistics.median(c["secs"] for t in every for c in t["cells"].values() if c["secs"]) \
            if any(c["secs"] for t in every for c in t["cells"].values()) else None
        saved = f", saving ~{_dur(secs * len(every))} per model" if secs else ""
        actions.append(f"{len(every)} question{'s' * (len(every) > 1)} every model passes: they don't separate models; "
                       f"move them to a regression pack{saved}.")
    if split:
        actions.append(f"{len(split)} question{'s' * (len(split) > 1)} split models: these decide rankings; "
                       "write more like the top ones.")
    return {"title": title, "lines": lines, "actions": actions}


def _dur(secs):
    secs = int(secs)
    return f"{secs // 3600}h{secs % 3600 // 60:02d}m" if secs >= 3600 else f"{secs // 60}m" if secs >= 60 else f"{secs}s"


# ---------------------------------------------------------------- 7 next runs
def next_runs(engine, models, ranking, limit=4, skip=()):
    """ranking: models best first (by the reference comparison, else accuracy). The Certify runs that
    would settle the open verdicts of the strongest `limit` models, with their time."""
    from . import verdict
    title = "Next runs"
    table = verdict.readiness(engine, models)
    times = engine.answer_times()
    med = {l: {p: statistics.median(ts) for p, ts in pp.items() if ts} for l, pp in times.items()}
    lines, actions, total = [], [], 0
    shown = 0
    for m in ranking:
        if m not in table or m in skip or shown >= limit:
            continue
        todo = []
        for g, (v, packs) in table[m].items():
            for name, pv in packs.items():
                ev = pv.evidence
                if pv.status != "INCONCLUSIVE" or not ev or ev.get("certified"):
                    continue
                left = max(0, ev["tests_total"] * ev["want_repeat"] - ev["trials"])
                spa = med.get(m, {}).get(name) or statistics.median(
                    [x[name] for x in med.values() if name in x] or [60.0])
                todo.append((name, left, left * spa))
        if not todo:
            continue
        shown += 1
        answers, secs = sum(x[1] for x in todo), sum(x[2] for x in todo)
        label = ", ".join(engine.packs[n].label if n in engine.packs else n for n, *_ in todo)
        lines.append(f"[b]{shown}. Certify {_esc(m)}[/b] on {_esc(label)}: ~{answers} answers, ~{_dur(secs)}")
        lines.append(f"   [dim]tuieval run --tier certify --only {_esc(m)} --packs {','.join(n for n, *_ in todo)}[/dim]")
        total += secs
    if shown:
        actions.append(f"Certifying {'these ' + str(shown) + ' contenders' if shown > 1 else 'it'} takes ~{_dur(total)} "
                       "in all; the rest are behind them.")
    if not lines:
        lines = ["[dim]Nothing open among the strongest models: their verdicts are settled or not run yet.[/dim]"]
    return {"title": title, "lines": lines, "actions": actions}


# ---------------------------------------------------------------- all of them
def ranking(rows, models, ref):
    """Models best first: by the difference to the reference (the reference first), else by accuracy."""
    diff = {x["model"]: x["diff"] for x in compare.vs_reference(rows, ref) if x["pack"] == "all"} if ref else {}
    acc = {m: statistics.fmean(r["ok"] for r in rows if r["model"] == m) for m in models
           if any(r["model"] == m for r in rows)}
    key = (lambda m: -(diff.get(m, 0) if m != ref else 0)) if diff else (lambda m: -acc.get(m, 0))
    return sorted(acc, key=lambda m: (key(m), m))


def panels(rows, models, packs, ref=None, bases=None, by="tokens", engine=None):
    """All seven (the last needs the engine: verdicts and answer times). Models are shown best first."""
    order = ranking(rows, models, ref)
    order += [m for m in models if m not in order]
    hosted = {m["label"] for m in engine.cfg["models"] if engine.hosted(m)} if engine is not None else set()
    out = [vs_reference(rows, order, ref), use_cases(rows, order, packs), vs_base(rows, order, bases or {}, packs),
           failure_mix(rows, order), accuracy_cost(rows, order, by, hosted=hosted), question_health(rows, order)]
    if engine is not None:
        out.append(next_runs(engine, order, order, skip={ref}))
    return out


def render(panel, number=None):
    """A panel as Rich markup: title, chart, then its action lines."""
    head = f"[b reverse] {f'{number} ' if number else ''}{panel['title']} [/]"
    acts = [f"[bold cyan]→[/] {_esc(a)}" for a in panel["actions"][:MAX_ACTIONS]]
    return "\n".join([head, ""] + panel["lines"] + ([""] + acts if acts else []))
