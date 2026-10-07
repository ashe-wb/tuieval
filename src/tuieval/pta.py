"""PTA index: parsimony (tokens), time and accuracy per model, a bar each (longer = better).

    P  parsimony (tokens)  the tokens (reasoning and answer) used on every question compared (each
                           question at its median over repeats), shown as how many times the leanest
                           model's ("2.4×"). Unlike T it doesn't depend on the machine.
    T  time                the total time to answer those questions, as how many times the fastest
                           model's ("28×").
    A  accuracy            the share of answers to those questions that passed (%).

Behind the P and T bars is a 0-100 score: 100 for the best, POINTS_PER_DOUBLING points less for every
doubling (0 at about 100 times more), so models far behind the best still get a visible bar.

Models are compared on the questions all of them answered, so one that skipped some never looks
faster. Server and connection errors aren't answers. By default a model with fewer than half as many
answers as the most-answered model is left out (listed, so it can be picked), or it would shrink the
shared questions for everyone. There's no combined number: the three stay side by side, and verdicts
and critical failures stay where they are (verdict.py).
"""
import math

from rich.markup import escape as rich_escape

from . import compare

MIN_SHARE = 0.5   # fewer answers than this share of the most-answered model's: left out by default
POINTS_PER_DOUBLING = 15


def score(total, best):
    """P and T: 100 for the best (lowest) total, POINTS_PER_DOUBLING less for every doubling, never below 0."""
    if not total or not best:
        return None
    return max(0.0, 100 - POINTS_PER_DOUBLING * math.log2(total / best))


def memory_gb(infos):
    """{model: the most memory its server held in any run (GB)}, for models tuieval served (hosted: none).
    On a discrete GPU that's its peak VRAM (vram_peak_gb), where the model lives; otherwise its resident memory."""
    out = {}
    for i in infos:
        gb = i.get("vram_peak_gb") or (i["peak_rss_mb"] / 1024 if i.get("peak_rss_mb") else None)
        if gb:
            out[i["label"]] = max(out.get(i["label"], 0), gb)
    return out


def index(rows, models=None, memory=None):
    """{"questions": n shared, "models": [{model, P, T, A, tokens, total_s, passed, answers, alongside,
    fewest, fastest, tps, ttft, memory_gb}],
    "left_out": [(model, questions answered)], "no_answers": [model], "most": questions answered by the
    most-answered model}, best accuracy first. models: the ones to compare (default: every model in rows
    with enough answers; models named here are compared whatever their count). memory: memory_gb()."""
    no_answers = sorted({r["model"] for r in rows} - {r["model"] for r in rows if not r.get("error")})
    rows = [r for r in rows if not r.get("error")]     # a server or connection error isn't an answer
    tests = compare.per_question(rows)
    counts = {}
    for t in tests:
        for m in t["cells"]:
            counts[m] = counts.get(m, 0) + 1
    most = max(counts.values(), default=0)
    if models:
        no_answers = [m for m in no_answers if m in models]
        models, left_out = sorted(m for m in models if m in counts), []
    else:
        models = sorted(m for m in counts if counts[m] >= MIN_SHARE * most)
        left_out = sorted(((m, n) for m, n in counts.items() if m not in models), key=lambda x: (-x[1], x[0]))
    shared = [t for t in tests if models and all(m in t["cells"] for m in models)]
    out = []
    for m in models:
        cells = [t["cells"][m] for t in shared]
        answers = sum(c["n"] for c in cells)
        out.append({"model": m, "tokens": sum(c["tokens"] or 0 for c in cells),
                    "total_s": sum(c["secs"] or 0 for c in cells),
                    "passed": sum(c["passed"] for c in cells), "answers": answers,
                    "A": 100 * sum(c["passed"] for c in cells) / answers if answers else None,
                    # generation speed and time to first token over the same answers (medians)
                    "tps": compare.median(r.get("gen_tps") for c in cells for r in c["rows"]),
                    "ttft": compare.median(r.get("ttft") for c in cells for r in c["rows"]),
                    "memory_gb": (memory or {}).get(m),
                    # answers that ran while other models were served shared the machine's speed
                    "alongside": any(r.get("alongside") for c in cells for r in c["rows"])})
    fastest = min((x["total_s"] for x in out if x["total_s"] > 0), default=None)
    fewest = min((x["tokens"] for x in out if x["tokens"] > 0), default=None)
    for x in out:
        x["P"], x["T"] = score(x["tokens"], fewest), score(x["total_s"], fastest)
        # shown on the bars: how many times the leanest / fastest model's total (1.0 = the best)
        x["tokens_x"] = x["tokens"] / fewest if fewest and x["tokens"] else None
        x["time_x"] = x["total_s"] / fastest if fastest and x["total_s"] else None
        x["fewest"] = bool(fewest) and x["tokens"] == fewest
        x["fastest"] = bool(fastest) and x["total_s"] == fastest
    out.sort(key=lambda x: (-(x["A"] or 0), -(x["T"] or 0), x["model"]))
    return {"questions": len(shared), "models": out, "left_out": left_out, "no_answers": no_answers,
            "most": most}


# (letter, name, what the number beside the bar shows, colour); the bar itself is the 0-100 score
LETTERS = (("P", "parsimony (tokens)", "tokens_x", "cyan"), ("T", "time", "time_x", "magenta"),
           ("A", "accuracy", "A", "green"))


def times(v):
    """A multiple of the best: "1.0×", "2.4×", "28×"."""
    return "-" if v is None else f"{v:.1f}×" if v < 9.95 else f"{v:.0f}×"


def bars(result, width=20, markup=True):
    """Lines with a row per model and a bar per score (P, T, A), longer = better: the at-a-glance view.
    Beside P and T, how many times the leanest / fastest model's tokens / time; beside A, the percentage.
    markup=False gives plain text (the report)."""
    if not result["models"]:
        return []
    name_w = max(len(x["model"]) for x in result["models"])
    # each column: the bar, a space, "100%" or "9.8×" (5), then a gap; at least as wide as its heading
    cell_w = max(width + 9, *(len(f"{k} {name}") + 2 for k, name, _, _ in LETTERS))
    head = " " * (name_w + 2) + "".join(f"{f'{k} {name}':<{cell_w}}" for k, name, _, _ in LETTERS)
    lines = [f"[b]{head.rstrip()}[/b]" if markup else head.rstrip()]
    for x in result["models"]:
        cells = []
        for k, _, shown, color in LETTERS:
            v = x[k]
            full = 0 if v is None else round(width * max(0, min(100, v)) / 100)
            if v is not None and v < 100:   # full only when perfect (accuracy) or the best (P, T)
                full = min(full, width - 1)
            bar, rest = "█" * full, "░" * (width - full)
            num = (("-" if v is None else compare.pct(v / 100)) if shown == "A" else times(x.get(shown)))
            bar = f"[{color}]{bar}[/][grey37]{rest}[/]" if markup else bar + rest
            cells.append(f"{bar} {num:>6}" + " " * (cell_w - width - 7))
        name = f"{x['model']:<{name_w}}"
        lines.append((f"{rich_escape(name) if markup else name}  " + "".join(cells)).rstrip())
    return lines


def left_out_lines(result):
    """Plain sentences on the models not compared (too few answers, or only server errors) and on
    answers that shared the machine with other models."""
    out = []
    if result["left_out"]:
        out.append("Left out (fewer than half as many answers as the most-answered model; pick them under "
                   "Models or --only to include them): "
                   + ", ".join(f"{m} ({n} of {result['most']})" for m, n in result["left_out"]))
    if result["no_answers"]:
        out.append("No answers, only server errors: " + ", ".join(result["no_answers"]))
    shared = [x["model"] for x in result["models"] if x["alongside"]]
    if shared:
        out.append("Some answers ran alongside other models, which shared the machine's speed: " + ", ".join(shared))
    return out


def table(result):
    """(header, rows) for a plain table of the index."""
    n = result["questions"] or 1
    header = ["model", "tokens/answer", "time/answer", "answers right", "tok/s", "TTFT s", "memory GB"]
    rows = [[x["model"], f"{round(x['tokens'] / n):,}" + (" ★" if x.get("fewest") else ""),
             fmt_secs(x["total_s"] / n) + (" ★" if x.get("fastest") else ""),
             f"{x['passed']}/{x['answers']}", _num(x.get("tps"), ".0f"), _num(x.get("ttft"), ".1f"),
             _num(x.get("memory_gb"), ".1f")] for x in result["models"]]
    return header, rows


def fmt_secs(secs):
    """Time per answer: "2.5s", "69s", "4m10s"."""
    if not secs:
        return "-"
    return f"{secs:.1f}s" if secs < 10 else f"{secs:.0f}s" if secs < 60 else fmt_total(secs)


def _num(v, fmt):
    return "-" if v is None else format(v, fmt)


def fmt_total(secs):
    if not secs:
        return "-"
    if secs < 60:
        return f"{secs:.1f}s"
    secs = round(secs)
    h, rem = divmod(secs, 3600)
    return f"{h}h{rem // 60:02d}m" if h else f"{rem // 60}m{rem % 60:02d}s" if rem >= 60 else f"{rem}s"


def scope_note(result, models):
    """What the scores compare, in a sentence."""
    if not result["questions"]:
        return "These models have no question in common yet, so there's nothing to compare."
    return (f"Compared on the {result['questions']} questions all {len(result['models'])} model(s) answered. "
            "Longer bars are better. P parsimony (tokens) and T time: how many times the leanest / fastest "
            "model's (1.0×, ★ in the table). A accuracy: answers right.")


def for_engine(e, labels=None, packs=None, results_dir=None):
    """(result, models) from a workspace's finished results (results on their way out excluded)."""
    paths = [p for p in compare.default_paths(results_dir or e.results_dir) if not e.superseded(p)]
    rows, infos = compare.load(paths) if paths else ([], [])
    if packs:
        rows = [r for r in rows if r["suite"] in packs]
    models = sorted({r["model"] for r in rows} & set(labels) if labels else {r["model"] for r in rows})
    return index(rows, models if labels else None, memory_gb(infos)), models


def markdown(result, models):
    """The index as a markdown section (tuieval report)."""
    lines = ["## PTA index", "", scope_note(result, models), ""]
    if not result["questions"]:
        return lines
    lines += ["```"] + bars(result, markup=False) + ["```", ""]
    header, rows = table(result)
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    for line in left_out_lines(result):
        lines += ["", line]
    return lines
