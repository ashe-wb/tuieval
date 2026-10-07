"""PTA index: parsimony, speed (time) and accuracy per model, 0-100 each and higher is better, shown
as a bar per score for each model.

    P  parsimony  a score out of 100, from the tokens (reasoning and answer) used on every question
                  compared (each question at its median over repeats): 100 for the model that used the
                  fewest, POINTS_PER_DOUBLING points less for every doubling (0 at about 100 times more).
                  Unlike T it doesn't depend on the machine: it's how much a model says to get there.
    T  speed      a score out of 100, the same from the total time to answer those questions: 100 for
                  the fastest model. Log scales, so models that are all far behind the best still differ.
    A  accuracy   a percentage: the share of answers to those questions that passed.

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


def index(rows, models=None):
    """{"questions": n shared, "models": [{model, P, T, A, tokens, total_s, passed, answers, alongside,
    fewest, fastest}],
    "left_out": [(model, questions answered)], "no_answers": [model], "most": questions answered by the
    most-answered model}, best accuracy first. models: the ones to compare (default: every model in rows
    with enough answers; models named here are compared whatever their count)."""
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
                    # answers that ran while other models were served shared the machine's speed
                    "alongside": any(r.get("alongside") for c in cells for r in c["rows"])})
    fastest = min((x["total_s"] for x in out if x["total_s"] > 0), default=None)
    fewest = min((x["tokens"] for x in out if x["tokens"] > 0), default=None)
    for x in out:
        x["P"], x["T"] = score(x["tokens"], fewest), score(x["total_s"], fastest)
        x["fewest"] = bool(fewest) and x["tokens"] == fewest
        x["fastest"] = bool(fastest) and x["total_s"] == fastest
    out.sort(key=lambda x: (-(x["A"] or 0), -(x["T"] or 0), x["model"]))
    return {"questions": len(shared), "models": out, "left_out": left_out, "no_answers": no_answers,
            "most": most}


LETTERS = (("P", "parsimony", "/100", "cyan"), ("T", "speed", "/100", "magenta"), ("A", "accuracy", "%", "green"))


def bars(result, width=20, markup=True):
    """Lines with a row per model and a bar per score (P, T, A), longer = better, each with its number:
    the at-a-glance view. markup=False gives plain text (the report)."""
    if not result["models"]:
        return []
    name_w = max(len(x["model"]) for x in result["models"])
    cell_w = width + 6                          # the bar, a space and "100%" or " 100"
    head = " " * (name_w + 2) + "".join(f"{f'{k} {name}':<{cell_w}}" for k, name, _, _ in LETTERS)
    lines = [f"[b]{head.rstrip()}[/b]" if markup else head.rstrip()]
    for x in result["models"]:
        cells = []
        for k, _, unit, color in LETTERS:
            v = x[k]
            full = 0 if v is None else round(width * max(0, min(100, v)) / 100)
            bar, rest = "█" * full, "░" * (width - full)
            num = "-" if v is None else f"{v:.0f}{'%' if unit == '%' else ''}"
            bar = f"[{color}]{bar}[/][grey37]{rest}[/]" if markup else bar + rest
            cells.append(f"{bar} {num:>4}")
        name = f"{x['model']:<{name_w}}"
        lines.append(f"{rich_escape(name) if markup else name}  " + "  ".join(cells))
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


def _score(v, unit=""):
    """A is a percentage (%), P and T scores out of 100 (/100: on a log scale, not percentages)."""
    return "-" if v is None else f"{v:.0f}{unit}"


def table(result):
    """(header, rows) for a plain table of the index."""
    header = ["model", "P parsimony /100", "T speed /100", "A accuracy %", "tokens", "total time",
              "answers right"]
    rows = [[x["model"], _score(x["P"]), _score(x["T"]), _score(x["A"], "%"),
             f"{round(x['tokens']):,}" + (" ★" if x.get("fewest") else ""),
             fmt_total(x["total_s"]) + (" ★" if x.get("fastest") else ""),
             f"{x['passed']}/{x['answers']}"] for x in result["models"]]
    return header, rows


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
            f"A is a percentage. P parsimony and T speed are scores out of 100, not percentages: 100 = the fewest "
            f"tokens / the fastest total time (★), {POINTS_PER_DOUBLING} points less for each doubling; a better "
            "model added lowers the others.")


def for_engine(e, labels=None, packs=None, results_dir=None):
    """(result, models) from a workspace's finished results (results on their way out excluded)."""
    paths = [p for p in compare.default_paths(results_dir or e.results_dir) if not e.superseded(p)]
    rows, _ = compare.load(paths) if paths else ([], [])
    if packs:
        rows = [r for r in rows if r["suite"] in packs]
    models = sorted({r["model"] for r in rows} & set(labels) if labels else {r["model"] for r in rows})
    return index(rows, models if labels else None), models


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
