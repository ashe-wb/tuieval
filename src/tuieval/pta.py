"""PTA index: privacy, time and accuracy per model, 0-100 each, with each model a dot on a triangle.

    P  privacy   100 when prompts stay on machines you control: tuieval starts the server, its URL
                 is this machine, or the server is marked private = true in models.toml. 0 otherwise
                 (a hosted API: there's no partial privacy once prompts leave).
    T  time      the total time to answer every question compared (each question at its median over
                 repeats), relative to the fastest model: 100 x fastest total / this model's total.
    A  accuracy  the share of answers to those questions that passed.

Models are compared on the questions all of them answered, so one that skipped some never looks
faster. Server and connection errors aren't answers. By default a model with fewer than half as many
answers as the most-answered model is left out (listed, so it can be picked), or it would shrink the
shared questions for everyone. There's no combined number: the three stay side by side, and verdicts
and critical failures stay where they are (verdict.py).
"""
import colorsys
import urllib.parse

from . import compare

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
MIN_SHARE = 0.5   # fewer answers than this share of the most-answered model's: left out by default


def colors(n):
    """n distinct colours (#rrggbb), evenly spread around the colour wheel, readable on dark and light."""
    out = []
    for i in range(n):
        r, g, b = colorsys.hls_to_rgb((i / max(n, 1) + 0.55) % 1, 0.58, 0.85)
        out.append(f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}")
    return out


def privacy(cfg, label):
    """100 if a model's prompts stay on machines you control, else 0 (see the module docstring);
    None for a model that's no longer in models.toml (where it ran isn't known)."""
    m = next((m for m in cfg["models"] if m["label"] == label), None)
    server = cfg["servers"].get(m["server"], {}) if m else {}
    if not server:
        return None
    if server.get("cmd") or server.get("private"):
        return 100
    host = urllib.parse.urlparse(server.get("url", "")).hostname or ""
    return 100 if host.lower() in LOCAL_HOSTS else 0


def index(rows, privacy_of, models=None):
    """{"questions": n shared, "models": [{model, P, T, A, total_s, passed, answers, alongside, color}],
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
        out.append({"model": m, "P": privacy_of(m), "total_s": sum(c["secs"] or 0 for c in cells),
                    "passed": sum(c["passed"] for c in cells), "answers": answers,
                    "A": 100 * sum(c["passed"] for c in cells) / answers if answers else None,
                    # answers that ran while other models were served shared the machine's speed
                    "alongside": any(r.get("alongside") for c in cells for r in c["rows"])})
    fastest = min((x["total_s"] for x in out if x["total_s"] > 0), default=None)
    for x in out:
        x["T"] = 100 * fastest / x["total_s"] if fastest and x["total_s"] > 0 else None
    out.sort(key=lambda x: (-(x["A"] or 0), -(x["T"] or 0), x["model"]))
    for x, c in zip(out, colors(len(out))):
        x["color"] = c
    return {"questions": len(shared), "models": out, "left_out": left_out, "no_answers": no_answers,
            "most": most}


def position(x):
    """Where a model's dot sits, as weights on the (P, T, A) corners summing to 1: pulled toward each
    corner by its score, so the dot shows the balance of the three. None when every score is 0 or unknown."""
    s = [x[k] or 0 for k in ("P", "T", "A")]
    total = sum(s)
    return tuple(v / total for v in s) if total else None


def strong(x):
    """Whether a model's scores average 50 or more: drawn filled (●), otherwise hollow (○)."""
    return sum(x[k] or 0 for k in ("P", "T", "A")) / 3 >= 50


# ---------------------------------------------------------------- the triangle (braille, no dependencies)
_DOTS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))   # braille dot bits by (row, column)


class _Canvas:
    """Braille cells, 2 dots wide and 4 tall each; every cell takes the colour of the last line through it."""
    def __init__(self, cols, rows):
        self.cols, self.rows = cols, rows
        self.bits = [[0] * cols for _ in range(rows)]
        self.color = [[None] * cols for _ in range(rows)]
        self.marks = {}

    def dot(self, x, y, color):
        cx, cy = int(x) // 2, int(y) // 4
        if 0 <= cx < self.cols and 0 <= cy < self.rows:
            self.bits[cy][cx] |= _DOTS[int(y) % 4][int(x) % 2]
            self.color[cy][cx] = color

    def line(self, a, b, color, dotted=False):
        steps = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) or 1
        for i in range(steps + 1):
            if dotted and i % 3:
                continue
            self.dot(a[0] + (b[0] - a[0]) * i / steps, a[1] + (b[1] - a[1]) * i / steps, color)

    def mark(self, cx, cy, char, color):
        """A whole cell for a model's dot; a second dot in the same cell turns it into a count."""
        self.marks.setdefault((cx, cy), []).append((char, color))

    def text(self):
        lines = []
        for y, (bits, colors) in enumerate(zip(self.bits, self.color)):
            out = []
            for x, (b, c) in enumerate(zip(bits, colors)):
                here = self.marks.get((x, y))
                if here:
                    out.append(f"[{here[0][1]}]{here[0][0]}[/]" if len(here) == 1 else
                               f"[b]{len(here) if len(here) < 10 else '+'}[/b]")
                    continue
                ch = chr(0x2800 + b) if b else " "
                out.append(f"[{c}]{ch}[/]" if b and c else ch)
            lines.append("".join(out).rstrip())
        return lines


def _geometry(width):
    cols = max(20, width)
    w = cols * 2 - 2                            # dots across
    h = int(w * 0.866)                          # an equilateral triangle (braille dots are about square)
    return cols, w, h


def spots(result, width=60):
    """{model: (column, row)}: the character cell each model's dot is drawn in (None if it has no dot)."""
    cols, w, h = _geometry(width)
    rows = h // 4 + 1
    corners = ((w / 2, 0), (0, h), (w, h))
    out = {}
    for x in result["models"]:
        wts = position(x)
        if wts:
            px = sum(k[0] * v for k, v in zip(corners, wts))
            py = sum(k[1] * v for k, v in zip(corners, wts))
            out[x["model"]] = (min(cols - 1, max(0, int(px) // 2)), min(rows - 1, max(0, int(py) // 4)))
    return out


def triangle(result, width=60):
    """The triangle as Rich markup lines: P at the top, T bottom left, A bottom right. Each model is
    a dot in its own colour, pulled toward each corner by its score (position()); filled when its
    scores average 50 or more, hollow below that. A number is that many models on the same spot."""
    cols, w, h = _geometry(width)
    rows = h // 4 + 1
    top, left, right = (w / 2, 0), (0, h), (w, h)
    center = (w / 2, h * 2 / 3)
    c = _Canvas(cols, rows)
    for corner in (top, left, right):           # lines to the corners from the middle, then the frame
        c.line(center, corner, "grey37", dotted=True)
    for a, b in ((top, left), (left, right), (right, top)):
        c.line(a, b, "grey70")
    where = spots(result, width)
    for x in result["models"]:
        if x["model"] in where:
            c.mark(*where[x["model"]], "●" if strong(x) else "○", x["color"])
    lines = c.text()
    pad = " " * max(0, cols // 2 - 5)
    return ([f"{pad}[b]P[/b] privacy"] + lines
            + [f"[b]T[/b] time{' ' * max(1, cols - 15)}[b]A[/b] accuracy"])


def legend(result, width=60):
    """One Rich markup line per model: its dot on the triangle (and the models it shares a spot with,
    drawn as a count), its three scores, then any models left out of the comparison and why."""
    where = spots(result, width)
    out = []
    for x in result["models"]:
        mark = f"[{x['color']}]{'●' if strong(x) else '○'}[/]" if x["model"] in where else " "
        same = [m for m, at in where.items() if m != x["model"] and at == where.get(x["model"])]
        out.append(f"{mark} {x['model']}  P {_score(x['P'])} · T {_score(x['T'])} · A {_score(x['A'])}"
                   + (f"  [dim](same spot as {', '.join(same)}: drawn as {len(same) + 1})[/dim]" if same else "")
                   + ("  [yellow](some answers ran alongside other models)[/yellow]" if x["alongside"] else ""))
    return out + [f"[dim]{line}[/dim]" for line in left_out_lines(result)]


def left_out_lines(result):
    """Plain sentences on the models not compared: too few answers, or only server errors."""
    out = []
    if result["left_out"]:
        out.append("Left out (fewer than half as many answers as the most-answered model; pick them under "
                   "Models or --only to include them): "
                   + ", ".join(f"{m} ({n} of {result['most']})" for m, n in result["left_out"]))
    if result["no_answers"]:
        out.append("No answers, only server errors: " + ", ".join(result["no_answers"]))
    return out


def _score(v):
    return "-" if v is None else f"{v:.0f}"


def table(result):
    """(header, rows) for a plain table of the index."""
    header = ["model", "P privacy", "T time", "A accuracy", "total time", "answers right"]
    rows = [[x["model"], _score(x["P"]), _score(x["T"]), _score(x["A"]), fmt_total(x["total_s"]),
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
            "T is relative to the fastest total, so adding a faster model lowers the others.")


def for_engine(e, labels=None, packs=None, results_dir=None):
    """(result, models) from a workspace's finished results (results on their way out excluded)."""
    paths = [p for p in compare.default_paths(results_dir or e.results_dir) if not e.superseded(p)]
    rows, _ = compare.load(paths) if paths else ([], [])
    if packs:
        rows = [r for r in rows if r["suite"] in packs]
    models = sorted({r["model"] for r in rows} & set(labels) if labels else {r["model"] for r in rows})
    return index(rows, lambda m: privacy(e.cfg, m), models if labels else None), models


def markdown(result, models):
    """The index as a markdown section (tuieval report)."""
    lines = ["## PTA index", "", scope_note(result, models), ""]
    if not result["questions"]:
        return lines
    header, rows = table(result)
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    if any(x["alongside"] for x in result["models"]):
        lines += ["", "Some answers ran alongside other models, which shared the machine's speed."]
    for line in left_out_lines(result):
        lines += ["", line]
    return lines
