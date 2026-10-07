"""PTA index: privacy, time and accuracy per model, 0-100 each, drawn as a radar triangle.

    P  privacy   100 when prompts stay on machines you control: tuieval starts the server, its URL
                 is this machine, or the server is marked private = true in models.toml. 0 otherwise
                 (a hosted API: there's no partial privacy once prompts leave).
    T  time      the total time to answer every question compared (each question at its median over
                 repeats), relative to the fastest model: 100 x fastest total / this model's total.
    A  accuracy  the share of answers to those questions that passed.

Models are compared on the questions all of them answered, so one that skipped some never looks
faster. There's no combined number: the three stay side by side, and verdicts and critical
failures stay where they are (verdict.py).
"""
import urllib.parse

from . import compare

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
# one colour per model on the triangle; more models than colours are listed but not drawn
COLORS = ["cyan", "magenta", "yellow", "green", "red", "blue", "bright_white", "bright_cyan"]


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
    """{"questions": n shared, "models": [{model, P, T, A, total_s, passed, answers, alongside}]},
    best accuracy first. models: the ones to compare (default: every model in rows)."""
    tests = compare.per_question(rows)
    models = sorted(models or {m for t in tests for m in t["cells"]})
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
    return {"questions": len(shared), "models": out}


# ---------------------------------------------------------------- the triangle (braille, no dependencies)
_DOTS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))   # braille dot bits by (row, column)


class _Canvas:
    """Braille cells, 2 dots wide and 4 tall each; every cell takes the colour of the last line through it."""
    def __init__(self, cols, rows):
        self.cols, self.rows = cols, rows
        self.bits = [[0] * cols for _ in range(rows)]
        self.color = [[None] * cols for _ in range(rows)]

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

    def text(self):
        lines = []
        for bits, colors in zip(self.bits, self.color):
            out = []
            for b, c in zip(bits, colors):
                ch = chr(0x2800 + b) if b else " "
                out.append(f"[{c}]{ch}[/]" if b and c else ch)
            lines.append("".join(out).rstrip())
        return lines


def triangle(result, width=60):
    """The radar triangle as Rich markup lines: P at the top, T bottom left, A bottom right; a
    model's triangle reaches each corner as far as its score (bigger = better all round)."""
    cols = max(20, width)
    w = cols * 2 - 2                            # dots across
    h = int(w * 0.866)                          # an equilateral triangle (braille dots are about square)
    rows = h // 4 + 1
    top, left, right = (w / 2, 0), (0, h), (w, h)
    center = (w / 2, h * 2 / 3)
    c = _Canvas(cols, rows)

    def at(corner, score):
        return (center[0] + (corner[0] - center[0]) * score / 100, center[1] + (corner[1] - center[1]) * score / 100)

    for corner in (top, left, right):           # axes, the 50% ring, then the frame
        c.line(center, corner, "grey37", dotted=True)
    half = [at(k, 50) for k in (top, left, right)]
    for a, b in zip(half, half[1:] + half[:1]):
        c.line(a, b, "grey37", dotted=True)
    for a, b in ((top, left), (left, right), (right, top)):
        c.line(a, b, "grey70")
    for i, x in enumerate(result["models"][:len(COLORS)][::-1]):   # the best ends on top
        color = COLORS[len(result["models"][:len(COLORS)]) - 1 - i]
        pts = [at(k, s or 0) for k, s in ((top, x["P"]), (left, x["T"]), (right, x["A"]))]
        for a, b in zip(pts, pts[1:] + pts[:1]):
            c.line(a, b, color)
    lines = c.text()
    pad = " " * max(0, cols // 2 - 5)
    return ([f"{pad}[b]P[/b] privacy"] + lines
            + [f"[b]T[/b] time{' ' * max(1, cols - 15)}[b]A[/b] accuracy"])


def legend(result):
    """One Rich markup line per model: its colour on the triangle and its three scores."""
    out = []
    for i, x in enumerate(result["models"]):
        mark = f"[{COLORS[i]}]■[/]" if i < len(COLORS) else " "
        out.append(f"{mark} {x['model']}  P {_score(x['P'])} · T {_score(x['T'])} · A {_score(x['A'])}"
                   + ("  [yellow](some answers ran alongside other models)[/yellow]" if x["alongside"] else ""))
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
    return (f"Compared on the {result['questions']} questions all {len(models)} model(s) answered. "
            "T is relative to the fastest total, so adding a faster model lowers the others.")


def for_engine(e, labels=None, packs=None, results_dir=None):
    """(result, models) from a workspace's finished results (results on their way out excluded)."""
    paths = [p for p in compare.default_paths(results_dir or e.results_dir) if not e.superseded(p)]
    rows, _ = compare.load(paths) if paths else ([], [])
    if packs:
        rows = [r for r in rows if r["suite"] in packs]
    models = sorted({r["model"] for r in rows} & set(labels) if labels else {r["model"] for r in rows})
    return index(rows, lambda m: privacy(e.cfg, m), models), models


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
    return lines
