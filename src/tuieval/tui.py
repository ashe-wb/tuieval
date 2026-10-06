"""Terminal UI for the evals: pick eval types and models, then watch the run live.

    tuieval            (in a workspace folder; see tuieval init)

Setup screen:  space toggles, s starts, a adds a model, m scans for new models, r shows results, q quits.
               t tunes the ticked models' server speed flags for this machine.
               Type in the filter box to narrow the models by label or tag.
               x hides/unhides the highlighted model (results kept; 'Show hidden' lists them).
Run screen:    live reasoning and answer, per-test ✓/✗, TTFT and tokens/s, progress and ETA.
               k skips the current model, c cancels everything, f pauses auto-scroll.
               r shows results at any time (also mid-run); n starts a new run when it ends.
               Enter on a Recent results row shows that answer in full.
Results:       ctrl+r reloads (it also reloads when a run's pack finishes). Per question compares
               each model's tokens and seconds on the same question. Enter on a Failures row or a
               Per question cell shows the answer: grading and checks, reasoning, answer, question.
Answer detail: [ and ] step to the previous/next answer; esc/q close.
Anywhere:      esc/q go back (q quits only on Setup), ctrl+q quits, w lists this session's runs and
               tunes. Leaving a live run or tune keeps it going; the header shows its progress.
"""
import argparse
import glob
import json
import os
import re
import threading
import time

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import (Button, Checkbox, Collapsible, DataTable, Footer, Header, Input, Label, Log, OptionList,
                             ProgressBar, RadioButton, RadioSet, Select, SelectionList, Static, TabbedContent,
                             TabPane, TextArea)
from textual.widgets.option_list import Option
from textual.widgets.selection_list import Selection

import yaml

from . import compare
from . import engine
from . import verdict

NO_PACKS = [
    "[yellow]No packs yet.[/yellow] A pack is a folder of your own questions.",
    "[dim]In a terminal:[/dim]  tuieval new-pack my-first-pack",
    "[dim]Edit[/dim] packs/my-first-pack/tests.yaml",
    "[dim]Check it:[/dim]  tuieval selftest",
    "[dim]Then open tuieval again. Guide: docs/writing-packs.md[/dim]",
]

VERDICT_STYLE = {"PASS": "bold green", "FAIL": "bold red", "INCONCLUSIVE": "yellow", "NO DATA": "dim"}


def model_verdicts(e):
    """{label: [(use case, status, when)]} in pack order: each use case's verdict from current
    results (when = None) or, failing that, the last recorded one (when = its date)."""
    current = verdict.readiness(e)
    last = verdict.latest(verdict.load_history(e))
    groups = list(dict.fromkeys(pk.group for pk in e.packs.values()))
    out = {}
    for m in e.cfg["models"]:
        label, rows = m["label"], []
        now = current.get(label, {})
        for g in groups + sorted({g for (l, g) in last if l == label} - set(groups)):
            if g in now and now[g][0].status != "NO DATA":
                rows.append((g, now[g][0].status, None))
            elif (label, g) in last:
                rows.append((g, last[label, g]["status"], last[label, g]["time"][:10]))
        out[label] = rows
    return out


VERDICT_MARK = {"PASS": "✓", "FAIL": "✗", "INCONCLUSIVE": "?"}


def group_codes(e):
    """{use case: short code}: its first letter, or its words' initials, or more letters, whichever
    is the shortest that keeps every code unique (Coding -> C; Demo answer, Demo code -> DA, DC)."""
    groups = list(dict.fromkeys(pk.group for pk in e.packs.values()))

    def initials(g, extra=0):
        words = g.split() or [g]
        return ("".join(w[0] for w in words[:-1]) + words[-1][:1 + extra]).upper()
    for code in (lambda g: g[:1].upper(), initials, lambda g: g[:2].upper(), lambda g: initials(g, 1),
                 lambda g: g[:3].upper(), lambda g: initials(g, 2)):
        codes = {g: code(g) for g in groups}
        if len(set(codes.values())) == len(codes):
            return codes
    return {g: f"{g[:1].upper()}{i + 1}" for i, g in enumerate(groups)}


def verdict_codes(rows, codes):
    """Compact markup, e.g. C✗ D? T✓ (colour = verdict, dim = from earlier results), one fixed
    column per use case so rows line up; blank where a use case has no verdict."""
    by_group = {g: (status, when) for g, status, when in rows}
    out = []
    for g in list(codes) + [g for g in by_group if g not in codes]:
        code = codes.get(g, g[:1].upper())
        if g not in by_group:
            out.append(" " * (len(code) + 1))
            continue
        status, when = by_group[g]
        text = f"{code}{VERDICT_MARK.get(status, '·')}"
        out.append(f"[dim]{text}[/dim]" if when else f"[{VERDICT_STYLE.get(status, 'dim')}]{text}[/]")
    return " ".join(out)


def verdict_rank(rows):
    """Sort key: more passes first, then fewer failures (untested models sit above failing ones)."""
    statuses = [s for _, s, _ in rows]
    return -statuses.count("PASS"), statuses.count("FAIL")


def short_note(note, width=60):
    """A job note for the queue table: a server that never started says so briefly (the notice has the
    detail), and long notes end in … rather than mid-word."""
    if " while loading" in note and note.startswith("server exited"):
        note = "didn't start: " + note.split(" while loading", 1)[1].lstrip(": ") or "didn't start"
    return note if len(note) <= width else note[:width - 1].rsplit(" ", 1)[0] + "…"


def quant_text(ep):
    """A pinned endpoint's quantization, as the provider declares it."""
    q = (ep or {}).get("quantization")
    return "quantization undeclared by provider" if q in (None, "unknown") else q


def serving_badge(e, m, newcomer=False):
    """One-line markup: how the model fits and is tuned on this machine; for a hosted model, the
    provider endpoint and quantization its answers come from. newcomer (nothing has run yet): only
    what stops a run (doesn't fit), not context sizes and tuning."""
    try:
        ep = e.endpoint(m)
        if ep:
            return f"  [magenta]{quant_text(ep)}[/magenta] [dim]via {ep['tag']}[/dim]"
        sv = e.serving(m)
    except Exception:
        return ""
    if sv.perf_source == "n/a":
        return ""
    if not sv.fits:
        return "  [bold red]doesn't fit here[/bold red]"
    if newcomer:
        return ""
    ctx = f"{sv.ctx // 1024}k" if sv.ctx else ""
    src = {"tuned": "[green]tuned[/green]", "seeded": "[green]hand-tuned[/green]",
           "untuned": "[yellow]untuned[/yellow]"}.get(sv.perf_source, "[yellow]retune[/yellow]")
    return f"  [dim]{ctx}[/dim] {src}"


LAT_STYLE = {"OK": "green", "TOO SLOW": "bold red", "DOESN'T FIT": "bold red", "NO DATA": "dim"}


def latency_cell(x):
    if x is None:
        return "[dim]-[/dim]"
    if x.status in ("NO DATA", "DOESN'T FIT"):
        return f"[{LAT_STYLE[x.status]}]{x.status.lower()}[/]"
    return f"[{LAT_STYLE[x.status]}]{x.status} {x.p90_s:.0f}s[/] [dim]{x.kind}[/dim]"


DIFF_STYLE = {"easy": "green", "medium": "yellow", "hard": "bold red", "unrated": "dim"}


def diff_badge(level):
    return f"[{DIFF_STYLE.get(level, 'dim')}]{level.upper() if level != 'unrated' else 'unrated'}[/]"


STATUS_STYLE = {"waiting": "dim", "loading": "yellow", "running": "bold cyan", "done": "green",
                "failed": "bold red", "skipped": "dim"}


def fmt_secs(secs):
    secs = int(secs)
    if secs >= 3600:
        return f"{secs // 3600}h{secs % 3600 // 60:02d}m"
    return f"{secs // 60}m{secs % 60:02d}s" if secs >= 60 else f"{secs}s"


class StreamView(TextArea):
    """Read-only, word-wrapped text that grows as tokens stream in. Scroll it with the mouse."""

    can_focus = False

    MAX_CHARS = 200_000  # keep very long reasoning responsive: drop the oldest text

    def __init__(self, **kw):
        super().__init__(read_only=True, soft_wrap=True, show_cursor=False, max_checkpoints=1, **kw)

    def append(self, text, follow=True):
        self.insert(text, self.document.end)
        if len(self.text) > self.MAX_CHARS:
            self.load_text("…" + self.text[-self.MAX_CHARS // 2:])
        if follow:
            self.scroll_end(animate=False)


# ------------------------------------------------------------------ answer detail
def answer_entry(row):
    """An answer for AnswerScreen from a compare row (None when the row has no stored answer)."""
    if not row.get("raw"):
        return None
    return {"model": row["model"], "pack": row["suite"], "record": row["raw"], "partial": row.get("partial", False),
            "pack_fp": row.get("pack_fp")}


def read_log_sections(path):
    """(reasoning, answer) from a logs/live/ request file, or None if unreadable."""
    try:
        with open(path, errors="replace") as f:
            text = f.read()
    except OSError:
        return None
    head, sep, rest = text.partition("\n\n=== REASONING ===\n")
    if not sep:
        return None
    reasoning, sep, answer = rest.partition("\n\n=== ANSWER ===\n")
    return reasoning, answer if sep else ""


def find_reasoning(e, entry):
    """(reasoning text, log file name) for an answer, or (None, why). Newer results name their log
    file; for older ones the log is found by model, pack and test, and matched on the answer text."""
    live = os.path.join(e.log_dir, "live")
    rec = entry["record"]
    if rec.get("log"):
        got = read_log_sections(os.path.join(live, rec["log"]))
        if got:
            return got[0], rec["log"]
        return None, f"its log file logs/live/{rec['log']} is gone"
    pattern = glob.escape(f"_{entry['model']}_{entry['pack']}_{rec['test'][:40]}") + ".txt"
    names = sorted((os.path.basename(p) for p in glob.glob(os.path.join(glob.escape(live), "*" + pattern))),
                   reverse=True)
    answer = rec.get("answer") or ""
    for name in names[:300]:
        got = read_log_sections(os.path.join(live, name))
        if got and got[1].startswith(answer + "\n"):
            return got[0], name
    return None, "no log file in logs/live/ matches this answer"


def check_lines(rec, test):
    """Markup lines for a code test's checks: from the result when it names them, otherwise
    rebuilt from the test's check names and the grading reason (older results)."""
    if rec.get("checks"):
        return [f"[green]✓[/green] {k}" if v is None else f"[red]✗ {k}[/red]  {rich_escape(str(v))}"
                for k, v in rec["checks"].items()]
    if not test or test.get("grader") != "code" or not test.get("hidden_tests"):
        return []
    from .graders import code as code_grader
    names = [n for n, _ in code_grader._split_checks(test["hidden_tests"])[1]]
    reason = rec.get("reason", "")
    if rec.get("pass"):
        return [f"[green]✓[/green] {n}" for n in names]
    if not reason.startswith("Failed "):
        return [f"[dim]· {n}  (not run)[/dim]" for n in names]
    return [f"[red]✗ {n}[/red]" if f"{n}: " in reason else f"[green]✓[/green] {n}" for n in names]


def rich_escape(text):
    return text.replace("[", r"\[")


class BlockDumper(yaml.SafeDumper):
    """Multi-line strings (reference code, expected text) as readable | blocks, not escaped \\n."""


BlockDumper.add_representer(str, lambda d, v: d.represent_scalar("tag:yaml.org,2002:str", v,
                                                                   style="|" if "\n" in v else None))


GLOSSARY = """[b]Words tuieval uses[/b]
  [b]pack[/b]        a folder of your own questions with checkable answers (packs/<name>/)
  [b]use case[/b]    a pack's group; a model PASSES a use case when all its packs pass
  [b]Smoke[/b]       3 questions per pack, once: checks the model and server work (minutes)
  [b]Screen[/b]      a sample of each pack, once: drops weak models fast; can say FAIL, not PASS
  [b]Certify[/b]     every question, with repeats: the only tier that can say PASS
  [b]PASS[/b]        enough answers right, with zero critical failures, to trust the model for this
  [b]FAIL[/b]        it got too much wrong, or broke a hard rule (a critical failure)
  [b]INCONCLUSIVE[/b] not enough answers yet to decide; the verdict says what's missing
  [b]critical[/b]    a failure that disqualifies on its own (a forbidden action, an invented answer)
  [b]gate[/b]        what PASS means for a pack (pack.toml \\[gate]); docs/writing-packs.md explains it"""

HELP = {
    "SetupScreen": """[b]Setup: choose what to run[/b]

1. Tick [b]packs[/b] on the left and [b]models[/b] on the right (space; type to filter models).
2. Pick a [b]tier[/b]: Smoke first to check the setup, then Screen, then Certify for finalists.
3. Press [b]s[/b]. The line above the buttons says how many answers that is and roughly how long.

[b]Models at a time[/b]: how many of the ticked models run side by side (blank: this machine's
parallel_models in models.toml, default 1). Answers note the models they ran alongside.

[b]Keys[/b]
  s  start (or queue it behind a run that's going)     r  results
  a  add a model (a GGUF, a model id, openrouter:<id>)  m  scan model folders for new GGUFs
  e  pick which tests of the highlighted pack run       p  save or load a selection (preset)
  t  tune the ticked models' speed flags here (after   x  hide or unhide a model
     a first run)
  w  runs and queue of this session                     q  quit

Something not working? In a terminal: [b]tuieval doctor[/b] checks servers, models and keys.""",
    "RunScreen": """[b]A run in progress[/b]

The top shows progress, ETA and the live score per model and pack. Below: the current question
with the model's reasoning and answer as they stream, then recent results with the grader's reason
(enter on one opens it in full).

[b]Keys[/b]
  k  skip the current model          c  cancel everything (finished answers are kept; it resumes next time)
  f  pause or resume auto-scroll     r  results     n  set up a new run (this one keeps going)
  w  runs and queue                  esc  back to Setup (the run keeps going)
  v  with several models at a time: stream the next one (k skips the one streaming)""",
    "ResultsScreen": """[b]Results[/b]

Start with [b]Production readiness[/b]: one verdict per model and use case, and what's missing when
it's INCONCLUSIVE. The other tabs are the evidence:
  Scorecard              accuracy, critical failures and truncation per model and pack
  Speed & tokens         time and tokens per answer (★ = nothing beats it on both accuracy and time)
  Per question           every question, model by model
  Is the difference real? whether one model is really better than another, or it's noise
  Tests that separate    the questions that tell models apart
  Failures               every wrong answer with the grader's reason; enter opens it in full
  Test quality           tests nobody fails, everybody fails, or that look broken

[b]Keys[/b]  esc back · ctrl+r refresh · w runs""",
    "TuneScreen": """[b]Tuning speed flags[/b]

tuieval tries speed-only server flags (batch sizes, flash attention, speculative decoding …) on this
machine and keeps the fastest set. A guard rejects any flag that changes the model's answers, so
tuning never changes results, only speed. It takes several minutes per model.

[b]Keys[/b]  c cancel · esc back to Setup · w runs""",
    "AnswerScreen": """[b]One answer in full[/b]

The grade and the grader's reason at the top, then tabs: the model's reasoning and answer, the
question as sent, and what the grader checked.

[b]Keys[/b]  [ and ] previous / next answer · esc close""",
    "PickTestsScreen": """[b]Pick tests[/b]

Tick the tests of this pack to run (space). Type to filter by id, description or category, enter to
go back to the list, and [b]a[/b] ticks every test the filter shows. With none ticked the tier's usual
tests run. Runs of different tests add up; the pack can PASS once all its tests have run.""",
}


class HelpScreen(ModalScreen):
    """What the screen underneath is for, its keys, and the words tuieval uses."""
    BINDINGS = [Binding("escape", "close", "Close"), Binding("question_mark", "close", "Close", show=False),
                Binding("q", "close", "Close", show=False)]

    def __init__(self, screen_name):
        super().__init__()
        self.text = HELP.get(screen_name, "[b]Help[/b]\n\nesc closes this dialog.")

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog", classes="wide"):
            with VerticalScroll(id="help-body"):
                yield Static(self.text + "\n\n" + GLOSSARY)
            yield Label("[dim]esc closes · docs: README.md and docs/ in the tuieval repository[/dim]")

    def action_close(self):
        self.dismiss()


class AnswerScreen(ModalScreen):
    """Everything about one answer: grading, timing, the question, the reasoning and the answer.
    [ and ] step through the other answers of the list it was opened from."""
    BINDINGS = [
        Binding("escape", "close", "Close", priority=True),
        Binding("q", "close", "Close", show=False, priority=True),
        Binding("left_square_bracket", "step(-1)", "Previous", priority=True),
        Binding("right_square_bracket", "step(1)", "Next", priority=True),
    ]

    def __init__(self, entries, index=0, what="answer"):
        super().__init__()
        self.entries, self.index, self.what = entries, index, what

    def check_action(self, action, parameters):
        return len(self.entries) > 1 if action == "step" else True

    def compose(self) -> ComposeResult:
        with Vertical(id="answer-dialog"):
            yield Static(id="answer-head")
            with VerticalScroll(id="answer-grading"):
                yield Static(id="answer-grading-text")
            with TabbedContent(id="answer-tabs"):
                with TabPane("Reasoning & answer", id="answer-tab-text"):
                    with Horizontal():
                        with Vertical(classes="stream"):
                            yield Label(id="answer-reasoning-label")
                            yield TextArea(id="answer-reasoning", read_only=True, soft_wrap=True, show_cursor=False)
                        with Vertical(classes="stream"):
                            yield Label("[b]Answer[/b]")
                            yield TextArea(id="answer-answer", read_only=True, soft_wrap=True, show_cursor=False)
                with TabPane("Question", id="answer-tab-question"):
                    yield TextArea(id="answer-question", read_only=True, soft_wrap=True, show_cursor=False)
                with TabPane("What's checked", id="answer-tab-checked"):
                    yield TextArea(id="answer-checked", read_only=True, soft_wrap=True, show_cursor=False)
            yield Footer()

    def on_mount(self):
        self.show()

    def action_close(self):
        self.dismiss()

    def action_step(self, delta):
        self.index = (self.index + delta) % len(self.entries)
        self.show()

    def show(self):
        e = self.app.engine
        entry = self.entries[self.index]
        rec = entry["record"]
        pack = e.packs.get(entry["pack"])
        test = next((t for t in pack.tests if t["id"] == rec["test"]), None) if pack else None
        mark = "[bold green]✓ PASS[/]" if rec["pass"] else "[bold red]✗ FAIL[/]"
        if not rec["pass"] and rec.get("severity") == "critical":
            mark += " [bold red]CRITICAL[/]"
        pos = f"   [dim]{self.what} {self.index + 1} of {len(self.entries)}  (\\[ and ] step)[/dim]" \
            if len(self.entries) > 1 else ""
        partial = "  [cyan](pack still running or unfinished: not in scores yet)[/cyan]" if entry.get("partial") else ""
        quant = f"  [magenta]{rec['quantization']}[/magenta]" if rec.get("quantization") else ""
        finish = rec.get("finish") or "?"
        finish = f"[bold red]{finish} (cut off at max_tokens)[/]" if finish == "length" else finish
        toks = rec.get("completion_tokens")
        split = ", ".join(f"{k} {rec[k + '_tokens']:,}" for k in ("reasoning", "answer")
                          if rec.get(k + "_tokens") is not None)
        facts = [f"finish {finish}",
                 (f"{toks:,} tokens" + (f" ({split})" if split else "")) if toks else "",
                 f"prompt {rec['prompt_tokens']:,} tokens" if rec.get("prompt_tokens") else "",
                 (f"TTFT {rec['ttft_s']:.1f}s" + (f" [yellow]⟲ {rec['cached_tokens']} cached[/yellow]"
                                                   if rec.get("cached_tokens") else "")) if rec.get("ttft_s") is not None else "",
                 f"{rec['gen_tps']:.0f} tok/s" if rec.get("gen_tps") else "",
                 f"{rec['total_s']:.0f}s total" if rec.get("total_s") is not None else "",
                 f"on {rec['machine']}" if rec.get("machine") else "",
                 f"[yellow]ran alongside {', '.join(rec['ran_alongside'])}[/yellow]" if rec.get("ran_alongside") else "",
                 f"seed {rec['seed']}" if "seed" in rec else ""]
        self.query_one("#answer-head", Static).update(
            f"{mark}  {diff_badge(rec.get('difficulty', 'unrated'))}  [b]{rich_escape(rec.get('description', rec['test']))}[/b]"
            f"{partial}{pos}\n"
            f"[b]{entry['model']}[/b]{quant} · {pack.label if pack else entry['pack']} · {rec['test']} · "
            f"repeat {rec.get('repeat', 0) + 1}\n"
            + " · ".join(f for f in facts if f))
        grading = [f"[b]Grading:[/b] {rich_escape(rec.get('reason', ''))}"]
        grading += ["  " + line for line in check_lines(rec, test)]
        if rec.get("identical_to_repeat") is not None:
            grading.append(f"[red]Exact copy of repeat {rec['identical_to_repeat'] + 1}: possibly a cached answer.[/red]")
        if test is None:
            grading.append("[yellow]This test is no longer in the pack; question and checks can't be shown.[/yellow]")
        elif entry.get("pack_fp") and pack and entry["pack_fp"] != pack.fingerprint:
            grading.append("[yellow]The pack's questions changed since this answer; Question and What's checked show "
                           "the current version.[/yellow]")
        self.query_one("#answer-grading-text", Static).update("\n".join(grading))

        reasoning, source = find_reasoning(e, entry)
        if reasoning is None:
            label = f"[b]Reasoning[/b]  [dim]{rec.get('reasoning_chars', 0):,} chars; not available: {source}[/dim]"
            reasoning = ""
        else:
            label = f"[b]Reasoning[/b]  [dim]logs/live/{rich_escape(source)}[/dim]"
        self.query_one("#answer-reasoning-label", Label).update(label)
        answer = rec.get("answer") or ""
        if rec.get("tool_calls"):
            answer += "\n\n=== TOOL CALLS ===\n" + json.dumps(rec["tool_calls"], indent=2)
        for wid, text in (("#answer-reasoning", reasoning), ("#answer-answer", answer or "(empty)")):
            self.query_one(wid, TextArea).load_text(text)

        question, checked = "(test not found)", "(test not found)"
        if test:
            parts = []
            if pack.system:
                parts.append("=== SYSTEM ===\n" + pack.system.strip())
            if pack.tools:
                parts.append("=== TOOLS ===\n" + ", ".join(t.get("function", t).get("name", "?") for t in pack.tools))
            parts.append("=== USER ===\n" + str(test["input"]).strip())
            if test.get("image"):
                parts.append(f"=== IMAGE ===\n{pack.asset(test['image'])}")
            question = "\n\n".join(parts)
            rest = {k: v for k, v in test.items() if k not in ("id", "input", "description", "hidden_tests")}
            checked = f"Grader: {test.get('grader', '?')}\n\n" + yaml.dump(rest, Dumper=BlockDumper, sort_keys=False,
                                                                          allow_unicode=True, width=100)
            if test.get("hidden_tests"):
                checked += "\n=== HIDDEN TESTS (run against the last ```python block) ===\n" + test["hidden_tests"]
        self.query_one("#answer-question", TextArea).load_text(question)
        self.query_one("#answer-checked", TextArea).load_text(checked)


# ------------------------------------------------------------------ setup
class SetupScreen(Screen):
    BINDINGS = [
        Binding("s", "start", "Start"),
        Binding("a", "add_model", "Add model"),
        Binding("m", "scan", "Scan for models"),
        Binding("p", "presets", "Presets", show=False),
        Binding("e", "pick_tests", "Pick tests"),
        Binding("t", "tune", "Tune speed"),
        Binding("r", "results", "Results"),
        Binding("x", "hide_model", "Hide/unhide", show=False),
        Binding("w", "runs", "Runs"),
        Binding("q", "quit", "Quit"),
    ]

    def check_action(self, action, parameters):
        if action == "runs":
            return bool(self.app.sessions)
        if action == "tune":   # speed tuning matters once a model runs; ? lists it, and tuieval tune works
            return self.app.engine.has_results()
        return True

    def action_quit(self):
        # q means "back" everywhere else, so a q too many shouldn't silently drop this session's runs
        if self.app.sessions and not self.app.active_session:
            self.app.push_screen(ConfirmScreen(f"Quit? This session's {len(self.app.sessions)} run(s) stay "
                                               "in Results, but the w list closes."
                                               + self.app.queue_loss_text()),
                                 lambda yes: yes and self.app.exit())
        else:
            self.app.action_quit_app()

    def action_runs(self):
        self.app.open_sessions()

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(id="machine")
        with Horizontal(id="pickers"):
            with Vertical(classes="picker"):
                yield Label("[b]Eval packs[/b]  (space to toggle · e picks tests)")
                yield SelectionList(id="suites")
            with Vertical(classes="picker"):
                yield Label(id="models-legend")
                with Horizontal(id="filter-row"):
                    yield Input(placeholder="filter by label or tag…", id="filter")
                    yield Checkbox("Show hidden", id="show-hidden")
                yield SelectionList(id="models")
                yield Static(id="model-detail")
        with Horizontal(id="tier-row"):
            yield Label("Tier")
            with RadioSet(id="tier"):
                yield RadioButton("Smoke (3 tests, check setup)", id="tier-smoke")
                yield RadioButton("Screen (a sample: drop weak models)", id="tier-screen", value=True)
                yield RadioButton("Certify (everything, with repeats: finalists)", id="tier-certify")
        with Horizontal(id="options"):
            yield Label("Repeats")
            yield Input("", id="repeat", type="integer", max_length=2, placeholder="1")
            yield Label("Models at a time")
            yield Input("", id="parallel", type="integer", max_length=1, placeholder="1")
            yield Checkbox("Rerun existing results", id="force")
        yield Static(id="estimate")
        with Horizontal(id="buttons"):
            yield Button("Start  [s]", id="start", variant="success")
            yield Button("Add model  [a]", id="add")
            yield Button("Results  [r]", id="results")
            yield Button("Quit  [q]", id="quit", variant="error")
        yield Footer()

    def on_mount(self):
        self.selected_models = set()
        self.picks = {}    # pack -> test ids to run instead of the tier's usual tests
        self.refresh_machine()
        self.refresh_packs()
        self.refresh_models()
        state = self.app.load_state()
        self.apply_selection(state, quiet=True)
        if not state:
            self.first_time_defaults()
        self.query_one("#suites", SelectionList).focus()

    def first_time_defaults(self):
        """With nothing chosen before: tick the only pack and the only local model, and start on
        Smoke until something has run (a 3-question check of the setup)."""
        e = self.app.engine
        if len(e.packs) == 1:
            self.query_one("#suites", SelectionList).select(next(iter(e.packs)))
        local = [m["label"] for m in e.cfg["models"] if not m.get("remote")]
        if len(local) == 1:
            self.selected_models.add(local[0])
            self.refresh_models()
        if not e.has_results():
            self.query_one("#tier-smoke", RadioButton).value = True

    def refresh_machine(self):
        e = self.app.engine
        here, others = e.machines()[0], [mc.id for mc in e.machines()[1:]]
        self.query_one("#machine", Static).update(
            f"[b]Machine[/b] {here.id}  [dim]{here.summary}"
            + (f" · also known: {', '.join(others)}" if others else "")
            + (" · t tunes the ticked models here" if e.has_results() else "") + "[/dim]")

    def refresh_packs(self):
        e = self.app.engine
        packs = self.query_one("#suites", SelectionList)
        keep, at = set(packs.selected), packs.highlighted
        packs.clear_options()
        for name, pk in e.packs.items():
            picked = self.current_picks().get(name)
            count = (f"[b cyan]{len(picked)}/{len(pk.tests)} tests picked[/]" if picked
                     else f"[dim]{len(pk.tests)} tests[/dim]")
            needs = f"  [magenta]{', '.join(pk.needs)}[/magenta]" if pk.needs else ""
            mix = {lvl: sum(t.get("difficulty") == lvl for t in pk.tests) for lvl in ("easy", "medium", "hard")}
            mixed = " ".join(f"[{DIFF_STYLE[l]}]{n}{l[0].upper()}[/]" for l, n in mix.items() if n)
            packs.add_option(Selection(f"[dim]{pk.group} ·[/dim] {pk.label}  {count} {mixed}{needs}",
                                       name, name in keep))
        if at is not None and at < packs.option_count:
            packs.highlighted = at
        if not e.packs:
            for i, line in enumerate(NO_PACKS):
                packs.add_option(Selection(line, f"\x00nopacks:{i}", disabled=True))

    def refresh_models(self, select=None):
        """Local models, then each hosted API's models tried so far, best verdicts first. Hidden
        models only show with "Show hidden" (or while ticked); the line under the list shows the
        highlighted model in full."""
        e = self.app.engine
        e.reload()
        models = self.query_one("#models", SelectionList)
        if select:
            self.selected_models.add(select)
        text = self.query_one("#filter", Input).value.strip().lower()
        show_hidden = self.query_one("#show-hidden", Checkbox).value
        keep = models.highlighted_option.value if models.highlighted_option else None
        keep_at = models.highlighted
        hidden = e.hidden()
        newcomer = not e.has_results()   # nothing run yet: leave out context sizes and tuning
        try:
            self.verdicts = model_verdicts(e)
        except Exception as err:  # a broken results file must not hide the model list
            self.verdicts = {}
            self.app.log(f"verdicts failed: {err!r}")
        codes = group_codes(e)
        self.query_one("#models-legend", Label).update(
            "[b]Models[/b]  [dim][green]✓[/green] pass  [red]✗[/red] fail  [yellow]?[/yellow] inconclusive  "
            "grey = earlier  ·  x hides[/dim]\n[dim]"
            + "  ".join(f"{c} {g}" for g, c in codes.items()) + "[/dim]")
        n_hidden = sum(m["label"] in hidden for m in e.cfg["models"])
        self.query_one("#show-hidden", Checkbox).label = f"Show hidden ({n_hidden})"
        sections = {}
        for i, m in enumerate(e.cfg["models"]):
            hay = " ".join([m["label"], m["server"], *m["tags"]]).lower()
            if text and not all(word in hay for word in text.split()):
                continue
            if m["label"] in hidden and not show_hidden and m["label"] not in self.selected_models:
                continue
            section = f"{m['server']} models tried" if m.get("remote") else "Local models"
            sections.setdefault(section, []).append((verdict_rank(self.verdicts.get(m["label"], [])), i, m))
        width = max((len(m["label"]) for rows in sections.values() for _, _, m in rows), default=0)
        models.clear_options()
        for section in sorted(sections, key=lambda s: s != "Local models"):
            models.add_option(Selection(f"[b $accent]{section}[/]", "\x00section:" + section, disabled=True))
            for _, _, m in sorted(sections[section], key=lambda r: r[:2]):
                label = m["label"]
                states = [e.result_status(label, n) for n in e.packs]
                flags = ("  [yellow]~[/yellow]" if any(st and st.startswith("outdated") for st in states) else "") + \
                        ("  [yellow]…[/yellow]" if "partial" in states else "")
                extra = ("  [magenta]vision[/magenta]" if m["vision"] else "") + \
                        ("  [yellow]no-think[/yellow]" if m.get("thinking") is False else "") + \
                        ("  [dim italic]hidden[/dim italic]" if label in hidden else "")
                where = "" if m.get("remote") else f"  [dim]{m['server']}[/dim]{serving_badge(e, m, newcomer)}"
                rows = self.verdicts.get(label, [])
                vcodes = verdict_codes(rows, codes) if rows else f"[dim]{'not run':<{len(verdict_codes([], codes))}}[/dim]"
                # a Selection shows one line only: keep it short, the detail line has the rest
                models.add_option(Selection(f"{label:<{width}}  {vcodes}{flags}{where}{extra}",
                                            label, label in self.selected_models))
        # One row per hosted API that serves any model (OpenRouter): ticking it opens its model list.
        for name, server in e.cfg["servers"].items():
            if server.get("any_model") and (not text or all(w in f"{name} any model" for w in text.split())):
                models.add_option(Selection(f"[b]+ {name}[/b]: any model  [dim]tick to choose one[/dim]", PICK + name, False))
        values = [models.get_option_at_index(i).value for i in range(models.option_count)]
        enabled = [i for i in range(models.option_count) if not models.get_option_at_index(i).disabled]
        if enabled:
            at = values.index(keep) if keep in values else min(keep_at or 0, models.option_count - 1)
            models.highlighted = min((i for i in enabled if i >= at), default=enabled[-1])
        self.show_detail()
        self.update_estimate()

    @on(SelectionList.SelectionHighlighted, "#models")
    def show_detail(self, *_):
        """Everything about the highlighted model: verdicts in full, serving, per-pack results."""
        e, detail = self.app.engine, self.query_one("#model-detail", Static)
        opt = self.query_one("#models", SelectionList).highlighted_option
        label = opt.value if opt else None
        if not label or not any(m["label"] == label for m in e.cfg["models"]):
            detail.update("[dim]Highlight a model to see its verdicts and results.[/dim]")
            return
        m = e.model(label)
        mark = {"certified": "[green]✓[/green]", "screened": "[cyan]◐[/cyan]", "partial": "[yellow]…[/yellow]"}
        packs, why = [], {}
        for n in e.packs:
            st = e.result_status(label, n)
            if st:
                packs.append(f"{mark.get(st, '[yellow]~[/yellow]')}{n}")
                if st.startswith("outdated"):
                    why.setdefault(st.removeprefix("outdated: "), []).append(n)
        if why:  # each reason once, not once per pack
            packs.append("[dim]· outdated: " + "; ".join(
                (f"{r} ({', '.join(ns)})" if len(why) > 1 else r) for r, ns in why.items()) + "[/dim]")
        verdicts = "  ".join(f"[{VERDICT_STYLE.get(s, 'dim')}]{g} {s}[/]" if not when
                             else f"[dim]{g} {s} ({when})[/dim]"
                             for g, s, when in getattr(self, "verdicts", {}).get(label, [])) or "[dim]no verdicts yet[/dim]"
        hidden = "  [dim italic]hidden (x unhides)[/dim italic]" if label in e.hidden() else ""
        tags = [t for t in m["tags"] if t != m["server"]]
        hidden = (f"  [cyan]{' '.join(tags)}[/cyan]" if tags else "") + hidden
        detail.update(f"[b]{label}[/b]  [dim]{m['server']} · {m.get('model') or ''}[/dim]"
                      f"{serving_badge(e, m, not e.has_results())}{hidden}\n{verdicts}\n"
                      + ("[dim]results (✓ certified ◐ screened ~ outdated … partial):[/dim] " + "  ".join(packs)
                         if packs else "[dim]no results yet[/dim]"))

    @on(Checkbox.Changed, "#show-hidden")
    def toggle_hidden(self, _):
        self.refresh_models()

    def action_hide_model(self):
        models = self.query_one("#models", SelectionList)
        opt = models.highlighted_option
        e = self.app.engine
        if self.focused is not models or not opt or not any(m["label"] == opt.value for m in e.cfg["models"]):
            self.notify("Highlight a model in the Models list first, then press x.", severity="warning")
            return
        label, hide = opt.value, opt.value not in e.hidden()
        e.set_hidden(label, hide)
        if hide:
            self.selected_models.discard(label)
            shown = self.query_one("#show-hidden", Checkbox).value
            self.notify(f"Hid {label}; its results are kept."
                        + ("" if shown else " Tick 'Show hidden' to see it again (x unhides)."))
        else:
            self.notify(f"{label} is listed again.")
        self.refresh_models()

    @on(Input.Changed, "#filter")
    def filter_models(self, _):
        self.refresh_models()

    @on(SelectionList.SelectedChanged, "#models")
    def remember_models(self, _):
        """Keep ticks for models hidden by the filter; sync the visible ones."""
        models = self.query_one("#models", SelectionList)
        picks = [v for v in models.selected if v.startswith(PICK)]
        if picks:
            models.deselect(picks[0])
            def done(label):
                if label:
                    self.refresh_models(select=label)
                    self.notify(f"Added {label} for this session. It stays listed once it has results.")
            self.app.push_screen(RemoteModelScreen(picks[0].removeprefix(PICK)), done)
            return
        visible = {models.get_option_at_index(i).value for i in range(models.option_count)}
        self.selected_models = (self.selected_models - visible) | set(models.selected)
        self.update_estimate()

    def tier(self):
        pressed = self.query_one("#tier", RadioSet).pressed_button
        return pressed.id.removeprefix("tier-") if pressed else "screen"

    def settings(self):
        try:
            repeat = max(1, int(self.query_one("#repeat", Input).value)) or None
        except ValueError:
            repeat = None  # blank: the tier default (1, or each pack's certification repeats)
        order = [m["label"] for m in self.app.engine.cfg["models"]]
        return ([l for l in order if l in self.selected_models],
                list(self.query_one("#suites", SelectionList).selected), repeat,
                self.tier(), self.query_one("#force", Checkbox).value)

    def parallel(self):
        """Models at a time for the next run: the field, or blank for this machine's parallel_models."""
        try:
            return max(1, int(self.query_one("#parallel", Input).value)) or None
        except ValueError:
            return None

    def current_picks(self, suites=None):
        """{pack: picked test ids} for packs that still have those tests (and, given suites, are ticked)."""
        e = self.app.engine
        out = {}
        for name, ids in getattr(self, "picks", {}).items():
            pk = e.packs.get(name)
            known = [i for i in ids if pk and any(t["id"] == i for t in pk.tests)]
            if known and (suites is None or name in suites):
                out[name] = known
        return out

    def action_pick_tests(self):
        """Choose which tests of the highlighted pack run (none ticked: the tier's usual tests)."""
        packs = self.query_one("#suites", SelectionList)
        opt = packs.highlighted_option
        if opt is None or opt.value not in self.app.engine.packs:
            self.notify("Highlight a pack first.", severity="warning")
            return
        name = opt.value

        def done(ids):
            if ids is None:
                return
            if ids:
                self.picks[name] = ids
                packs.select(name)
            else:
                self.picks.pop(name, None)
            self.refresh_packs()
            self.update_estimate()
        self.app.push_screen(PickTestsScreen(self.app.engine.packs[name], self.current_picks().get(name, [])), done)

    @on(SelectionList.SelectedChanged)
    @on(Input.Changed)
    @on(Checkbox.Changed)
    @on(RadioSet.Changed)
    def update_estimate(self, *_):
        labels, suites, repeat, tier, force = self.settings()
        est = self.query_one("#estimate", Static)
        e = self.app.engine
        defaults = sorted({e.default_repeat(e.packs[p], tier) for p in suites}) or [1]
        self.query_one("#repeat", Input).placeholder = str(defaults[0]) if len(defaults) == 1 else "pack"
        self.query_one("#parallel", Input).placeholder = str(e.parallel_models())
        if not e.packs:
            est.update("[yellow]No packs yet.[/yellow] [dim]Quit, run tuieval new-pack <name>, "
                       "and open tuieval again.[/dim]")
            return
        if not e.cfg["models"]:
            est.update("[yellow]No models yet.[/yellow] [dim]Press a to add one (a GGUF path, a model id on "
                       "a running server, or openrouter:<model id>).[/dim]")
            return
        if not labels or not suites:
            est.update("Tick at least one [b]pack[/b] (left) and one [b]model[/b] (right) with space, then press "
                       "[b]s[/b] to start." + self.queue_hint())
            return
        picks = self.current_picks(suites)
        jobs = self.app.engine.plan(labels, suites, repeat, tier, force, picks)
        rerun = not any(j.status == "waiting" for j in jobs)
        if rerun:   # everything has results: Start runs it again, so estimate that
            jobs = self.app.engine.plan(labels, suites, repeat, tier, True, picks)
        run = [j for j in jobs if j.status == "waiting"]
        skipped = [j for j in jobs if j.status == "skipped"]
        secs, notes = self.app.engine.estimate_seconds(jobs)
        todo = sum(j.total for j in run)
        reps = sorted({j.repeat for j in jobs})
        reps_text = "/".join(map(str, reps)) + (" (pack default)" if repeat is None and len(reps) > 1 else "")
        text = (f"[b]{tier.title()}[/b]: [b]{len(labels)}[/b] model(s) × [b]{len(suites)}[/b] pack(s) × "
                f"[b]{reps_text}[/b] repeat(s) = "
                f"[b]{todo:,}[/b] answers · about [b]{fmt_secs(secs)}[/b]"
                + (f" [dim](rough: {'; '.join(notes)})[/dim]" if notes else ""))
        if tier == "smoke" and not e.has_results():
            text += "\n[green]First run: Smoke asks 3 questions per pack to check the setup works. Press s.[/green]"
        if picks:
            text += f"\n[cyan]Picked tests only in {', '.join(e.packs[p].label for p in picks)}[/cyan]" \
                    "[dim] (e changes; a pack with picked tests can't PASS until all its tests have run)[/dim]"
        elif tier == "screen":
            text += "\n[dim]Screening can only say FAIL or promising; PASS needs Certify.[/dim]"
        finished = [j for j in jobs if j.status == "done"]
        if finished:
            text += f"   [dim]{len(finished)} pack(s) already done[/dim]"
        if skipped:
            reasons = {}
            for j in skipped:
                reasons.setdefault(j.note, []).append(j.key)
            text += "   [dim]skipping: " + "; ".join(f"{len(v)} {k}" for k, v in reasons.items()) + "[/dim]"
        if not run:
            text += "\n[yellow]Nothing selected can run (see skipping).[/yellow]"
        elif rerun:
            text += "\n[yellow]Everything selected has results; Start runs it again.[/yellow]"
        resumed = [j for j in run if "done earlier" in j.note]
        if resumed:
            text += f"   [yellow]{len(resumed)} pack(s) continue from earlier results[/yellow]"
        at_once = min(self.parallel() or e.parallel_models(), len({j.label for j in run}))
        if at_once > 1:
            text += (f"\n[cyan]Up to {at_once} models at a time[/cyan][dim] (the time above assumes one at a "
                     "time). Answers note the models they ran alongside, since those share the machine's "
                     "speed.[/dim]")
        est.update(text + self.queue_hint())

    def queue_hint(self):
        """Says what Start will do when something is already running, here or in another window."""
        app = self.app
        if app.is_busy():
            return (f"\n[cyan]{app.busy_text()}: Start adds this to the queue as #{len(app.queue) + 1} "
                    "(w shows the queue).[/cyan]")
        holder = app.engine.machine_holder()
        if holder:
            return f"\n[yellow]Start waits: {engine.holder_text(holder)}.[/yellow]"
        return ""

    def apply_selection(self, sel, quiet=False):
        """Tick the given models/packs (unknown names are ignored) and set the options."""
        if not sel:
            return
        e = self.app.engine
        self.selected_models = {l for l in sel.get("models", []) if any(m["label"] == l for m in e.cfg["models"])}
        self.picks = {p: list(ids) for p, ids in (sel.get("tests") or {}).items()}
        self.refresh_packs()
        packs = self.query_one("#suites", SelectionList)
        packs.deselect_all()
        for name in sel.get("packs", []):
            if name in e.packs:
                packs.select(name)
        self.query_one("#repeat", Input).value = str(sel["repeat"]) if sel.get("repeat") else ""
        if "parallel" in sel:
            self.query_one("#parallel", Input).value = str(sel["parallel"]) if sel["parallel"] else ""
        tier = sel.get("tier") or ("smoke" if sel.get("smoke") else None)
        if tier in engine.TIERS:
            self.query_one(f"#tier-{tier}", RadioButton).value = True
        if "force" in sel:
            self.query_one("#force", Checkbox).value = bool(sel["force"])
        self.query_one("#filter", Input).value = ""
        self.refresh_models()
        if not quiet:
            self.notify(f"Selected {len(self.selected_models)} model(s) and {len(packs.selected)} pack(s).")

    def action_presets(self):
        labels, packs, repeat, _, _ = self.settings()
        tests = self.current_picks(packs)

        def done(result):
            if result and result[0] == "load":
                self.apply_selection(result[1])
            elif result and result[0] == "saved":
                self.notify(f"Saved preset '{result[1]}'. Also usable as tuieval run --preset {result[1]}")
        self.app.push_screen(PresetsScreen(labels, packs, repeat, tests), done)

    def action_scan(self):
        def done(added):
            if added:
                for label in added:
                    self.selected_models.add(label)
                self.refresh_models()
                self.notify(f"Added {len(added)} model(s). Try the Smoke tier first.")
        self.app.push_screen(ScanScreen(), done)

    @on(Button.Pressed, "#start")
    def action_start(self):
        """Start now, or queue it behind the run or tune that's going (one at a time)."""
        labels, suites, repeat, tier, force = self.settings()
        if not labels or not suites:
            self.notify("Select at least one eval pack and one model.", severity="warning")
            return
        picks = self.current_picks(suites)
        sel = {"models": labels, "packs": suites, "repeat": repeat, "tier": tier, "force": force, "tests": picks,
               "parallel": self.parallel()}
        jobs = self.app.engine.plan(labels, suites, repeat, tier, force, picks)
        rerun = False
        if not any(j.status == "waiting" for j in jobs):
            # Everything selected already has results: Start means run it again (earlier results
            # are kept in history/). Packs skipped for another reason (no vision, …) stay skipped.
            jobs = self.app.engine.plan(labels, suites, repeat, tier, True, picks)
            if not any(j.status == "waiting" for j in jobs):
                reasons = sorted({j.note for j in jobs if j.status == "skipped"})
                self.notify(f"Nothing selected can run: {'; '.join(reasons)}", severity="warning")
                return
            rerun = True
        self.app.save_state(sel)
        lost = [(j, what, n, why) for j in jobs if j.status == "waiting" for what, n, why in j.discards]
        item = QueuedRun(sel, rerun, {(j.key, what) for j, what, _, _ in lost}, jobs, self.app.engine)
        if not lost:
            self.app.start_or_queue(item)
            return

        def answer(yes):
            if yes:
                self.app.start_or_queue(item)
        lines = "\n".join(f"• {j.label} · {j.pack.label}: {n} {what} answers — {why}" for j, what, n, why in lost)
        self.app.push_screen(ConfirmScreen(
            "These packs would start over: their earlier answers ran with other questions or settings.\n\n"
            f"{lines}\n\nThe earlier answers move to results/<model>/history/ (kept, but no longer counted). "
            "To resume instead, pick No and put the setting back in models.toml.\n\nStart over?"), answer)

    def action_tune(self):
        e = self.app.engine
        local = [m["label"] for m in e.cfg["models"] if e.serving(m).perf_source != "n/a"]
        labels = [l for l in local if l in self.selected_models]
        if not labels:
            hidden = e.hidden()
            labels = [l for l in local if l not in hidden
                      and e.serving(e.model(l)).perf_source not in ("tuned", "seeded")]
        if not labels:
            self.notify("Tick the models to tune (every local model already has a profile here).",
                        severity="warning")
            return

        def confirmed(yes):
            if yes:
                self.app.start_or_queue(QueuedTune(labels))
        later = (f"\n\n{self.app.busy_text()}: this goes in the queue as #{len(self.app.queue) + 1}."
                 if self.app.is_busy() else "")
        self.app.push_screen(ConfirmScreen(
            f"Tune {', '.join(labels)} on {e.machine().id}? About 8-15 server starts per model "
            f"(~20-30 min each for a large model).{later}"), confirmed)

    @on(Button.Pressed, "#add")
    def action_add_model(self):
        def done(label):
            if label:
                self.refresh_models(select=label)
                self.notify(f"Added {label}. Try the Smoke tier first.")
        self.app.push_screen(AddModelScreen(), done)

    @on(Button.Pressed, "#results")
    def action_results(self):
        self.app.push_screen(ResultsScreen())

    @on(Button.Pressed, "#quit")
    def quit_pressed(self):
        self.action_quit()

    def on_screen_resume(self):
        self.refresh_bindings()  # "w Runs" appears once there is a run
        self.refresh_machine()
        self.refresh_packs()
        self.refresh_models()
        self.update_estimate()   # the queue hint changes as runs start and end


# ------------------------------------------------------------------ tune
class TuneScreen(Screen):
    """Runs tune.py for some models on this machine and shows its progress."""
    BINDINGS = [
        Binding("c", "cancel", "Cancel", priority=True),
        Binding("w", "runs", "Runs", priority=True),
        Binding("escape", "back", "Setup", priority=True),
        Binding("q", "back", "Back", show=False, priority=True),
    ]
    kind = "Tune"

    def __init__(self, labels):
        super().__init__()
        self.labels = labels
        self.running = True
        self.started = time.time()
        self.current = labels[0]
        self.summary = ""
        self.waiting = None

    def check_action(self, action, parameters):
        return self.running if action == "cancel" else True

    def live_text(self):
        if self.waiting:
            return "Tune waiting for another tuieval window"
        i = self.labels.index(self.current) + 1 if self.current in self.labels else 1
        return f"Tuning {self.current} ({i}/{len(self.labels)}) · {fmt_secs(time.time() - self.started)}"

    def session_line(self):
        what = f"{time.strftime('%H:%M', time.localtime(self.started))}  Tune  {', '.join(self.labels)}"
        if self.running:
            return f"{what}  ● waiting for another tuieval window" if self.waiting else f"{what}  ● running"
        return f"{what}  {self.summary}"

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(id="tune-status")
        yield Log(id="tune-log", max_lines=5000)
        yield Footer()

    def on_mount(self):
        e = self.app.engine
        self.status(f"Tuning {len(self.labels)} model(s) on {e.machine().id}. Each option starts the server, "
                    "runs a fixed built-in workload (short, medium and one long prompt) and keeps what is fastest without changing "
                    "answers. c cancels.")
        self.app.start_tune(self.labels, self.on_engine_event, self.finished)

    def status(self, text):
        self.query_one("#tune-status", Static).update(text)

    def write(self, text):
        self.query_one("#tune-log", Log).write_line(text)

    def on_engine_event(self, kind, **d):
        if self.app.closing:
            return
        if kind.startswith("tune_") or kind in ("model_failed", "tune_model", "model_ready", "server_facts",
                                                 "server_log", "server_stall", "machine_busy", "machine_free"):
            self.app.call_from_thread(self.handle, kind, d)

    def handle(self, kind, d):
        if kind == "machine_busy":
            self.waiting = d["holder"]
            msg = engine.holder_text(d["holder"])
            self.status(f"[yellow]Waiting:[/yellow] {msg}. c cancels.")
            self.write("waiting: " + msg)
            self.notify(f"Waiting: {msg}.", severity="warning", timeout=15)
        elif kind == "machine_free":
            self.waiting = None
            self.started = time.time()
            self.write("the machine is free; starting")
        elif kind == "tune_model":
            self.current = d["label"]
            self.write(f"\n━━ {d['label']}")
        elif kind == "model_ready":
            self.write(f"    ready after {d['load_s']:.0f}s" if d.get("load_s") else "    ready")
        elif kind == "server_stall":
            self.write("    STALL: " + d["message"])
        elif kind == "server_facts":
            self.write("    " + " · ".join(f"{k} {v}" for k, v in d["facts"].items()))
        elif kind == "tune_step" and d.get("start"):
            self.status(f"{d.get('label', '')}: server start {d['start']} of at most {d['max_starts']}")
            self.write(d["message"])
        elif kind == "tune_progress":
            self.write(d["message"])
            self.status(f"{d.get('label', '')}: {d['message'].strip()}")
        elif kind == "server_log":
            self.write("    | " + d["line"][:160])
        elif kind in ("tune_step", "tune_result", "tune_done"):
            self.write(d["message"])
        elif kind == "tune_error":
            self.write(f"[not tuned] {d['label']}: {d['message']}")

    def finished(self, summary):
        self.running = False
        self.summary = summary
        self.status("Done. " + summary + "  Esc goes back.")
        self.write("\n" + summary)
        self.refresh_bindings()
        self.app.refresh_live()
        self.notify(summary + ("" if self.is_current else "  Press w to open it."), timeout=10)
        self.app.session_finished(self)

    def action_cancel(self):
        if self.running:
            def stop():
                self.app.engine.cancel()
                self.status("Cancelling… (stopping the server)")
            self.app.ask_cancel("Cancel the tune? Models already tuned keep their profiles.", stop, confirm=False)

    def action_runs(self):
        self.app.open_sessions()

    def action_back(self):
        if self.running:
            self.notify("Tuning keeps going; the header shows its progress. Press w to come back.")
        self.app.back_to_setup()


# ------------------------------------------------------------------ add model
class AddModelScreen(ModalScreen):
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def compose(self) -> ComposeResult:
        e = self.app.engine
        servers = [("Auto (.gguf → llama)", "")] + [(n, n) for n in e.cfg["servers"]]
        with Vertical(id="dialog"):
            yield Label("[b]Add a model[/b]")
            yield Static("[dim]Looking for servers running on this machine…[/dim]", id="detected-label")
            yield OptionList(id="detected")
            yield Label("GGUF path or model id")
            yield Input(placeholder="~/models/Some-Tune-Q4_K_M.gguf  or  org/model-id  or  openrouter:qwen/qwen3-32b",
                        id="model")
            yield Label("Server")
            yield Select(servers, id="server", allow_blank=False, value="")
            yield Label("Label (names the results; lowercase)")
            yield Input(placeholder="derived from the model", id="label")
            with Collapsible(title="More options: vision, thinking, tags", collapsed=True, id="more"):
                yield Checkbox("Can read images (enables vision packs)", id="vision")
                yield Checkbox("Thinking off (adds -nothink to the label)", id="nothink")
                yield Label("Tags (comma-separated, e.g. 27b, moe, q4)")
                yield Input(placeholder="optional", id="tags")
                yield Label("mmproj file (llama vision models only)")
                yield Input(placeholder="optional: ~/models/…-mmproj.gguf", id="mmproj")
            yield Static(id="error")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Save", id="save", variant="success")
                yield Button("Cancel", id="cancel")

    def on_mount(self):
        self.found = {}    # model id -> the running server listing it
        self.query_one("#detected", OptionList).display = False
        threading.Thread(target=self._detect, daemon=True).start()

    def _detect(self):
        e = self.app.engine
        try:
            found = engine.detect_servers(engine.detect_ports(e.cfg))
        except Exception:   # detection is a convenience; the dialog works without it
            found = []
        self.app.call_from_thread(self._show_detected, found)

    def _show_detected(self, found):
        from . import onboard
        items = onboard.unregistered(self.app.engine.cfg, found)
        label, ol = self.query_one("#detected-label", Static), self.query_one("#detected", OptionList)
        if not items:
            label.update("[dim]No new models on running servers (LM Studio, Ollama, vLLM…). "
                         "Type a GGUF path, a model id, or openrouter:<model id>.[/dim]" if not found else
                         "[dim]Every model on the running servers is already added.[/dim]")
            return
        self.found = {mid: f for mid, f in items}
        label.update("[b]Running now[/b] [dim](enter picks one)[/dim]")
        ol.add_options([Option(f"{rich_escape(mid)}  [dim]{rich_escape(onboard.where(f))}[/dim]", id=mid)
                        for mid, f in items])
        ol.display = True

    @on(OptionList.OptionSelected, "#detected")
    def detected_picked(self, event):
        self.query_one("#model", Input).value = event.option.id
        self.query_one("#server", Select).value = ""
        self.query_one("#label", Input).focus()

    @on(Input.Changed, "#model")
    def suggest_label(self, event):
        self.query_one("#label", Input).placeholder = engine.slug(event.value) or "derived from the model"

    @on(Button.Pressed, "#save")
    @on(Input.Submitted)
    def save(self):
        get = lambda i: self.query_one(i, Input).value  # noqa: E731
        e, server = self.app.engine, self.query_one("#server", Select).value
        spec = f"{server}:{get('#model').strip()}" if e.cfg["servers"].get(server, {}).get("any_model") \
            else get("#model").strip()
        if e.cfg["servers"].get(spec.partition(":")[0], {}).get("any_model"):
            # A hosted API's model (openrouter:<id>): used this session, not written to models.toml;
            # once it has results it stays listed. Vision, tools and context come from its model list.
            try:
                label = e.resolve(spec)
            except engine.ConfigError as err:
                self.query_one("#error", Static).update(f"[red]{err}[/red]")
                return
            self.dismiss(label)
            return
        server = self.query_one("#server", Select).value or None
        model = get("#model").strip()
        if not server and not model.lower().endswith(".gguf") and not os.path.exists(os.path.expanduser(model)):
            f = self.found.get(model) or next(iter(engine.find_running(e.cfg, model)), None) if model else None
            if f:   # the running server that lists it (added to models.toml if it isn't there yet)
                server = engine.ensure_server(e.models_path, f["url"], f["port"])
                e.reload()
        try:
            label, warnings = engine.add_model(
                self.app.engine.models_path, get("#model"), server,
                get("#label") or None, self.query_one("#vision", Checkbox).value, get("#mmproj") or None,
                False if self.query_one("#nothink", Checkbox).value else None,
                [t.strip() for t in get("#tags").split(",")])
        except engine.ConfigError as err:
            self.query_one("#error", Static).update(f"[red]{err}[/red]")
            return
        for w in warnings:
            self.app.notify(w, severity="warning", timeout=8)
        self.dismiss(label)

    @on(Button.Pressed, "#cancel")
    def action_cancel(self):
        self.dismiss(None)


PICK = "\x00pick:"   # models-list value of the "+ openrouter: any model" row (labels can't contain it)


class RemoteModelScreen(ModalScreen):
    """Pick any model a hosted API serves (e.g. OpenRouter) from its live model list."""
    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    SHOWN = 200

    def __init__(self, server):
        super().__init__()
        self.server, self.catalogue = server, {}

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog", classes="wide"):
            yield Label(f"[b]Choose a {self.server} model[/b]  [dim]type to filter · ↓ then enter picks · "
                        "enter in the box takes the exact id typed[/dim]")
            yield Input(placeholder="e.g. qwen3  or  deepseek  or  qwen/qwen3-235b-a22b-2507", id="q")
            yield OptionList(id="ids")
            yield Static("[dim]Loading the model list…[/dim]", id="error")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Cancel", id="cancel")

    def on_mount(self):
        self.query_one("#q", Input).focus()
        e = self.app.engine
        def load():
            try:
                cat = e.catalogue(self.server)
                self.app.call_from_thread(self.loaded, cat, None)
            except engine.ConfigError as err:
                self.app.call_from_thread(self.loaded, {}, str(err))
        threading.Thread(target=load, daemon=True).start()

    def loaded(self, cat, err):
        self.catalogue = cat
        self.query_one("#error", Static).update(f"[red]{err}[/red]" if err else "")
        self.fill()

    @staticmethod
    def describe(mid, meta):
        price = meta.get("pricing") or {}
        try:
            cost = f"${float(price['prompt']) * 1e6:.2f}/${float(price['completion']) * 1e6:.2f} per M in/out"
        except (KeyError, TypeError, ValueError):
            cost = ""
        ctx = f"{meta['context_length'] // 1000}k ctx" if meta.get("context_length") else ""
        vision = "vision" if "image" in ((meta.get("architecture") or {}).get("input_modalities") or []) else ""
        extra = "  ".join(x for x in (ctx, cost, vision) if x)
        return f"{mid}  [dim]{extra}[/dim]"

    @on(Input.Changed, "#q")
    def fill(self, *_):
        words = self.query_one("#q", Input).value.strip().lower().split()
        hits = [(mid, meta) for mid, meta in sorted(self.catalogue.items())
                if all(w in f"{mid} {meta.get('name', '')}".lower() for w in words)]
        ids = self.query_one("#ids", OptionList)
        ids.clear_options()
        ids.add_options([Option(self.describe(mid, meta), id=mid) for mid, meta in hits[:self.SHOWN]])
        if self.catalogue:
            more = f", showing {self.SHOWN}; type more to narrow" if len(hits) > self.SHOWN else ""
            self.query_one("#error", Static).update(f"[dim]{len(hits)} of {len(self.catalogue)} models{more}[/dim]")

    @on(OptionList.OptionSelected, "#ids")
    def picked(self, event):
        self.choose(event.option.id)

    @on(Input.Submitted, "#q")
    def typed(self):
        value = self.query_one("#q", Input).value.strip()
        ids = self.query_one("#ids", OptionList)
        if value in self.catalogue or ids.option_count != 1:
            self.choose(value)
        else:   # the filter left exactly one model
            self.choose(ids.get_option_at_index(0).id)

    def choose(self, mid):
        try:
            label = self.app.engine.resolve(f"{self.server}:{mid}")
        except engine.ConfigError as err:
            self.query_one("#error", Static).update(f"[red]{err}[/red]")
            return
        self.dismiss(label)

    @on(Button.Pressed, "#cancel")
    def action_cancel(self):
        self.dismiss(None)


class PresetsScreen(ModalScreen):
    """Load a saved selection, or save the current one (presets.toml)."""
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, labels, packs, repeat, tests=None):
        super().__init__()
        self.current = (labels, packs, repeat, tests or {})

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("[b]Presets[/b]  [dim]enter loads the highlighted one[/dim]")
            yield OptionList(id="presets")
            yield Label("Save the current selection as")
            yield Input(placeholder="e.g. nightly", id="name")
            yield Static(id="error")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Save", id="save", variant="success")
                yield Button("Close", id="cancel")

    def on_mount(self):
        self.presets = engine.load_presets(self.app.presets_path)
        opts = self.query_one("#presets", OptionList)
        for name, pr in self.presets.items():
            who = ", ".join(pr.get("models", [])) or ("tags " + ", ".join(pr["tags"]) if pr.get("tags") else "all models")
            opts.add_option(f"[b]{name}[/b]  [dim]{', '.join(pr.get('packs', ['all packs']))} · {who}"
                            f"{' · repeat ' + str(pr['repeat']) if pr.get('repeat') else ''}"
                            f"{' · picked tests in ' + ', '.join(pr['tests']) if pr.get('tests') else ''}[/dim]")
        if not self.presets:
            opts.add_option("[dim]No presets yet: select models and packs, then save them here.[/dim]")
            opts.disabled = True

    @on(OptionList.OptionSelected, "#presets")
    def load(self, event):
        name = list(self.presets)[event.option_index]
        labels, packs, repeat, tests = engine.resolve_preset(self.app.engine.cfg, self.presets[name],
                                                             list(self.app.engine.packs))
        self.dismiss(("load", {"models": labels, "packs": packs, "repeat": repeat, "tests": tests}))

    @on(Button.Pressed, "#save")
    @on(Input.Submitted, "#name")
    def save(self):
        name = self.query_one("#name", Input).value.strip()
        labels, packs, repeat, tests = self.current
        err = self.query_one("#error", Static)
        if not name or not engine.LABEL_RE.match(name):
            err.update("[red]Use lowercase letters, digits, '.', '_' or '-'.[/red]")
            return
        if not labels or not packs:
            err.update("[red]Select at least one model and one pack first.[/red]")
            return
        engine.save_preset(self.app.presets_path, name, labels, packs, repeat, tests)
        self.dismiss(("saved", name))

    @on(Button.Pressed, "#cancel")
    def action_cancel(self):
        self.dismiss(None)


class ScanScreen(ModalScreen):
    """GGUFs in the model folders that aren't registered yet; tick and add."""
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def compose(self) -> ComposeResult:
        dirs = ", ".join(self.app.engine.cfg["defaults"].get("model_dirs", ["~/models"]))
        with Vertical(id="dialog", classes="wide"):
            yield Label(f"[b]Unregistered GGUFs[/b] in {dirs}  [dim](model_dirs in models.toml)[/dim]")
            yield SelectionList(id="found")
            yield Static(id="error")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Add selected", id="add", variant="success")
                yield Button("Cancel", id="cancel")

    def on_mount(self):
        self.found = engine.scan_models(self.app.engine.cfg)
        lst = self.query_one("#found", SelectionList)
        for i, f in enumerate(self.found):
            proj = "  [magenta]+ mmproj (vision)[/magenta]" if f["mmproj"] else ""
            lst.add_option(Selection(f"{f['model']}{proj}", i, True))
        if not self.found:
            self.query_one("#error", Static).update("Nothing new found.")

    @on(Button.Pressed, "#add")
    def add(self):
        added, errors = [], []
        for i in self.query_one("#found", SelectionList).selected:
            f = self.found[i]
            try:
                label, _ = engine.add_model(self.app.engine.models_path, f["model"], mmproj=f["mmproj"])
                added.append(label)
            except engine.ConfigError as e:
                errors.append(str(e))
        for e in errors:
            self.app.notify(e, severity="warning", timeout=8)
        self.dismiss(added)

    @on(Button.Pressed, "#cancel")
    def action_cancel(self):
        self.dismiss([])


# ------------------------------------------------------------------ run
def run_title(tier, models, packs):
    models, packs = sorted(models), sorted(packs)
    return (f"Run {tier}  {', '.join(models) if len(models) <= 2 else f'{len(models)} models'} · "
            f"{', '.join(packs) if len(packs) <= 2 else f'{len(packs)} packs'}")


class QueuedRun:
    """A run waiting its turn. It's planned again when it starts, so it picks up what the runs before
    it finished; it never sets aside answers that weren't confirmed when it was queued."""
    kind = "Run"

    def __init__(self, sel, rerun, accepted, jobs, eng):
        self.sel, self.rerun, self.accepted = sel, rerun, accepted
        self.queued = time.time()
        self.title = run_title(sel["tier"], {j.label for j in jobs}, {j.pack.label for j in jobs})
        self.jobs = jobs
        self.est = eng.estimate_seconds(jobs)[0]

    def tests(self, eng):
        """The picked tests that are still in their packs (a pack edited while queued drops the rest)."""
        out = {}
        for p, ids in (self.sel.get("tests") or {}).items():
            pk = eng.packs.get(p)
            known = [i for i in ids if pk and any(t["id"] == i for t in pk.tests)]
            if known:
                out[p] = known
        return out

    def line(self):
        return f"{self.title}  [dim]~{fmt_secs(self.est)} · queued {time.strftime('%H:%M', time.localtime(self.queued))}[/dim]"

    def make(self, eng):
        """(RunScreen, None), or (None, why it can't start) when its turn comes."""
        sel = self.sel
        jobs = eng.plan(sel["models"], sel["packs"], sel["repeat"], sel["tier"], sel["force"] or self.rerun,
                        self.tests(eng))
        waiting = [j for j in jobs if j.status == "waiting"]
        if not waiting:
            return None, "nothing left to run (earlier runs finished it)"
        new = sorted({j.key for j in waiting for what, _, _ in j.discards if (j.key, what) not in self.accepted})
        if new:
            return None, (f"it would now start over {', '.join(new)}, which wasn't confirmed when it was queued. "
                          "Start it again from Setup to see why")
        return RunScreen(jobs, sel.get("parallel")), None


class QueuedTune:
    kind = "Tune"

    def __init__(self, labels):
        self.labels = labels
        self.queued = time.time()
        self.title = f"Tune  {', '.join(labels)}"

    def line(self):
        return f"{self.title}  [dim]queued {time.strftime('%H:%M', time.localtime(self.queued))}[/dim]"

    def make(self, eng):
        return TuneScreen(self.labels), None


class ChoiceScreen(ModalScreen):
    """A question with several buttons; dismisses with the picked id (None on esc)."""
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, question, choices):
        super().__init__()
        self.question, self.choices = question, choices

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog", classes="wide"):
            yield Label(self.question)
            with Horizontal(classes="dialog-buttons"):
                for cid, label, variant in self.choices:
                    yield Button(label, id=cid, variant=variant)

    @on(Button.Pressed)
    def answer(self, event):
        self.dismiss(event.button.id)

    def action_cancel(self):
        self.dismiss(None)


class ConfirmScreen(ModalScreen):
    def __init__(self, question):
        super().__init__()
        self.question = question

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(self.question)
            with Horizontal(classes="dialog-buttons"):
                yield Button("Yes", id="yes", variant="error")
                yield Button("No", id="no")

    @on(Button.Pressed)
    def answer(self, event):
        self.dismiss(event.button.id == "yes")


def _words(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def test_note(t):
    """What a test's row shows after its id: the description's words the id doesn't already say,
    and the category unless the id or description already names it."""
    desc, ids = str(t.get("description") or ""), [w for w in _words(t["id"]) if not re.fullmatch(r"v\d+", w)]
    note = desc
    tokens = list(re.finditer(r"[a-z0-9]+", desc.lower()))
    if set(w.group() for w in tokens) <= set(_words(t["id"])):
        note = ""                                   # the id in other words
    elif len(ids) > 1:                              # "normal entry #1, pretty JSON" for normal-entry-1-v1
        n = 0
        while n < len(tokens) and n < len(ids) and tokens[n].group() == ids[n]:
            n += 1
        if n == len(ids):
            note = desc[tokens[n - 1].end():].strip(" ,;:·-#")
    cat = str(t.get("category") or "")
    said = " ".join(_words(t["id"]) + _words(note))
    if cat and " ".join(_words(cat)) not in said:
        note = f"{note} · {cat}" if note else cat
    return note


class PickTestsScreen(ModalScreen):
    """Pick which tests of one pack run. Dismisses with the ticked ids in pack order (empty = the
    tier's usual tests), or None to keep the current pick."""
    BINDINGS = [Binding("escape", "cancel", "Cancel"),
                Binding("space", "toggle", "Tick", show=False),
                Binding("a", "toggle_shown", "Tick all shown")]

    def __init__(self, pack, chosen):
        super().__init__()
        self.pack, self.chosen, self.shown = pack, set(chosen), []

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog", classes="wide"):
            yield Label(f"[b]Pick tests: {rich_escape(self.pack.label)}[/b]\n[dim]space ticks · a ticks all shown · "
                        "type to filter, enter to go to the list · none ticked = the tier's usual tests[/dim]")
            yield Input(placeholder="filter by id, description or category", id="pick-filter")
            yield OptionList(id="pick-tests")
            yield Static(id="pick-count")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Use these", id="pick-apply", variant="success")
                yield Button("Clear (usual tests)", id="pick-clear")
                yield Button("Cancel", id="pick-cancel")

    def on_mount(self):
        self.fill("")
        self.query_one("#pick-tests").focus()

    def row(self, t):
        on = t["id"] in self.chosen
        mark = "[b green]\\[✓][/]" if on else "[dim]\\[ ][/dim]"
        tid = f"[b]{rich_escape(t['id'])}[/b]" if on else rich_escape(t["id"])
        note = test_note(t)
        return f"{mark} {diff_badge(t.get('difficulty', 'unrated'))} {tid}" + (f"  [dim]{rich_escape(note)}[/dim]" if note else "")

    def fill(self, text):
        ol = self.query_one("#pick-tests", OptionList)
        ol.clear_options()
        text = text.lower()
        self.shown = [t for t in self.pack.tests
                      if text in " ".join(str(t.get(k, "")) for k in ("id", "description", "category")).lower()]
        ol.add_options([Option(self.row(t), id=t["id"]) for t in self.shown])
        if self.shown:
            ol.highlighted = 0
        self.count()

    def count(self):
        shown = f"  [dim]({len(self.shown)} shown)[/dim]" if len(self.shown) != len(self.pack.tests) else ""
        self.query_one("#pick-count", Static).update(
            (f"[b]{len(self.chosen)}[/b] of {len(self.pack.tests)} ticked" if self.chosen
             else "[dim]None ticked: the tier's usual tests run.[/dim]") + shown)

    def redraw(self, indexes):
        ol = self.query_one("#pick-tests", OptionList)
        for i in indexes:
            ol.replace_option_prompt_at_index(i, self.row(self.shown[i]))
        self.count()

    def action_toggle(self):
        ol = self.query_one("#pick-tests", OptionList)
        if not ol.has_focus or ol.highlighted is None:
            return
        self.chosen.symmetric_difference_update({self.shown[ol.highlighted]["id"]})
        self.redraw([ol.highlighted])

    @on(OptionList.OptionSelected, "#pick-tests")
    def selected(self, event):   # enter on a row ticks it too
        self.chosen.symmetric_difference_update({event.option.id})
        self.redraw([event.option_index])

    def action_toggle_shown(self):
        """Tick every shown test, or untick them all when they're all ticked."""
        if self.query_one("#pick-filter", Input).has_focus:
            return
        ids = {t["id"] for t in self.shown}
        if ids <= self.chosen:
            self.chosen -= ids
        else:
            self.chosen |= ids
        self.redraw(range(len(self.shown)))

    def check_action(self, action, parameters):
        # while typing in the filter, space and a are text
        if action in ("toggle", "toggle_shown") and self.query_one("#pick-filter", Input).has_focus:
            return None
        return True

    @on(Input.Changed, "#pick-filter")
    def filter_changed(self, event):
        self.fill(event.value)

    @on(Input.Submitted, "#pick-filter")
    def to_list(self):
        self.query_one("#pick-tests").focus()

    @on(Button.Pressed, "#pick-apply")
    def apply(self):
        self.dismiss([t["id"] for t in self.pack.tests if t["id"] in self.chosen])

    @on(Button.Pressed, "#pick-clear")
    def clear(self):
        self.dismiss([])

    @on(Button.Pressed, "#pick-cancel")
    def action_cancel(self):
        self.dismiss(None)


class ModelPickScreen(ModalScreen):
    """Pick which models Per question compares. Dismisses with a set of labels (empty = all) or None."""
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, models, counts, chosen):
        super().__init__()
        self.models, self.counts, self.chosen = models, counts, set(chosen)

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog", classes="wide"):
            yield Label("[b]Compare models[/b]  [dim]space ticks · type to filter · none ticked = all[/dim]")
            yield Input(placeholder="filter by name", id="pick-filter")
            yield SelectionList(id="pick-models")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Compare", id="pick-apply", variant="success")
                yield Button("Clear (all models)", id="pick-clear")
                yield Button("Cancel", id="pick-cancel")

    def on_mount(self):
        self.fill("")
        self.query_one("#pick-models").focus()

    def fill(self, text):
        sl = self.query_one("#pick-models", SelectionList)
        sl.clear_options()
        for m in self.models:
            if text.lower() in m.lower():
                sl.add_option(Selection(f"{m}  [dim]{self.counts.get(m, 0)} questions[/dim]", m, m in self.chosen))
        if sl.option_count:
            sl.highlighted = 0  # otherwise the first space press does nothing

    @on(Input.Changed, "#pick-filter")
    def filter_changed(self, event):
        self.fill(event.value)

    @on(SelectionList.SelectionToggled, "#pick-models")
    def toggled(self, event):
        m = event.selection.value
        self.chosen.symmetric_difference_update({m})

    @on(Button.Pressed, "#pick-apply")
    @on(Input.Submitted, "#pick-filter")
    def apply(self):
        self.dismiss(set(self.chosen))

    @on(Button.Pressed, "#pick-clear")
    def clear(self):
        self.dismiss(set())

    @on(Button.Pressed, "#pick-cancel")
    def action_cancel(self):
        self.dismiss(None)


class SessionsScreen(ModalScreen):
    """The queue (next first), then this session's runs and tunes, newest first; picking a run reopens it."""
    BINDINGS = [Binding("escape", "cancel", "Close"), Binding("w", "cancel", "Close", show=False),
                Binding("delete", "remove", "Remove from queue"),
                Binding("shift+up", "move(-1)", "Earlier"), Binding("shift+down", "move(1)", "Later")]
    QUEUE_ACTIONS = {"remove", "move"}

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog", classes="wide"):
            yield Label(id="sessions-title")
            yield OptionList(id="sessions")

    def on_mount(self):
        self.fill()

    def check_action(self, action, parameters):
        return bool(self.app.queue) if action in self.QUEUE_ACTIONS else True

    def fill(self, keep=None):
        """(Re)build the list; keep = the queued item to leave highlighted."""
        app = self.app
        ol = self.query_one("#sessions", OptionList)
        ol.clear_options()
        hint = "enter opens · esc closes"
        if app.queue:
            hint += " · del removes · shift+↑/↓ moves"
            ol.add_option(Option("[b]Queued[/b] [dim](runs top to bottom, one at a time, after the "
                                 "current one)[/dim]", disabled=True))
            for i, item in enumerate(app.queue):
                ol.add_option(Option(f"  ⏳ #{i + 1}  {item.line()}", id=f"q{i}"))
            ol.add_option(Option("[b]This session[/b]", disabled=True))
        for i, sess in reversed(list(enumerate(app.sessions))):
            ol.add_option(Option(sess.session_line(), id=f"s{i}"))
        self.query_one("#sessions-title", Label).update(f"[b]Runs in this session[/b]  [dim]{hint}[/dim]")
        target = f"q{app.queue.index(keep)}" if keep in app.queue else None
        idx = next((k for k in range(ol.option_count) if not ol.get_option_at_index(k).disabled
                    and (target is None or ol.get_option_at_index(k).id == target)), None)
        if idx is not None:
            ol.highlighted = idx
        self.refresh_bindings()

    def highlighted_queue_index(self):
        ol = self.query_one("#sessions", OptionList)
        if ol.highlighted is None:
            return None
        oid = ol.get_option_at_index(ol.highlighted).id or ""
        return int(oid[1:]) if oid.startswith("q") else None

    def action_remove(self):
        i = self.highlighted_queue_index()
        if i is None:
            self.notify("Highlight a queued item (⏳) to remove it.")
            return
        item = self.app.queue.pop(i)
        self.notify(f"Removed from the queue: {item.title}")
        self.app.refresh_live()
        self.fill(self.app.queue[min(i, len(self.app.queue) - 1)] if self.app.queue else None)

    def action_move(self, delta):
        i = self.highlighted_queue_index()
        if i is None:
            return
        q = self.app.queue
        k = i + delta
        if 0 <= k < len(q):
            q[i], q[k] = q[k], q[i]
            self.fill(q[k])

    @on(OptionList.OptionSelected)
    def pick(self, event):
        oid = event.option.id
        if oid.startswith("s"):
            self.dismiss(self.app.sessions[int(oid[1:])])
        else:
            self.notify("Queued: it starts by itself when the ones before it finish.")

    def action_cancel(self):
        self.dismiss(None)


class RunScreen(Screen):
    # priority: these work whichever pane has focus (the text panes would otherwise swallow keys)
    BINDINGS = [
        Binding("k", "skip_model", "Skip model", priority=True),
        Binding("c", "cancel_all", "Cancel all", priority=True),
        Binding("f", "toggle_follow", "Pause scroll", priority=True),
        Binding("v", "watch_next", "Next model", priority=True),
        Binding("n", "new_run", "New run", priority=True),
        Binding("r", "results", "Results", priority=True),
        Binding("w", "runs", "Runs", priority=True),
        Binding("escape", "back", "Setup", priority=True),
        Binding("q", "back", "Back", show=False, priority=True),
    ]
    kind = "Run"
    LIVE_ONLY = {"skip_model", "cancel_all", "toggle_follow"}
    DONE_ONLY = {"new_run"}

    def __init__(self, jobs, parallel=None):
        super().__init__()
        self.jobs = jobs
        self.parallel = parallel      # models at a time (None: this machine's parallel_models)
        self.watch = None             # the model whose answers stream (several run at a time)
        self.running = True
        self.follow = True
        self.buf = {"reasoning": [], "answer": []}
        self.buf_lock = threading.Lock()
        self.started = time.time()
        self.req_started = None
        self.req_tokens = 0
        self.current = ""
        self.req_first = None
        self.stats = []
        self.mem = {}
        self.outcome = ""  # "Finished" or "Cancelled" once the queue is done
        self.ended = None
        self.recent_entries = {}  # Recent results row key -> answer (for AnswerScreen)
        self.waiting = None       # set while another tuieval window holds the machine
        self.failed_packs = {}    # label -> [pack labels] failed, waiting to be reported in one notice
        self.server_logs = {}     # label -> its server log, for failure notices
        self.total = sum(j.total for j in jobs if j.status in ("waiting", "done"))   # done: finished earlier

    def notify_failure(self, label, message, model=False):
        """One notice per failure: what went wrong, which packs it took down, and where to look.
        Packs a failed model never ran are folded into the model's notice."""
        packs = self.failed_packs.pop(label, [])
        if not packs and not model:
            return   # already reported with the model's failure
        message = message.removeprefix(f"{label}: ")
        reason, _, rest = message.partition(" (see ")
        reason = reason.rstrip(".")
        lines = [reason[:1].upper() + reason[1:] + "."]
        if packs:
            shown = " · ".join(packs[:4]) + (f" · and {len(packs) - 4} more" if len(packs) > 4 else "")
            lines.append(f"{'Skipped' if model else 'Failed'} ({len(packs)}): {shown}")
        log = self.server_logs.get(label)
        if log:
            root = self.app.engine.root + os.sep
            home = os.path.expanduser("~") + os.sep
            lines.append("Server log: " + (log[len(root):] if log.startswith(root) else
                                           "~/" + log[len(home):] if log.startswith(home) else log))
        title = f"{label} didn't start" if "while loading" in message or "not found" in message else \
            f"{label} failed" if model else f"{label}: pack failed"
        self.notify("\n".join(lines), title=title, severity="error", timeout=30)

    def check_action(self, action, parameters):
        if action == "watch_next":
            return self.running and self.side_by_side
        if action in self.LIVE_ONLY:
            return self.running
        if action in self.DONE_ONLY:
            return not self.running
        return True

    @property
    def side_by_side(self):
        """Several models run at a time in this run."""
        return min(self.parallel or self.app.engine.parallel_models(),
                   len({j.label for j in self.jobs if j.status != "skipped"})) > 1

    def active(self):
        """Models being loaded or run now, in queue order."""
        return list(dict.fromkeys(j.label for j in self.jobs if j.status in ("loading", "running")))

    def counts(self):
        return (sum(j.done for j in self.jobs), sum(j.passed for j in self.jobs), sum(j.failed for j in self.jobs))

    def live_text(self):
        if self.waiting:
            return f"{self.jobs[0].tier.title()} run waiting for another tuieval window"
        done = self.counts()[0]
        return f"Running {self.jobs[0].tier}: {done:,}/{self.total:,} · {fmt_secs(time.time() - self.started)}"

    def session_line(self):
        what = (f"{time.strftime('%H:%M', time.localtime(self.started))}  "
                + run_title(self.jobs[0].tier, {j.label for j in self.jobs}, {j.pack.label for j in self.jobs}))
        done, passed, failed = self.counts()
        if self.running:
            if self.waiting:
                return f"{what}  ● waiting for another tuieval window"
            return f"{what}  ● running {done:,}/{self.total:,}"
        return f"{what}  {self.outcome} {passed} ✓ {failed} ✗ in {fmt_secs(self.ended - self.started)}"

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="top"):
            yield Static(id="overall")
            yield ProgressBar(id="bar", show_eta=False)
            yield DataTable(id="queue", cursor_type="none", zebra_stripes=True)
        yield Static(id="current")
        with Horizontal(id="done-bar"):
            yield Button("New run  [n]", id="done-new", variant="success")
            yield Button("Results  [r]", id="done-results")
            yield Button("Runs  [w]", id="done-runs")
        with Horizontal(id="streams"):
            with Vertical(classes="stream"):
                yield Label("[b]Reasoning[/b]")
                yield StreamView(id="reasoning")
            with Vertical(classes="stream"):
                yield Label("[b]Answer[/b]")
                yield StreamView(id="answer")
        with TabbedContent(id="tabs"):
            with TabPane("Recent results", id="tab-recent"):
                yield DataTable(id="recent", cursor_type="row", zebra_stripes=True)
            with TabPane("Server log", id="tab-server"):
                yield Log(id="serverlog", max_lines=2000)
            with TabPane("Events", id="tab-events"):
                yield Log(id="events", max_lines=2000)
        yield Footer()

    def on_mount(self):
        q = self.query_one("#queue", DataTable)
        for label, key in (("Model", "model"), ("Eval", "suite"), ("Status", "status"), ("Progress", "progress"),
                           ("✓", "pass"), ("✗", "fail"), ("Time", "time"), ("Note", "note")):
            q.add_column(label, key=key)
        for j in self.jobs:
            ep = self.app.engine.endpoint(j.model)
            name = f"{j.label}  [magenta]{quant_text(ep)}[/magenta] [dim]via {ep['tag']}[/dim]" if ep else j.label
            q.add_row(name, j.pack.label, "", "", "", "", "", "", key=j.key)
            self.update_job(j)
        r = self.query_one("#recent", DataTable)
        for label, key in (("", "mark"), ("Level", "level"), ("Test", "test"), ("Model", "model"), ("Tokens", "tokens"),
                           ("TTFT", "ttft"), ("tok/s", "tps"), ("Secs", "secs"), ("Reason", "reason")):
            r.add_column(label, key=key)
        self.query_one("#bar", ProgressBar).update(total=max(self.total, 1), progress=0)
        self.set_interval(0.1, self.flush_streams)
        self.set_interval(1.0, self.tick)
        self.app.start_run(self.jobs, self.on_engine_event, self.parallel)
        self.tick()

    # -- events from the engine thread
    def on_engine_event(self, kind, **d):
        if self.app.closing:
            return  # the app is shutting down; the UI thread may be waiting on the engine
        if kind in ("request_started", "model_loading") and self.watch not in self.active():
            self.watch = d["job"].label if "job" in d else d["label"]   # follow the next model streaming
        if kind == "delta":  # very frequent: buffer, flushed every 100 ms on the UI thread
            if d.get("label", self.watch) != self.watch:
                return       # another model running alongside: v switches to it
            with self.buf_lock:
                self.buf[d["stream"]].append(d["text"])
            return
        self.app.call_from_thread(self.handle, kind, d)

    def flush_streams(self):
        with self.buf_lock:
            parts = {k: "".join(v) for k, v in self.buf.items()}
            for v in self.buf.values():
                v.clear()
        for stream, text in parts.items():
            if text:
                self.query_one(f"#{stream}", StreamView).append(text, self.follow)
                self.req_tokens += max(1, len(text) // 4)
                if self.req_first is None and self.req_started:
                    self.req_first = time.time()

    def event(self, text):
        self.query_one("#events", Log).write_line(time.strftime("%H:%M:%S ") + text)

    def watched(self, kind, d):
        """Whether an event belongs to the streaming model (always, with one model at a time)."""
        label = d["job"].label if "job" in d else d.get("label")
        return label is None or label == self.watch or not self.side_by_side

    def handle(self, kind, d):
        if kind == "request_started" and not self.watched(kind, d):
            pass         # another model running alongside: its rows still show in Recent results
        elif kind == "request_started":
            self.flush_streams()
            self.query_one("#reasoning", StreamView).clear()
            self.query_one("#answer", StreamView).clear()
            self.req_started, self.req_tokens, self.req_first = time.time(), 0, None
            job = d["job"]
            rep_note = f" · repeat {d['repeat'] + 1}/{job.repeat}" if job.repeat > 1 else ""
            self.current = (f"[b]{job.label}[/b] · {job.pack.label} · test {job.done + 1}/{job.total}{rep_note}"
                            f"{'  [magenta]🖼 image[/magenta]' if d['has_image'] else ''}\n"
                            f"{diff_badge(d.get('difficulty', 'unrated'))}  {d['test']}")
        elif kind == "request_done":
            if self.watched(kind, d):
                self.flush_streams()
            job, r = d["job"], d["record"]
            mark = "[green]✓[/green]" if r["pass"] else "[red]✗[/red]"
            recent = self.query_one("#recent", DataTable)
            # follow new rows unless you're browsing an older one
            at_end = not recent.has_focus or recent.cursor_row >= recent.row_count - 1
            calls = f"  [cyan]→ {r['tool_calls'][0]['name']}()[/cyan]" if r.get("tool_calls") else ""
            row_key = recent.add_row(mark, diff_badge(r.get("difficulty", "unrated")), r["description"][:50], job.label,
                           str(r.get("completion_tokens") or ""),
                           (f"{r['ttft_s']:.1f}" + (" ⟲" if r.get("cached_tokens") else ""))
                           if r.get("ttft_s") is not None else "",
                           f"{r['gen_tps']:.0f}" if r.get("gen_tps") else "", f"{r['total_s']:.0f}",
                           r["reason"][:90] + calls
                           + (f"  [red]copy of repeat {r['identical_to_repeat'] + 1}[/red]"
                              if r.get("identical_to_repeat") is not None else ""))
            self.stats.append(r)
            self.recent_entries[row_key] = {"model": job.label, "pack": job.pack.name, "record": r,
                                            "pack_fp": job.pack.fingerprint}
            if recent.row_count > 500:
                oldest = next(iter(recent.rows))
                recent.remove_row(oldest)
                self.recent_entries.pop(oldest, None)
            if self.follow and at_end:
                recent.move_cursor(row=recent.row_count - 1)
            if job:
                self.update_job(job)
            if self.watched(kind, d):
                self.req_started = None
            self.update_overall()
        elif kind in ("job_started", "job_update", "job_done"):
            job = d["job"]
            self.update_job(job)
            if kind == "job_started" and d.get("resumed"):
                earlier = (f" ({fmt_secs(d['earlier_s'])} over {d['sittings']} earlier sitting(s))"
                           if d.get("sittings") else "")
                self.event(f"{job.key}: resuming after {d['resumed']} finished requests{earlier}")
            if kind == "job_done":
                self.event(f"{job.key}: {job.status} {job.note}")
                for screen in self.app.screen_stack:  # a Results screen open during the run picks it up
                    if isinstance(screen, ResultsScreen):
                        screen.reload(f"{job.label} · {job.pack.label} finished; results refreshed.")
                if job.status == "failed":
                    # A model that fails to start fails all its packs at once, followed by model_failed:
                    # collect them so one notice covers the lot instead of one per pack.
                    first = job.label not in self.failed_packs
                    self.failed_packs.setdefault(job.label, []).append(job.pack.label)
                    if first:
                        self.set_timer(1.0, lambda label=job.label, note=job.note: self.notify_failure(label, note))
            self.update_overall()
        elif kind == "machine_busy":
            self.waiting = d["holder"]
            msg = engine.holder_text(d["holder"])
            self.current = f"[yellow]Waiting:[/yellow] {msg}.\n[dim]c cancels.[/dim]"
            self.event("waiting: " + msg)
            self.notify(f"Waiting: {msg}.", severity="warning", timeout=15)
        elif kind == "machine_free":
            self.waiting = None
            self.started = time.time()   # the run's clock starts when it really starts
            self.current = "The other window finished; starting…"
            self.event("the machine is free; starting")
        elif kind == "model_loading":
            self.event(f"{d['label']}: starting server")
            if d.get("log"):
                self.server_logs[d["label"]] = d["log"]
            self.query_one("#serverlog", Log).write_line(self.log_prefix(d) + f"$ {' '.join(d['command'])}")
            if self.watched(kind, d):
                self.current = f"Loading [b]{d['label']}[/b]…"
                self.query_one("#reasoning", StreamView).clear()
                self.query_one("#answer", StreamView).clear()
        elif kind == "model_waiting":
            self.event(f"[yellow]{d['message']}[/yellow]")
        elif kind == "server_log":
            self.query_one("#serverlog", Log).write_line(self.log_prefix(d) + d["line"])
        elif kind == "server_stall":
            self.event(f"[bold red]STALL[/bold red] {d['message']}")
            self.notify(d["message"], severity="error", timeout=30)
        elif kind == "server_facts":
            facts = " · ".join(f"{k} {v}" for k, v in d["facts"].items())
            self.event(f"{d['label']}: {facts}")
            if d["facts"].get("memory_limit") == "live-available":
                self.notify(f"{d['label']} sized itself by the memory free at startup ({facts}). "
                            "Other open apps are costing it speed.", severity="warning", timeout=12)
        elif kind == "model_ready":
            loaded = f" in {d['load_s']:.0f}s" if d.get("load_s") else ""
            self.event(f"{d['label']}: ready as {d['ids']}{loaded}")
            if self.watched(kind, d):
                self.current = f"[b]{d['label']}[/b] ready{loaded}, starting evals…"
        elif kind == "resource":
            self.mem[d["label"]] = d["rss_mb"]
        elif kind == "server_retry":
            self.event(f"[yellow]{d['message']}[/yellow]")
        elif kind == "model_failed":
            self.event(f"{d['label']}: FAILED: {d['message']}")
            if "by user" not in d["message"]:
                self.notify_failure(d["label"], d["message"], model=True)
        elif kind == "prompt_reused":
            label, pack, tokens = d["label"], d["pack"], d["tokens"]
            if d.get("hosted"):
                msg = (f"{label} · {pack}: the provider reused {tokens} cached prompt tokens from an earlier request. "
                       "Hosted APIs cache shared prompt openings and it can't be switched off; the answer is computed "
                       "the same way, but its TTFT is left out of the speed numbers. Marked ⟲ in Recent results.")
            else:
                msg = (f"{label} · {pack}: the server reused {tokens} prompt tokens from an earlier "
                       f"request, so answers aren't computed from scratch. Set request = {{ cache_prompt = false }} "
                       f"(or its equivalent) under [servers.{d['server']}] in models.toml.")
            self.event("[bold yellow]REUSE[/bold yellow] " + msg)
            self.notify(msg, severity="warning", timeout=20)
        elif kind == "identical_answer":
            msg = (f"{d['label']} · {d['pack']} · {d['test']}: repeat {d['repeat'] + 1} is an exact copy of repeat "
                   f"{d['same'] + 1}, so the server may be returning cached answers rather than fresh ones.")
            self.event("[bold red]COPY[/bold red] " + msg)
            self.notify(msg, severity="error", timeout=20)

        elif kind == "verdicts":
            for h in d["changes"]:
                msg = f"{h['model']} · {h['use_case']}: {h.get('previous') or 'new'} → {h['status']}"
                self.event("verdict " + msg)
                self.notify(msg, title="Verdict", timeout=15,
                            severity="error" if h["status"] == "FAIL" else "information")
        elif kind == "queue_done":
            self.running = False
            self.waiting = None
            self.req_started = None
            self.ended = time.time()
            self.outcome = "Cancelled" if d["cancelled"] else "Finished"
            done = sum(j.status == "done" for j in self.jobs)
            failed = sum(j.status == "failed" for j in self.jobs)
            self.current = (f"[b]{self.outcome}[/b] in {fmt_secs(self.ended - self.started)}: "
                            f"{done} done, {failed} failed.")
            self.event("queue finished")
            self.query_one("#done-bar").display = True
            self.refresh_bindings()
            self.app.refresh_live()
            self.notify(f"Run {self.outcome.lower()}. " +
                        ("Press r for results, n for a new run." if self.is_current else "Press w to open it."),
                        timeout=10)
            self.app.session_finished(self)
        self.refresh_current()

    def log_prefix(self, d):
        return f"[{d['label']}] " if self.side_by_side and d.get("label") else ""

    def update_job(self, j):
        q = self.query_one("#queue", DataTable)
        style = STATUS_STYLE.get(j.status, "")
        elapsed = (j.finished or time.time()) - j.started if j.started else 0
        elapsed += sum(s.get("wall_s") or 0 for s in j.earlier)   # a resumed pack: its earlier sittings too
        # update_width: columns start at their header's width; "12" under ✓ would otherwise show as "1"
        for col, value in (("status", f"[{style}]{j.status}[/{style}]"),
                           ("progress", f"{j.done}/{j.total}" if j.status != "skipped" or j.done else ""),
                           ("pass", f"[green]{j.passed}[/green]" if j.passed else ""),
                           ("fail", f"[red]{j.failed}[/red]" if j.failed else ""),
                           ("time", fmt_secs(elapsed) if elapsed else ""),
                           ("note", short_note(j.note))):
            q.update_cell(j.key, col, value, update_width=True)

    def update_overall(self):
        done = sum(j.done for j in self.jobs)
        passed = sum(j.passed for j in self.jobs)
        failed = sum(j.failed for j in self.jobs)
        self.query_one("#bar", ProgressBar).update(progress=done)
        elapsed = time.time() - self.started
        eta = ""
        done_now = sum(j.done - j.resumed for j in self.jobs if j.started)   # answers from earlier sittings
        if self.running and done_now > 0 and self.total > done:              # took no time in this one
            eta = f" · ETA {fmt_secs(elapsed / done_now * (self.total - done))}"
        pct = f"{100 * passed / (passed + failed):.0f}%" if passed + failed else "-"
        recent = self.stats[-50:]
        tps = [r["gen_tps"] for r in recent if r.get("gen_tps")]
        ttfts = [r["ttft_s"] for r in recent if r.get("ttft_s") is not None and not r.get("cached_tokens")]
        speed = ""
        if tps:
            speed += f" · {sorted(tps)[len(tps) // 2]:.0f} tok/s"
        if ttfts:
            speed += f" · TTFT {sorted(ttfts)[len(ttfts) // 2]:.1f}s"
        active = self.active()
        mem = (sum(self.mem.get(l, 0) for l in active) if self.side_by_side and active   # models side by side add up
               else max(self.mem.values()) if self.mem else 0)
        if mem:
            speed += f" · mem {mem / 1024:.1f} GB" if mem >= 1024 else f" · mem {mem:.0f} MB"
        self.query_one("#overall", Static).update(
            f"[b]{done:,} / {self.total:,}[/b] requests · live score [green]{passed} ✓[/green] "
            f"[red]{failed} ✗[/red] ({pct}){speed} · elapsed {fmt_secs(elapsed)}{eta}"
            f"{'' if self.follow else '  [yellow](scroll paused, f to resume)[/yellow]'}")

    def refresh_current(self):
        extra = ""
        if self.req_started:
            now = time.time()
            ttft = f"TTFT {self.req_first - self.req_started:.1f}s · " if self.req_first else "waiting for first token · "
            rate = ""
            if self.req_first and now - self.req_first > 1:
                rate = f" · ~{self.req_tokens / (now - self.req_first):.0f} tok/s"
            extra = f"   [dim]{fmt_secs(now - self.req_started)} · {ttft}~{self.req_tokens} tokens{rate}[/dim]"
        others = [l for l in self.active() if l != self.watch] if self.running and self.side_by_side else []
        if others:
            def where(label):
                j = next((j for j in self.jobs if j.label == label and j.status in ("loading", "running")), None)
                return f"{label} ({j.pack.label} {j.done}/{j.total})" if j and j.status == "running" else f"{label} (loading)"
            extra += f"\n[dim]Also running: {' · '.join(map(where, others))} · v streams the next one[/dim]"
        self.query_one("#current", Static).update(self.current + extra)

    def tick(self):
        for j in self.jobs:
            if j.status in ("running", "loading"):
                self.update_job(j)
        self.update_overall()
        self.refresh_current()

    # -- actions
    def action_skip_model(self):
        if self.running:
            if self.side_by_side and self.watch:
                self.app.engine.skip_model(self.watch)
                self.notify(f"Skipping {self.watch}; the others keep going…")
            else:
                self.app.engine.skip_model()
                self.notify("Skipping the current model…")

    def action_watch_next(self):
        """Stream the next model running alongside (from its next answer)."""
        active = self.active()
        if len(active) < 2:
            self.notify("Only one model is running right now.")
            return
        self.watch = active[(active.index(self.watch) + 1) % len(active)] if self.watch in active else active[0]
        with self.buf_lock:
            for v in self.buf.values():
                v.clear()
        self.query_one("#reasoning", StreamView).clear()
        self.query_one("#answer", StreamView).clear()
        self.req_started = None
        self.current = f"Streaming [b]{self.watch}[/b] from its next answer…"
        self.refresh_current()

    def action_cancel_all(self):
        if not self.running:
            return

        def stop():
            self.app.engine.cancel()
            self.notify("Cancelling: stopping the current request and the model server…")
        self.app.ask_cancel("Cancel the whole run? Finished suites keep their results.", stop, confirm=True)

    def action_toggle_follow(self):
        self.follow = not self.follow
        self.update_overall()

    @on(DataTable.RowSelected, "#recent")
    def recent_selected(self, event):
        keys = [k for k in event.data_table.rows if k in self.recent_entries]
        if event.row_key in self.recent_entries:
            self.app.push_screen(AnswerScreen([self.recent_entries[k] for k in keys], keys.index(event.row_key)))

    @on(Button.Pressed, "#done-results")
    def action_results(self):
        # Works during the run too: finished packs are saved as they end; the one running shows as unfinished.
        # Always opens all your results; a smoke run's own results (kept apart) are one button away.
        main = os.path.abspath(self.app.engine.results_dir)
        smoke = {os.path.abspath(os.path.dirname(os.path.dirname(j.out_path))) for j in self.jobs} - {main}
        self.app.push_screen(ResultsScreen(smoke_dir=smoke.pop() if len(smoke) == 1 else None))

    @on(Button.Pressed, "#done-new")
    def action_new_run(self):
        self.app.back_to_setup()  # Setup still holds the last selection
        self.app.screen.query_one("#models").focus()

    @on(Button.Pressed, "#done-runs")
    def action_runs(self):
        self.app.open_sessions()

    def action_back(self):
        if self.running:
            self.notify("The run keeps going; the header shows its progress. Press w to come back.")
        self.app.back_to_setup()


# ------------------------------------------------------------------ results
class ResultsScreen(Screen):
    BINDINGS = [Binding("escape", "app.pop_screen", "Back"), Binding("q", "app.pop_screen", "Back", show=False),
                Binding("w", "runs", "Runs"), Binding("ctrl+r", "refresh", "Refresh")]
    SORTS = [("Sort: token gap", "tokens"), ("Sort: time gap", "secs"),
             ("Sort: most tokens", "most"), ("Sort: pack order", "order")]

    def check_action(self, action, parameters):
        return bool(self.app.sessions) if action == "runs" else True

    def action_runs(self):
        self.app.open_sessions()

    READINESS_HELP = ("[dim]PASS needs a Certify run with zero critical failures over enough trials and an "
                      "accuracy lower bound that clears the pack's gate (pack.toml). Screening can only FAIL "
                      "or look promising.[/dim]")

    def __init__(self, results_dir=None, smoke_dir=None):
        """results_dir: which results to show (default: all of them, smoke runs excluded).
        smoke_dir: a smoke run's own results, offered with a button to switch to and back."""
        super().__init__()
        self.results_dir = results_dir
        self.smoke_dir = smoke_dir

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="results-top"):
            yield Static(id="results-info")
            if self.smoke_dir:
                yield Button("Show this Smoke run's results", id="toggle-smoke")
        with TabbedContent():
            with TabPane("Production readiness"):
                yield Static(id="readiness-short")
                yield Static(self.READINESS_HELP, id="readiness-help")
                yield DataTable(id="readiness", zebra_stripes=True)
                yield Static("[b]Fast enough?[/b] [dim]p90 seconds per answer against the pack's limit, per machine: "
                             "measured there, or projected from token counts and that machine's tuned speeds.[/dim]")
                yield DataTable(id="latency", zebra_stripes=True)
                yield DataTable(id="readiness-detail", zebra_stripes=True, cursor_type="row")
            with TabPane("Verdict history"):
                yield Static("[dim]Every change in a model's verdict, newest first. Kept even after results are rerun "
                             "or questions change; the evidence behind older verdicts is in results/<model>/history/.[/dim]")
                yield DataTable(id="history", zebra_stripes=True, cursor_type="row")
            with TabPane("Scorecard"):
                yield DataTable(id="scorecard", zebra_stripes=True)
            with TabPane("Speed & tokens"):
                yield Select([("As measured (all machines)", "")] +
                             [(f"{mc.id}{' (this machine)' if i == 0 else ''}", mc.id)
                              for i, mc in enumerate(self.app.engine.machines())],
                             id="speed-machine", allow_blank=False, value="")
                yield Static("[dim]★ = no other model is both more accurate and faster per answer. "
                             "Memory is the server's resident size (includes GPU-mapped weights on Apple Silicon). "
                             "On another machine, times are measured there if the model ran there, otherwise "
                             "projected from each answer's token counts and that machine's tuned speeds.[/dim]")
                yield DataTable(id="speed", zebra_stripes=True)
            with TabPane("Per question", id="tab-per-question"):
                with Horizontal(id="pq-controls"):
                    yield Select([("All packs", "")], id="pq-pack", allow_blank=False, value="")
                    yield Select([("All levels", "")] + [(l.upper(), l) for l in compare.LEVELS + ("unrated",)],
                                 id="pq-level", allow_blank=False, value="")
                    yield Select(self.SORTS, id="pq-sort", allow_blank=False, value="tokens")
                    yield Button("Models: all", id="pq-models")
                    yield Checkbox("Only questions all answered", False, id="pq-shared")
                    yield Checkbox("Include unfinished", True, id="pq-partial")
                yield Static(id="pq-compare-title")
                yield DataTable(id="pq-summary", zebra_stripes=True, cursor_type="none")
                yield Static(id="pq-legend")
                yield DataTable(id="per-question", zebra_stripes=True, cursor_type="cell")
            with TabPane("Is the difference real?"):
                yield DataTable(id="pairwise", zebra_stripes=True)
            with TabPane("Tests that separate models"):
                yield DataTable(id="separating", zebra_stripes=True)
            with TabPane("By difficulty"):
                yield Static("[dim]Pass rate per difficulty level. The level is your label only; models never see it. "
                             "A strong model should hold up on HARD, not just EASY.[/dim]")
                yield DataTable(id="difficulty", zebra_stripes=True)
            with TabPane("Failures"):
                yield Static("[dim]Enter on a row shows the whole answer, its reasoning and what was checked.[/dim]")
                yield DataTable(id="failures", zebra_stripes=True, cursor_type="row")
            with TabPane("Test quality"):
                yield Static("[dim]no signal: every model passed · suspicious: every model failed (check the test) · "
                             "inverted: the weakest model beat the strongest · flaky: same model, different results "
                             "across repeats[/dim]")
                yield DataTable(id="quality", zebra_stripes=True)
        yield Footer()

    def on_mount(self):
        self.rows, self.infos, self.partial_rows, self.superseded = [], [], [], set()
        if not hasattr(self.app, "pq_models"):
            self.app.pq_models = set()
        self.fill()

    def on_screen_resume(self):
        message = getattr(self, "pending_reload", None)
        if message:
            self.pending_reload = None
            self.call_after_refresh(self.reload, message)

    def action_refresh(self):
        self.reload("Results refreshed.")

    @on(Button.Pressed, "#toggle-smoke")
    def toggle_smoke(self, event):
        """Switch between all your results and the smoke run's own (kept apart in results/smoke/)."""
        showing_smoke = self.results_dir == self.smoke_dir
        self.results_dir = None if showing_smoke else self.smoke_dir
        event.button.label = "Show this Smoke run's results" if showing_smoke else "Show all results"
        self.reload("Showing all your results." if showing_smoke else
                    "Showing this Smoke run's results (they never count toward verdicts).")

    def reload(self, message=None):
        """Re-read the result files (tables keep their tab; cursors go back to the top). Under
        another screen (an open answer, the run) it waits until Results is shown again: tables
        rebuilt while hidden keep header-only column widths."""
        if self.app.screen is not self:
            self.pending_reload = message or "Results refreshed."
            return
        for t in self.query(DataTable):
            t.clear(columns=True)
        self.fill()
        if message:
            self.notify(message, timeout=4)

    def run_banner(self, results_dir):
        """What a run going on now hasn't saved yet (the pack it's on and those still waiting)."""
        s = self.app.active_session
        if not isinstance(s, RunScreen):
            return ""
        same = [j for j in s.jobs if os.path.abspath(os.path.dirname(os.path.dirname(j.out_path)))
                == os.path.abspath(results_dir)]
        running = [f"{j.label} · {j.pack.label} {j.done}/{j.total}" for j in same if j.status in ("running", "loading")]
        waiting = sum(j.status == "waiting" for j in same)
        if not running and not waiting:
            return ""
        return ("\n[cyan]● Run in progress: not in these results yet: "
                + ", ".join(running + ([f"{waiting} waiting"] if waiting else []))
                + ". Answers so far are in Per question (◐). Refreshes when a pack finishes; ctrl+r any time.[/cyan]")

    def fill(self):
        results_dir = self.results_dir or self.app.engine.results_dir
        self.fill_history()
        paths = compare.default_paths(results_dir)
        self.partial_rows = compare.load_partials(results_dir)
        self.superseded = {p for p in compare.default_paths(results_dir) if self.app.engine.superseded(p)}
        info = self.query_one("#results-info", Static)
        banner = self.run_banner(results_dir)
        if not paths:
            info.update("No results yet. Run an eval first." + banner)
            self.rows, self.infos = [], []
            self.fill_per_question()
            return
        rows, infos = compare.load(paths)
        notes = compare.settings_notes(infos)
        smoke = os.path.abspath(results_dir) != os.path.abspath(self.app.engine.results_dir)
        info.update(f"{len(paths)} {'Smoke ' if smoke else ''}result files in {os.path.relpath(results_dir)}/" + banner
                    + "".join(f"\n[yellow]⚠ {n}[/yellow]" for n in notes))
        header, table = compare.scorecard(rows)
        t = self.query_one("#scorecard", DataTable)
        t.add_columns(*header)
        t.add_rows(table)
        self.rows, self.infos = rows, infos
        self.fill_speed(self.query_one("#speed-machine", Select).value or "")
        p = self.query_one("#pairwise", DataTable)
        p.add_columns("Model A", "Model B", "A − B", "95% CI", "Verdict", "Shared tests")
        for a, b, diff, lo, hi, call, n in compare.pairwise(rows):
            color = "green" if call == "clear" else "dim"
            p.add_row(a, b, f"{100 * diff:+.1f} pts", f"[{100 * lo:+.1f}, {100 * hi:+.1f}]",
                      f"[{color}]{call}[/{color}]", str(n))
        split, _ = compare.separating(rows)
        sep = self.query_one("#separating", DataTable)
        models = sorted({r["model"] for r in rows})
        sep.add_columns("Test", *models)
        for test, d in split:
            sep.add_row(test, *[f"{sum(d[m])}/{len(d[m])}" if m in d else "-" for m in models])
        f = self.query_one("#failures", DataTable)
        f.add_columns("Model", "Level", "Test", "Reason")
        # compare.failures lists the failed rows in order, so they line up with fail_rows
        self.fail_rows = [r for r in rows if not r["ok"]]
        for model, test, reason, level in compare.failures(rows):
            f.add_row(model, diff_badge(level), test, rich_escape(reason[:130]))
        dh, dt = compare.by_difficulty(rows)
        dtab = self.query_one("#difficulty", DataTable)
        dtab.add_columns("Model", "Pack", *[diff_badge(l) for l in compare.LEVELS])
        for line in dt:
            dtab.add_row(*line)
        qt = self.query_one("#quality", DataTable)
        qt.add_columns("Flags", "Test", *models)
        if len(models) >= 2:
            for test, flags, rates in compare.items(rows):
                qt.add_row(", ".join(flags), test, *[f"{100 * rates[m]:.0f}%" if m in rates else "-" for m in models])
        self.fill_per_question()
        self.fill_readiness(results_dir)

    @on(DataTable.RowSelected, "#failures")
    def failure_selected(self, event):
        entries = [answer_entry(r) for r in self.fail_rows]
        i = event.cursor_row
        if not entries[i]:
            self.notify("This result is in the old format and has no stored answer.", severity="warning")
            return
        keep = [x for x in entries if x]
        self.app.push_screen(AnswerScreen(keep, keep.index(entries[i]), what="failure"))

    # -- per question: tokens and time for every model on the same question
    @on(Select.Changed, "#pq-pack")
    @on(Select.Changed, "#pq-level")
    @on(Select.Changed, "#pq-sort")
    @on(Checkbox.Changed, "#pq-partial")
    @on(Checkbox.Changed, "#pq-shared")
    def per_question_changed(self):
        if getattr(self, "rows", None) is not None:
            self.fill_per_question()

    @on(Button.Pressed, "#pq-models")
    def pick_models(self):
        counts = {}
        for t in getattr(self, "pq_all_tests", []):
            for m in t["cells"]:
                counts[m] = counts.get(m, 0) + 1
        models = sorted(counts, key=lambda m: (-counts[m], m))

        def picked(chosen):
            if chosen is None:
                return
            first = chosen and not self.app.pq_models
            self.app.pq_models = chosen  # kept for this session, so reopening Results keeps it
            if first:  # a fair comparison needs the same questions: turn it on (you can untick it)
                with self.query_one("#pq-shared", Checkbox).prevent(Checkbox.Changed):
                    self.query_one("#pq-shared", Checkbox).value = True
            # after the picker has closed: a table rebuilt under a modal keeps its header-only widths
            self.call_after_refresh(self.fill_per_question)
        self.app.push_screen(ModelPickScreen(models, counts, self.app.pq_models), picked)

    def fill_per_question(self):
        # a pack being rerun under new questions or settings: its old result file is on its way out,
        # so it isn't compared (it moves to history/ when the new run of that pack finishes)
        rows = [r for r in self.rows if r.get("path") not in self.superseded]
        if self.query_one("#pq-partial", Checkbox).value and self.partial_rows:
            # a pack being rerun: its new answers replace the saved ones for the same question and repeat
            fresh = {(r["model"], r["test"], r["repeat"]) for r in self.partial_rows}
            rows = [r for r in rows if (r["model"], r["test"], r["repeat"]) not in fresh] + self.partial_rows
        packs = sorted({r["suite"] for r in rows})
        pack_sel = self.query_one("#pq-pack", Select)
        options = [("All packs", "")] + [(self.app.engine.packs[p].label if p in self.app.engine.packs else p, p)
                                         for p in packs]
        if [v for _, v in options] != getattr(self, "pq_pack_values", None):
            self.pq_pack_values = [v for _, v in options]
            current = pack_sel.value
            with pack_sel.prevent(Select.Changed):
                pack_sel.set_options(options)
                pack_sel.value = current if current in self.pq_pack_values else ""
        pack, level = pack_sel.value or "", self.query_one("#pq-level", Select).value or ""
        sort = self.query_one("#pq-sort", Select).value or "tokens"
        all_tests = compare.per_question(rows)
        self.pq_all_tests = all_tests
        chosen = {m for m in self.app.pq_models if any(m in t["cells"] for t in all_tests)}
        self.query_one("#pq-models", Button).label = (
            f"Models: {len(chosen)} picked" if chosen else "Models: all")
        tests = []
        for t in all_tests:
            if (pack and t["suite"] != pack) or (level and t["difficulty"] != level):
                continue
            if chosen:
                t = compare.with_models(t, chosen)
                if not t["cells"]:
                    continue
            tests.append(t)
        shown = sorted(chosen) if chosen else sorted({m for t in tests for m in t["cells"]})
        if self.query_one("#pq-shared", Checkbox).value:
            tests = [t for t in tests if all(m in t["cells"] for m in shown)]
        self.fill_pq_summary(tests, shown)
        order = {}
        for name, pk in self.app.engine.packs.items():
            for i, t in enumerate(pk.tests):
                order[f"{name}: {t['id']}"] = (name, i)
        if sort in ("tokens", "secs"):
            tests.sort(key=lambda t: -t["spread"][sort])
        elif sort == "most":
            tests.sort(key=lambda t: -max((c["tokens"] or 0) for c in t["cells"].values()))
        else:
            tests.sort(key=lambda t: order.get(t["test"], (t["suite"], 1e9)))
        models = shown
        here = self.app.engine.machine().id
        table = self.query_one("#per-question", DataTable)
        table.clear(columns=True)
        table.add_columns("Test", "Pack", "Level", "×tokens", *models)
        self.pq_tests, self.pq_models = {}, models
        for t in tests:
            cells = []
            best = min((c["tokens"] for c in t["cells"].values() if c["tokens"] and c["passed"]), default=None)
            for m in models:
                c = t["cells"].get(m)
                cells.append(self.pq_cell(c, best, here) if c else "[dim]-[/dim]")
            spread = t["spread"]["tokens"]
            sp = f"[yellow]{spread:.1f}×[/yellow]" if spread >= 3 else (f"{spread:.1f}×" if len(t["cells"]) > 1 else "")
            desc = t["cells"][next(iter(t["cells"]))]["rows"][0].get("raw", {}).get("description") or t["name"]
            table.add_row(rich_escape(desc[:48]), t["suite"], diff_badge(t["difficulty"]), sp, *cells, key=t["test"])
            self.pq_tests[t["test"]] = t
        self.query_one("#pq-legend", Static).update(
            f"[dim]{len(tests)} questions. Each cell: result (passes/answers if repeated), median tokens, median seconds. "
            "[green]bold[/green] = fewest tokens among passing answers · [red]cut[/red] = hit max_tokens · "
            "◐ = pack unfinished (not in scores) · dim seconds = hosted, or measured on another machine; tokens compare "
            "across machines, seconds don't. ×tokens = most / fewest tokens across models. Enter on a cell opens the "
            "answers.[/dim]" + ("\n[yellow]Left out, being rerun with other questions or settings: " + ", ".join(
                f"{os.path.basename(os.path.dirname(p))} · {os.path.basename(p)[:-5]}" for p in sorted(self.superseded))
                + ". Only the new run's answers (◐) are shown for these.[/yellow]" if self.superseded else ""))

    def fill_pq_summary(self, tests, models):
        """One row per shown model over the questions they ALL answered, so totals compare fairly;
        with two models, a head-to-head line."""
        table = self.query_one("#pq-summary", DataTable)
        table.clear(columns=True)
        title = self.query_one("#pq-compare-title", Static)
        shared = [t for t in tests if all(m in t["cells"] for m in models)]
        if len(models) < 2 or len(models) > 8:
            table.display = False
            title.update("[dim]Pick 2 to 8 models (Models button) for a side-by-side summary.[/dim]"
                         if len(models) > 8 else "")
            return
        table.display = True
        if not shared:
            title.update("[yellow]These models have no question in common (with the filters above).[/yellow]")
            table.display = False
            return
        rows = compare.compare_models(shared, models)
        table.add_columns("Model", "Total time", "Passed", "Median tokens", "Total tokens", "Tokens per ✓", "Median s",
                          "Avg prompt tok/s", "Avg gen tok/s", "Fewest tokens", "Fastest", "Cut off")
        best = {k: max((r[k] for r in rows if r[k] is not None), default=None)
                for k in ("passed", "fewest", "fastest", "avg_pp_tps", "avg_gen_tps")}
        low = {k: min(r[k] for r in rows if r[k] is not None) if any(r[k] is not None for r in rows) else None
               for k in ("med_tokens", "total_tokens", "tokens_per_pass", "med_secs", "total_secs")}

        def hi(v, text, key, good):
            ref = best.get(key) if good == "max" else low.get(key)
            return f"[bold green]{text}[/bold green]" if v is not None and v == ref and len(rows) > 1 else text
        for r in rows:
            tt = r["total_secs"]
            table.add_row(r["model"],
                          hi(tt, ("-" if not tt else f"{tt:.1f}s" if tt < 60 else fmt_secs(tt)), "total_secs", "min"),
                          hi(r["passed"], f"{r['passed']}/{r['n']}", "passed", "max"),
                          hi(r["med_tokens"], f"{r['med_tokens']:,.0f}" if r["med_tokens"] else "-", "med_tokens", "min"),
                          hi(r["total_tokens"], f"{r['total_tokens']:,.0f}", "total_tokens", "min"),
                          hi(r["tokens_per_pass"], f"{r['tokens_per_pass']:,.0f}" if r["tokens_per_pass"] else "-",
                             "tokens_per_pass", "min"),
                          hi(r["med_secs"], f"{r['med_secs']:.1f}" if r["med_secs"] else "-", "med_secs", "min"),
                          hi(r["avg_pp_tps"], f"{r['avg_pp_tps']:,.0f}" if r["avg_pp_tps"] else "-", "avg_pp_tps", "max"),
                          hi(r["avg_gen_tps"], f"{r['avg_gen_tps']:.1f}" if r["avg_gen_tps"] else "-", "avg_gen_tps",
                             "max"),
                          hi(r["fewest"], str(r["fewest"]), "fewest", "max"),
                          hi(r["fastest"], str(r["fastest"]), "fastest", "max"),
                          f"[red]{r['cut']}[/red]" if r["cut"] else "0")
        line = f"[b]Comparison[/b] over the {len(shared)} questions all {len(models)} answered"
        if len(models) == 2:
            h = compare.head_to_head(shared, *models)
            a, b = models
            line += (f"\n{a} passed [b]{h['only_a']}[/b] that {b} failed; {b} passed [b]{h['only_b']}[/b] that {a} "
                     f"failed; both passed {h['both']}, both failed {h['neither']}. Fewer tokens: {a} on "
                     f"{h['fewer_a']}, {b} on {h['fewer_b']}" + (f" (median {a}/{b} ratio {h['ratio']:.2f}×)"
                                                               if h["ratio"] else "") + ".")
        shared_speed = sorted({c_m for t in shared for c_m, c in t["cells"].items() if c_m in models
                               and any(r.get("alongside") for r in c["rows"])})
        if shared_speed:
            line += (f"\n[yellow]Some answers of {', '.join(shared_speed)} ran alongside other models, which "
                     "shared the machine's speed: compare their times with care.[/yellow]")
        title.update(line + "  [dim]Fewest tokens / Fastest = questions won; ties count for both. Total tokens / "
                     "Total time add up each question's median. Avg tok/s = all tokens over all the time they took "
                     "(cached prompts left out).[/dim]")

    @staticmethod
    def pq_cell(c, best, here):
        mark = ("[green]✓[/green]" if c["passed"] == c["n"] else "[red]✗[/red]" if not c["passed"]
                else f"[yellow]{c['passed']}/{c['n']}[/yellow]")
        if c["n"] > 1 and c["passed"] in (0, c["n"]):
            mark += f"[dim]{c['n']}[/dim]"
        tok = c["tokens"]
        tok_s = "-" if not tok else (f"{tok / 1000:.1f}k" if tok >= 1000 else f"{tok:.0f}")
        if tok and best and tok == best and c["passed"]:
            tok_s = f"[bold green]{tok_s}[/bold green]"
        secs = "-" if not c["secs"] else f"{c['secs']:.0f}s" if c["secs"] >= 10 else f"{c['secs']:.1f}s"
        remote = any(r.get("machine") not in (None, here) or (r.get("raw") or {}).get("quantization")
                     for r in c["rows"])
        if remote:
            secs = f"[dim]{secs}[/dim]"
        return (f"{mark} {tok_s} {secs}" + (" [red]cut[/red]" if c["truncated"] else "")
                + (" [cyan]◐[/cyan]" if c["partial"] else ""))

    @on(DataTable.CellSelected, "#per-question")
    def per_question_selected(self, event):
        col = event.coordinate.column - 4
        t = self.pq_tests.get(event.cell_key.row_key.value)
        if not t or col < 0:
            self.notify("Pick a model's cell to open its answers.")
            return
        c = t["cells"].get(self.pq_models[col])
        entries = [x for x in (answer_entry(r) for r in (c or {}).get("rows", [])) if x]
        if not entries:
            self.notify("No stored answer for this model on this question.", severity="warning")
            return
        self.app.push_screen(AnswerScreen(entries, what="repeat"))

    @on(Select.Changed, "#speed-machine")
    def machine_changed(self, event):
        if getattr(self, "rows", None) is not None:
            self.fill_speed(event.value or "")

    def fill_speed(self, machine_id):
        e = self.app.engine
        speeds = {}
        if machine_id:
            mc = next(x for x in e.machines() if x.id == machine_id)
            speeds = {m["label"]: (e.serving(m, mc).profile or {}).get("measured") for m in e.cfg["models"]}
        header, table, _ = compare.speed(self.rows, self.infos, machine_id or None, speeds)
        sp = self.query_one("#speed", DataTable)
        sp.clear(columns=True)
        sp.add_columns(*header)
        for line in table:
            sp.add_row(*([f"[yellow]{line[0]}[/yellow]"] + line[1:]))

    def fill_history(self):
        e = self.app.engine
        t = self.query_one("#history", DataTable)
        t.add_columns("When", "Model", "Use case", "Change", "Questions", "Why")
        rank = {"FAIL": 0, "INCONCLUSIVE": 1, "NO DATA": 1, "PASS": 2}
        for h in reversed(verdict.load_history(e)):
            prev, now = h.get("previous"), h["status"]
            change = f"[{VERDICT_STYLE[now]}]{now}[/]" if not prev else \
                f"{prev} → [{VERDICT_STYLE[now]}]{now}[/]"
            if prev and rank.get(now, 1) < rank.get(prev, 1):
                change += "  [bold red]⚠ worse[/bold red]"
            qs = "current" if verdict.is_current(h, e) else "[yellow]changed since[/yellow]"
            t.add_row(h["time"].replace("T", " "), h["model"], h["use_case"], change, qs, "; ".join(h["reasons"])[:120])

    def fill_readiness(self, results_dir):
        style = {"PASS": "bold green", "FAIL": "bold red", "INCONCLUSIVE": "yellow", "NO DATA": "dim"}
        matrix = self.query_one("#readiness", DataTable)
        detail = self.query_one("#readiness-detail", DataTable)
        help_text = self.query_one("#readiness-help", Static)
        if os.path.abspath(results_dir) != os.path.abspath(self.app.engine.results_dir):
            help_text.update("[yellow]These are smoke-test results; they don't count toward production readiness."
                             "[/yellow]")
            return
        help_text.update(self.READINESS_HELP)
        table = verdict.readiness(self.app.engine)
        short = ["[b]In short[/b]"]
        for label, sentence, todo in verdict.plain_summary(table):
            short.append(f"  [b]{rich_escape(label)}[/b]: {rich_escape(sentence)}")
            short += [f"     → {rich_escape(g)}: {rich_escape(what)}" for g, what, _ in todo]
        if any("Certify" in what for _, _, todo in verdict.plain_summary(table) for _, what, _ in todo):
            short.append("  [dim]To run Certify: in Setup tick the model and pack, choose Certify, press s.[/dim]")
        self.query_one("#readiness-short", Static).update("\n".join(short) if table else
                                                          "[dim]No verdicts yet: run something from Setup.[/dim]")
        groups = sorted({g for t in table.values() for g in t})
        matrix.add_columns("Model", *groups)
        detail.add_columns("Model", "Use case", "Pack", "Verdict", "Why")
        lat = verdict.latency_table(self.app.engine, None, table)
        machine_ids = [mc.id for mc in self.app.engine.machines()]
        lt = self.query_one("#latency", DataTable)
        lt.add_columns("Model", "Use case", *[f"{mid}{' (here)' if i == 0 else ''}" for i, mid in enumerate(machine_ids)])
        for label, gs in table.items():
            cells = []
            for g in groups:
                v = gs[g][0] if g in gs else None
                cells.append(f"[{style[v.status]}]{v.status}[/]" if v else "[dim]-[/dim]")
            matrix.add_row(label, *cells)
            for g in gs:
                row = lat.get(label, {}).get(g, {})
                lt.add_row(label, g, *[latency_cell(row.get(mid)) for mid in machine_ids])
            for g, (_, packs) in gs.items():
                for name, pv in packs.items():
                    if pv.status != "NO DATA":
                        detail.add_row(label, g, name, f"[{style[pv.status]}]{pv.status}[/]", "; ".join(pv.reasons)[:160])


# ------------------------------------------------------------------ app
class EvalsApp(App):
    TITLE = "tuieval"
    CSS = """
    #pickers { height: 1fr; }
    .picker { width: 1fr; padding: 0 1; }
    .picker SelectionList { height: 1fr; }
    #models-legend { width: 100%; }
    #filter-row { height: 3; }
    #filter { width: 1fr; }
    #show-hidden { width: auto; }
    #model-detail { height: auto; min-height: 3; max-height: 8; padding: 0 1; background: $boost; }
    #tier-row { height: 3; padding: 0 1; }
    #tier-row Label { padding: 1 1 0 0; }
    #tier-row RadioSet { layout: horizontal; width: auto; height: auto; }
    #tier-row RadioButton { width: auto; margin-right: 2; }
    #options { height: 3; padding: 0 1; align-vertical: middle; }
    #options Label { padding: 1 1 0 0; }
    #options Input { width: 12; }
    #options #parallel { width: 8; }
    #estimate { padding: 0 2; height: auto; min-height: 2; }
    #buttons { height: 3; padding: 0 1; }
    #buttons Button { margin-right: 2; }
    #top { height: auto; max-height: 45%; }
    #overall { padding: 0 1; }
    #bar { padding: 0 1; width: 100%; }
    #bar Bar { width: 1fr; }
    #queue { height: auto; max-height: 12; }
    #current { height: 3; padding: 0 1; background: $boost; }
    #done-bar { height: 3; padding: 0 1; background: $boost; display: none; }
    #done-bar Button { margin-right: 2; }
    #streams { height: 1fr; }
    .stream { width: 1fr; padding: 0 1; }
    #reasoning { color: $text-muted; }
    StreamView { height: 1fr; border: none; padding: 0; }
    #tabs { height: 12; }
    AddModelScreen, ConfirmScreen { align: center middle; }
    SelectionList > .selection-list--button, SelectionList > .selection-list--button-highlighted {
        color: $panel; background: $panel;
    }
    SelectionList > .selection-list--button-selected, SelectionList > .selection-list--button-selected-highlighted {
        color: $success; background: $panel; text-style: bold;
    }
    Checkbox > .toggle--button { color: $panel; }
    Checkbox.-on > .toggle--button { color: $success; text-style: bold; }
    #dialog { width: 80; height: auto; max-height: 90%; padding: 1 2; border: thick $primary; background: $surface; }
    #dialog.wide { width: 120; }
    #dialog SelectionList, #dialog OptionList { height: auto; max-height: 20; }
    PresetsScreen, ScanScreen, SessionsScreen, ChoiceScreen, PickTestsScreen, HelpScreen { align: center middle; }
    #help-body { height: auto; max-height: 30; }
    #dialog Input, #dialog Select { margin-bottom: 1; }
    .dialog-buttons { height: 3; margin-top: 1; }
    .dialog-buttons Button { margin-right: 2; }
    #results-info { padding: 0 1; height: auto; }
    #readiness-short { padding: 0 1 1 1; height: auto; }
    #machine { padding: 0 1; height: auto; }
    #tune-status { padding: 0 1; height: auto; min-height: 2; background: $boost; }
    #tune-log { height: 1fr; }
    #speed-machine { width: 60; margin: 0 1; }
    #results-top { height: auto; }
    #results-info { width: 1fr; height: auto; }
    #toggle-smoke { width: auto; min-width: 32; }
    #pq-controls { height: 3; }
    #pq-controls Checkbox { width: auto; }
    #pq-controls Select { width: 28; margin-right: 1; }
    #pq-models { margin-right: 1; min-width: 20; }
    #pq-compare-title { padding: 0 1; height: auto; }
    #pq-summary { height: auto; max-height: 10; }
    #pq-legend { padding: 0 1; height: auto; }
    /* Tabs must read as tabs, not as a line of text: each tab is a button-like chip on its own band,
       the active one filled with the accent colour. (Textual's default: dimmed words on the background.) */
    Tabs { background: $panel; }
    Tab { color: $foreground 85%; background: $boost; padding: 0 1; margin: 0 1 0 0; }
    Tab:ansi { text-style: not dim; }
    Tab:hover { color: $foreground; background: $primary 40%; }
    Tab.-active { color: $text; background: $accent; text-style: bold; }
    Tab.-active:hover { background: $accent; }
    Underline > .underline--bar { color: $accent; background: $panel; }
    AnswerScreen { align: center middle; }
    #answer-dialog { width: 96%; height: 94%; border: thick $primary; background: $surface; padding: 0 1; }
    #answer-head { height: auto; }
    #answer-grading { height: auto; max-height: 9; background: $boost; }
    #answer-grading-text { height: auto; padding: 0 1; }
    #answer-tabs { height: 1fr; }
    #answer-tabs ContentSwitcher { height: 1fr; }
    #answer-tabs TabPane { height: 1fr; padding: 0; }
    #answer-tabs TabPane > Horizontal { height: 1fr; }
    #answer-tabs TextArea { height: 1fr; border: none; }
    #answer-reasoning { color: $text-muted; }
    """
    BINDINGS = [Binding("ctrl+q", "quit_app", "Quit", show=False), Binding("question_mark", "help", "Help")]

    def action_help(self):
        if not isinstance(self.screen, HelpScreen):
            self.push_screen(HelpScreen(type(self.screen).__name__))

    def __init__(self, engine_kwargs):
        super().__init__()
        self.engine = engine.Engine(**engine_kwargs)
        self.worker = None
        self.closing = False
        self.sessions = []  # RunScreen/TuneScreen started since launch, oldest first; kept until quit
        self.queue = []     # QueuedRun/QueuedTune waiting their turn, next first; dropped on quit
        self._finished_last = None

    @property
    def presets_path(self):
        return os.path.join(os.path.dirname(os.path.abspath(self.engine.models_path)), "presets.toml")

    @property
    def state_path(self):
        return os.path.join(self.engine.log_dir, "tui_state.json")

    def load_state(self):
        try:
            with open(self.state_path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def save_state(self, state):
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, "w") as f:
                json.dump(state, f)
        except OSError:
            pass

    def on_mount(self):
        self.push_screen(SetupScreen())
        for err in self.engine.pack_errors:
            self.notify(err, severity="error", timeout=15)
        self.set_interval(1.0, self.refresh_live)

    # -- sessions: every run and tune started here stays reopenable (w) until the app quits
    def add_session(self, screen, show=True):
        self.sessions.append(screen)
        screen.session_name = f"session-{len(self.sessions)}"
        self.install_screen(screen, screen.session_name)  # installed screens survive being popped
        if show:
            self.push_screen(screen.session_name)
        else:
            self.run_worker(self._mount_behind(screen), exclusive=False)

    async def _mount_behind(self, screen):
        """Start a queued run without taking over the screen you're on: a screen only starts its
        work once mounted, so mount it and step straight back (it keeps going, like after esc)."""
        await self.push_screen(screen.session_name)
        if self.screen is screen:
            self.pop_screen()

    @property
    def active_session(self):
        return next((s for s in self.sessions if s.running), None)

    def is_busy(self):
        """A run or tune owns the model servers (one at a time), or queued ones are about to."""
        return bool(self.active_session or (self.worker and self.worker.is_alive()) or self.queue)

    def busy_text(self):
        s = self.active_session
        what = s.live_text() if s else "The last run is still stopping its server"
        return what + (f" (+{len(self.queue)} queued)" if self.queue else "")

    # -- the queue: Start or t while something runs adds to it; each item starts when the one before ends
    def start_or_queue(self, item):
        if not self.is_busy():
            screen, problem = item.make(self.engine)
            if problem:
                self.notify(f"Can't start {item.title}: {problem}", severity="warning", timeout=15)
                return
            if item.kind == "Run" and item.rerun:
                self.notify("Everything selected had results; running it again.")
            self.add_session(screen)
            return
        ahead = self.busy_text()
        self.queue.append(item)
        self.notify(f"Queued #{len(self.queue)}: {item.title}. It starts by itself after: {ahead}. "
                    "w shows the queue.", timeout=10)
        self.refresh_live()
        if isinstance(self.screen, SetupScreen):
            self.screen.update_estimate()

    def session_finished(self, screen):
        """A run or tune ended (finished, failed or cancelled): start the next queued item, if any."""
        self._finished_last = screen
        if self.queue:
            self.set_timer(0.3, self.start_next)

    def start_next(self):
        if self.closing or not self.queue or self.active_session:
            return
        if self.worker and self.worker.is_alive():   # still stopping the last server
            self.set_timer(0.3, self.start_next)
            return
        item = self.queue.pop(0)
        screen, problem = item.make(self.engine)
        if problem:
            self.notify(f"Skipped queued {item.title}: {problem}", severity="error", timeout=30)
            self.start_next()
            self.refresh_sessions_list()
            return
        # show it if you were watching the one that just ended (or are on Setup); otherwise it starts behind
        show = isinstance(self.screen, SetupScreen) or self.screen is self._finished_last
        if show:
            self.back_to_setup()
        self.add_session(screen, show=show)
        more = f" · {len(self.queue)} more queued" if self.queue else ""
        self.notify(f"Started queued {item.title}{more}" + ("" if show else " · w to watch"), timeout=10)
        self.refresh_sessions_list()
        self.refresh_live()

    def refresh_sessions_list(self):
        for sc in self.screen_stack:
            if isinstance(sc, SessionsScreen):
                sc.fill()

    def ask_cancel(self, question, stop, confirm):
        """Cancel the current run or tune; with a queue, also ask whether the queue goes too."""
        if not self.queue:
            if confirm:
                self.push_screen(ConfirmScreen(question), lambda yes: yes and stop())
            else:
                stop()
            return

        def picked(choice):
            if choice == "all":
                self.queue.clear()
                self.refresh_live()
            if choice in ("this", "all"):
                stop()
        n = len(self.queue)
        self.push_screen(ChoiceScreen(
            f"{question}\n\n{n} more queued: {'; '.join(i.title for i in self.queue[:3])}"
            f"{' …' if n > 3 else ''}.\nCancel only this one (the queue goes on), or the queue as well?",
            [("this", "Cancel this one", "warning"), ("all", f"Cancel it and the {n} queued", "error"),
             ("keep", "Keep going", "default")]), picked)

    def queue_loss_text(self):
        if not self.queue:
            return ""
        return (f"\n\nThe {len(self.queue)} queued item(s) are dropped (the queue isn't kept after quitting): "
                + "; ".join(i.title for i in self.queue) + ".")

    def back_to_setup(self):
        while len(self.screen_stack) > 2 and not isinstance(self.screen, SetupScreen):
            self.pop_screen()

    def show_session(self, screen):
        if screen is None or screen is self.screen:
            return
        self.back_to_setup()
        self.push_screen(screen.session_name)

    def open_sessions(self):
        if not self.sessions:
            self.notify("No runs or tunes yet in this session.")
            return
        self.push_screen(SessionsScreen(), self.show_session)

    def refresh_live(self):
        """Header line for a run or tune that's going on behind the current screen."""
        s = self.active_session
        queued = f" · {len(self.queue)} queued" if self.queue else ""
        self.sub_title = f"● {s.live_text()}{queued} · w to watch" if s and s is not self.screen else queued.lstrip(" ·")

    def start_tune(self, labels, on_event, finished):
        """Tune models one after another on a worker thread (tune.py); finished(summary) on the UI thread."""
        from . import tune
        e = self.engine
        e.on_event = on_event
        e._cancel.clear()
        e._skip.clear()

        def work():
            done, failed = [], []
            waited_out = False
            try:
                with e.machine_lock(f"tune: {', '.join(labels)}"):
                    for label in labels:
                        if e._cancel.is_set():
                            break
                        on_event("tune_model", label=label)
                        try:
                            p = tune.tune(e, label, lambda kind, **d: on_event(kind, label=label, **d))
                            ms = p["measured"]
                            gain = f" ({tune.gain_text(ms)})" if tune.gain_text(ms) else ""
                            if p["meta"].get("warnings"):
                                gain += " (warning: it pushes other apps' memory to swap; see the log)"
                            if not p["meta"].get("answer_guard", True):
                                gain += " (answers depend on these settings: rerun its evals here)"
                            done.append(f"{label}{gain}")
                        except (engine.ModelFailed, engine.Cancelled) as ex:
                            failed.append(label)
                            on_event("tune_error", label=label, message=str(ex) or "cancelled")
                        except Exception as ex:  # never leave the screen hanging
                            failed.append(label)
                            on_event("tune_error", label=label, message=f"unexpected error: {ex!r}")
            except engine.Cancelled:
                waited_out = True
            summary = ("Cancelled while waiting for another tuieval window. " if waited_out else "") + \
                (f"Tuned: {', '.join(done)}." if done else "Nothing tuned.") + \
                (f" Not tuned: {', '.join(failed)}." if failed else "")
            if not self.closing:
                self.call_from_thread(finished, summary)
        self.worker = threading.Thread(target=work, name="tune")
        self.worker.start()

    def start_run(self, jobs, on_event, parallel=None):
        self.engine.on_event = on_event
        # A plain (non-daemon) thread, so quitting can wait for servers to be stopped.
        self.worker = threading.Thread(target=self.engine.run, args=(jobs, parallel), name="engine")
        self.worker.start()

    def on_unmount(self):
        """However the app ends, never leave a model server running."""
        if self.worker and self.worker.is_alive():
            self.closing = True
            self.engine.cancel()
            self.worker.join(90)

    def action_quit_app(self):
        if self.worker and self.worker.is_alive():
            def confirmed(yes):
                if yes:
                    self.queue.clear()   # nothing new may start while it stops
                    self.notify("Stopping the current request and the model server…")
                    self.engine.cancel()
                    # Don't join() here: the engine thread still delivers events to this UI thread.
                    self.set_interval(0.2, lambda: None if self.worker.is_alive() else self.exit())
            self.push_screen(ConfirmScreen("A run is in progress. Stop it and quit?" + self.queue_loss_text()),
                             confirmed)
        else:
            self.exit()


def main(argv=None):
    p = argparse.ArgumentParser(prog="tuieval", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", help="models.toml (default: the workspace's)")
    p.add_argument("--packs-dir", help="packs folder (default: the workspace's packs/)")
    p.add_argument("--results-dir")
    p.add_argument("--log-dir")
    a = p.parse_args(argv)
    EvalsApp(dict(models_path=a.models, packs_dir=a.packs_dir, results_dir=a.results_dir,
                  log_dir=a.log_dir)).run()


if __name__ == "__main__":
    main()
