"""Command-line runner (the TUI, tuieval, does the same interactively).

    tuieval run                                   # screen every model on every pack (fast)
    tuieval run --tier certify --only a,b         # full certification for finalists (overnight)
    tuieval verdict                               # PASS / FAIL / INCONCLUSIVE per model and use case
    tuieval report                                # the same, with evidence, as a markdown file
    tuieval history                               # every verdict change over time
    tuieval items                                 # tests that don't separate models or look broken
    tuieval selftest                              # check every test's reference and wrong answers
    tuieval capture logs/live/<file> --pack X     # turn a real failure into a new test
    tuieval run --only my-model --packs my-pack,other-pack
    tuieval run --tags moe --packs my-pack        # models tagged "moe"
    tuieval run --packs my-pack --tests a,b       # only these tests of a pack (or --tests my-pack:a,other:b)
    tuieval run --preset nightly                  # a selection saved in the TUI (presets.toml)
    tuieval run --tier smoke --only my-model      # 3 tests per pack, 1 repeat: check paths and flags
    tuieval run --tier smoke --only openrouter:qwen/qwen3-32b   # any OpenRouter model, no models.toml edit
    tuieval run --dry-run                         # print the plan and server commands only
    tuieval run --parallel 2                      # serve two models at a time (parallel_models in models.toml)
    tuieval add ~/models/New-Model-Q4_K_M.gguf    # register a model (--vision/--mmproj, --no-think, --tags)
    tuieval scan                                  # GGUFs in model_dirs that aren't registered yet
    tuieval list                                  # the models tuieval knows; hidden ones listed separately
    tuieval remove --hidden                       # clean up hidden models (archived in removed/)
    tuieval regrade                               # re-score stored answers with the current graders
    tuieval regrade --only my-model --packs coding   # just one model's results (and packs)
    tuieval machines                              # this machine, known machines, fit and tuning per model
    tuieval tune my-model                         # find the fastest speed flags for a model on this machine
    tuieval tune --untuned                        # tune every model that has no profile here yet
    tuieval export pi my-model                    # serve it in the pi coding agent with the tuned flags

Each model's server is started once, every selected pack runs against it, then it is stopped.
Results go to results/<label>/<pack>.json. Interrupted packs resume where they stopped.
"""
import argparse
import os
import time
import shlex
import signal
import sys
import threading

from . import engine
from . import packs as packs_mod
from . import workspace

BOLD, DIM, RED, GREEN, CYAN, YELLOW, RESET = "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[36m", "\033[33m", "\033[0m"


def say(msg, color=BOLD):
    print(f"{color}[tuieval] {msg}{RESET}", flush=True)


class Printer:
    """Terminal view of engine events: live reasoning/answer plus one line per finished test."""

    def __init__(self):
        self.lock = threading.Lock()
        self.mode = None
        self.side_by_side = False   # several models at a time: one line per answer, no live streams

    def __call__(self, kind, **d):
        with self.lock:
            getattr(self, "on_" + kind, lambda **_: None)(**d)

    def on_machine_busy(self, holder):
        say("waiting: " + engine.holder_text(holder) + " (Ctrl+C gives up)", "\033[33m")

    def on_machine_free(self):
        say("the machine is free; starting")

    def on_model_loading(self, label, command, log):
        say(f"{label}: starting server (log: {os.path.relpath(log)})" if log else f"{label}: {command[0]}")

    def on_model_ready(self, label, ids, load_s):
        say(f"{label}: ready as {ids}" + (f" after {load_s:.0f}s" if load_s else ""))

    def on_server_stall(self, label, message):
        say(f"STALL: {message}", RED)

    def on_model_waiting(self, label, message):
        say(message, YELLOW)

    def on_server_retry(self, label, message):
        say(message, RED)

    def on_model_failed(self, label, message):
        say(f"{label}: FAILED: {message}", RED)

    def on_prompt_reused(self, label, pack, tokens, server, hosted=False):
        if hosted:
            say(f"{label} · {pack}: the provider reused {tokens} cached prompt tokens from an earlier request. "
                "Hosted APIs cache shared prompt openings and it can't be switched off; the answer is computed "
                "the same way, but its TTFT is left out of the speed numbers", YELLOW)
        else:
            say(f"{label} · {pack}: the server reused {tokens} prompt tokens from an earlier request; set "
                f"request = {{ cache_prompt = false }} (or its equivalent) under [servers.{server}]", RED)

    def on_identical_answer(self, label, pack, test, repeat, same):
        say(f"{label} · {pack} · {test}: repeat {repeat + 1} is an exact copy of repeat {same + 1}, "
            "so the server may be returning cached answers rather than fresh ones", RED)

    def on_job_started(self, job, resumed, earlier_s=0, sittings=0):
        say(f"{job.key}: {job.count} tests x {job.repeat} repeats" + (f", resuming after {resumed}" if resumed else "")
            + (f" ({earlier_s / 60:.0f} min over {sittings} earlier sitting(s))" if sittings else ""))

    def on_request_started(self, job, test, test_id, has_image, repeat, difficulty="unrated"):
        if self.side_by_side:
            return
        tag = {"easy": GREEN, "medium": "\033[33m", "hard": RED}.get(difficulty, DIM) + f"[{difficulty}]" + RESET
        print(f"\n{BOLD}{CYAN}━━ {job.key} [{job.done + 1}/{job.total}]{RESET} {tag} {BOLD}{CYAN}{test}{RESET}", flush=True)
        self.mode = None

    def on_delta(self, stream, text, label=None):
        if self.side_by_side:
            return
        if self.mode != stream:
            sys.stdout.write(f"{DIM}[thinking] " if stream == "reasoning" else f"{RESET}\n{BOLD}[answer]{RESET} ")
            self.mode = stream
        sys.stdout.write(text)
        sys.stdout.flush()

    def on_request_done(self, job, record):
        r = record
        mark = f"{GREEN}✓" if r["pass"] else f"{RED}✗"
        calls = f" tool={r['tool_calls'][0]['name']}" if r.get("tool_calls") else ""
        speed = f"{r['gen_tps']:.0f} tok/s, " if r.get("gen_tps") else ""
        ttft = f"TTFT {r['ttft_s']:.1f}s, " if r.get("ttft_s") is not None else ""
        where = f"{CYAN}{job.key} [{job.done}/{job.total}]{RESET} " if self.side_by_side else ""
        print(f"{RESET}{'' if self.side_by_side else chr(10)}{where}{mark} {r['description']}{RESET}{calls} "
              f"{DIM}({r.get('completion_tokens')} tokens, {ttft}{speed}{r['total_s']:.1f}s) {r['reason'][:120]}{RESET}",
              flush=True)

    def on_job_done(self, job):
        color = GREEN if job.status == "done" else RED if job.status == "failed" else ""
        say(f"{job.key}: {job.status} {job.note}", color)


def models_file(path):
    return path or workspace.path("models.toml")


def cmd_add(argv):
    p = argparse.ArgumentParser(prog="tuieval add", description="Register a model in models.toml. Without a "
                                "model, lists the models on servers running on this machine to choose from.")
    p.add_argument("model", nargs="?", help="GGUF path (llama) or model id (other servers)")
    p.add_argument("--server", help="server from models.toml; default: llama for .gguf, else the other one")
    p.add_argument("--label", help="name for results/<label>/; default: derived from the model")
    p.add_argument("--vision", action="store_true", help="model can read images (enables vision packs)")
    p.add_argument("--mmproj", help="vision projector file for llama (implies --vision)")
    p.add_argument("--no-think", action="store_true", help="run with thinking off (label gets -nothink)")
    p.add_argument("--tags", help="comma-separated, e.g. 27b,moe,q4")
    p.add_argument("--models", default=None, help="default: the workspace's models.toml")
    p.add_argument("--yes", "-y", action="store_true", help="without a model: add every one found, no questions")
    a = p.parse_args(argv)
    path = models_file(a.models)
    if not a.model:
        from . import onboard
        labels = onboard.add_detected(path, a.yes)
        if labels:
            print(f"next: tuieval run --tier smoke --only {','.join(labels)}   (or open the TUI: tuieval)")
        return
    server = a.server
    if not server and not a.model.lower().endswith(".gguf") and not os.path.exists(os.path.expanduser(a.model)):
        cfg = engine.load_config(path)
        hits = engine.find_running(cfg, a.model)
        if hits:   # the server that has it, added to models.toml if it isn't there yet
            server = engine.ensure_server(path, hits[0]["url"], hits[0]["port"])
            print(f"found {a.model} on the server at {hits[0]['url']} ([servers.{server}])")
        elif engine.infer_server(cfg, a.model):
            s = engine.infer_server(cfg, a.model)
            print(f"note: no running server lists {a.model}; it's set to [servers.{s}] at "
                  f"{cfg['servers'][s].get('url')}. Start that server before running (tuieval doctor checks).")
    try:
        label, warnings = engine.add_model(path, a.model, server, a.label, a.vision, a.mmproj,
                                           False if a.no_think else None, (a.tags or "").split(","))
    except engine.ConfigError as e:
        sys.exit(str(e))
    for w in warnings:
        print("warning:", w)
    print(f"added {label!r}")
    print(f"next: tuieval run --tier smoke --only {label}    then: tuieval run --only {label}")


def cmd_scan(argv):
    p = argparse.ArgumentParser(prog="tuieval scan", description="List GGUFs not yet in models.toml.")
    p.add_argument("dirs", nargs="*", help="folders to scan (default: model_dirs in models.toml)")
    p.add_argument("--add", action="store_true", help="add every one found")
    p.add_argument("--models", default=None, help="default: the workspace's models.toml")
    a = p.parse_args(argv)
    cfg = engine.load_config(models_file(a.models))
    found = engine.scan_models(cfg, a.dirs or None)
    if not found:
        print("no unregistered GGUFs found in", ", ".join(a.dirs or cfg["defaults"].get("model_dirs", ["~/models"])))
    for f in found:
        print(f"{f['model']}" + (f"   (mmproj: {f['mmproj']})" if f["mmproj"] else ""))
        if a.add:
            try:
                label, _ = engine.add_model(models_file(a.models), f["model"], mmproj=f["mmproj"])
                print(f"  added as {label}")
            except engine.ConfigError as e:
                print(f"  not added: {e}")
    if found and not a.add:
        print(f"\n{len(found)} found. Add them with tuieval scan --add, or one by one with tuieval add <path>.")


def cmd_list(argv):
    p = argparse.ArgumentParser(prog="tuieval list", description="The models tuieval knows, with the hidden ones "
                                                                  "listed separately.")
    p.add_argument("--packs", action="store_true", help="list the packs instead")
    p.add_argument("--models", default=None, help="default: the workspace's models.toml")
    a = p.parse_args(argv)
    e = engine.Engine(models_path=a.models)
    if a.packs:
        print("Packs:" if e.packs else "Packs: none yet (tuieval new-pack <name>)")
        for n, pk in e.packs.items():
            needs = f"  needs {', '.join(pk.needs)}" if pk.needs else ""
            print(f"  {n:12s} {pk.label:28s} {len(pk.tests):4d} tests  grader={pk.grader}{needs}")
        return
    home = os.path.expanduser("~")
    hidden = e.hidden()
    widths = [max((len(m[k]) for m in e.cfg["models"]), default=0) for k in ("label", "server")]

    def table(models):
        for m in models:
            where = "~" + m["model"][len(home):] if m["model"].startswith(home + os.sep) else m["model"]
            print(f"  {m['label'].ljust(widths[0])}  {m['server'].ljust(widths[1])}  {where}")
    shown = [m for m in e.cfg["models"] if m["label"] not in hidden]
    hid = [m for m in e.cfg["models"] if m["label"] in hidden]
    print(f"Models ({len(shown)}):" if shown or hid else "Models: none yet (tuieval add <GGUF or model id>)")
    table(shown)
    if hid:
        print(f"\nHidden ({len(hid)}; tuieval remove <model> or tuieval remove --hidden cleans them up):")
        table(hid)


def cmd_remove(argv):
    p = argparse.ArgumentParser(prog="tuieval remove",
                                description="Remove models: their models.toml entry, results, tuning profiles and "
                                            "hidden mark. Nothing is deleted: it all moves to removed/<model>-<time>/.")
    p.add_argument("labels", nargs="*", help="models to remove")
    p.add_argument("--hidden", action="store_true", help="every hidden model")
    p.add_argument("--yes", action="store_true", help="don't ask")
    p.add_argument("--models", default=None, help="default: the workspace's models.toml")
    a = p.parse_args(argv)
    from . import remove
    e = engine.Engine(models_path=a.models)
    labels = list(a.labels)
    if a.hidden:
        labels += [m["label"] for m in e.cfg["models"] if m["label"] in e.hidden() and m["label"] not in labels]
    if not labels:
        sys.exit("name models to remove, or --hidden" if not a.hidden else "no hidden models")
    try:
        plans = [remove.plan(e, label) for label in labels]
    except ValueError as ex:
        sys.exit(str(ex))
    for pl in plans:
        say(pl.label)
        for line in pl.describe(e.root) or ["  (nothing stored)"]:
            print(line)
    diff, _ = remove.models_diff(plans, e.models_path)
    if diff:
        print(diff)
    if not a.yes:
        if not sys.stdin.isatty():
            sys.exit("not a terminal: pass --yes to remove")
        if input(f"Remove {len(plans)} model(s)? Everything moves to removed/. [y/N] ").strip().lower() not in ("y", "yes"):
            sys.exit("nothing removed")
    folders = remove.apply(e, plans)
    say(f"removed {len(plans)} model(s); kept in {os.path.relpath(os.path.dirname(folders[0]), e.root)}/", GREEN)


def cmd_regrade(argv):
    p = argparse.ArgumentParser(prog="tuieval regrade", description="Re-score stored answers with current graders.")
    p.add_argument("paths", nargs="*", help="result files (default: all in results/)")
    p.add_argument("--only", help="comma-separated model labels")
    p.add_argument("--packs", help="comma-separated packs")
    a = p.parse_args(argv)
    from . import compare
    paths = a.paths or compare.default_paths()
    for flag, value, part in (("--only", a.only, lambda q: os.path.basename(os.path.dirname(q))),
                              ("--packs", a.packs, lambda q: os.path.basename(q)[:-len(".json")])):
        if not value:
            continue
        wanted = [v.strip() for v in value.split(",") if v.strip()]
        known = sorted({part(q) for q in paths})
        missing = [v for v in wanted if v not in known]
        if missing:
            sys.exit(f"{flag}: no results for {', '.join(missing)}. With results: {', '.join(known) or 'none'}")
        paths = [q for q in paths if part(q) in wanted]
    for path in paths:
        try:
            before, after = engine.regrade(path)
            print(f"{os.path.relpath(path)}: {before} -> {after} passed")
        except (engine.ConfigError, KeyError, ValueError) as e:
            print(f"{os.path.relpath(path)}: skipped ({e})")
    from . import verdict
    for h in verdict.record_history(engine.Engine()):
        print(f"verdict changed: {h['model']} {h['use_case']}: {h.get('previous')} -> {h['status']}")


MARK = {"PASS": "\033[32mPASS\033[0m", "FAIL": "\033[31mFAIL\033[0m",
        "INCONCLUSIVE": "\033[33mINCONCLUSIVE\033[0m", "NO DATA": "-"}


def cmd_verdict(argv):
    p = argparse.ArgumentParser(prog="tuieval verdict", description="Production readiness per model and use case.")
    p.add_argument("--only", help="comma-separated model labels")
    p.add_argument("--models")
    p.add_argument("--results-dir")
    a = p.parse_args(argv)
    from . import verdict
    e = engine.Engine(models_path=a.models, results_dir=a.results_dir)
    table = verdict.readiness(e, a.only.split(",") if a.only else None)
    if not table:
        sys.exit("no results yet: tuieval run first")
    summary = verdict.plain_summary(table)
    print(f"{BOLD}In short{RESET}")
    certify = {}
    for label, sentence, todo in summary:
        print(f"  {label}: {sentence}")
        for g, what, pending in todo:
            print(f"     → {g}: {what}")
            if "Certify" in what:
                certify.setdefault(tuple(pending), []).append(label)
    for pending, labels in certify.items():
        print(f"  next: tuieval run --tier certify --only {','.join(labels)} --packs {','.join(pending)}")
    print(f"\n{DIM}Details (PASS needs a Certify run, zero critical failures and an accuracy whose 95% lower bound "
          f"clears the pack's gate):{RESET}")
    for label, groups in table.items():
        print(f"\n{BOLD}{label}{RESET}")
        lat = verdict.latency_table(e, [label], {label: groups}).get(label, {})
        for group, (v, packs) in groups.items():
            print(f"  {group:18s} {MARK[v.status]}")
            for name, pv in packs.items():
                if pv.status == "NO DATA":
                    continue
                print(f"      {name:14s} {pv.status:12s} {'; '.join(pv.reasons)[:150]}")
            for mid, x in lat.get(group, {}).items():
                color = GREEN if x.status == "OK" else RED if x.status in ("TOO SLOW", "DOESN'T FIT") else DIM
                print(f"      {DIM}speed on{RESET} {mid:18s} {color}{x.summary}{RESET}")


def report_markdown(e, labels=None):
    from . import verdict
    table = verdict.readiness(e, labels)
    lines = ["# Production readiness report", "", f"Generated {time.strftime('%Y-%m-%d %H:%M')}.", "",
             "PASS needs a full certification run, zero critical failures across enough trials, and accuracy "
             "whose 95% lower bound clears the gate. See verdict.py for the rules and each pack.toml for its gate.", ""]
    lines += ["## In short", ""]
    for label, sentence, todo in verdict.plain_summary(table):
        lines.append(f"- **{label}**: {sentence}" + "".join(f"; {g}: {what}" for g, what, _ in todo))
    lines.append("")
    groups = sorted({g for t in table.values() for g in t})
    lines += ["| model | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
    for label, gs in table.items():
        lines.append(f"| {label} | " + " | ".join(gs[g][0].status if g in gs else "-" for g in groups) + " |")
    lat = verdict.latency_table(e, labels, table)
    for label, gs in table.items():
        lines += ["", f"## {label}"]
        for group, (v, packs) in gs.items():
            lines += ["", f"### {group}: {v.status}"]
            for mid, x in lat.get(label, {}).get(group, {}).items():
                lines.append(f"- speed on **{mid}**: {x.summary}")
            for name, pv in packs.items():
                if pv.status == "NO DATA":
                    lines.append(f"- **{name}**: not run")
                    continue
                ev = pv.evidence
                lines.append(f"- **{name}: {pv.status}**. " + "; ".join(pv.reasons))
                lines.append(f"  - accuracy {100 * ev['accuracy']:.1f}% (95% CI {100 * ev['acc_lo']:.1f}-{100 * ev['acc_hi']:.1f}%) "
                             f"over {ev['trials']} answers; {ev['tests_run']}/{ev['tests_total']} tests, {ev['repeats']} repeat(s)")
                lines.append(f"  - critical failures {ev['critical_failures']} in {ev['critical_trials']} critical trials"
                             + (f"; p90 {ev['p90_s']:.1f}s" if ev.get("p90_s") is not None else "")
                             + f"; truncated {100 * ev['truncation']:.1f}%"
                             + (f"; consistency {100 * ev['consistency']:.0f}% of {ev['groups']} groups" if ev.get("consistency") is not None else ""))
                recs = e.records(label, name)
                levels = []
                for lvl in ("easy", "medium", "hard"):
                    x = [r["pass"] for r in recs if r.get("difficulty") == lvl]
                    if x:
                        levels.append(f"{lvl} {sum(x)}/{len(x)} ({100 * sum(x) / len(x):.0f}%)")
                if levels:
                    lines.append("  - by difficulty: " + ", ".join(levels))
                for ex in ev.get("critical_examples", []):
                    lines.append(f"  - critical: {ex}")
    from . import pta
    lines += [""] + pta.markdown(*pta.for_engine(e, labels))
    return "\n".join(lines) + "\n"


def cmd_pta(argv):
    p = argparse.ArgumentParser(prog="tuieval pta",
                                description="The PTA index: privacy, speed and accuracy per model (0-100, higher is better), as a "
                                            "triangle (a dot per model) and a table, over the questions all the models answered.")
    p.add_argument("--only", help="comma-separated model labels (default: every model with results)")
    p.add_argument("--packs", help="comma-separated packs (default: all)")
    p.add_argument("--width", type=int, default=48, help="triangle width in characters (default 48)")
    p.add_argument("--models")
    p.add_argument("--results-dir")
    a = p.parse_args(argv)
    from rich.console import Console
    from . import pta
    e = engine.Engine(models_path=a.models, results_dir=a.results_dir)
    labels = [x.strip() for x in a.only.split(",")] if a.only else None
    packs = [x.strip() for x in a.packs.split(",")] if a.packs else None
    result, models = pta.for_engine(e, labels, packs)
    out = Console(highlight=False)
    if not models:
        sys.exit("no finished results to compare" + (" for those models or packs" if labels or packs else ""))
    out.print(pta.scope_note(result, models))
    if not result["questions"]:
        for line in pta.left_out_lines(result):
            out.print(line, markup=False)
        return
    out.print()
    for line in pta.triangle(result, a.width) + [""] + pta.legend(result, a.width):
        out.print(line)
    header, rows = pta.table(result)
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    out.print()
    for line in [header] + rows:
        out.print("  ".join(c.ljust(w) for c, w in zip(line, widths)).rstrip(), markup=False)


def cmd_report(argv):
    p = argparse.ArgumentParser(prog="tuieval report", description="Write the readiness report as markdown.")
    p.add_argument("-o", "--output", help="default: reports/readiness-<date>.md")
    p.add_argument("--only", help="comma-separated model labels")
    p.add_argument("--models")
    p.add_argument("--results-dir")
    a = p.parse_args(argv)
    e = engine.Engine(models_path=a.models, results_dir=a.results_dir)
    text = report_markdown(e, a.only.split(",") if a.only else None)
    out = a.output or workspace.path("reports", time.strftime("readiness-%Y%m%d-%H%M.md"))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        f.write(text)
    print(f"wrote {os.path.relpath(out)}")


def cmd_items(argv):
    p = argparse.ArgumentParser(prog="tuieval items", description="Tests that don't separate models or look broken.")
    p.add_argument("paths", nargs="*")
    a = p.parse_args(argv)
    from . import compare
    rows, _ = compare.load(a.paths or compare.default_paths())
    if len({r["model"] for r in rows}) < 2:
        sys.exit("needs results from at least two models")
    flagged = compare.items(rows)
    counts = {}
    for _, flags, _ in flagged:
        for f in flags:
            counts[f] = counts.get(f, 0) + 1
    print(f"{len(flagged)} flagged tests: " + ", ".join(f"{v} {k}" for k, v in counts.items()))
    for test, flags, rates in flagged:
        print(f"  [{', '.join(flags)}] {test}: " + ", ".join(f"{m} {100 * r:.0f}%" for m, r in sorted(rates.items())))


def cmd_history(argv):
    p = argparse.ArgumentParser(prog="tuieval history", description="Every verdict change, newest first.")
    p.add_argument("--only", help="comma-separated model labels")
    p.add_argument("--models")
    p.add_argument("--results-dir")
    a = p.parse_args(argv)
    from . import verdict
    e = engine.Engine(models_path=a.models, results_dir=a.results_dir)
    rows = [h for h in verdict.load_history(e) if not a.only or h["model"] in a.only.split(",")]
    if not rows:
        sys.exit("no verdict history yet: it's recorded after each Screen or Certify run")
    for h in reversed(rows):
        change = f"{h['previous']} -> {h['status']}" if h.get("previous") else f"first: {h['status']}"
        old = "" if verdict.is_current(h, e) else "  (questions changed since)"
        print(f"{h['time'].replace('T', ' ')}  {h['model']:22s} {h['use_case']:16s} {change:28s}{old}")
        print(f"{'':21s}{'; '.join(h['reasons'])[:150]}")


def cmd_selftest(argv):
    from . import selftest
    sys.exit(0 if selftest.run(argv or None) else 1)


def cmd_capture(argv):
    p = argparse.ArgumentParser(prog="tuieval capture",
                                description="Turn a request log (logs/live/*.txt) into a test skeleton to fill in.")
    p.add_argument("log", help="a file from logs/live/")
    p.add_argument("--pack", required=True, help="pack to add it to (written to packs/<pack>/captured.yaml)")
    p.add_argument("--packs-dir", help="default: packs/")
    a = p.parse_args(argv)
    import yaml
    from . import graders
    from .yamlout import dump_tests
    text = open(a.log).read()
    prompt = text.split("PROMPT: ", 1)[1].split("\n\n=== REASONING ===", 1)[0]
    answer = text.split("=== ANSWER ===\n", 1)[1].rsplit("\n\n=== ", 1)[0].split("\n=== TOOL CALLS ===")[0].strip()
    calls = text.split("=== TOOL CALLS ===\n", 1)[1].split("\n", 1)[0] if "=== TOOL CALLS ===" in text else ""
    e = engine.Engine(packs_dir=a.packs_dir)
    if a.pack not in e.packs:
        sys.exit(f"unknown pack {a.pack!r}; packs: {', '.join(e.packs)}")
    grader = e.packs[a.pack].grader
    path = os.path.join(e.packs[a.pack].path, "captured.yaml")
    tests = yaml.safe_load(open(path)) if os.path.isfile(path) else []
    tid = f"captured-{len(tests) + 1}-{time.strftime('%Y%m%d')}"
    entry = {"id": tid, "description": "TODO: what this checks", "category": "captured", "input": prompt,
             **graders.TEMPLATES.get(grader, {"TODO": "the fields your grader checks"}),
             "captured_from": os.path.basename(a.log)}
    if answer and grader != "tool_call":
        entry["wrong"] = [answer]            # the failure itself: the grader must reject it
    else:
        entry["captured_answer"] = answer or f"(tool calls) {calls}"
    tests.append(entry)
    dump_tests(tests, path, "# Tests captured from real use. Fill in every TODO; tuieval selftest flags them until then.\n"
                            "# The model's original answer is kept under `wrong` (it was the failure), or under\n"
                            "# `captured_answer` for tool calls. Neither is sent to the model.\n\n")
    print(f"added {tid} to {os.path.relpath(path)}; fill in the TODOs, then tuieval selftest {a.pack}")


def cmd_machines(argv):
    p = argparse.ArgumentParser(prog="tuieval machines",
                                description="This machine, the others recorded in tuning/, and how each model "
                                            "fits and is tuned on each.")
    p.add_argument("--models", default=None, help="default: the workspace's models.toml")
    a = p.parse_args(argv)
    e = engine.Engine(models_path=a.models)
    known = e.machines()
    for i, mc in enumerate(known):
        print(f"{BOLD}{mc.id}{RESET}{'  (this machine)' if i == 0 else ''}  {DIM}{mc.summary} · GPU keeps "
              f"~{e.gpu_residency_gb(mc):.0f} GB resident{RESET}")
        for m in e.cfg["models"]:
            sv = e.serving(m, mc)
            if sv.perf_source == "n/a":
                print(f"  {m['label']:24s} {DIM}{m['server']} picks its own context and speed settings{RESET}")
                continue
            fit = f"{RED}doesn't fit{RESET}" if not sv.fits else (f"ctx {sv.ctx // 1024}k" if sv.ctx else "ctx default")
            src = {"tuned": GREEN + "tuned" + RESET, "seeded": "seeded (hand-tuned flags)"}.get(
                sv.perf_source, f"{DIM}{sv.perf_source}{RESET}")
            speed = (sv.profile or {}).get("measured", {})
            sp = f"  gen {speed['tg_tps']} tok/s, prompt {speed['pp_tps']} tok/s" if speed.get("tg_tps") else ""
            print(f"  {m['label']:24s} {fit:12s} {src}{sp}")
            if not sv.fits:
                print(f"  {'':24s} {DIM}{sv.fit_note}{RESET}")
    print(f"\n{DIM}Machines appear here after tuieval has run on them once (tuning/<id>/machine.toml; sync the"
          f" workspace to share it). Set EVALS_MACHINE to rename this one.{RESET}")


class TunePrinter:
    """Everything the tuner does, as it happens: server start and its log, each request's
    progress, each candidate's result."""
    def __call__(self, kind, **d):
        t = time.strftime("%H:%M:%S")
        if kind.startswith("tune_"):
            color = DIM if kind == "tune_progress" else ""
            print(f"{DIM}{t}{RESET} {color}" + ("  " if kind == "tune_result" else "") + d.get("message", "")
                  + RESET, flush=True)
        elif kind == "model_loading":
            print(f"{DIM}{t}    starting the server…{RESET}", flush=True)
        elif kind == "server_log":
            print(f"{DIM}{t}    | {d['line'][:160]}{RESET}", flush=True)
        elif kind == "model_ready":
            print(f"{DIM}{t}{RESET}    ready" + (f" after {d['load_s']:.0f}s" if d.get("load_s") else ""), flush=True)
        elif kind == "server_facts":
            print(f"{DIM}{t}{RESET}    " + " · ".join(f"{k} {v}" for k, v in d["facts"].items()), flush=True)
        elif kind == "server_stall":
            say("STALL: " + d.get("message", ""), RED)
        elif kind == "model_failed":
            say(d.get("message", ""), RED)
        elif kind == "machine_busy":
            say("waiting: " + engine.holder_text(d["holder"]) + " (Ctrl+C gives up)", "\033[33m")
        elif kind == "machine_free":
            say("the machine is free; starting")


def cmd_tune(argv):
    p = argparse.ArgumentParser(prog="tuieval tune",
                                description="Find the fastest speed-only server flags for models on this machine "
                                            "(tune.py). Takes ~8-15 server starts per model, or ~4-6 when a model of the "
                                            "same architecture is already tuned here (its flags are the starting point).")
    p.add_argument("labels", nargs="*", help="models to tune")
    p.add_argument("--untuned", action="store_true", help="every local model without a current profile here")
    p.add_argument("--max-starts", type=int, default=16, help="server starts per model (default 16)")
    p.add_argument("--no-bench", action="store_true", help="skip llama-bench even if it is installed")
    p.add_argument("--export-pi", action="store_true",
                   help="after tuning, export each model's serving settings to the pi coding agent (asks first)")
    p.add_argument("--cold", action="store_true",
                   help="tune every knob from the defaults, even if a model of the same architecture is tuned here")
    p.add_argument("--models", default=None, help="default: the workspace's models.toml")
    a = p.parse_args(argv)
    from . import tune
    e = engine.Engine(TunePrinter(), models_path=a.models)
    labels = a.labels
    if a.untuned:
        labels += [m["label"] for m in e.cfg["models"] if m["label"] not in labels
                   and e.cfg["servers"][m["server"]].get("cmd") and e.serving(m).perf_source not in ("tuned",)]
    bad = [x for x in labels if x not in {m["label"] for m in e.cfg["models"]}]
    if bad or not labels:
        sys.exit(f"unknown model(s) {bad}; see tuieval list" if bad else "name models to tune, or --untuned")
    signal.signal(signal.SIGINT, lambda *_: (say("stopping…", RED), e.cancel()))
    say(f"tuning on {e.machine().id}: {e.machine().summary}")
    failed = 0
    try:
        lock = e.machine_lock(f"tune: {', '.join(labels)}")
        lock.__enter__()   # held until this process exits
    except engine.Cancelled:
        sys.exit(1)
    for label in labels:
        say(label)
        result = {}

        def work(label=label, result=result):
            try:
                result["profile"] = tune.tune(e, label, e.on_event, max_starts=a.max_starts, use_bench=not a.no_bench,
                                                 warm=not a.cold)
            except (engine.ModelFailed, engine.Cancelled) as ex:
                result["error"] = str(ex) or "cancelled"
        t = threading.Thread(target=work)
        t.start()
        while t.is_alive():
            t.join(0.5)
        if "error" in result:
            failed += 1
            say(f"{label}: not tuned: {result['error']}", RED)
            if e._cancel.is_set():
                break
            continue
        pr = result["profile"]
        ms = pr["measured"]
        gain = ", " + tune.gain_text(ms) if tune.gain_text(ms) else ""
        result = f"{ms['tg_tps']} tok/s decode" if ms.get("objective") == "decode" else \
            f"{ms['projected_s']}s projected for the workload" if ms.get("objective") == "projected" and ms.get("projected_s") \
            else f"{ms['total_s']}s for the workload"
        say(f"{label}: {' '.join(pr['args']) or '(no knobs)'}  ->  {result}{gain}", GREEN)
        if pr["meta"].get("warm_start"):
            ws = pr["meta"]["warm_start"]
            say(f"{label}: started from {ws['from']}'s flags; re-tried {', '.join(ws['retested']) or 'nothing'}")
        if not pr["meta"].get("answer_guard", True):
            say(f"{label}: its answers depend on these settings; rerun its evals on this machine", "\033[33m")
        for w in pr["meta"].get("warnings", []):
            say(f"{label}: warning: {w}", YELLOW)
        if pr["meta"].get("rejected"):
            say(f"{label}: rejected because answers changed: {'; '.join(pr['meta']['rejected'])}")
        if a.export_pi and not export_pi(e, label):
            failed += 1
    sys.exit(1 if failed else 0)


def export_pi(e, label, model_id=None, name=None, dry_run=False, yes=False):
    """Show what exporting a model to pi changes, ask, then write it. False if it failed or was declined."""
    from . import export
    try:
        changes, notes = export.plan(e, label, model_id, name)
    except (export.ExportError, ValueError) as ex:
        say(f"{label}: not exported: {ex}", RED)
        return False
    for c in changes:
        print(c.diff())
    for n in notes:
        say(f"note: {n}", YELLOW)
    if not changes:
        say(f"{label}: pi already has these settings", GREEN)
        return True
    if dry_run:
        say("dry run: nothing written")
        return True
    if not yes:
        if not sys.stdin.isatty():
            say("not a terminal: pass --yes to write", RED)
            return False
        if input(f"Write {len(changes)} file(s)? [y/N] ").strip().lower() not in ("y", "yes"):
            say("nothing written")
            return False
    try:
        backups = export.apply(changes)
    except (export.ExportError, OSError) as ex:
        say(f"{label}: not exported: {ex}", RED)
        return False
    say(f"{label}: exported to pi; backups: {', '.join(backups)}", GREEN)
    return True


def cmd_export(argv):
    p = argparse.ArgumentParser(prog="tuieval export",
                                description="Write a model's serving settings (model file, output-affecting flags, "
                                            "context and the speed flags tuned on this machine) to another tool. "
                                            "Shows the diff and asks first; every file is backed up.")
    p.add_argument("target", choices=["pi"], help="pi: the pi coding agent (llama.cpp router presets and pi's "
                                                  "models.json)")
    p.add_argument("labels", nargs="+", help="models to export")
    p.add_argument("--id", help="the model id pi sees (default: models.toml pi_id, else the GGUF's name)")
    p.add_argument("--name", help="the name pi shows (default: models.toml pi_name, else the id)")
    p.add_argument("--dry-run", action="store_true", help="show the changes only")
    p.add_argument("--yes", action="store_true", help="write without asking")
    p.add_argument("--models", default=None, help="default: the workspace's models.toml")
    a = p.parse_args(argv)
    if (a.id or a.name) and len(a.labels) > 1:
        sys.exit("--id and --name take one model")
    e = engine.Engine(TunePrinter(), models_path=a.models)
    bad = [x for x in a.labels if x not in {m["label"] for m in e.cfg["models"]}]
    if bad:
        sys.exit(f"unknown model(s) {bad}; see tuieval list")
    ok = all([export_pi(e, label, a.id, a.name, a.dry_run, a.yes) for label in a.labels])
    sys.exit(0 if ok else 1)


COMMANDS = {"add": cmd_add, "list": cmd_list, "scan": cmd_scan, "regrade": cmd_regrade,
            "verdict": cmd_verdict, "report": cmd_report, "items": cmd_items, "selftest": cmd_selftest,
            "capture": cmd_capture, "history": cmd_history, "machines": cmd_machines, "tune": cmd_tune,
            "export": cmd_export, "remove": cmd_remove, "pta": cmd_pta}


def parse_tests(text, pack_names, known):
    """({pack: [test ids]}, packs to run) from --tests. A bare id needs exactly one pack chosen;
    pack:id names its pack, and without --packs the packs named are the ones that run."""
    tests = {}
    for item in filter(None, (x.strip() for x in text.split(","))):
        pack, _, tid = item.rpartition(":")
        if not pack:
            if not pack_names or len(pack_names) != 1:
                sys.exit(f"--tests {item}: say which pack, as pack:{item} (or choose one pack with --packs)")
            pack = pack_names[0]
        if pack not in known:
            sys.exit(f"--tests {item}: unknown pack {pack!r}; packs are {', '.join(known)}")
        tests.setdefault(pack, []).append(tid)
    if pack_names is None:
        pack_names = list(tests)
    missing = [p for p in tests if p not in pack_names]
    if missing:
        sys.exit(f"--tests names pack(s) {missing} that aren't in --packs")
    return tests, pack_names


def main(argv=None):
    """tuieval run [options] (argv without the "run")."""
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(prog="tuieval run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", help="comma-separated model labels, or openrouter:<model id> "
                                  "(default: all models in models.toml)")
    p.add_argument("--tags", help="comma-separated tags; models having any of them")
    p.add_argument("--preset", help="a saved selection from presets.toml")
    p.add_argument("--packs", help="comma-separated packs (default: all)")
    p.add_argument("--tests", help="comma-separated test ids to run instead of the tier's sample: id with one "
                                   "pack in --packs, else pack:id (and --packs defaults to those packs)")
    p.add_argument("--force", action="store_true", help="rerun packs (or the --tests) that already have current results")
    p.add_argument("--repeat", type=int, help="times each question is asked, any tier (default: 1 for smoke and "
                                                  "screen, the pack's certification count for certify)")
    p.add_argument("--tier", choices=engine.TIERS, default="screen",
                   help="smoke: 3 tests; screen (default): a spread sample; certify: everything, with repeats")
    p.add_argument("--smoke", action="store_const", const="smoke", dest="tier", help="same as --tier smoke")
    p.add_argument("--parallel", type=int, metavar="N",
                   help="models served at a time (default: parallel_models for this machine in models.toml, else 1)")
    p.add_argument("--dry-run", action="store_true", help="print the plan and commands without running")
    p.add_argument("--brief", action="store_true", help="end with one line per model instead of the scorecards")
    p.add_argument("--models", default=None, help="default: the workspace's models.toml")
    p.add_argument("--packs-dir")
    p.add_argument("--results-dir")
    p.add_argument("--log-dir", help="server and request logs (default: logs/)")
    args = p.parse_args(argv)

    printer = Printer()
    e = engine.Engine(printer, models_path=args.models, packs_dir=args.packs_dir,
                      results_dir=args.results_dir, log_dir=args.log_dir)
    if args.parallel is not None and args.parallel < 1:
        sys.exit("--parallel must be 1 or more")
    parallel = args.parallel or e.parallel_models()
    models = e.cfg["models"]
    labels = [m["label"] for m in models if not m.get("remote")]   # hosted APIs cost money: only by name
    if args.only:
        try:   # labels, or e.g. openrouter:qwen/qwen3-32b for any model a hosted API serves
            labels = [e.resolve(x) for x in args.only.split(",") if x.strip()]
        except engine.ConfigError as err:
            sys.exit(str(err))
    if args.tags:
        want = {t.strip() for t in args.tags.split(",")}
        labels = [l for l in labels if want & set(e.model(l)["tags"])]
        if not labels:
            sys.exit(f"no models tagged {sorted(want)}")
    pack_names, tests = list(e.packs), {}
    if not pack_names:
        sys.exit("no packs in this workspace yet: tuieval new-pack <name> creates one (see docs/writing-packs.md)")
    if not labels and not args.preset:
        sys.exit("no models to run yet: tuieval add <GGUF path or model id>, or name a hosted one with "
                 "--only openrouter:<model id>" if not models else
                 "no local models; hosted ones (they cost money) run only when named: --only <label>")
    if args.preset:
        presets = engine.load_presets(os.path.join(os.path.dirname(os.path.abspath(models_file(args.models))),
                                                   "presets.toml"))
        if args.preset not in presets:
            sys.exit(f"unknown preset {args.preset!r}; presets: {', '.join(presets) or 'none (save one in the TUI with p)'}")
        labels, pack_names, preset_repeat, tests = engine.resolve_preset(e.cfg, presets[args.preset], list(e.packs))
        args.repeat = args.repeat or preset_repeat
    if args.packs:
        pack_names = [x.strip() for x in args.packs.split(",")]
        bad = [x for x in pack_names if x not in e.packs]
        if bad:
            sys.exit(f"unknown pack(s) {bad}; packs are {', '.join(e.packs)}")
    if args.tests:
        tests, pack_names = parse_tests(args.tests, pack_names if (args.packs or args.preset) else None, e.packs)
    try:
        jobs = e.plan(labels, pack_names, args.repeat, args.tier, args.force, tests)
    except packs_mod.PackError as err:
        sys.exit(str(err))

    if args.dry_run:
        seen = set()
        print(f"# machine: {e.machine().id} ({e.machine().summary})")
        for j in jobs:
            if j.label not in seen:
                seen.add(j.label)
                cmd, cwd, url = e.server_command(j.model)
                sv = e.serving(j.model)
                print(f"\n# {j.label}  ->  {url}   speed flags: {sv.perf_source}"
                      + (f"; {sv.fit_note}" if sv.fit_note else ""))
                print(f"(cd {shlex.quote(cwd or e.root)} && {shlex.join(cmd)})" if cmd else "(already running)")
            print(f"  {j.pack.name:12s} {j.status:8s} {j.total:5d} requests  {j.note}")
        return

    for j in jobs:
        if j.status == "skipped":
            say(f"{j.key}: skipping ({j.note})")
        elif j.status == "done":
            say(f"{j.key}: done earlier ({j.passed}/{j.done} passed)")

    def stop(*_):
        say("stopping…", RED)
        e.cancel()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    side_by_side = min(parallel, len({j.label for j in jobs if j.status == "waiting"}))
    if side_by_side > 1:
        printer.side_by_side = True
        say(f"up to {side_by_side} models at a time: one line per answer; each answer notes the models "
            "it ran alongside, since they share this machine's speed")
    worker = threading.Thread(target=e.run, args=(jobs, parallel))
    worker.start()
    while worker.is_alive():
        worker.join(0.5)

    print()
    width = max((len(j.key) for j in jobs), default=0)
    for j in jobs:
        color = GREEN if j.status == "done" else RED if j.status == "failed" else ""
        secs = (j.finished - j.started) if j.started and j.finished else 0
        print(f"  {color}{j.status:<8}{RESET} {j.key:<{width}} {secs / 60:6.0f} min  {j.note}", flush=True)
    results = sorted({j.out_path for j in jobs if os.path.isfile(j.out_path)})
    if results and args.brief:
        print()
        for label in dict.fromkeys(j.label for j in jobs):
            mine = [j for j in jobs if j.label == label and j.done]
            if mine:
                right, asked = sum(j.passed for j in mine), sum(j.done for j in mine)
                print(f"{label}: {right} of {asked} answers right"
                      + (" (a setup check, not a verdict)" if args.tier == "smoke" else ""))
    elif results:
        print()
        from . import compare
        compare.main([*results, "--speed"])
    sys.exit(1 if any(j.status == "failed" for j in jobs) else 0)


if __name__ == "__main__":
    main()
