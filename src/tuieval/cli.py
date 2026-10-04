"""tuieval: evaluate local and frontier models on your own eval packs, in the terminal.

Usage: tuieval [--workspace DIR] [command] [options]

  tuieval                     open the TUI
  tuieval init [DIR]          create a workspace (models.toml, packs/, graders/)
  tuieval doctor              check servers, models, packs and API keys, and say what to fix
  tuieval new-pack NAME       a new pack from a starter template (--grader answer|rag|reply|tool_call|code)
  tuieval add PATH|ID         register a model in models.toml (a GGUF, or a model id)
  tuieval scan                list GGUFs in model_dirs that aren't in models.toml yet
  tuieval list                the models tuieval knows, hidden ones separately (--packs: the packs)
  tuieval remove MODEL        remove models, or every hidden one (--hidden); archived in removed/
  tuieval run [...]           run evals from the command line (tuieval run --help)
  tuieval compare [...]       scorecards (--speed, --pairwise, --failures)
  tuieval regrade             re-score stored answers after changing a grader
  tuieval verdict             production readiness (PASS/FAIL per model and use case)
  tuieval report              the readiness verdicts with their evidence, as a markdown file
  tuieval history             every verdict change, newest first
  tuieval selftest            check every test's reference and wrong answers against its grader
  tuieval items               tests that don't separate models or look broken
  tuieval capture LOG --pack  turn a real failure (logs/live/*.txt) into a new test
  tuieval machines            this machine, other recorded machines, and fit and tuning per model
  tuieval tune MODEL          find the fastest speed-only server flags for a model on this machine
  tuieval export pi MODEL     serve a model in the pi coding agent with its tuned flags
  tuieval watch               a proxy that shows the reasoning of any app using your server
  tuieval help                this help
  tuieval --version

The workspace is --workspace DIR, else $TUIEVAL_HOME, else the current folder.
"""
import os
import sys

from . import __version__
from . import workspace

NEEDS_NO_WORKSPACE = {"init", "watch", "help", "-h", "--help", "--version", "-V"}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("--workspace", "-w"):
        if len(argv) < 2:
            sys.exit("--workspace needs a folder")
        os.environ[workspace.ENV] = os.path.abspath(os.path.expanduser(argv[1]))
        argv = argv[2:]
    elif argv and argv[0].startswith("--workspace="):
        os.environ[workspace.ENV] = os.path.abspath(os.path.expanduser(argv[0].split("=", 1)[1]))
        argv = argv[1:]
    cmd, rest = (argv[0], argv[1:]) if argv and not argv[0].startswith("--") else ("", argv)
    if cmd in ("help", "-h", "--help") or (not cmd and rest[:1] in (["-h"], ["--help"])):
        print(__doc__.strip())
        return 0
    if "--version" in argv[:1] or "-V" in argv[:1]:
        print(f"tuieval {__version__}")
        return 0
    if cmd not in NEEDS_NO_WORKSPACE and cmd != "new-pack" and not workspace.is_workspace():
        sys.exit(f"{workspace.root()} isn't a tuieval workspace (no models.toml).\n"
                 "Create one here with `tuieval init`, or point to one with --workspace DIR or TUIEVAL_HOME.")
    if cmd == "":
        from . import tui
        return tui.main(rest)
    if cmd == "init":
        from . import scaffold
        return scaffold.cmd_init(rest)
    if cmd == "new-pack":
        from . import scaffold
        if not workspace.is_workspace():
            sys.exit(f"{workspace.root()} isn't a tuieval workspace; run `tuieval init` first")
        return scaffold.cmd_new_pack(rest)
    if cmd == "doctor":
        from . import doctor
        return doctor.cmd_doctor(rest)
    if cmd == "run":
        from . import run_evals
        return run_evals.main(rest)
    if cmd == "compare":
        from . import compare
        return compare.main(rest)
    if cmd == "watch":
        from . import watch_proxy
        return watch_proxy.main(rest)
    from . import run_evals
    if cmd in run_evals.COMMANDS:
        return run_evals.COMMANDS[cmd](rest)
    sys.exit(f"unknown command: {cmd} (try tuieval help)")


if __name__ == "__main__":
    sys.exit(main())
