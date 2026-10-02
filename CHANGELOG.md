# Changelog

## 0.1.5

- The tuieval icon (the Scorecard Mosaic), a README header and a social preview; all brand files are in docs/images/brand/.

## 0.1.4

- Pressing `r` during a run always opens all your results. During a Smoke run, a button switches to that run's own results (kept apart in results/smoke/) and back.
- The readiness help text no longer sticks on the Smoke note after viewing Smoke results.

## 0.1.3

- Graders can add their own Scorecard columns with `@scorecard_column("title")` (see docs/writing-packs.md).

## 0.1.2

- A model that fails to start raises one notice (the reason, the packs it skipped, and its server log) instead of one per pack.
- Server errors show the specific error a server logged, not the generic line that follows it.
- Tabs in the TUI look like tabs: shaded chips on their own band, with the active tab highlighted.
- README: install with uv too; release steps moved to RELEASING.md; the Homebrew template was removed.

## 0.1.1

- Package description and README: tuieval evaluates local and frontier (OpenRouter) models alike, and explains where eval questions come from.

## 0.1.0

First public release.

- TUI and CLI for evaluating local models on your own eval packs: accuracy, speed and token use, with PASS / FAIL / INCONCLUSIVE verdicts per use case.
- Workspaces (`tuieval init`, `--workspace`, `TUIEVAL_HOME`) keep your packs, models and results apart from the tool.
- `tuieval new-pack` starter templates for the built-in graders: answer, rag, reply, tool_call, code.
- Custom graders from the workspace's `graders/` folder or a pack's own `grader.py`.
- Servers: llama.cpp (with per-machine fit check and speed tuning), any OpenAI-compatible server, OpenRouter (pinned to one provider endpoint).
- Speed tuning uses a built-in workload, so it needs no packs.
