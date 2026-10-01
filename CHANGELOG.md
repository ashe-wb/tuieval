# Changelog

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
