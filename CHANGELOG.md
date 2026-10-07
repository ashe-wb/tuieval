# Changelog

## Unreleased

- Nvidia and AMD GPUs on Linux: VRAM from `nvidia-smi`, `amd-smi` or `rocm-smi` sizes models (`gpu_memory_gb` overrides it), runs record peak VRAM, and `tuieval doctor` shows the GPU.
- Results has four tabs: PTA index, Production readiness, Per question and Failures. The PTA table adds tok/s, TTFT and memory; Per question can show only where models disagree; the settings warning is one line. Scorecard, pairwise, history and test quality stay in `tuieval compare`, `history` and `items`.
- The run's ETA adds up each pack's answers left × that model's time per answer there (its history, then its own times once it has 5 answers), instead of extrapolating from the first few answers.
- Faster start on Macs: the GPU core count comes from `ioreg` instead of `system_profiler`, which took seconds on some Macs.
- A model a server won't serve (HTTP 401-404, e.g. OpenRouter's "No endpoints found") is never judged on it: those answers don't count, and the model stops at once instead of failing every question.
- PTA index: parsimony (tokens), time and accuracy per model as bars, with tokens and time shown as multiples of the best (2.4×): the first Results tab (where `r` lands), `tuieval pta` and the report. Server errors aren't answers, and models with few answers are left out unless picked.
- Run several models at a time: `parallel_models` per machine, **Models at a time** in the TUI, or `tuieval run --parallel N` (default 1). Models wait for memory to fit, and answers note what ran alongside them.
- More evals during a run: `n` opens Setup, and Start adds them to the run going (same tier) or queues them after it.
- A new workspace's `models.toml` is 23 lines: the llama, local and OpenRouter servers and the defaults are built in (docs/models.md lists them in full; a server defined in `models.toml` replaces the built-in one with that name). Existing workspaces keep their settings; built-in servers they don't define become available, and adding a model still prefers the workspace's own servers.
- Less to take in at first: until something has run, the setup screen leaves out context sizes, tuning state and the Tune key; Presets and Hide are off the footer (`?` lists every key); `tuieval help` groups commands into Start here, Day to day and Advanced.
- Results lead with a plain summary per model ("ready for Support; not decided yet for Coding → run Certify to finish"), in the TUI, `tuieval verdict` (with the command to run next) and the report; the statistics follow.
- First packs without hand-written YAML: `tuieval new-pack <name> --from questions.csv` (question and answer columns) and `--about "<topic>"` (writes a prompt for a strong model to draft the tests). A test can take its prompt from a file (`input_file`). `tuieval selftest` errors say how to fix them.
- `tuieval init` in a terminal is a guided setup: it finds models on servers already running (LM Studio, Ollama, llama-server, vLLM), adds the ones you pick, creates a starter pack and runs a Smoke check (`--yes` for no questions).
- `tuieval add` without a model lists the models on running servers to pick from; `tuieval add <id>` finds the server that has it and adds that server to models.toml. The TUI's add dialog lists them too, with rarely needed options folded away.
- `?` on any TUI screen explains what it's for, its keys, and the words tuieval uses (tiers, verdicts, critical failures, gates).
- The setup screen is ready on first open: the only pack and model are ticked, the tier is Smoke until something has run, and ticked boxes are clearly marked (unticked ones are empty).
- First runs fail fast with the fix instead of hanging: a server that isn't running fails in seconds (it used to wait up to 20 minutes), a missing `llama-server` says how to install it, and `tuieval run` with no models says how to add one.
- `tuieval doctor` checks servers, models, packs and API keys, and says what to fix.
- Run only some tests of a pack: `e` in the TUI, or `tuieval run --tests`. Runs add up; `--force` reruns just the picked tests. Presets keep the pick.
- `tuieval tune` scores options on projected full-length answers (`[tune] answer_tokens`, default 1024), so generation speed counts as in real runs.
- `tuieval tune` reports an option's memory cost instead of rejecting it; it rejects only on critical memory pressure or over `[tune] swap_limit_mb`.
- Warm-start tuning recognises fine-tunes whose GGUF describes the MTP draft layer differently (e.g. listing KV heads per layer) as the same model shape, so they start from an already-tuned sibling instead of tuning in full.
- The fit check (context sized from the GGUF header, llama.cpp's memory use) only applies to servers whose command takes `{ctx}`. Servers that size their own memory keep the model's `max_context` instead of an estimate that didn't apply to them; `fit_check = true|false` on a server overrides it.
- `tuieval export pi` has no default presets path any more: set `[export.pi] presets` to your llama.cpp router's `--models-preset` file.
- A new README screenshot from a demo workspace.
- Versions come from git tags: a tag `vX.Y.Z` is that release, and every commit on `main` after it is published automatically as a dev build `X.(Y+1).0.devN` (N = commits since the release). `pip install tuieval` keeps installing releases only; `pip install --pre tuieval` gets the latest build.

## 0.1.7 (replaces 0.1.6, withdrawn)

- `tuieval tune` starts warm for fine-tunes: when a model with the same architecture and tensor shapes is already tuned on this machine, its flags are the starting point and only speculative decoding and micro-batch are re-tried (~4-6 server starts instead of 8-15). It tunes in full if the inherited flags fail, change answers or are slower than the defaults. `--cold` (or `[tune] warm_start = false`) always tunes in full; `[tune] warm_retest` names the re-tried knobs.
- `tuieval export pi <model>` (and `tuieval tune --export-pi`) writes a model's serving settings to the pi coding agent: a llama.cpp router presets section and pi's models.json. It shows the diff, asks, and backs every file up. See docs/models.md.
- `tuieval list` lists the models, with hidden ones separately (`--packs` lists the packs).
- `tuieval remove <model>` / `--hidden` removes models: their models.toml entry, results, tuning profiles and hidden mark, archived in removed/.
- `tuieval --help` gives every command its own line.

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
