# Models, servers and machines

Everything about models lives in your workspace's `models.toml`. `tuieval init` writes a short one; three servers are built in and ready to use (their full definitions are at the end of this page, under [Built-in servers and defaults](#built-in-servers-and-defaults)):

| server | what it is |
|---|---|
| `llama` | [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`, started by tuieval for each GGUF model, with a per-machine fit check and speed tuning |
| `local` | any OpenAI-compatible server that's already running (LM Studio, Ollama, vLLM, a remote box): just a `url` |
| `openrouter` | any model on [OpenRouter](https://openrouter.ai/models), named at run time, pinned to one provider endpoint |

## Adding models

```bash
tuieval add ~/models/Some-Model-Q4_K_M.gguf --tags 9b,dense,q4      # GGUF -> llama
tuieval add ~/models/Some-Model-Q4_K_M.gguf --no-think              # same model, thinking off (label …-nothink)
tuieval add ~/models/VL-Q4.gguf --mmproj ~/models/VL-mmproj.gguf    # vision model
tuieval add                                                          # pick from the models on servers already running
tuieval add qwen3:8b                                                 # finds the running server that has it
tuieval scan --add                                                   # every new GGUF under model_dirs
tuieval list                                                         # models; hidden ones listed separately
tuieval remove --hidden                                              # clean up every hidden model
```

`tuieval remove <model>` (or `--hidden` for every model hidden with `x` in the TUI) takes a model out of the workspace: its `models.toml` block (with the comment right above it), its results, smoke results and tuning profiles, and its hidden mark. It shows what it will do and asks first. Nothing is deleted: everything moves to `removed/<model>-<time>/`, with the `models.toml` block saved as `models.toml-entry.txt`, so you can put it back by hand.

In the TUI, `a` adds a model and `m` scans your model folders. Each model is one `[[models]]` block:

| field | meaning |
|---|---|
| `server` | which `[servers.*]` block serves it |
| `model` | GGUF path (llama) or model id (other servers) |
| `label` | names `results/<label>/`; default: the file or repo name, lowercased. Never rename a label that has results. |
| `served_name` | the model name sent with each request (default: the label for llama, the model id elsewhere) |
| `vision`, `mmproj` | the model reads images; `mmproj` is llama's vision projector and implies `vision` |
| `thinking = false` | run with thinking off |
| `tags = [...]` | filter by them in the TUI and with `--tags` |
| `sampling = {...}` | override `[sampling]` for this model |
| `server_args = [...]` | extra server flags (treated as answer-changing) |
| `ctx` (llama) | cap the context; the fit check may lower it further per machine |
| `max_context` (other servers) | the context the server provides; packs whose prompts need more are skipped |
| `kv_type` (llama) | e.g. `"q8_0"`: a quantized KV cache, which makes it a different model (give it its own label) |
| `tools = false` | packs that need tool calling are skipped |

Results record the settings each model actually ran with, and the results screen notes any differences between models.

## OpenRouter

OpenRouter models need no block. Name any model from openrouter.ai/models when you run, e.g. `tuieval run --tier smoke --only openrouter:qwen/qwen3-235b-a22b-2507`. In the TUI, tick **+ openrouter: any model** at the bottom of the model list and pick from the live list (filter by typing; it shows context and price), or press `a` and type `openrouter:<model id>`. A wrong id is refused with the closest matches.

- Needs `OPENROUTER_API_KEY` exported in the shell that starts tuieval. A plain `tuieval run` never includes OpenRouter models, since they cost money.
- Vision, tool calling and context size come from OpenRouter's model list; packs the model can't do are skipped.
- **One provider per model.** Left alone, OpenRouter spreads requests over providers running different quantizations. `pin_endpoint = true` picks one endpoint per model (closest to the released weights first: bf16 > fp8 > undeclared > lower; then tool and sampling support, uptime and price), sends every request only there with no fallback, keeps it while it's offered, and records it with the results. Answers from any other provider aren't counted.
- Hosted providers cache shared prompt openings and that can't be switched off. It doesn't change answers, but those answers (⟲ in the run) are left out of TTFT and prompt-speed numbers.
- To keep providers that store or train on prompts (and could learn your tests) out entirely, add `request = { provider = { data_collection = "deny" } }` under `[servers.openrouter]`.

## Every answer is independent

Each request is a fresh single-turn conversation: the system prompt and one question, never an earlier answer. Repeats go round by round, each round asking the tests in its own fixed shuffled order, so a question never follows itself. The llama server is told not to reuse earlier prompts (`request = { cache_prompt = false }`). Each answer records the messages sent and any prompt tokens the server reports reusing, and the run warns if that's ever not 0. If a question gets the exact same long answer twice with sampling on, the run flags it as a probable cached response. `tuieval selftest` checks all of this.

## Answer-changing vs speed-only flags

The best server flags differ per model and per machine, so tuieval splits them by what they change:

| Changes **answers**: same everywhere, part of each result's fingerprint | Changes **only speed**: tuned per model per machine |
|---|---|
| model file and quant, KV-cache type, `--jinja`, `--reasoning-format`, sampling, mmproj (`cmd` in `models.toml`) | threads, batch sizes, flash attention, cache reuse (`perf` and `[servers.llama.tune]`) |

**Quality results travel.** Tuning never invalidates results, and a result from one machine counts on another as long as the answer-changing settings match. So you can Certify on your fastest machine and only tune (and optionally Screen) on the others: sync the workspace folder between them.

## Machines

- **Machine id** is detected automatically (e.g. `m3max-64gb`, `m2-16gb`, or `rtx4090-128gb` on a Linux machine with a discrete GPU; set `EVALS_MACHINE` to rename it). Every result records the machine, the server version and the exact speed flags used. `tuieval machines` lists this machine and every other machine that has run the evals (they record themselves in `tuning/`).
- **Fit check (llama):** before starting a GGUF, tuieval reads its header (layers, KV heads, hybrid attention layers) and picks the largest context that fits this machine's GPU memory (a Mac's unified memory, or a discrete GPU's VRAM), capped at `max_ctx`. A model that can't fit at 8k context is skipped with *doesn't fit on <machine>* instead of swapping. Packs that need more context than fits are skipped. It never quantizes the KV cache on its own, since that changes answers. Servers that size their own memory (their `cmd` doesn't take `{ctx}`) are left alone: their context is the model's `max_context`; `fit_check` on the server changes that.
- **Per-machine settings** go under `[machines.<id>]`: `memory_headroom_gb` (GPU memory kept free: default 4 on a Mac, 1.5 in a discrete GPU's VRAM), `gpu_memory_gb` (the GPU memory to size models against, when detection gets it wrong or you want to hold some back), `gpu_residency_gb` (see the stall guard) and `parallel_models` (see below).

### Discrete GPUs (Linux: Nvidia, AMD)

With `nvidia-smi` (Nvidia) or `amd-smi` / `rocm-smi` (AMD) installed, tuieval reads each card's name and VRAM. Several cards add up, since llama.cpp splits a model across them. `tuieval doctor` shows what it found.

- Models are sized to fit **entirely in VRAM** (servers keep `-ngl 99`); splitting a model between GPU and CPU isn't supported. A model that doesn't fit is skipped with what it needs and what's free.
- **Memory** in results is the server's peak VRAM: its own processes' on Nvidia, the whole machine's on AMD (its tools don't report it per process reliably, so it's only exact when nothing else uses the GPU).
- `tuieval tune` rejects flags that leave VRAM nearly full. The macOS memory-pressure checks and the stall guard don't apply.
- A machine that recorded itself before its GPU was detected keeps that id, so its results and tuning still match; `tuieval doctor` names the new id if you want to switch (`EVALS_MACHINE`).
- Without the vendor tools, models are sized against system RAM, as on any other machine; set `gpu_memory_gb` instead.

Windows isn't supported (WSL2 works like Linux).

## Several models at a time

By default a run serves one model at a time. On a machine with room for more, set `parallel_models = 2` (or more) under `[machines.<id>]`, type a number in **Models at a time** on the TUI's setup screen (blank uses the machine's setting), or pass `tuieval run --parallel N`.

- Models start in queue order, each in its own lane. The next one waits while its fit-check memory estimate wouldn't fit next to the ones running (without one, e.g. a server that sizes its own memory, the setting decides). Contexts are never shrunk to make room.
- A second model on the same server gets a free port, so a server's `cmd` must take `{port}`.
- A server with `before_start` (which may stop other servers) always runs on its own. Tuning is always one model at a time.
- Answers are judged exactly as in a run of one model. Each answer records the models served while it ran (`ran_alongside`), and a load time measured next to others is marked (`loaded_alongside`), since models side by side share the machine's GPU and memory bandwidth. The speed table and the answer view show it.
- In the TUI, one model's answers stream at a time: `v` switches to the next, `k` skips the one streaming. `tuieval run` prints one line per answer instead of streaming.
- Only one run uses the machine at a time across tuieval windows; models at a time applies within a run.

## PTA index

The **PTA index** puts privacy, time and accuracy side by side, 0-100 each, as a radar triangle (Results → PTA index, `tuieval pta`, and a section in `tuieval report`). Each model is a triangle reaching each corner as far as its score: bigger means better all round. There's no combined number; verdicts and critical failures stay in Production readiness.

- **P privacy**: 100 when prompts stay on machines you control (tuieval starts the server, its `url` is this machine, or the server has `private = true`, e.g. your own box on the network); 0 for a hosted API.
- **T time**: the total time to answer every question compared (each question at its median over repeats), relative to the fastest model: `100 × fastest total ÷ this model's total`. Adding a faster model lowers the others.
- **A accuracy**: the share of answers to those questions that passed.

Models are compared on the questions all of them answered, so one that skipped some never looks faster. Pick the models and pack to compare in the TUI, or use `--only` and `--packs`.

## Tuning

`tuieval tune <model>` (or `t` on the setup screen, for the ticked models) finds the fastest speed flags for a model on this machine:

1. If `llama-bench` is installed, it sweeps threads, micro-batch and flash attention first (fast, no server starts).
2. Then it starts the real server with one knob changed at a time and times a fixed **built-in** workload (three short prompts, three medium ones and one ~8k-token prompt), so tuning needs no packs and speeds are comparable between workspaces. Options are scored on the **projected** time: each prompt's measured reading time plus a full-length answer (`[tune] answer_tokens`, default 1024) at its measured generation speed. The workload itself generates 128 tokens per prompt, but eval answers run to thousands, so generation speed counts as much as in real runs.
3. An **output guard** rejects any option that changes greedy answers beyond noise. Memory is a cost, not a reason to reject: an option that makes macOS push other apps' idle memory to swap can still win, and the result warns about it. An option is rejected only if macOS memory pressure turns critical while it runs, or, with `[tune] swap_limit_mb` set, if it swaps more than that beyond the defaults.

Expect 8–15 server starts, about 20–30 minutes for a 27B model, once per model per machine. The result is saved in `tuning/<machine>/<model>.toml` and used by every later run there. Models without a profile run with each knob's first option and show *untuned*. A profile is marked for retuning when the model file or server version changes.

**Fine-tunes start warm.** When another model on the same server is already tuned on this machine and its GGUF has the same architecture and tensor shapes (a fine-tune or another quant of the same base model), its flags are the starting point. After the defaults (still timed, as the answer guard's reference), the tuner checks the inherited flags and then re-tries only the knobs that depend on the weights: speculative decoding (a fine-tune may have retrained or dropped its MTP layers) and micro-batch (the quant mix shifts it). Expect ~4–6 server starts. The closest file size wins when several models qualify. If the inherited flags fail, change answers or are slower than the defaults, it tunes in full. The profile records which model it started from (`meta.warm_start`). `tuieval tune --cold` always tunes in full; in `models.toml`, `[tune] warm_start = false` turns warm starts off and `[tune] warm_retest = ["spec", "ubatch"]` names the knobs a warm start re-tries.

The knobs are `[servers.<name>.tune]` in `models.toml`: each knob is a list of options, each option a list of flags. Placeholders: `{p}` P-cores, `{p_minus_2}`, `{all}` all cores, `{gpu_safe_gb}` (the GPU residency limit minus 2 GB; also `_minus_1`, `_minus_2`, `_plus_1`).

## Exporting to the pi coding agent

`tuieval export pi <model>` makes pi serve a model the way the evals did: the same file, the output-affecting flags, the context from the fit check, the model's sampling, and the speed flags tuned on this machine. `tuieval tune <model> --export-pi` does it right after tuning.

It writes a `[<id>]` section in the model presets file of a llama.cpp router (`llama-server --models-preset <file>`; keys that equal its `[*]` section are left out) and an entry under pi's `llama` provider `modelOverrides` (name, context window, vision, reasoning). It exports models on llama servers.

It shows the diff and the notes first (untuned or outdated tuning, other preset sections that name missing files, a reasoning effort to pick in pi), asks, backs every file up as `<file>.bak-tuieval-<time>`, and refuses to write a file that changed in the meantime. `--dry-run` only shows; `--yes` doesn't ask. Restart the router afterwards so it reads the new presets.

The id pi sees defaults to the GGUF's name; set `pi_id` and `pi_name` on the model (or pass `--id`/`--name`). Paths and provider names come from `[export.pi]` in `models.toml`:

```toml
[export.pi]
presets = "/path/to/presets.ini"           # required: the router's --models-preset file
pi_models = "~/.pi/agent/models.json"       # pi's default
llama_provider = "llama"     # pi's provider name for the router
servers = ["llama"]          # models.toml servers the router can serve
```

## Speed verdicts

Speed is judged separately, per machine: the same model can be production-grade in quality and still too slow on a smaller machine. Results → Production readiness has a **Fast enough?** table: p90 seconds per answer against each pack's `max_p90_s`, for every machine. It's *measured* where the model ran, and otherwise *projected* from each answer's token counts and that machine's tuned speeds. `tuieval compare --machine <id>` shows the same on the command line.

## The stall guard (Apple Silicon)

On Apple Silicon Macs, once the system's GPU allocations pass about half of RAM, the GPU driver evicts and re-maps memory on every GPU job, the server spends 70–95% of its CPU in the kernel while the GPU idles, and servers that submit many small GPU jobs slow to a crawl. For servers with `stall_guard = true`, tuieval samples the server's own vs kernel CPU time and the system's GPU allocation every 5 s during runs and tuning. If more than 70% of its CPU goes to the kernel for a minute, the run stops with an explanation instead of crawling for hours. Finished answers are kept and resume next time. If your machine behaves differently, set `gpu_residency_gb` under `[machines.<id>]`.

## Server options

| option | meaning |
|---|---|
| `cmd` | command that starts the server (no `cmd` = an already-running server at `url`). Placeholders: `{model}`, `{served_name}`, `{port}`, `{mmproj}`, `{ctx}`, `{kv_type}`, `{root}` (the workspace), and the tune placeholders. An argument `env:NAME=value` sets an environment variable instead. |
| `url` | an already-running server's base URL |
| `private` | `true` for a server on a machine you control (e.g. your own box on the network): its models score 100 for privacy in the PTA index |
| `port` | the port `cmd` serves on |
| `cwd` | folder to start `cmd` in |
| `model_is_path` | `model` is a file or folder: check it exists before starting |
| `fit_check` | size the context from the GGUF header and skip models that don't fit (the fit check; it assumes llama.cpp's memory use). Default: on for servers whose `cmd` takes `{ctx}`, off for others, whose context is the model's `max_context` |
| `vision_args` | flags appended for models with `mmproj` |
| `kv_type`, `max_ctx` | llama defaults: KV-cache type, upper bound on context |
| `ctx_flag` | the server's context flag, so long packs are skipped if a configured context is too small |
| `request` | fields added to every request (e.g. `{ cache_prompt = false }`) |
| `perf` | speed-only flags always applied |
| `tune` | speed-only knobs for `tuieval tune` |
| `tune_objective` | `projected` (default: workload time with full-length answers), `total` (workload time as run) or `decode` (tokens/s after a warm-up pass) |
| `version_cmd` | prints the server version, recorded with every result |
| `health` | readiness path for servers without `/v1/models` (must return JSON with `model`) |
| `before_start` | a command run before starting the server (e.g. to free memory another process holds) |
| `env` | environment variables for the server |
| `log_facts` | `{name = "regex"}` read from the server's startup log and recorded with every run |
| `require_facts` | `{name = "regex"}` the startup log must show, or the run refuses to start (e.g. a setting that must be off for fair evals) |
| `stall_guard` | watch for GPU-driver stalls (see above) |
| `outputs_depend_on_machine` | answers depend on this machine's memory settings: results only count on the machine that produced them, and tuned flags are fingerprinted |
| `any_model`, `label_prefix` | any model the server lists can be named at run time as `<server>:<id>` (OpenRouter) |
| `pin_endpoint` | pin each model to one provider endpoint (OpenRouter) |
| `api_key_env` | environment variable holding the API key |
| `thinking_param` | `reasoning` to send `[sampling] enable_thinking` as OpenRouter's `reasoning.enabled` |
| `headers` | extra HTTP headers |

## Built-in servers and defaults

A workspace's `models.toml` only needs what differs from these. `[defaults]` fill in key by key. `[sampling]` applies only when `models.toml` has no `[sampling]` section at all, because sampling is part of every result's fingerprint. A server you define replaces the built-in one with the same name; to change one setting, copy the whole block into `models.toml` and edit it. `tuieval add` adds a `[servers.<name>]` block with just a `url` when it finds a model on a server that's already running.

```toml
[defaults] fill in key by key; [sampling] only when models.toml has no [sampling] at all (it's part of
# every result's fingerprint); a server here is replaced whole by one of the same name in models.toml.

[defaults]
repeat = 3
request_timeout_ms = 1800000   # per request; long thinking runs need it
ready_timeout_s = 1200         # how long a model may take to load before giving up
model_dirs = ["~/models"]      # where "Scan for models" looks for new GGUFs

# Sent with every request. A model may override it (see `thinking` / `sampling` below); results
# record the settings each model actually ran with, and the scorecard shows any difference.
[sampling]
temperature = 0.7
top_p = 0.95
top_k = 20
min_p = 0.0
max_tokens = 16384
enable_thinking = true         # test every model in the SAME mode

# llama.cpp's server (https://github.com/ggml-org/llama.cpp), started for each GGUF model.
[servers.llama]
# Output-affecting flags only. Anything here must be identical wherever results are compared,
# so it's part of each result's fingerprint. Speed-only flags live in `perf` and `tune` below.
cmd = [
  "llama-server",
  "-m", "{model}",
  "--alias", "{served_name}",
  "--host", "127.0.0.1", "--port", "{port}",
  "--jinja",
  "--metrics",
  "-np", "1",
  "-ctk", "{kv_type}", "-ctv", "{kv_type}",
  "-c", "{ctx}",
  "--reasoning-format", "auto",
]
port = 8080
model_is_path = true           # check the GGUF exists and fits before starting
vision_args = ["--mmproj", "{mmproj}"]   # appended only for models that set mmproj
kv_type = "f16"                # a model's own `kv_type` (e.g. "q8_0") makes it a different model entry
max_ctx = 65536                # upper bound; the fit check lowers it per machine (see machines.py)
version_cmd = ["llama-server", "--version"]   # recorded with every result
# Added to every request. cache_prompt = false: each answer is computed from scratch, so a repeat of
# the same question reuses nothing from the last time it was asked (and its TTFT is honest).
# Answers record any prompt tokens the server reused.
request = { cache_prompt = false }

# Speed-only flags, always applied (not tuned).
perf = ["-ngl", "99"]

# Speed-only knobs `tuieval tune` tries, one at a time from the best so far. The first option of
# each knob is the default for models not tuned on this machine yet. Placeholders: {p} P-cores,
# {p_minus_2}, {all} all cores. Options that stop the server from starting are skipped.
# Check the flags against `llama-server --help` for your build.
[servers.llama.tune]
threads = [["-t", "{p}", "-tb", "{p}"], ["-t", "{p_minus_2}", "-tb", "{p}"], ["-t", "{all}", "-tb", "{all}"]]
ubatch = [["-ub", "512"], ["-ub", "256"], ["-ub", "1024"]]
batch = [["-b", "2048"], ["-b", "4096"]]
flash_attn = [["-fa", "on"], ["-fa", "off"]]

# A server that's already running (LM Studio, Ollama, vLLM, a remote box): give its URL and no cmd.
# Its models are added with `tuieval add <model id> --server local`.
[servers.local]
url = "http://127.0.0.1:1234"   # LM Studio's default; Ollama is http://127.0.0.1:11434, vLLM :8000

# OpenRouter: ANY model it serves, with no [[models]] entry. Name it when you run:
#     tuieval run --tier smoke --only openrouter:qwen/qwen3-235b-a22b-2507
#     TUI: tick "+ openrouter: any model" in the model list, or press "a" and type openrouter:<model id>
# Results go to results/or-<id>/ and the model stays listed once it has results. A plain
# `tuieval run` never includes OpenRouter models (they cost money); name them.
# Needs OPENROUTER_API_KEY in the environment of the shell that starts tuieval.
# pin_endpoint: every answer of a model comes from one provider endpoint (the closest to the released
# weights), with no fallback, since providers run different quantizations.
# To keep providers that store or train on prompts (and could learn your tests) out entirely, add
#   request = { provider = { data_collection = "deny" } }
[servers.openrouter]
url = "https://openrouter.ai/api/v1"
any_model = true
label_prefix = "or"
pin_endpoint = true
api_key_env = "OPENROUTER_API_KEY"
thinking_param = "reasoning"     # [sampling] enable_thinking -> OpenRouter's reasoning.enabled
headers = { "X-Title" = "tuieval" }
```

`[defaults] detect_ports` (default `[1234, 11434, 8080, 8000]`, the usual LM Studio, Ollama, llama-server and vLLM ports) is where `tuieval add`, `tuieval init` and `tuieval doctor` look for servers that are already running. `TUIEVAL_DETECT_PORTS` (comma-separated; empty means none) overrides it.

Optional sections:

```toml
# Optional tuner settings.
# [tune]
# guard_similarity = 0.6       # greedy answers under a candidate flag must stay this similar to the defaults'
# request_timeout_s = 600      # a tuning request that takes longer fails that candidate
# warm_start = true            # start from a tuned model with the same architecture and shapes (a fine-tune)
# warm_retest = ["spec", "ubatch"]   # the knobs a warm start re-tries; the rest are inherited
# answer_tokens = 1024         # answer length options are scored for (projected objective)
# swap_limit_mb = 500          # reject options that swap more than this beyond the defaults (default: report only)

# Optional: llama-bench for the tuner's fast first stage (found on PATH as llama-bench otherwise).
# [bench]
# cmd = ["~/llama.cpp/build/bin/llama-bench"]

# Optional per-machine settings, by machine id (tuieval machines shows the ids).
# [machines.m3max-64gb]
# memory_headroom_gb = 3       # GPU memory kept free for macOS and other apps (default 4)
# gpu_residency_gb = 12        # GPU memory the driver keeps resident before churning (default: half of RAM)
# parallel_models = 2          # models a run serves at a time (default 1)
```

