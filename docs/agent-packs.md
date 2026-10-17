# Agent packs

An ordinary pack sends one prompt and grades one answer. That ranks models fairly, but it can't show how a model does inside a coding agent: reading files, editing them, running the tests, reacting to failures and deciding when it's done. An **agent pack** tests that.

Each test is a small project with a task. tuieval serves the model as for any other pack, hands a scratch copy of the project to a command-line agent you choose, and grades what the agent leaves behind with tests. Results show up beside your other packs.

## How a task runs

1. **Copy the project.** The task's `repo/` is copied to a scratch folder and committed to git, so the agent's changes can be shown as a diff.
2. **Run the agent.** Its command runs in that folder. Every line it prints shows live in the run screen.
3. **Stop it at the time limit.** The agent is stopped at the task's time limit (`minutes`). It keeps to the turn limit (`max_turns`) itself.
4. **Grade the folder** with pytest. Two sets of tests run:
   - the task's **original** visible tests. If the agent edits or deletes them, it gains nothing, and the edit is recorded;
   - its **hidden** tests, which the agent never sees.

   The task passes when every test passes. Its score is the share of hidden tests passed, so a near miss is visible.

Server failures never count against the model. If the model's server stops, or the agent reports that the server failed, the task is asked again. If the server keeps failing, the pack stops, like any pack whose server keeps failing.

## The pack

```
packs/agentic-coding/
  pack.toml
  tasks/
    fix-fees/
      task.md          the instruction the agent gets
      task.toml        optional: description, category, difficulty, minutes, max_turns, visible_tests
      repo/            the project the agent works on, with its visible tests (tests/ by default)
      hidden/          tests the agent never sees, copied in only for grading
      solution/        optional: a reference fix, copied over repo/ (a .delete file lists files to remove)
      wrong/<name>/    optional: plausible wrong fixes that must fail
```

```toml
# pack.toml
label = "Agentic coding"
group = "Coding"
kind = "agent"            # task folders run by an agent
agent = "my-agent"        # optional: which [agents.<name>]; default: [defaults] agent
python = "agentic/.venv/bin/python"   # optional: the Python the tasks run with (relative to the workspace)
minutes = 15              # optional: default time limit per task
max_turns = 40            # optional: default turn limit per task
min_context = 32768       # optional: context a model needs for this pack (default 32768)

[certify]
repeat = 1
[gate]
min_accuracy = 0.0
```

```toml
# tasks/fix-fees/task.toml
description = "Fees counted twice"
category = "bug fix"
difficulty = "medium"
minutes = 15
max_turns = 40
```

- **Python:** the pack's Python must have pytest installed, plus whatever the tasks import. Without a `python` setting, tuieval's own interpreter is used (`pipx inject tuieval pytest` adds pytest to a pipx install).
- **Fingerprint:** it covers each task's `task.md`, its limits and every file under `repo/` and `hidden/`. Editing any of them reruns that pack. The description, category and difficulty are labels only.
- **Selftest:** `tuieval selftest <pack>` checks three things for every task: the untouched `repo/` fails, `solution/` passes, and every `wrong/<name>/` fails. A task that passes untouched, or a wrong fix that passes, can't tell good agents from bad ones.

## The agent

Define the agent in `models.toml`. Its command is any program; `{root}` is your workspace.

```toml
[agents.my-agent]
cmd = ["python3", "{root}/agents/my_agent.py"]

[defaults]
agent = "my-agent"
```

The agent's name and command are part of an agent pack's settings, so changing the command reruns the pack. The agent's version isn't: each answer records it, and `tuieval compare` marks a pack that ran on more than one version as MIXED.

Most agents need a small **adapter**: a script that reads what tuieval provides, starts the agent with it, and writes the stats file. pi, Aider, Codex CLI, OpenCode and similar agents can all be wrapped this way. The adapter gets everything from its environment:

| Variable | Contents |
|---|---|
| `TUIEVAL_BASE_URL` | OpenAI-compatible URL of the model (ends in `/v1`) |
| `TUIEVAL_MODEL_ID` | the model name to send |
| `TUIEVAL_API_KEY` | the API key for hosted models; empty for local ones |
| `TUIEVAL_PROMPT_FILE` | the task's instruction |
| `TUIEVAL_SAMPLING` | the model's sampling settings as JSON (temperature, top_p, max_tokens, reasoning_effort, …) |
| `TUIEVAL_CONTEXT` | the model's context size in tokens, if known |
| `TUIEVAL_MAX_TURNS` | the task's turn limit |
| `TUIEVAL_TASK` | the task's id |
| `TUIEVAL_STATS_FILE` | where to write the stats (below) |

- **Working folder:** the scratch copy of the task.
- **PATH:** starts with the pack's Python, so `python` and `pytest` work.
- **Exit code:** ignored. Only the tests decide.

**Stats file.** It's optional; without it, tuieval records only the time taken. Any of these keys may be written:

```json
{"agent": "my-agent 1.4.2", "turns": 14, "input_tokens": 182340, "output_tokens": 9120,
 "tool_calls": 22, "tool_errors": 1, "ran_tests": true, "stop": "done",
 "final_message": "Fixed: the ledger subtracted the fee twice."}
```

- **`tool_errors`:** count failed edits and malformed tool calls, not commands that exit nonzero.
- **`stop`:** `done`, `turn limit` or `error`.
- **`server_error`:** if the model's server failed rather than the model, write this key with the error text. The task isn't counted and is asked again.

A minimal adapter:

```python
#!/usr/bin/env python3
import json, os, subprocess

prompt = open(os.environ["TUIEVAL_PROMPT_FILE"]).read()
proc = subprocess.run(["my-agent", "--base-url", os.environ["TUIEVAL_BASE_URL"],
                       "--model", os.environ["TUIEVAL_MODEL_ID"], "--max-turns", os.environ["TUIEVAL_MAX_TURNS"],
                       "--print", prompt], capture_output=True, text=True)
print(proc.stdout)                      # shows in the run screen and the transcript
with open(os.environ["TUIEVAL_STATS_FILE"], "w") as f:
    json.dump({"agent": "my-agent", "stop": "done" if proc.returncode == 0 else "error",
               "final_message": proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""}, f)
```

For a live view in the run screen, print a line per turn and tool call as they happen rather than at the end.

## Results

Agent tasks appear wherever other answers do: Per question, Failures, the scorecard and the PTA index.

- **Answer view:**
  - the agent's stats: version, turns, tokens, tool errors, and whether it ran the tests;
  - its final message and its changes as a diff;
  - the full transcript, with the test output;
  - under **What's checked**, the visible and hidden tests.
- **`tuieval compare`:** adds an **Agent tasks** table per model and pack. It shows tasks passed, median minutes and turns, output tokens, tool errors, how often the tests were run, limits hit, tests edited, and the agent version.

`tuieval regrade` leaves agent results alone: they're graded when they run, so retest the pack instead.

## Safety

⚠️ The agent runs commands as you, with your permissions. It works in a scratch folder, but nothing stops it from reaching outside that folder. Run agent packs in a container or VM with no credentials in the environment. The hidden tests and reference fixes live in your pack folder, so an agent that searched your disk could find them. The transcript would show it.
