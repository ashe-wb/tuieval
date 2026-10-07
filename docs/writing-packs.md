# Writing eval packs

tuieval ships with no packs: you write the questions that matter for what you use models for. A pack is a folder in your workspace's `packs/` with a `pack.toml` and one or more test files. Every folder there with a `pack.toml` shows up in the TUI. Delete or rename a folder (prefix it with `_` to hide it) and it's gone.

The quickest start is a starter template with example tests that already pass `tuieval selftest`:

```bash
tuieval new-pack support-bot --grader reply      # answer | rag | reply | tool_call | code
```

Edit `packs/support-bot/tests.yaml`, replace the examples with your own questions, and run `tuieval selftest support-bot` after every change.

Two shortcuts to a first set of real questions:

- **From a spreadsheet:** `tuieval new-pack quiz --from questions.csv` makes one test per row from a `question` and an `answer` column (optional: `id`, `category`, `difficulty`, `tolerance`, and `wrong` with known-wrong answers separated by `|`). Numbers are checked with the tolerance, words case-insensitively, and `NOT_AVAILABLE` means the right answer is "the question doesn't say".
- **Drafted by a model:** `tuieval new-pack support-bot --grader reply --about "refund questions for a shoe store"` also writes `DRAFT-PROMPT.md`. Paste it into the strongest model you have, save the YAML it writes as `tests.yaml`, and run `tuieval selftest`: it catches drafted tests whose expected answer is wrong or whose grader can't tell a mistake from the right answer. Read the questions anyway; you know your domain better than the model.

To try a different set of questions, copy a pack (`cp -r packs/support-bot packs/support-bot-v2`), edit the copy, and pick whichever you want in the TUI.

## Fingerprints

Every pack has a fingerprint of what the model sees and what decides pass/fail (questions, images, system prompt, tools, expected answers, grader). Results record it, so after you edit a question the TUI marks older results `~ outdated` and reruns them instead of comparing different question sets. Editing a description, category, difficulty, reference answer, wrong answers or gate doesn't make results outdated.

## pack.toml

```toml
label = "Support bot"            # shown in the TUI
group = "Customer support"       # the use case it counts toward (see below)
description = "…"
grader = "reply"                 # default grader for its tests (see Graders)
system = "system.txt"            # system prompt file (optional)
tools = "tools.yaml"             # OpenAI-style tool definitions sent with every request (optional)
needs = ["vision"]               # vision | tools | long_context | <python module>: skipped where unavailable
order = 30                       # position in the list
tests = ["tests.yaml"]           # optional; default: tests.* first, then every other .yaml/.csv file
screen = 20                      # tests in a Screen run (spread across categories, one variant per group)

[certify]
repeat = 3                       # repeats in a Certify run (default: models.toml [defaults] repeat)

[gate]                           # release criteria; see "Gates" below
min_accuracy = 0.90              # the 95% lower bound of the pass rate must reach this to PASS
max_critical_rate = 0.01         # needs 3/rate failure-free critical trials (rule of three): 300 here
max_p90_s = 30                   # 90th-percentile seconds per answer, judged per machine
max_truncation = 0.01            # share of answers cut off by max_tokens
min_consistency = 0.95           # share of variant groups where every variant passes
```

`group` doubles as the **use case** in the readiness verdicts: all packs in a group must PASS for the use case to PASS. For example, a "Coding" use case could be a `coding` pack plus a `coding-sql` pack.

`needs` entries other than `vision`, `tools` and `long_context` name Python modules the pack's grading needs (for example `pandas`, when your hidden tests use it). The pack is skipped, with a note, where that module isn't installed in tuieval's Python environment.

## Tests (YAML)

```yaml
- id: refund-window               # optional, must be unique in the pack (default: from description)
  description: refund window       # shown while running and in results
  category: policy                 # optional grouping (Screen runs spread across categories)
  difficulty: medium               # easy | medium | hard: your label, shown in the TUI, never sent to the model
  input: |                         # the user message
    A customer bought shoes 20 days ago …
  image: images/receipt.png        # optional: sent as an image (needs vision)
  input_file: prompts/refund.txt   # optional, instead of input: the user message from a file in the pack
  grader: answer                   # optional: override the pack's grader
  group: refund-window-2           # optional: variants of one case share a group (consistency gate)
  critical: true                   # optional: any failure of this test is critical (disqualifying)
  expected: 30                     # grader-specific fields from here on
  tolerance: 0
  reference: "…\nANSWER: 30"       # a correct model answer (never sent to the model)
  wrong: ["ANSWER: 14"]            # known-bad answers the grader must reject
```

**`reference` and `wrong` make a test check itself.** `tuieval selftest` grades every reference (it must pass with full score) and every wrong answer (it must fail). A test whose expected answer is wrong, or whose grader can't tell a classic mistake from the right answer, is caught before it fails a good model. Give every new test a reference, and a `wrong` entry for the mistake you most expect.

`selftest` also checks that no two tests send the same prompt, that tool tests expect tools the pack defines, that no `TODO` placeholders are left, that each request is one fresh single-turn conversation, and that the gate is reachable (see below).

## Tests (CSV)

For question-and-answer packs, a spreadsheet is often easier. Columns: `id, input, expected, tolerance, expected_text, category` (and any other test field). Empty cells are ignored.

```csv
id,input,expected,tolerance,expected_text,category
q1,"What is 15% of 80? End with ANSWER: <number>",12,0,,math
q2,"Capital of Peru? End with ANSWER: <city>",,,Lima,geo
```

## Graders

| grader | checks | test fields |
|---|---|---|
| `answer` | the last `ANSWER: …` line | `expected` (number or `NOT_AVAILABLE`) + `tolerance`, or `expected_text` (word, case-insensitive; a list means any of them) |
| `code` | last ```` ```python ```` block against hidden tests | `hidden_tests` with `# SETUP` / `# CHECK: name` blocks; score = fraction of checks passed |
| `rag` | answer grounded in passages, with citations | as `answer`, plus `expected_sources: [P3]`; half credit if the answer is right but the citation isn't |
| `tool_call` | the tool the model called | `expect_tool: {name, arguments}`, or `expect_no_tool: true` (+ `clarify: true`) |
| `reply` | a free-form reply | `max_words`, `no_markdown`, `must_include` (`"a\|b"` = either), `must_not_include`, `must_match`, `ends_with_question` |

⚠️ The `code` grader executes model-written code on your machine. Run code packs inside a container or VM with no credentials in the environment.

### Your own graders

A grader is a Python function registered by name. Put it in your workspace's `graders/` folder (one `.py` file each), or in a pack's own `grader.py` so the pack carries its grading with it:

```python
# graders/ticket_route.py
import json

from tuieval.graders import grader, result


@grader("ticket_route", template={"expected_queue": "TODO"})
def ticket_route(answer, test, meta):
    """The model replies with JSON like {"queue": "billing"}. Security-report tests are marked
    `critical: true` in tests.yaml, so they count as critical trials."""
    try:
        queue = json.loads(answer).get("queue")
    except (ValueError, AttributeError):
        return result(False, "not a JSON object")
    if queue != test["expected_queue"]:
        # sending a security report anywhere but the security queue is disqualifying
        severe = test["expected_queue"] == "security"
        return result(False, f"routed to {queue!r}", severity="critical" if severe else None)
    return result(True, f"routed to {queue!r}")
```

- `answer` is the model's final answer (reasoning already removed); `test` is the test's fields; `meta` has `finish` and `tool_calls`.
- `result(ok, reason, score=None, severity=None)`: `score` defaults to 1.0 or 0.0; `severity="critical"` marks a disqualifying failure.
- `critical=True` makes every test of this grader a critical trial (any of its failures *can* be critical). Otherwise only tests with `critical: true` or `expected: NOT_AVAILABLE` are.
- `template` is the skeleton `tuieval capture` writes for a new test of this grader.

A grader file can also add **Scorecard columns** (`tuieval compare`), for numbers your use case cares about that pass/fail doesn't show:

```python
import json

from tuieval.graders import scorecard_column


@scorecard_column("routing: json only")
def json_only(rows):
    """Share of this model's support-bot answers that are nothing but a JSON object."""
    mine = [r for r in rows if r["suite"] == "support-bot"]
    if not mine:
        return None                      # shown as "-"
    ok = 0
    for r in mine:
        try:
            ok += isinstance(json.loads(r["output"].strip()), dict)
        except ValueError:
            pass
    return f"{100 * ok / len(mine):.0f}%"
```

`rows` is one model's answers across all packs; each has `suite` (the pack's folder name), `test`, `ok`, `score`, `output` (the answer text), `truncated` and `repeat`.

Truncated and empty answers are failed before your grader runs. After changing a grader, `tuieval regrade` re-scores the stored answers without rerunning any model.

## Gates

A pack **PASSes** only on a full Certify run, when all of these hold:

- **zero critical failures**, over enough critical trials to show the critical rate is below `max_critical_rate` (rule of three: 300 clean trials for 1%, 60 for 5%)
- the **95% lower bound** of accuracy clears `min_accuracy`. Because it's the lower bound, the observed score must be higher, more so for small packs: with 30 trials, clearing an 80% bar takes about 29 correct.
- truncation, consistency and (per machine) p90 latency are within their budgets

`tuieval selftest` rejects a gate that a flawless certification run couldn't pass, and says what to add (tests, repeats, or a looser rate).

## What makes a failure critical

Graders mark failures that should disqualify a model whatever its accuracy:

- `answer`, `rag`: inventing an answer where the right one is `NOT_AVAILABLE`
- `tool_call`: an action nobody asked for (wrong tool, an unwanted or extra call)
- any test with `critical: true`
- whatever your own graders return with `severity="critical"`

## Difficulty labels

Every test has a `difficulty` of easy, medium or hard. It's for you, not the model: requests are built only from `input`, `image`, the system prompt and tools, so the label never reaches the model. The TUI shows it in the pack list (e.g. `12E 20M 8H`), next to the current question, in Recent results and Failures, and as a filter in Results → Per question.

A useful convention: **easy** = one rule or a direct read; **medium** = arithmetic, one trap, or two conditions; **hard** = rules in conflict, multi-step reasoning, or misleading data. Once several models have run, `tuieval items` flags labels the results contradict ("easier than labelled": a hard test every model passes; "harder than labelled": an easy test most models fail). `tuieval selftest` warns about tests without a label.

## Writing tests that separate production-ready models

- **Spec every behaviour you check.** A correct answer written only from the prompt must pass.
- **Test the boundaries** (exactly at a limit, one past it) and the conflicts between rules, not only typical cases.
- **Write each case twice** in a different surface form (layout, wording, field order) with the same `group`: a production model must not flip its answer.
- **Put traps where real use has them:** missing data, injected instructions, misleading visuals.
- **Use invented names, products and numbers**, so a model can't score from memory.
- **Generate volume with a script.** Critical gates need hundreds of trials; a small generator with a fixed seed that writes `generated.yaml` (references and wrong answers included) scales far better than hand-writing. Rerun it rather than editing its output.
- **After a few models have run, `tuieval items`** lists tests nobody fails (no signal), everybody fails (check the test), weak models beat strong ones on (inverted), or one model flips on (flaky).
- **Turn real failures into tests:** `tuieval capture logs/live/<file> --pack <name>` writes a skeleton to `captured.yaml` with the model's bad answer under `wrong`; fill in the TODOs.
