"""tuieval init (a new workspace) and tuieval new-pack (a new pack from a starter template)."""
import argparse
import os
import re
import shutil
import sys

from . import workspace

TEMPLATES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
PACK_TEMPLATES = os.path.join(TEMPLATES, "packs")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

WORKSPACE_GITIGNORE = """\
# Written by tuieval. Keep results/ and tuning/ if you want to sync them between machines.
logs/
reports/
results/smoke/
*.partial.jsonl
*.sittings
"""


def graders_available():
    return sorted(d for d in os.listdir(PACK_TEMPLATES) if os.path.isdir(os.path.join(PACK_TEMPLATES, d)))


def init(folder):
    """Create a workspace in folder. Returns the files created (existing files are left alone)."""
    folder = os.path.abspath(os.path.expanduser(folder))
    os.makedirs(folder, exist_ok=True)
    created = []
    for name, src in (("models.toml", os.path.join(TEMPLATES, "models.toml")),):
        dst = os.path.join(folder, name)
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
            created.append(name)
    gitignore = os.path.join(folder, ".gitignore")
    if not os.path.exists(gitignore):
        with open(gitignore, "w") as f:
            f.write(WORKSPACE_GITIGNORE)
        created.append(".gitignore")
    for d in ("packs", "graders"):
        if not os.path.isdir(os.path.join(folder, d)):
            os.makedirs(os.path.join(folder, d))
            created.append(d + "/")
    return created


def new_pack(name, grader="answer", label=None, group=None, packs_dir=None):
    """Copy the starter template for a grader to packs/<name>/. Returns its path. Raises ValueError."""
    if not NAME_RE.match(name):
        raise ValueError(f"pack name {name!r} must be lowercase letters, digits, '.', '_' or '-'")
    if grader not in graders_available():
        raise ValueError(f"no starter template for grader {grader!r}; templates: {', '.join(graders_available())}")
    packs_dir = packs_dir or workspace.path("packs")
    dst = os.path.join(packs_dir, name)
    if os.path.exists(dst):
        raise ValueError(f"{dst} already exists")
    shutil.copytree(os.path.join(PACK_TEMPLATES, grader), dst)
    toml = os.path.join(dst, "pack.toml")
    with open(toml) as f:
        text = f.read()
    title = label or name.replace("-", " ").replace("_", " ").capitalize()
    text = text.replace("__LABEL__", title.replace('"', "'")).replace("__GROUP__", (group or title).replace('"', "'"))
    with open(toml, "w") as f:
        f.write(text)
    return dst


CSV_INPUT = ("question", "input", "prompt")
CSV_ANSWER = ("answer", "expected", "expected_answer")


def tests_from_csv(csv_path):
    """Answer-grader tests from a spreadsheet: a question column and an answer column, plus optional
    id, category, difficulty, tolerance and wrong (known-wrong answers separated by |). Raises ValueError."""
    import csv
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{csv_path} has no rows")
    cols = {c.strip().lower(): c for c in rows[0] if c}
    q = next((cols[c] for c in CSV_INPUT if c in cols), None)
    ans = next((cols[c] for c in CSV_ANSWER if c in cols), None)
    if not q or not ans:
        raise ValueError(f"{csv_path} needs a question column ({' or '.join(CSV_INPUT)}) and an answer column "
                         f"({' or '.join(CSV_ANSWER)}); it has: {', '.join(cols)}")
    get = lambda row, name: (row.get(cols[name]) or "").strip() if name in cols else ""  # noqa: E731
    tests = []
    for n, row in enumerate(rows, 1):
        question, answer = (row.get(q) or "").strip(), (row.get(ans) or "").strip()
        if not question:
            continue
        t = {"id": get(row, "id") or f"q{n}"}
        for field in ("category", "difficulty"):
            if get(row, field):
                t[field] = get(row, field).lower() if field == "difficulty" else get(row, field)
        t["input"] = question
        try:
            t["expected"] = int(answer) if answer.lstrip("-").isdigit() else float(answer)
            t["tolerance"] = float(get(row, "tolerance") or 0)
        except ValueError:
            t["expected"] = "NOT_AVAILABLE" if answer.upper() == "NOT_AVAILABLE" else answer
            if t["expected"] != "NOT_AVAILABLE":
                t["expected_text"] = t.pop("expected")
        t["reference"] = f"ANSWER: {answer}"
        wrong = [w.strip() for w in get(row, "wrong").split("|") if w.strip()]
        if wrong:
            t["wrong"] = [f"ANSWER: {w}" for w in wrong]
        tests.append(t)
    return tests


GRADER_RULES = {
    "answer": "Each question has one checkable answer: a number (with `expected` and `tolerance`) or a single word "
              "or name (`expected_text`). Include a few questions whose answer isn't in the question at all "
              "(`expected: NOT_AVAILABLE`), since inventing an answer is the mistake that matters most.",
    "rag": "Each test gives passages labelled [P1], [P2]… in the input and asks a question answered by one of them "
           "(`expected_sources`). Include questions the passages don't answer (`expected: NOT_AVAILABLE`).",
    "reply": "Each test is a message a user might send; the checks (`must_include`, `must_not_include`, `max_words` "
             "…) describe what a good reply must and mustn't do.",
    "tool_call": "Each test is a request; `expect_tool` is the one tool call (name and arguments) a good assistant "
                 "makes, or `expect_no_tool: true` when none should be made. Use only the tools in tools.yaml.",
    "code": "Each test asks for one Python function; `hidden_tests` holds `# SETUP` and `# CHECK: name` blocks of "
            "asserts that a correct implementation passes, including edge cases.",
}


def fit_gate(pack_dir, n_tests, repeat=3):
    """Lower a new pack's min_accuracy to what its few tests can reach (selftest rejects a gate no
    flawless run could pass), with a note to raise it as tests are added."""
    from . import verdict
    lo = int(verdict.wilson(n_tests * repeat, n_tests * repeat)[0] * 100) / 100
    toml = os.path.join(pack_dir, "pack.toml")
    with open(toml) as f:
        text = f.read()
    m = re.search(r"^min_accuracy = ([0-9.]+)", text, re.M)
    if m and lo < float(m.group(1)):
        text = text[:m.start()] + f"min_accuracy = {lo:.2f}              # what {n_tests} tests can reach; raise it as you add tests" \
            + text[text.index("\n", m.start()):]
        with open(toml, "w") as f:
            f.write(text)


def draft_prompt(pack_dir, grader, about):
    """A prompt to paste into a strong model so it drafts this pack's tests.yaml."""
    def read(name):
        p = os.path.join(pack_dir, name)
        return open(p).read().strip() if os.path.isfile(p) else ""
    tools = read("tools.yaml")
    return f"""Write evaluation tests for tuieval, a tool that checks whether AI models can be trusted with a task.

Topic: {about}

Write 20 tests as a YAML list, in exactly the format of the example below. Rules:
- {GRADER_RULES.get(grader, "Follow the example's fields.")}
- Every test is answerable from its input alone (plus the system prompt), and has one right answer.
- `reference`: a correct reply, written the way the system prompt asks.
- `wrong`: 1-3 plausible mistakes a weaker model would make (each must be judged wrong).
- `difficulty`: a mix of easy (one rule or a direct read), medium (one trap or two conditions) and hard
  (rules in conflict, several steps, misleading details).
- Write about a third of the cases twice, reworded, with the same `group`: a reliable model gets both right.
- Use realistic details from the topic, not textbook examples. Don't reuse the example's questions.
Output only the YAML, nothing else.

System prompt the model will see:
---
{read("system.txt")}
---
{"Tools the model has (tools.yaml):" + chr(10) + "---" + chr(10) + tools + chr(10) + "---" + chr(10) if tools else ""}
Example tests (format only):
---
{read("tests.yaml")}
---
"""


def cmd_init(argv):
    p = argparse.ArgumentParser(prog="tuieval init", description="Create a workspace: models.toml, packs/, graders/.")
    p.add_argument("folder", nargs="?", default=None, help="default: the workspace (current folder or --workspace)")
    p.add_argument("--yes", "-y", action="store_true",
                   help="guided setup without questions: add every running model, a starter pack, run Smoke")
    a = p.parse_args(argv)
    folder = os.path.abspath(os.path.expanduser(a.folder or workspace.root()))
    created = init(folder)
    shown = os.path.relpath(folder) if folder.startswith(os.getcwd()) else folder
    if created:
        print(f"workspace ready in {shown}: created {', '.join(created)}")
    else:
        print(f"{shown} is already a workspace; nothing changed")
    if a.yes or (sys.stdin.isatty() and sys.stdout.isatty()):
        from . import onboard
        onboard.guided_init(folder, a.yes)
        return
    cd = "" if folder == os.getcwd() else f"cd {shown}\n  "
    print(f"next:\n  {cd}tuieval new-pack my-first-pack       # a pack of example questions to edit\n"
          "  tuieval add ~/models/Some-Model.gguf   # or press a in the TUI\n"
          "  tuieval                                # open the TUI")


def cmd_new_pack(argv):
    p = argparse.ArgumentParser(prog="tuieval new-pack",
                                description="Create packs/<name>/ from a starter template with example tests.")
    p.add_argument("name", help="folder name, e.g. support-bot")
    p.add_argument("--grader", default="answer", choices=graders_available(),
                   help="how answers are scored (default: answer)")
    p.add_argument("--label", help="name shown in the TUI (default: from the folder name)")
    p.add_argument("--group", help="use case it counts toward (default: the label)")
    p.add_argument("--packs-dir", help="default: the workspace's packs/")
    p.add_argument("--from", dest="source", metavar="CSV",
                   help="make the tests from a spreadsheet with question and answer columns (answer grader)")
    p.add_argument("--about", metavar="TOPIC",
                   help="also write DRAFT-PROMPT.md: paste it into a strong model to draft tests on TOPIC")
    a = p.parse_args(argv)
    if a.source and a.grader != "answer":
        sys.exit("--from makes answer-grader tests (question and answer columns); leave out --grader")
    try:
        tests = tests_from_csv(a.source) if a.source else None
        path = new_pack(a.name, a.grader, a.label, a.group, a.packs_dir)
    except (ValueError, OSError) as e:
        sys.exit(str(e))
    shown = os.path.relpath(path)
    if tests is not None:
        from .yamlout import dump_tests
        dump_tests(tests, os.path.join(path, "tests.yaml"),
                   f"# Made from {os.path.basename(a.source)}. Add wrong answers (mistakes a weak model makes) and\n"
                   "# difficulty labels where you can; fields: docs/writing-packs.md\n\n")
        fit_gate(path, len(tests))
        print(f"created {shown}/ with {len(tests)} tests from {a.source}")
    else:
        print(f"created {shown}/ with example tests for the {a.grader} grader")
    if a.about:
        with open(os.path.join(path, "DRAFT-PROMPT.md"), "w") as f:
            f.write(draft_prompt(path, a.grader, a.about))
        print(f"wrote {shown}/DRAFT-PROMPT.md: paste it into your strongest model and save the YAML it writes as "
              f"{shown}/tests.yaml")
    print(("next:" if tests is not None else f"next: replace the examples in {shown}/tests.yaml with your own "
           "questions, then") + f"\n  tuieval selftest {a.name}     # checks every reference passes and every "
          "wrong answer fails\nguide: docs/writing-packs.md"
          + ("" if tests is not None or a.about else
             "\ntip: --from questions.csv makes tests from a spreadsheet; --about \"topic\" writes a prompt for a "
             "model to draft them"))
