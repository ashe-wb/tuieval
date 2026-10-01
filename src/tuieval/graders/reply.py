"""reply: checks a free-form reply against simple rules (made for spoken assistant replies).

Test fields (all optional; each is one check):
  max_words: 30            no_markdown: true         ends_with_question: true
  must_include: ["12", "twelve|12"]   every item required; "a|b" means either
  must_not_include: ["as an AI"]      must_match: "regex"
Score = fraction of checks passed; pass = all passed.
"""
import re

from . import grader, result

MARKDOWN_RE = re.compile(r"(^\s*[-*+]\s|^\s*\d+\.\s|^#{1,6}\s|\*\*|__|```|^\s*\|.*\|\s*$)", re.MULTILINE)


@grader("reply", template={"must_include": ["TODO"]})
def grade_reply(text, test, meta):
    checks = []
    words = len(re.findall(r"\b[\w'’-]+\b", text))
    if "max_words" in test:
        checks.append((words <= test["max_words"], f"{words} words (max {test['max_words']})"))
    if test.get("no_markdown"):
        m = MARKDOWN_RE.search(text)
        checks.append((m is None, "no markdown" if m is None else f"markdown found: {m.group(0).strip()!r}"))
    low = text.lower()
    for item in test.get("must_include", []):
        alts = [a.strip().lower() for a in str(item).split("|")]
        hit = any(a in low for a in alts)
        checks.append((hit, f"mentions {item!r}" if hit else f"missing {item!r}"))
    for item in test.get("must_not_include", []):
        hit = str(item).lower() in low
        checks.append((not hit, f"avoids {item!r}" if not hit else f"contains {item!r}"))
    if test.get("must_match"):
        hit = re.search(test["must_match"], text, re.IGNORECASE | re.DOTALL) is not None
        checks.append((hit, f"matches /{test['must_match']}/" if hit else f"doesn't match /{test['must_match']}/"))
    if test.get("ends_with_question"):
        hit = text.rstrip().rstrip('"”').endswith("?")
        checks.append((hit, "ends with a question" if hit else "doesn't end with a question"))
    if not checks:
        return result(True, "no checks defined")
    passed = sum(ok for ok, _ in checks)
    failed = [why for ok, why in checks if not ok]
    return result(passed == len(checks), "; ".join(failed) if failed else "; ".join(why for _, why in checks),
                  passed / len(checks))
