"""answer: the final 'ANSWER: x' line, checked as a number (expected ± tolerance), a word
(expected_text, case-insensitive; a list means any of them), or NOT_AVAILABLE."""
import math
import re

from . import grader, result

ANSWER_RE = re.compile(r"ANSWER:\s*\**\s*([^\n*]+)", re.IGNORECASE)


def final_answer(text):
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    raw = matches[-1].strip().rstrip(".").strip()
    return raw


def _number(raw):
    token = raw.split()[0] if raw.split() else raw
    return float(token.replace(",", "").replace("%", "").replace("$", ""))


def check(raw, test):
    """(ok, reason, severity) for a raw answer string against the test's expectation.
    Inventing an answer where the data isn't there is critical."""
    if "expected_text" in test:
        wants = test["expected_text"] if isinstance(test["expected_text"], list) else [test["expected_text"]]
        got = raw.strip().strip("\"'`").upper()
        ok = any(got == str(w).upper() or got.split()[0:1] == [str(w).upper()] for w in wants)
        return ok, f"got {raw}, expected {' or '.join(map(str, wants))}", None
    expected = str(test["expected"])
    if expected == "NOT_AVAILABLE":
        ok = raw.split()[0].upper().strip(".") == "NOT_AVAILABLE" if raw.split() else False
        return ok, "Correctly declined" if ok else f"Invented an answer: {raw}", None if ok else "critical"
    try:
        got = _number(raw)
    except (ValueError, IndexError):
        return False, f"Unparseable answer: {raw}", None
    want, tol = float(expected), float(test.get("tolerance", 0.01))
    candidates = [got, abs(got)] if test.get("sign_agnostic") else [got]
    ok = any(math.isclose(c, want, abs_tol=tol) for c in candidates)
    return ok, f"got {got}, expected {want} (±{tol})", None


@grader("answer", template={"expected": "TODO", "tolerance": 0})
def grade_answer(text, test, meta):
    raw = final_answer(text)
    if raw is None:
        return result(False, "No 'ANSWER:' line found")
    ok, reason, severity = check(raw, test)
    return result(ok, reason, severity=severity)
