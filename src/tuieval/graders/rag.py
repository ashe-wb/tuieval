"""rag: answer grounded in the provided passages, with citations.

The reply must end with
    ANSWER: <answer or NOT_AVAILABLE>
    SOURCES: P2, P5            (or: SOURCES: none)
Test fields: expected / expected_text / tolerance (as in the answer grader) and
expected_sources: [P3]  — every one must be cited. For NOT_AVAILABLE no source is required.
Score: 1 answer and sources right, 0.5 answer right but sources wrong, 0 otherwise.
"""
import re

from . import grader, result
from .answer import check, final_answer

SOURCES_RE = re.compile(r"SOURCES?:\s*([^\n]*)", re.IGNORECASE)


@grader("rag", template={"expected": "TODO", "expected_sources": ["TODO"]})
def grade_rag(text, test, meta):
    raw = final_answer(text)
    if raw is None:
        return result(False, "No 'ANSWER:' line found")
    ok, reason, severity = check(raw, test)
    if not ok:
        return result(False, reason, severity=severity)
    if str(test.get("expected")) == "NOT_AVAILABLE":
        return result(True, reason)
    found = SOURCES_RE.findall(text)
    cited = {s.upper() for s in re.findall(r"P\d+", found[-1], re.IGNORECASE)} if found else set()
    missing = [s for s in test.get("expected_sources", []) if s.upper() not in cited]
    if missing:
        return result(False, f"{reason}, but didn't cite {', '.join(missing)} (cited: {', '.join(sorted(cited)) or 'none'})", 0.5)
    return result(True, f"{reason}; cited {', '.join(sorted(cited))}")
