"""tool_call: did the model call the right tool with the right arguments (or rightly call none)?

Test fields:
  expect_tool:    {name: set_timer, arguments: {duration_seconds: 1200, label: pasta}}
                  numbers match within arg_tolerance (default 0); strings match case-insensitively
                  if either contains the other; lists must contain every expected item. Extra
                  arguments are allowed. Exactly one call is expected.
  expect_no_tool: true   -> any tool call fails
  clarify:        true   -> (with expect_no_tool) the reply must ask a question
Native tool calls are used; a JSON object {"name"|"tool": …, "arguments": {…}} in the text
counts too, for servers without tool-calling support.
Score: 1 all correct, 0.5 right tool with wrong arguments, 0 otherwise.
Critical: any action that wasn't asked for (a call when none was expected, the wrong tool, extra calls).
"""
import json
import re

from . import grader, result


def _calls_from_text(text):
    calls = []
    decoder, i = json.JSONDecoder(), 0
    text = re.sub(r"```(?:json)?", "", text)
    while (i := text.find("{", i)) != -1:
        try:
            obj, n = decoder.raw_decode(text[i:])
        except ValueError:
            i += 1
            continue
        if isinstance(obj, dict) and (obj.get("name") or obj.get("tool")) and isinstance(obj.get("arguments", {}), dict):
            calls.append({"name": obj.get("name") or obj.get("tool"), "arguments": obj.get("arguments", {})})
        i += n
    return calls


def _match(want, got, tol):
    if isinstance(want, bool) or isinstance(got, bool):
        return want == got
    if isinstance(want, (int, float)):
        try:
            return abs(float(got) - want) <= tol
        except (TypeError, ValueError):
            return False
    if isinstance(want, list):
        got = got if isinstance(got, list) else [got]
        return all(any(_match(w, g, tol) for g in got) for w in want)
    w, g = str(want).lower().strip(), str(got).lower().strip()
    return bool(w) and (w in g or g in w) if g else False


@grader("tool_call", critical=True, template={"expect_tool": {"name": "TODO", "arguments": {}}})
def grade_tool_call(text, test, meta):
    calls = meta.get("tool_calls") or _calls_from_text(text)
    if test.get("expect_no_tool"):
        if calls:
            return result(False, f"called {calls[0]['name']}() but no tool call was expected", severity="critical")
        if test.get("clarify") and "?" not in text:
            return result(False, "should have asked a clarifying question")
        return result(True, "correctly answered without a tool" + (" and asked for details" if test.get("clarify") else ""))
    want = test["expect_tool"]
    if not calls:
        return result(False, f"no tool call; expected {want['name']}()")
    if len(calls) > 1:
        return result(False, f"{len(calls)} tool calls ({', '.join(c['name'] for c in calls)}); expected one", 0.25,
                      severity="critical")
    call = calls[0]
    if call["name"] != want["name"]:
        return result(False, f"called {call['name']}(), expected {want['name']}()", severity="critical")
    tol = float(test.get("arg_tolerance", 0))
    bad = [f"{k}={call['arguments'].get(k)!r} (want {v!r})" for k, v in (want.get("arguments") or {}).items()
           if not _match(v, call["arguments"].get(k), tol)]
    if bad:
        return result(False, f"{want['name']}() with wrong arguments: {'; '.join(bad)}", 0.5)
    return result(True, f"{want['name']}({json.dumps(call['arguments'])})")
