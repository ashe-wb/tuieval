"""code: extracts the last ```python block and runs it against the test's hidden_tests.

hidden_tests is split into a `# SETUP` block and `# CHECK: <name>` blocks; each check runs
separately, so the score is the fraction passed and failures name the check.
"""
import json
import os
import re
import secrets
import subprocess
import sys
import tempfile

from . import grader, result


def _extract_code(text):
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.DOTALL)
    return blocks[-1] if blocks else text  # last block = the final answer, not a draft


CHECK_RE = re.compile(r"^# (SETUP|CHECK: *(.+?)) *$")


def _split_checks(tests):
    """Split hidden_tests into (setup, [(name, code), ...]) on '# SETUP' / '# CHECK: name' lines.

    Tests without markers become a single check named 'all'.
    """
    setup, checks, current = [], [], None
    for line in tests.splitlines():
        m = CHECK_RE.match(line)
        if m and m.group(2):
            current = []
            checks.append((m.group(2), current))
        elif m:
            current = setup
        elif current is None:
            setup.append(line)
        else:
            current.append(line)
    if not checks:
        return "", [("all", "\n".join(setup))]
    return "\n".join(setup), [(name, "\n".join(lines)) for name, lines in checks]


HARNESS = """
import json, traceback
ns = {"__name__": "candidate"}
results = {}
def _err(e):
    last = traceback.format_exception_only(type(e), e)[-1].strip()
    return last[:200]
try:
    exec(compile(CANDIDATE, "candidate.py", "exec"), ns)
    exec(compile(SETUP, "setup.py", "exec"), ns)
    setup_error = None
except BaseException as e:
    setup_error = _err(e)
for name, code in CHECKS:
    if setup_error:
        results[name] = setup_error
        continue
    try:
        exec(compile(code, "check_" + name + ".py", "exec"), ns)
        results[name] = None
    except BaseException as e:
        results[name] = _err(e)
print(MARKER + json.dumps(results))
"""


@grader("code", template={"hidden_tests": "# CHECK: TODO\nassert False, 'TODO'\n"})
def grade_code(text, test, meta):
    """Runs the model's code, then each hidden check separately; score = fraction passed."""
    code = _extract_code(text)
    setup, checks = _split_checks(test["hidden_tests"])
    marker = "RESULTS_" + secrets.token_hex(8) + ":"
    program = (f"CANDIDATE = {code!r}\nSETUP = {setup!r}\nCHECKS = {checks!r}\n"
               f"MARKER = {marker!r}\n" + HARNESS)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "harness.py")
        with open(path, "w") as f:
            f.write(program)
        try:
            proc = subprocess.run(
                [sys.executable, path], cwd=tmp, capture_output=True, text=True, timeout=20
            )
        except subprocess.TimeoutExpired:
            return result(False, "Timed out after 20s", 0)
    lines = [l for l in proc.stdout.splitlines() if l.startswith(marker)]
    if not lines:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        return result(False, "Harness crashed: " + " | ".join(tail), 0)
    results = json.loads(lines[-1][len(marker):])
    failed = {k: v for k, v in results.items() if v is not None}
    score = (len(results) - len(failed)) / len(results)
    if not failed:
        out = result(True, f"All {len(results)} checks passed", 1)
    else:
        detail = "; ".join(f"{k}: {v}" for k, v in failed.items())
        out = result(False, f"Failed {len(failed)}/{len(results)} checks: {detail}", score)
    out["checks"] = results  # {check name: None if passed, else its error}; shown in the TUI's answer detail
    return out
