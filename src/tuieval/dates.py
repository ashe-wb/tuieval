"""When each answer was given, and when each test was added to its pack.

Answers record `answered_at` (local time with its UTC offset). Older answers have none, but their
log file name starts with the time the answer finished (logs/live/20261014_143741_…), so they get it
from that.

A test's date is its `added` field when it has one (tests.yaml: `added: 2026-10-14`). Otherwise
tuieval notes the day it first sees the test, in test-dates.json at the workspace root. Tests that were
already there when that file was started get the day of their first answer, or the day the file was
started, as "on or before" dates: the day they were really written isn't known.
"""
import datetime
import glob
import json
import os
import re

FILE = "test-dates.json"
_LOG_STAMP = re.compile(r"(\d{8}_\d{6})_")
_HISTORY = re.compile(r"-\d{8}-\d{6}(\.partial\.jsonl|\.json)$")


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def answered_at(rec):
    """When an answer finished (an aware datetime), or None if its result doesn't say."""
    if rec.get("answered_at"):
        try:
            return datetime.datetime.fromisoformat(rec["answered_at"])
        except ValueError:
            return None
    m = _LOG_STAMP.match(rec.get("log") or "")
    if m:
        try:
            return datetime.datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").astimezone()
        except ValueError:
            return None
    return None


def day(dt):
    return dt.strftime("%Y-%m-%d") if dt else ""


def minute(dt):
    return dt.strftime("%Y-%m-%d %H:%M") if dt else ""


def span(dts):
    """'2026-10-14', or '2026-10-01 – 10-07' for answers given over several days."""
    dts = [d for d in dts if d]
    if not dts:
        return ""
    first, last = day(min(dts)), day(max(dts))
    if first == last:
        return first
    return f"{first} – {last[5:] if first[:4] == last[:4] else last}"


def _explicit(test):
    v = test.get("added")
    if isinstance(v, datetime.datetime):
        return v.date().isoformat()
    if isinstance(v, datetime.date):
        return v.isoformat()
    if isinstance(v, str):
        try:
            return datetime.date.fromisoformat(v.strip()).isoformat()
        except ValueError:
            return None
    return None


def first_answers(results_dir):
    """{(pack, test id): the earliest day it was answered} over every result, history and archive."""
    first = {}

    def note(pack, rec):
        dt = answered_at(rec)
        if pack and dt and rec.get("test"):
            key, d = (pack, rec["test"]), day(dt)
            if key not in first or d < first[key]:
                first[key] = d
    for path in glob.glob(os.path.join(glob.escape(results_dir), "**", "*.json*"), recursive=True):
        base = os.path.basename(path)
        try:
            if base.endswith(".partial.jsonl"):
                pack = _HISTORY.sub("", base) if _HISTORY.search(base) else base[:-len(".json.partial.jsonl")]
                with open(path) as f:
                    for line in f:
                        rec = json.loads(line)
                        if "test" in rec:
                            note(pack, rec)
            elif base.endswith(".json"):
                with open(path) as f:
                    data = json.load(f)
                if isinstance(data, dict) and data.get("format") == "evals/1":
                    for rec in data.get("results", []):
                        note(data["pack"]["name"], rec)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return first


class AddedDates:
    """Each test's added date: {(pack, test id): (YYYY-MM-DD, exact)}. Not exact = on or before."""

    def __init__(self, root, packs, results_dir):
        self.path = os.path.join(root, FILE)
        try:
            with open(self.path) as f:
                saved = json.load(f).get("tests", {})
            started = True
        except (OSError, ValueError, AttributeError):
            saved, started = {}, False
        self.dates = {}
        missing = []
        for name, pack in packs.items():
            for t in pack.tests:
                key = (name, t["id"])
                given = _explicit(t)
                if given:
                    self.dates[key] = (given, True)
                elif f"{name}/{t['id']}" in saved:
                    s = saved[f"{name}/{t['id']}"]
                    self.dates[key] = (s["added"], s.get("how") == "seen")
                else:
                    missing.append(key)
        if not missing:
            return
        today = datetime.date.today().isoformat()
        first = first_answers(results_dir)
        for key in missing:
            if key in first:           # answered before tuieval noted it: on or before its first answer
                saved[f"{key[0]}/{key[1]}"] = {"added": first[key], "how": "first answer"}
            elif started:              # new since the file was started
                saved[f"{key[0]}/{key[1]}"] = {"added": today, "how": "seen"}
            else:                      # already there when the file was started
                saved[f"{key[0]}/{key[1]}"] = {"added": today, "how": "already there"}
            s = saved[f"{key[0]}/{key[1]}"]
            self.dates[key] = (s["added"], s["how"] == "seen")
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"about": "The day tuieval first saw each test (how: seen), or for tests it found "
                                    "already there, a day on or before it. A test's own `added` field wins.",
                           "tests": dict(sorted(saved.items()))}, f, indent=1)
                f.write("\n")
            os.replace(tmp, self.path)
        except OSError:
            pass                       # a read-only workspace: the dates still show, they just aren't kept

    def get(self, pack, test_id):
        """(YYYY-MM-DD, exact) or None."""
        return self.dates.get((pack, test_id))

    def text(self, pack, test_id):
        """'2026-10-14', '≤ 2026-10-01' (on or before), or ''."""
        got = self.get(pack, test_id)
        if not got:
            return ""
        return got[0] if got[1] else f"≤ {got[0]}"
