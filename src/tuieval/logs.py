"""tuieval logs: follow the server log of the model running now, without naming it.

    tuieval logs              the newest server log, then every log that grows (a header says
                              which model each part is from), so it moves on with the run
    tuieval logs live         each answer (reasoning and answer, from logs/live/) as it finishes
    tuieval logs --list       every model's server log, newest first
    tuieval logs MODEL        one model's log (any unambiguous part of its label)

Read-only: it never touches servers or results. Only servers tuieval starts write a log here;
OpenRouter and servers that were already running don't.
"""
import argparse
import datetime
import os
import re
import sys
import time

from . import workspace

POLL_S = 0.5


def server_dir():
    return workspace.path("logs", "server")


def live_dir():
    return workspace.path("logs", "live")


def _files(d, ext):
    """{path: (size, mtime)} for the files in d with this extension (none if d doesn't exist)."""
    out = {}
    try:
        names = os.listdir(d)
    except OSError:
        return out
    for n in names:
        if n.endswith(ext):
            p = os.path.join(d, n)
            try:
                st = os.stat(p)
            except OSError:
                continue
            out[p] = (st.st_size, st.st_mtime)
    return out


def _label(path):
    return os.path.basename(path)[:-len(".log")]


def _tail_offset(path, lines):
    """Where the last `lines` lines of a file start."""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        end = pos = f.tell()
        found, chunk = 0, 8192
        while pos > 0 and found <= lines:
            pos = max(0, pos - chunk)
            f.seek(pos)
            found = f.read(end - pos).count(b"\n")
        if pos == 0 and found <= lines:
            return 0
        f.seek(pos)
        data = f.read(end - pos)
    cut = len(data)
    for _ in range(lines + 1):
        cut = data.rfind(b"\n", 0, cut)
    return pos + cut + 1


class Follower:
    """Follows server logs. `poll()` returns the text written since the last poll, with a
    `── label ──` header whenever it comes from another log than the text before it.
    only: follow just this log. Otherwise every log in the folder, including new ones."""

    def __init__(self, folder, lines=40, only=None):
        self.folder, self.only = folder, only
        self.offsets = {p: size for p, (size, _) in _files(folder, ".log").items()}
        self.last = None        # the log the previous text came from
        self.held = {}          # path -> size at the last poll, for a line still being written
        start = only or max(self.offsets, key=lambda p: os.path.getmtime(p), default=None)
        if start and start in self.offsets:
            self.offsets[start] = _tail_offset(start, lines)

    def poll(self):
        out = []
        files = _files(self.folder, ".log")
        if self.only:
            files = {p: v for p, v in files.items() if p == self.only}
        for path, (size, _) in sorted(files.items(), key=lambda kv: kv[1][1]):   # oldest change first
            off = self.offsets.setdefault(path, 0)
            if size < off:          # truncated or replaced: read it from the start
                off = 0
            if size == off:
                continue
            with open(path, "rb") as f:
                f.seek(off)
                data = f.read(size - off)
            cut = data.rfind(b"\n") + 1
            if not cut:             # no complete line yet: wait one poll for the rest of it
                if self.held.get(path) != size:
                    self.held[path] = size
                    continue
                cut = len(data)
            self.held.pop(path, None)
            self.offsets[path] = off + cut
            if path != self.last:
                out.append(f"── {_label(path)} ──\n")
                self.last = path
            out.append(data[:cut].decode("utf-8", "replace"))
        return "".join(out)


FOOTER = re.compile(r"^=== .*finish=.* ===$", re.M)


class LiveFollower:
    """New answers in logs/live/ as they're written: each one from its reasoning on (the prompt is
    in the file), under a header with the file name."""

    def __init__(self, folder, last=1):
        self.folder = folder
        files = sorted(_files(folder, ".txt").items(), key=lambda kv: kv[1][1])
        self.seen = {p for p, _ in files[:max(0, len(files) - last)]}

    def poll(self):
        out = []
        for path, (_, mtime) in sorted(_files(self.folder, ".txt").items(), key=lambda kv: kv[1][1]):
            if path in self.seen:
                continue
            try:
                with open(path, errors="replace") as f:
                    text = f.read()
            except OSError:
                continue
            if not FOOTER.search(text) and time.time() - mtime < 2:   # still being written
                continue
            self.seen.add(path)
            i = text.find("=== REASONING ===")
            out.append(f"── {os.path.basename(path)[:-4]} ──\n{text[i:] if i >= 0 else text}\n")
        return "".join(out)


def _ago(mtime):
    s = max(0, int(time.time() - mtime))
    if s < 60:
        return f"{s}s ago"
    if s < 3600:
        return f"{s // 60} min ago"
    if s < 86400:
        return f"{s // 3600} h ago"
    return datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")


def _size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def list_logs(folder):
    files = sorted(_files(folder, ".log").items(), key=lambda kv: -kv[1][1])
    if not files:
        return f"No server logs yet in {os.path.relpath(folder)}/."
    w = max(len(_label(p)) for p, _ in files)
    return "\n".join(f"{_label(p):<{w}}  {_size(size):>9}  {_ago(mtime)}" for p, (size, mtime) in files)


def find_log(folder, name):
    """The log whose label is `name`, or the only one containing it. Raises SystemExit otherwise."""
    labels = {_label(p): p for p in _files(folder, ".log")}
    if name in labels:
        return labels[name]
    hits = sorted(l for l in labels if name.lower() in l.lower())
    if len(hits) == 1:
        return labels[hits[0]]
    if not hits:
        sys.exit(f"No server log matches {name!r}; tuieval logs --list shows them.")
    sys.exit(f"{name!r} matches {len(hits)} logs; say which:\n  " + "\n  ".join(hits))


def follow(follower, waiting):
    """Print what the follower finds until ctrl-c."""
    said = False
    try:
        while True:
            text = follower.poll()
            if text:
                sys.stdout.write(text)
                sys.stdout.flush()
                said = True
            elif not said:
                print(waiting, flush=True)
                said = True
            time.sleep(POLL_S)
    except KeyboardInterrupt:
        return 0


def main(argv):
    p = argparse.ArgumentParser(prog="tuieval logs", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="ctrl-c stops following (a run going on isn't affected).")
    p.add_argument("what", nargs="?", help="'live' for answers, or a model (any unambiguous part of its label)")
    p.add_argument("--list", action="store_true", help="every model's server log, newest first")
    p.add_argument("-n", "--lines", type=int, default=40, help="lines shown before following (default 40)")
    a = p.parse_args(argv)
    if a.list:
        print(list_logs(server_dir()))
        return 0
    if a.what == "live":
        return follow(LiveFollower(live_dir()),
                      f"Waiting for answers in {os.path.relpath(live_dir())}/ (ctrl-c stops)")
    only = find_log(server_dir(), a.what) if a.what else None
    return follow(Follower(server_dir(), a.lines, only),
                  f"Waiting for {'that model' if only else 'a server'} to log in "
                  f"{os.path.relpath(server_dir())}/ (ctrl-c stops)")
