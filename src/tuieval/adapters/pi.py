"""The built-in adapter for pi (https://github.com/earendil-works/pi), for agent packs: runs bare pi on the
model tuieval serves.

    [agents.pi]          cmd = ["{tuieval_python}", "-m", "tuieval.adapters.pi"]              (built in)
    [agents.pi-scouts]   cmd = ["{tuieval_python}", "-m", "tuieval.adapters.pi", "--scouts"]  (built in)

pi gets a scratch config folder with one provider, "tuieval", pointing at TUIEVAL_BASE_URL (context size and
sampling from tuieval), and runs with no extensions, skills, context files, prompt templates, themes, MCP or saved
session, so your own pi setup never changes a result. One line per turn and tool call is printed for the live
view; the stats go to TUIEVAL_STATS_FILE.

On macOS pi runs inside a sandbox (sandbox-exec, see sandbox_profile): it can't read your home folder or the temp
folders, apart from the task's work folder, the adapter's own scratch folder and the pack's Python, and it can
write only those two folders. So a model can't look up hidden tests or reference fixes, or leave notes for the
next model. HOME and TMPDIR point into the scratch folder. TUIEVAL_NO_SANDBOX=1 turns the sandbox off (the run
line then says "sandbox OFF"); without sandbox-exec (other systems) there is none.

--scouts adds pi's example subagent extension (the one explicit -e; everything else stays off) and one agent,
"scout": read-only tools (read, grep, find, ls), the model tuieval serves at thinking "low", and SCOUT_PROMPT.
The lead gets one appended system line saying the tool exists (SCOUT_HINT); the task text is unchanged, and
whether to delegate is the lead's own choice. Scouts are child pi processes of the sandboxed pi, so the sandbox
covers them too, and they use the same scratch config. Their turns and tokens are added to the run's totals and
also written as subagent_* stats; the turn limit counts the lead's turns only.

Standard library only: it runs with tuieval's own Python, whatever the pack's Python is.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile

PI_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repeat_penalty")
# provider errors that are the server's fault, not the model's (tuieval asks the task again)
SERVER_ERRORS = ("ECONNREFUSED", "ECONNRESET", "socket hang up", "fetch failed", "502", "503",
                 "Internal Server Error", "Service Unavailable", "Bad Gateway")
SUBAGENT_STATS = ("subagent_calls", "subagent_tasks", "subagent_failed", "subagent_turns", "subagent_input_tokens",
                  "subagent_output_tokens")

SCOUT_HINT = ("You have a `subagent` tool with a read-only `scout` agent (read, grep, find, ls). When a task needs "
              "context from several independent parts of the code, you can send scouts in parallel to gather it "
              "first; edits, commands and tests stay with you.")
SCOUT_PROMPT = """You are a scout. Quickly investigate a codebase and return structured findings that another agent can use without re-reading everything.

Your output will be passed to an agent who has NOT seen the files you explored.

Thoroughness (infer from task, default medium):
- Quick: Targeted lookups, key files only
- Medium: Follow imports, read critical sections
- Thorough: Trace all dependencies, check tests/types

Strategy:
1. grep/find to locate relevant code
2. Read key sections (not entire files)
3. Identify types, interfaces, key functions
4. Note dependencies between files

Output format:

## Files Retrieved
List with exact line ranges:
1. `path/to/file.ts` (lines 10-50) - Description of what's here
2. `path/to/other.ts` (lines 100-150) - Description
3. ...

## Key Code
Critical types, interfaces, or functions:

```typescript
interface Example {
  // actual code from the files
}
```

```typescript
function keyFunction() {
  // actual implementation
}
```

## Architecture
Brief explanation of how the pieces connect.

## Start Here
Which file to look at first and why.
"""


def pi_version():
    try:
        out = subprocess.run(["pi", "--version"], capture_output=True, text=True, timeout=30).stdout.strip()
        return f"pi {out}" if out else "pi"
    except (OSError, subprocess.TimeoutExpired):
        return "pi"


def subagent_extension():
    """pi's example subagent extension, found from the pi on PATH (its package root holds examples/)."""
    d = os.path.dirname(os.path.realpath(shutil.which("pi") or "pi"))
    while d != os.path.dirname(d):
        pkg = os.path.join(d, "package.json")
        if os.path.isfile(pkg):
            try:
                name = json.load(open(pkg)).get("name", "")
            except ValueError:
                name = ""
            if name.endswith("/pi-coding-agent"):
                ext = os.path.join(d, "examples", "extensions", "subagent", "index.ts")
                if os.path.isfile(ext):
                    return ext
                break
        d = os.path.dirname(d)
    return None


def config(dest, sampling, model_id, scouts=False):
    """A pi config folder with one provider, "tuieval", serving model_id (and, for scouts, the scout agent)."""
    ctx = int(os.environ.get("TUIEVAL_CONTEXT") or 32768)
    params = {k: sampling[k] for k in SAMPLING_KEYS if k in sampling}
    if "enable_thinking" in sampling:
        params["chat_template_kwargs"] = {"enable_thinking": bool(sampling["enable_thinking"])}
    model = {"id": model_id, "name": model_id, "reasoning": True, "input": ["text"], "contextWindow": ctx,
             "maxTokens": min(int(sampling.get("max_tokens") or 16384), ctx), "samplingParams": params,
             # requests like tuieval's own: max_tokens (not max_completion_tokens), and reasoning_effort only
             # when the model sets one (pi would otherwise send its default level)
             "compat": {"maxTokensField": "max_tokens",
                        **({} if sampling.get("reasoning_effort") else {"supportsReasoningEffort": False})}}
    providers = {"tuieval": {"baseUrl": os.environ["TUIEVAL_BASE_URL"], "api": "openai-completions",
                             "apiKey": os.environ.get("TUIEVAL_API_KEY") or "none", "models": [model]}}
    os.makedirs(dest, mode=0o700, exist_ok=True)
    with open(os.path.join(dest, "models.json"), "w") as f:
        json.dump({"providers": providers}, f, indent=1)
    with open(os.path.join(dest, "settings.json"), "w") as f:
        json.dump({"quietStartup": True}, f)
    if scouts:
        os.makedirs(os.path.join(dest, "agents"), exist_ok=True)
        with open(os.path.join(dest, "agents", "scout.md"), "w") as f:
            f.write("---\nname: scout\ndescription: Fast codebase recon that returns compressed context for handoff "
                    f"to other agents\ntools: read, grep, find, ls\nmodel: tuieval/{model_id}:low\n---\n\n"
                    + SCOUT_PROMPT)


def sandbox_profile(work, scratch, python_dirs):
    """A macOS sandbox (sandbox-exec) profile for pi and everything it runs, so a model can't look up answers
    or leave notes for the next model:
      reads   nothing in your home folder, the temp folders or /Volumes, except the task's work folder, the
              adapter's own scratch folder (pi config, HOME, TMPDIR) and the pack's Python
      writes  only the work folder, the scratch folder and /dev
      escapes no Apple Events (Office, Terminal, System Events…), no LaunchServices (`open -a …`), no
              launchctl/at/crontab, no ssh to this Mac: each would run code outside the sandbox
    Everything else (the pi install, system libraries, the network to the model server) is allowed."""
    q = lambda p: json.dumps(os.path.realpath(p))   # SBPL strings use the same escapes as JSON
    home = os.path.realpath(os.path.expanduser("~"))
    hidden = [home, "/private/tmp", "/private/var/tmp", "/private/var/folders"]
    hidden += [p for p in ("/Volumes",) if os.path.isdir(p)]
    ok_rw = [work, scratch]
    rules = ["(version 1)", "(allow default)",
             "(deny file-read* " + " ".join(f"(subpath {q(p)})" for p in hidden) + ")",
             "(allow file-read* " + " ".join(f"(subpath {q(p)})" for p in ok_rw + python_dirs) + ")",
             "(allow file-read-metadata)",     # stat() on parent folders, which path lookups need; no contents
             "(deny file-write* (require-not (require-any "
             + " ".join(f"(subpath {q(p)})" for p in ok_rw) + ' (subpath "/dev"))))',
             "(deny appleevent-send)",
             '(deny mach-lookup (global-name "com.apple.coreservices.launchservicesd")'
             ' (global-name "com.apple.coreservices.quarantine-resolver"))',
             '(deny process-exec (literal "/bin/launchctl") (literal "/usr/bin/at") (literal "/usr/bin/crontab")'
             ' (literal "/usr/bin/open") (literal "/usr/bin/ssh") (literal "/usr/bin/osascript")'
             ' (literal "/usr/bin/shortcuts") (literal "/usr/bin/automator"))',
             '(deny network-outbound (remote tcp "localhost:22"))']
    return "\n".join(rules)


def python_dirs():
    """Folders the pack's Python needs to read: its venv (first on PATH, put there by tuieval) and the base
    install the venv points at."""
    dirs = []
    first = (os.environ.get("PATH") or "").split(os.pathsep)[0]
    venv = os.path.dirname(first)
    cfg = os.path.join(venv, "pyvenv.cfg")
    if os.path.isfile(cfg):
        dirs.append(venv)
        for line in open(cfg):
            k, _, v = line.partition("=")
            if k.strip() == "home" and v.strip():
                dirs.append(os.path.dirname(os.path.realpath(v.strip())))   # …/bin -> the install
    return dirs


def thinking(sampling):
    if sampling.get("enable_thinking") is False:
        return "off"
    effort = sampling.get("reasoning_effort")
    return effort if effort in PI_LEVELS else None


def short(v, n=140):
    v = " ".join(str(v).split())
    return v if len(v) <= n else v[:n - 1] + "…"


def add_subagents(stats, ev):
    """Count a finished subagent tool call (pi's example extension) into stats; returns its summary line."""
    rs = ((ev.get("result") or {}).get("details") or {}).get("results") or []
    stats["subagent_calls"] += 1
    turns = out = 0
    for r in rs:
        u = r.get("usage") or {}
        stats["subagent_tasks"] += 1
        stats["subagent_failed"] += int(r.get("exitCode") not in (0, None) or r.get("status") == "failed")
        tokens_in = (u.get("input") or 0) + (u.get("cacheRead") or 0)
        stats["subagent_turns"] += u.get("turns") or 0
        stats["subagent_input_tokens"] += tokens_in
        stats["subagent_output_tokens"] += u.get("output") or 0
        stats["input_tokens"] += tokens_in
        stats["output_tokens"] += u.get("output") or 0
        turns += u.get("turns") or 0
        out += u.get("output") or 0
    if ev.get("isError"):
        stats["tool_errors"] += 1
    return f"  subagents: {len(rs)} task(s), {turns} turns, {out} output tokens" + (" [error]" if ev.get("isError") else "")


def build_command(model_id, sampling, prompt, scouts, extension):
    cmd = ["pi", "--mode", "json", "--no-session", "--no-extensions", "--no-skills", "--no-context-files",
           "--no-prompt-templates", "--no-themes", "--no-mcp", "--no-approve", "--offline",
           "--model", "tuieval/" + model_id]
    if scouts:
        cmd += ["-e", extension, "--append-system-prompt", SCOUT_HINT]
    level = thinking(sampling)
    if level:
        cmd += ["--thinking", level]
    return cmd + ["--", prompt], level


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    scouts = "--scouts" in argv
    sampling = json.loads(os.environ.get("TUIEVAL_SAMPLING") or "{}")
    max_turns = int(os.environ.get("TUIEVAL_MAX_TURNS") or 40)
    prompt = open(os.environ["TUIEVAL_PROMPT_FILE"]).read()
    model_id = os.environ["TUIEVAL_MODEL_ID"]
    stats = {"agent": pi_version() + (" + scouts" if scouts else ""), "turns": 0, "input_tokens": 0,
             "output_tokens": 0, "tool_calls": 0, "tool_errors": 0, "ran_tests": False, "stop": "", "final_message": ""}
    extension = None
    if scouts:
        stats.update(dict.fromkeys(SUBAGENT_STATS, 0))
        extension = subagent_extension()
        if not extension:
            print("pi's examples/extensions/subagent/index.ts wasn't found next to the pi on PATH", flush=True)
            stats.update(stop="error", final_message="no subagent extension")
            save(stats)
            return 1
    tmp = tempfile.mkdtemp(prefix="pi-adapter-")
    cfg_dir = os.path.join(tmp, "pi-config")
    config(cfg_dir, sampling, model_id, scouts)
    cmd, level = build_command(model_id, sampling, prompt, scouts, extension)
    env = dict(os.environ, PI_CODING_AGENT_DIR=cfg_dir, PI_OFFLINE="1", PI_TELEMETRY="0")
    shown = " ".join(cmd[:-1])
    sandbox = os.environ.get("TUIEVAL_NO_SANDBOX") != "1" and shutil.which("sandbox-exec")
    if sandbox:
        fake_home, fake_tmp = os.path.join(tmp, "home"), os.path.join(tmp, "tmp")
        os.makedirs(fake_home)
        os.makedirs(fake_tmp)
        env.update(HOME=fake_home, TMPDIR=fake_tmp + "/", ZDOTDIR=fake_home)
        pi = shutil.which("pi") or "pi"
        cmd = ["sandbox-exec", "-p", sandbox_profile(os.getcwd(), tmp, python_dirs()), pi] + cmd[1:]
    print(f"$ {shown} <task>   (thinking {level or 'default'}, max {max_turns} turns, "
          f"sandbox {'on' if sandbox else 'OFF'})", flush=True)
    try:
        proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, start_new_session=True)
    except OSError as e:
        shutil.rmtree(tmp, ignore_errors=True)
        stats.update(stop="error", final_message=f"can't start pi: {e}")
        save(stats)
        return 1

    def stop_pi(*_):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    signal.signal(signal.SIGTERM, lambda *a: (stop_pi(), save(stats), sys.exit(1)))
    error = ""
    try:
        for raw in iter(proc.stdout.readline, b""):
            try:
                ev = json.loads(raw)
            except ValueError:
                continue
            kind = ev.get("type")
            if kind == "message_end" and ev.get("message", {}).get("role") == "assistant":
                m = ev["message"]
                u = m.get("usage") or {}
                stats["turns"] += 1
                stats["output_tokens"] += u.get("output") or 0
                stats["input_tokens"] += (u.get("input") or 0) + (u.get("cacheRead") or 0)
                text = "".join(c.get("text", "") for c in m.get("content", []) if c.get("type") == "text").strip()
                if text:
                    stats["final_message"] = text
                if m.get("errorMessage"):
                    error = str(m["errorMessage"])
                print(f"turn {stats['turns']}: {u.get('output') or 0} tokens, {m.get('stopReason')}"
                      + (f": {short(text)}" if text else "") + (f" [error: {short(error)}]" if error else ""),
                      flush=True)
                if stats["turns"] >= max_turns and m.get("stopReason") == "toolUse":
                    stats["stop"] = "turn limit"
                    print(f"turn limit ({max_turns}) reached", flush=True)
                    break
            elif kind == "tool_execution_start":
                stats["tool_calls"] += 1
                args = ev.get("args") or {}
                what = args.get("command") or args.get("path") or json.dumps(args)
                if "pytest" in json.dumps(args) or "unittest" in json.dumps(args):
                    stats["ran_tests"] = True
                print(f"  {ev.get('toolName')}: {short(what)}", flush=True)
            elif kind == "tool_execution_end" and ev.get("toolName") == "subagent" and scouts:
                print(add_subagents(stats, ev), flush=True)
            elif kind == "tool_execution_end" and ev.get("isError"):
                if ev.get("toolName") != "bash":    # a failed edit or a bad call, not a failing command
                    stats["tool_errors"] += 1
                    out = " ".join(c.get("text", "") for c in (ev.get("result") or {}).get("content", []))
                    print(f"  ! {ev.get('toolName')} failed: {short(out)}", flush=True)
            elif kind == "agent_settled":
                stats["stop"] = stats["stop"] or ("error" if ev.get("aborted") or error else "done")
                break
    finally:
        stop_pi()
        proc.wait()
        err = proc.stderr.read().decode("utf-8", "replace").strip()
        if err:
            print("pi stderr: " + short(err, 600), flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
    if not stats["stop"]:
        stats["stop"] = "error"
        error = error or f"pi exited ({proc.returncode})"
    if error and any(s in error for s in SERVER_ERRORS):
        stats["server_error"] = error     # the server failed, not the model: tuieval asks the task again
    save(stats)
    return 0


def save(stats):
    with open(os.environ["TUIEVAL_STATS_FILE"], "w") as f:
        json.dump(stats, f)


if __name__ == "__main__":
    sys.exit(main())
