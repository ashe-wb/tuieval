"""Find the fastest speed-only server flags for one model on this machine, and save them as its
profile in tuning/<machine-id>/<label>.toml. Eval runs then serve the model with those flags.

  Stage 1 (only if llama-bench is installed): sweep threads, micro-batch and flash attention
          with llama-bench (fast; no server starts). Knobs it settles are not retried in stage 2.
  Stage 2 (always): start the real server with candidate flags and time a fixed workload of
          built-in prompts (WORKLOAD below; it needs no packs, so speeds compare across users and
          workspaces). One knob at a time from the best so far (coordinate descent), so a model
          takes ~8-15 server starts rather than a full grid.

Warm start: when another model on the same server is already tuned on this machine and its GGUF
has the same architecture and shapes (a fine-tune or another quant of the same base), its flags are
the starting point. Threads, batch sizes and flash attention depend on the shapes and the machine,
not on the trained weights, so they carry over; only the knobs that do depend on the weights are
re-tried ([tune] warm_retest, default speculative decoding and micro-batch: a fine-tune may have
retrained or dropped its MTP layers, and its quant mix shifts the best micro-batch). That takes
~4-6 server starts. If the inherited flags fail, change answers or are slower than the defaults,
it tunes in full instead. `tuieval tune --cold` (or [tune] warm_start = false) always tunes in full.

One objective per server (models.toml `tune_objective`):
  total   (default) total seconds for the workload: short prompts, medium ones and one long
          (~8k-token) prompt, fixed generation length
  decode  generated tokens per second, measured on a second pass over a few prompts after a
          warm-up pass (for servers whose decode speed grows as their caches warm)
An output guard compares greedy answers with the default flags; an option that changes them beyond
noise is rejected, because speed flags must not change answers. Servers whose answers depend on
the machine's memory settings (`outputs_depend_on_machine`) skip the guard; their tuned flags
become part of the results fingerprint instead, so retuning marks that machine's results
outdated. Any candidate that makes macOS swap is rejected. Servers without tune knobs are only
measured.

The knobs and their options come from models.toml [servers.<name>.tune]; placeholders {p},
{p_minus_2} and {all} are this machine's core counts.
"""
import dataclasses
import difflib
import json
import os
import shutil
import subprocess
import threading
import time

from . import client
from . import engine as engine_mod
from . import machines
from . import profiles

GEN_TOKENS = 128          # generated per workload prompt
DECODE_TOKENS = 256      # generated per prompt for the decode objective
SWAP_LIMIT = 256 * 2**20  # a candidate that swaps out more than this is rejected
GPU_MARGIN_GB = 0.75      # a candidate whose GPU allocation comes this close to the residency limit
                          # is rejected even before it stalls: servers grow as requests arrive
REQUEST_LIMIT_S = 600     # a tuning request taking longer fails the candidate ([tune] request_timeout_s)
PROGRESS_EVERY_S = 10     # how often a running request reports progress
LONG_PROMPT_CHARS = 32000  # ~8k tokens
BENCH_FLAGS = {"-t": "-t", "-ub": "-ub", "-fa": "-fa"}   # knob flags llama-bench can sweep
IGNORED_IN_BENCH = {"-tb"}
WARM_RETEST = ["spec", "ubatch"]   # knobs a warm start re-tries ([tune] warm_retest)


@dataclasses.dataclass
class Measure:
    total_s: float = 0.0
    ttft_s: float | None = None     # mean over the short prompts
    pp_tps: float | None = None     # prompt tokens per second (whole workload)
    tg_tps: float | None = None     # generated tokens per second (whole workload)
    texts: list = dataclasses.field(default_factory=list)
    load_s: float | None = None
    error: str = ""
    facts: dict = dataclasses.field(default_factory=dict)   # from the server log, e.g. expert capacity

    def score(self, objective):
        """Lower is better."""
        if objective == "decode":
            return 1 / self.tg_tps if self.tg_tps else float("inf")
        return self.total_s

    def summary(self):
        facts = "".join(f" · {k} {v}" for k, v in self.facts.items())
        return (f"{self.total_s:.1f}s total · gen {self.tg_tps} tok/s · prompt {self.pp_tps} tok/s"
                f" · ttft {self.ttft_s}s{facts}")


# ------------------------------------------------------------------ workload
SHORT_SYSTEM = "You are a helpful assistant. Answer briefly."
SHORT_PROMPTS = [
    "Turn on the kitchen lights and set them to 40 percent.",
    "What's a good name for a grey cat?",
    "Remind me to call the dentist tomorrow at nine.",
]
MEDIUM_SYSTEM = "You are a careful analyst. Think step by step, then give a short answer."
MEDIUM_PROMPTS = [
    "A warehouse ships 1,240 parcels on Monday, 15% more on Tuesday, and 230 fewer than Tuesday on "
    "Wednesday. Parcels heavier than 5 kg are 18% of each day's total and cost 2.40 to ship; the rest "
    "cost 1.10. Work out the total shipping cost for the three days, and say which day was the most "
    "expensive and by how much compared with the cheapest day.",
    "Here is a meeting schedule. Team A: Mon 9-11, Wed 14-16, Fri 10-12. Team B: Mon 10-12, Tue 9-10, "
    "Fri 11-13. Team C: Wed 15-17, Thu 9-11, Fri 9-10. A room fits one meeting at a time. List every "
    "clash (two teams in the room at once), and propose the smallest change that removes all clashes.",
    "Summarise the pros and cons of a relational database versus a document store for an app that keeps "
    "user profiles, an activity feed and monthly invoices. Cover consistency, schema changes, querying "
    "across records and operating cost, then recommend one with a one-sentence reason.",
]
DECODE_PROMPTS = [
    "Write a short story (about 200 words) about a lighthouse keeper who finds a message in a bottle.",
    "Explain how a hash table works to a new programmer, with a small example.",
    "Describe a week-long walking trip through a mountain region of your choice, day by day.",
]


def long_prompt(chars):
    """A ~chars-long synthetic document with one fact to find in it (the same text every time)."""
    words = ("river", "station", "ledger", "orchard", "signal", "harbour", "copper", "meadow", "lantern",
             "quarry", "window", "valley", "engine", "garden", "bridge", "tower")
    paras, i = [], 0
    while sum(len(p) for p in paras) < chars:
        w = [words[(i * 7 + k * 3) % len(words)] for k in range(12)]
        paras.append(f"Section {i + 1}. The {w[0]} near the {w[1]} kept a {w[2]} of every {w[3]}, and the "
                     f"{w[4]} by the {w[5]} was painted {w[6]} after the {w[7]} flooded. Visitors to the "
                     f"{w[8]} counted {(i * 37) % 900 + 100} steps to the {w[9]}, past a {w[10]} facing the "
                     f"{w[11]}.")
        i += 1
    paras.insert(len(paras) * 2 // 3, "Note: the archive room code is 4817.")
    return "\n\n".join(paras) + "\n\nWhat is the archive room code? Answer with the number only."


def workload(eng, ctx=None):
    """[(name, messages)]: 3 short prompts, 3 medium ones, and 1 long one sized to the context."""
    items = [(f"short-{i + 1}", [{"role": "system", "content": SHORT_SYSTEM}, {"role": "user", "content": p}])
             for i, p in enumerate(SHORT_PROMPTS)]
    items += [(f"medium-{i + 1}", [{"role": "system", "content": MEDIUM_SYSTEM}, {"role": "user", "content": p}])
              for i, p in enumerate(MEDIUM_PROMPTS)]
    limit = min(LONG_PROMPT_CHARS, ((ctx or 10**9) - GEN_TOKENS - 1024) * 3)
    if limit >= 4000:
        items.append(("long", [{"role": "user", "content": long_prompt(limit)}]))
    return items


def decode_workload(eng):
    """A few ordinary prompts for timing generation (no long prompt: prefill isn't measured)."""
    return [(f"decode-{i + 1}", [{"role": "user", "content": p}]) for i, p in enumerate(DECODE_PROMPTS)]


def _body(m, sampling, msgs, max_tokens=GEN_TOKENS, request=None):
    """request: the server's per-request fields (models.toml `request`), so tuning measures the
    same conditions as the evals (e.g. no prompt reuse on llama)."""
    return {"model": m["served_name"], "messages": msgs, "temperature": 0.0, "top_k": 1, "seed": 1,
            "max_tokens": max_tokens, "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": bool(sampling.get("enable_thinking", True))},
            **(request or {})}


def _request(eng, base_url, body, name, emit, limit):
    """One streamed request with live progress (every PROGRESS_EVERY_S: waiting for the first
    token, or ~tokens so far and tok/s) and a wall-clock limit. Returns client.stream_chat's result."""
    st, t0 = client.Stream(), time.time()
    state = {"chunks": 0, "first": None}   # streamed chunks: one token each on llama.cpp
    done = threading.Event()
    eng._stream = st   # so cancelling aborts the request at once

    def on_delta(kind, text):
        if state["first"] is None:
            state["first"] = time.time()
            emit("tune_progress", message=f"      first token after {state['first'] - t0:.1f}s")
        state["chunks"] += 1

    def report_progress():
        while not done.wait(PROGRESS_EVERY_S):
            now = time.time()
            if now - t0 > limit:
                state["timed_out"] = True
                st.close()
                return
            if state["first"] is None:
                emit("tune_progress", message=f"      {name}: waiting for the first token… {now - t0:.0f}s "
                                              "(processing the prompt)")
            else:
                toks = state["chunks"]
                rate = toks / max(now - state["first"], 1e-3)
                emit("tune_progress", message=f"      {name}: ~{toks:.0f} tokens, ~{rate:.1f} tok/s, {now - t0:.0f}s")
    threading.Thread(target=report_progress, daemon=True).start()
    try:
        return client.stream_chat(base_url, body, on_delta, timeout=limit, stream=st)
    except client.Cancelled:
        eng._check()   # a user cancel wins
        return {"error": f"no complete answer within {limit}s", "answer": "", "reasoning": ""}
    finally:
        done.set()
        eng._stream = None


def measure(eng, m, base_url, work, gen_tokens=GEN_TOKENS, passes=1, emit=lambda *a, **k: None,
            limit=REQUEST_LIMIT_S):
    """Time the workload (after a warm-up request). With passes > 1 the earlier passes warm the
    server's caches and only the last pass is reported."""
    out = Measure()
    for p in range(passes):
        if passes > 1:
            emit("tune_progress", message=f"    pass {p + 1} of {passes}" + (" (warm-up)" if p < passes - 1 else " (timed)"))
        out = _measure_once(eng, m, base_url, work, gen_tokens, emit, limit)
        if out.error:
            break
    return out


def _measure_once(eng, m, base_url, work, gen_tokens, emit, limit):
    sampling = engine_mod.effective_sampling(eng.cfg, m)
    request = eng.cfg["servers"][m["server"]].get("request", {})
    timeout = eng.cfg["defaults"]["request_timeout_ms"] / 1000
    out = Measure()
    timeout = min(timeout, limit)
    warm = _request(eng, base_url, _body(m, sampling, [{"role": "user", "content": "Say OK."}], 8, request),
                    "warm-up", emit, timeout)
    if warm["error"]:
        out.error = warm["error"]
        return out
    ttfts, p_tok, p_s, g_tok, g_s = [], 0, 0.0, 0, 0.0
    for name, msgs in work:
        eng._check()
        emit("tune_progress", message=f"    {name} ({gen_tokens} tokens)")
        res = _request(eng, base_url, _body(m, sampling, msgs, gen_tokens, request), name, emit, timeout)
        if res["error"]:
            out.error = f"{name}: {res['error']}"
            return out
        met = client.metrics(res)
        emit("tune_progress", message=f"      done: {met['completion_tokens']} tokens in {res['total_s']:.1f}s"
                                      + (f", {met['gen_tps']} tok/s" if met["gen_tps"] else ""))
        out.total_s += res["total_s"]
        out.texts.append((res["reasoning"] or "") + (res["answer"] or ""))
        if name != "long" and res.get("ttft_s") is not None:
            ttfts.append(res["ttft_s"])
        if met["prompt_tokens"] and met["prompt_tps"]:
            p_tok, p_s = p_tok + met["prompt_tokens"], p_s + met["prompt_tokens"] / met["prompt_tps"]
        if met["completion_tokens"] and met["gen_tps"]:
            g_tok, g_s = g_tok + met["completion_tokens"], g_s + met["completion_tokens"] / met["gen_tps"]
    out.total_s = round(out.total_s, 2)
    out.ttft_s = round(sum(ttfts) / len(ttfts), 3) if ttfts else None
    out.pp_tps = round(p_tok / p_s, 1) if p_s else None
    out.tg_tps = round(g_tok / g_s, 2) if g_s else None
    return out


def gain_pct(objective, base, best):
    """How much better the chosen settings are than the defaults: % more decode tok/s for the
    decode objective, % less total time otherwise."""
    if base is None:
        return None
    if objective == "decode":
        return round(100 * (best.tg_tps / base.tg_tps - 1)) if base.tg_tps and best.tg_tps else None
    return round(100 * (1 - best.total_s / base.total_s)) if base.total_s else None


def gain_text(measured):
    g = measured.get("gain_pct")
    if g is None:
        return ""
    if g <= 0:
        return "the defaults were already fastest"
    return f"{g}% faster decode than the defaults" if measured.get("objective") == "decode" else \
        f"{g}% less time than the defaults"


def similarity(texts, reference, chars=600):
    """Mean similarity of greedy outputs to the reference outputs (1.0 = identical)."""
    pairs = list(zip(texts, reference))
    if not pairs:
        return 1.0
    return sum(difflib.SequenceMatcher(None, a[:chars], b[:chars]).ratio() for a, b in pairs) / len(pairs)


# ------------------------------------------------------------------ knobs
def knobs(eng, m, sv=None):
    """{knob: [resolved option args, …]} for this model on this machine, with options that can't
    apply removed (draft-mtp on a model without MTP layers) and duplicates merged."""
    server = eng.cfg["servers"][m["server"]]
    sv = sv or eng.serving(m)
    vals = eng.knob_values(sv.machine)
    out = {}
    for name, options in server.get("tune", {}).items():
        seen, opts = set(), []
        for o in options:
            args = [a.format(**vals) for a in o]
            if sv.mtp_layers == 0 and any("draft-mtp" in a for a in args) and sv.identity.get("model_bytes"):
                continue  # the GGUF was read and has no MTP layers
            if tuple(args) not in seen:
                seen.add(tuple(args))
                opts.append(args)
        if opts:
            out[name] = opts
    return out


def resolve(knob_opts, choices):
    return [a for k, opts in knob_opts.items() for a in opts[choices.get(k, 0)]]


def option_index(opts, args):
    """Which of a knob's options a profile's args hold: the first non-empty option found in them as
    a contiguous run, else the knob's empty option (it was off). None if neither: the knob's options
    changed since, or the option isn't offered for this model (draft-mtp without MTP layers)."""
    empty = None
    for i, o in enumerate(opts):
        if not o:
            empty = i if empty is None else empty
        elif any(args[j:j + len(o)] == o for j in range(len(args) - len(o) + 1)):
            return i
    return empty


def family(path):
    """What decides which speed flags suit a GGUF: architecture and tensor shapes (not the weights).
    None if the header can't be read."""
    try:
        i = machines.read_gguf(engine_mod.expand(path))
    except (OSError, ValueError):
        return None
    # The MTP (nextn) draft layers at the end are left out: GGUFs of the same model describe them
    # differently, and whether draft-mtp is offered is decided per model (knobs) anyway.
    heads = i["kv_heads_per_layer"][:len(i["kv_heads_per_layer"]) - i["mtp_layers"]]
    return (i["architecture"], i["layers"], i["embedding"], i["experts"], i["experts_used"],
            i["head_dim_k"], i["head_dim_v"], tuple(heads))


@dataclasses.dataclass
class Sibling:
    label: str
    profile: dict
    choices: dict     # {knob: option index} for this model's knobs
    unmapped: list    # knobs whose inherited option isn't offered here (re-tried)


def find_sibling(eng, m, knob_opts, machine_id):
    """The best tuned profile to start from: a model on the same server and machine whose GGUF has the
    same family, with a current profile from a tune or hand-tuned flags. The closest file size wins
    (the most similar quant mix), then the newest profile. None if there isn't one."""
    own = family(m["model"])
    if own is None:
        return None
    version = eng.server_version(m)
    size = os.path.getsize(engine_mod.expand(m["model"]))
    found = []
    for o in eng.cfg["models"]:
        if o["label"] == m["label"] or o.get("server") != m["server"]:
            continue
        p = profiles.load(machine_id, o["label"], eng.tuning_dir)
        if not p or not p.get("args") or p.get("meta", {}).get("method") not in ("tuned", "seeded"):
            continue
        path = engine_mod.expand(o["model"])
        if not os.path.isfile(path) or profiles.stale(p, version, os.path.getsize(path)) or family(path) != own:
            continue
        choices, unmapped = {}, []
        for name, opts in knob_opts.items():
            i = option_index(opts, list(p["args"]))
            if i is None:
                unmapped.append(name)
                i = 0
            choices[name] = i
        found.append((abs(os.path.getsize(path) - size), str(p.get("meta", {}).get("date", "")),
                      Sibling(o["label"], p, choices, unmapped)))
    if not found:
        return None
    found.sort(key=lambda f: f[1], reverse=True)   # newest first among equal sizes (the sort is stable)
    found.sort(key=lambda f: f[0])
    return found[0][2]


def _flags(args):
    """["-t","8","-fa","on"] -> {"-t": "8", "-fa": "on"} (None when an arg isn't a flag/value pair)."""
    if len(args) % 2:
        return None
    return dict(zip(args[::2], args[1::2]))


# ------------------------------------------------------------------ stage 1: llama-bench
def bench_command(eng):
    cfg = eng.cfg.get("bench", {})
    if cfg.get("cmd"):
        cmd = [engine_mod.expand(a) for a in cfg["cmd"]]
        return cmd if shutil.which(cmd[0]) or os.path.isfile(cmd[0]) else None
    found = shutil.which("llama-bench")
    return [found] if found else None


def bench_stage(eng, m, knob_opts, work, emit=lambda *a, **k: None):
    """Settle the llama-bench-sweepable knobs. Returns ({knob: index}, info) or ({}, reason)."""
    sweep = {}   # knob -> {flag: [values per option]}
    for name, opts in knob_opts.items():
        parsed = [_flags(o) for o in opts]
        if all(p is not None and p and set(p) <= set(BENCH_FLAGS) | IGNORED_IN_BENCH for p in parsed) \
                and len(opts) > 1 and any(set(p) & set(BENCH_FLAGS) for p in parsed):
            sweep[name] = parsed
    if not sweep:
        return {}, None   # nothing llama-bench could help with (e.g. not a llama server)
    cmd = bench_command(eng)
    if not cmd:
        return {}, "llama-bench not found; using server timing only"
    lists = {}
    for parsed in sweep.values():
        for p in parsed:
            for flag, v in p.items():
                if flag in BENCH_FLAGS:
                    v = {"on": "1", "off": "0", "auto": "1"}.get(v, v) if flag == "-fa" else v
                    lists.setdefault(flag, [])
                    if v not in lists[flag]:
                        lists[flag].append(v)
    base = cmd + ["-m", engine_mod.expand(m["model"]), "-ngl", "99", "-r", "2", "-o", "json"]
    grid = lambda skip=(): [a for flag, vs in lists.items() if flag not in skip  # noqa: E731
                            for a in (BENCH_FLAGS[flag], ",".join(vs))]
    rows = []
    # prompt speed over the whole grid; generation speed doesn't depend on the micro-batch
    for extra, g in ((["-p", "512", "-n", "0"], grid()), (["-p", "0", "-n", "64"], grid(("-ub",)))):
        emit("tune_step", message="llama-bench " + " ".join(g + extra))
        try:
            out = subprocess.run(base + g + extra, capture_output=True, text=True, timeout=1800)
            rows += json.loads(out.stdout or "[]")
        except (OSError, subprocess.TimeoutExpired, ValueError) as e:
            return {}, f"llama-bench failed ({e}); using server timing only"
    key = lambda r: (str(r.get("n_threads")), str(r.get("n_ubatch")), str(int(bool(r.get("flash_attn")))))  # noqa: E731
    pp = {key(r): r["avg_ts"] for r in rows if r.get("n_prompt") and not r.get("n_gen")}
    tg = {(k[0], k[2]): r["avg_ts"] for r in rows if r.get("n_gen") and not r.get("n_prompt") for k in [key(r)]}
    if not pp or not tg:
        return {}, "llama-bench gave no usable numbers; using server timing only"
    prompt_tokens = sum(len(json.dumps(msgs)) // 4 for _, msgs in work)
    gen_tokens = GEN_TOKENS * len(work)
    scored = sorted((prompt_tokens / v + gen_tokens / tg[(t, fa)], (t, ub, fa)) for (t, ub, fa), v in pp.items()
                    if (t, fa) in tg)
    if not scored:
        return {}, "llama-bench results didn't line up; using server timing only"
    best_s, (t, ub, fa) = scored[0]
    want = {"-t": t, "-ub": ub, "-fa": fa}
    choices = {}
    for name, parsed in sweep.items():
        for i, p in enumerate(parsed):
            norm = {f: ({"on": "1", "off": "0", "auto": "1"}.get(v, v) if f == "-fa" else v)
                    for f, v in p.items() if f in BENCH_FLAGS}
            if all(want.get(f) == v for f, v in norm.items()):
                choices[name] = i
                break
    return choices, {"pp512_tps": round(pp[(t, ub, fa)], 1), "tg_tps": round(tg[(t, fa)], 2),
                     "estimate_s": round(best_s, 1), "combos": len(pp)}


# ------------------------------------------------------------------ stage 2 + driver
def tune(eng, label, emit=lambda *a, **k: None, max_starts=16, min_gain=0.03, use_bench=True, warm=True):
    """Tune one model on this machine and save its profile. Returns the profile dict. warm: start
    from a tuned model of the same family when there is one (see the module docstring)."""
    m = eng.model(label)
    server = eng.cfg["servers"][m["server"]]
    sv = eng.serving(m)
    if not sv.fits:
        raise engine_mod.ModelFailed(sv.fit_note)
    objective = server.get("tune_objective", "total")
    if objective == "decode":
        work, gen_tokens, passes = decode_workload(eng), DECODE_TOKENS, 2
    else:
        work, gen_tokens, passes = workload(eng, sv.ctx), GEN_TOKENS, 1
    settings = eng.cfg.get("tune", {})
    threshold = settings.get("guard_similarity", 0.6)
    guard = not server.get("outputs_depend_on_machine")
    knob_opts = knobs(eng, m, sv) if server.get("cmd") else {}
    fixed = list(server.get("perf", [])) if server.get("cmd") else []
    cache, starts, notes, rejected, bad = {}, [0], [], [], set()   # bad: (knob, option) that failed
    warned = []

    def evaluate(choices, why):
        k = tuple(sorted(choices.items()))
        if k in cache:
            return cache[k]
        if starts[0] >= max_starts:
            return None
        starts[0] += 1
        args = resolve(knob_opts, choices)
        emit("tune_step", message=f"[{starts[0]}] {why}: {' '.join(args) or '(server defaults)'}",
             start=starts[0], max_starts=max_starts)
        swap0 = machines.swapped_out_bytes()
        try:
            with eng.serve(m, perf_args=fixed + args, log_name=f"{label}.tune") as (url, info):
                r = measure(eng, m, url, work, gen_tokens, passes, emit,
                            settings.get("request_timeout_s", REQUEST_LIMIT_S))
                r.load_s = round(info["load_s"], 1) if info["load_s"] else None
                r.facts = dict(info["facts"])
                # Settled allocation after the timed pass (brief peaks while processing a prompt are
                # harmless; sustained allocation over the limit is what makes the driver churn).
                settled = machines.gpu_allocated_gb() if server.get("stall_guard") else None
                if settled:
                    r.facts["gpu_settled_gb"] = round(settled, 1)
                if info.get("kernel_share_max") is not None:
                    r.facts["kernel_share"] = info["kernel_share_max"]
                limit = eng.gpu_residency_gb()
                if settled and settled > limit - GPU_MARGIN_GB and not r.error:
                    r.error = (f"GPU allocation settled at {settled:.1f} GB, within {GPU_MARGIN_GB} GB of the "
                               f"~{limit:.0f} GB this machine keeps resident (it would stall as it grows)")
                if info["facts"].get("memory_limit") == "live-available" and not warned:
                    warned.append(1)
                    emit("tune_step", message="    note: the server sized itself by memory that was free at startup; "
                                              "close other apps for faster and fairer results")
        except engine_mod.ModelFailed as e:
            r = Measure(error=str(e).splitlines()[0])
        swap1 = machines.swapped_out_bytes()
        if not r.error and swap0 is not None and swap1 is not None and swap1 - swap0 > SWAP_LIMIT:
            r.error = f"macOS swapped {(swap1 - swap0) / 2**20:.0f} MB (memory too tight)"
        cache[k] = r
        emit("tune_result", message=f"    failed: {r.error}" if r.error else "    " + r.summary(), result=r, args=args)
        return r

    base = None
    if not server.get("cmd") or not knob_opts:
        r = evaluate({}, "measuring (no speed knobs to tune)")
        if r.error:
            raise engine_mod.ModelFailed(r.error)
        best, best_r = {}, r
        method, bench_info = "measured", None
    else:
        default = {k: 0 for k in knob_opts}
        base = evaluate(default, "defaults")
        if base is None or base.error:
            raise engine_mod.ModelFailed(f"the server doesn't start with the default flags: "
                                         f"{base.error if base else 'no starts left'}")
        settled, bench_info, warm_info = {}, None, None
        best, best_r, search = dict(default), base, list(knob_opts)
        sib = find_sibling(eng, m, knob_opts, sv.machine.id) \
            if warm and settings.get("warm_start", True) else None
        if sib:
            cand = {**default, **sib.choices}
            r = evaluate(cand, f"{sib.label}'s flags (same architecture and shapes)")
            why = "no server starts left" if r is None else f"they failed ({r.error})" if r.error else \
                "they changed answers" if guard and similarity(r.texts, base.texts) < threshold else \
                "they were slower than the defaults here" if r.score(objective) > base.score(objective) else None
            if why:
                notes.append(f"warm start from {sib.label} abandoned: {why}; tuned in full")
                emit("tune_step", message=f"    {sib.label}'s flags don't carry over ({why}): tuning in full")
            else:
                retest = settings.get("warm_retest", WARM_RETEST)
                search = [k for k in knob_opts if k in retest or k in sib.unmapped]
                best, best_r = cand, r
                warm_info = {"from": sib.label, "from_model_file": sib.profile.get("meta", {}).get("model_file"),
                             "inherited": [k for k in knob_opts if k not in search], "retested": search}
                emit("tune_step", message=f"    starting from {sib.label}'s flags; re-trying "
                                          f"{', '.join(search) or 'nothing'}")
        if use_bench and not warm_info:
            settled, bench_info = bench_stage(eng, m, knob_opts, work, emit)
            if isinstance(bench_info, str):
                notes.append(bench_info)
                emit("tune_step", message=bench_info)
                bench_info = None
        if settled:
            cand = {**default, **settled}
            r = evaluate(cand, "llama-bench pick")
            if r and not r.error and similarity(r.texts, base.texts) >= threshold \
                    and r.score(objective) < best_r.score(objective):
                best, best_r = cand, r
        for _pass in range(2):
            improved = False
            for name, opts in knob_opts.items():
                if name not in search:
                    continue
                if name in settled and _pass == 0 and best.get(name) == settled[name]:
                    continue
                for i in range(len(opts)):
                    if i == best[name] or (name, i) in bad:
                        continue
                    cand = {**best, name: i}
                    r = evaluate(cand, f"{name} = {' '.join(opts[i]) or '(off)'}")
                    if r is None:
                        break
                    if r.error:
                        bad.add((name, i))
                        notes.append(f"{name} {' '.join(opts[i]) or '(off)'}: {r.error}")
                        continue
                    sim = similarity(r.texts, base.texts)
                    if not guard:
                        emit("tune_step", message=f"    answers {sim:.2f} similar to the defaults (expected to "
                                                  "vary on this server; its results are tied to these settings)")
                    elif sim < threshold:
                        bad.add((name, i))
                        rejected.append(f"{' '.join(opts[i]) or name + ' off'} (answers changed: {sim:.2f} similar)")
                        emit("tune_step", message=f"    rejected: answers changed ({sim:.2f} similar)")
                        continue
                    if r.score(objective) < best_r.score(objective) * (1 - min_gain):
                        best, best_r, improved = cand, r, True
            if not improved or starts[0] >= max_starts:
                break
        method = "tuned"
        if warm_info:
            notes.insert(0, f"warm start from {warm_info['from']}: inherited "
                            f"{', '.join(warm_info['inherited']) or 'nothing'}")
    size = sv.identity.get("model_bytes")
    profile = {
        "args": resolve(knob_opts, best),
        "choices": {k: best.get(k, 0) for k in knob_opts},
        "measured": {"total_s": best_r.total_s, "ttft_s": best_r.ttft_s, "pp_tps": best_r.pp_tps,
                     "tg_tps": best_r.tg_tps, "load_s": best_r.load_s,
                     "default_total_s": base.total_s if knob_opts else None,
                     "default_tg_tps": base.tg_tps if knob_opts else None, "objective": objective,
                     **{f"server_{k}": v for k, v in best_r.facts.items()},
                     "gain_pct": gain_pct(objective, base, best_r)},
        "meta": {"method": method, "date": profiles.now(), "machine": sv.machine.summary,
                 "server_version": eng.server_version(m), "model_file": os.path.basename(m["model"]),
                 "model_bytes": size, "ctx": sv.ctx, "server_starts": starts[0],
                 "workload": f"{len(work)} prompts x {gen_tokens} tokens" + (f", {passes} passes" if passes > 1 else ""),
                 "llama_bench": bool(bench_info), "answer_guard": guard,
                 **({"warm_start": warm_info} if knob_opts and warm_info else {}),
                 "rejected": rejected, "notes": notes[:10]},
    }
    path = profiles.save(sv.machine.id, label, profile, eng.tuning_dir)
    eng._serving.clear()
    shown = os.path.relpath(path, eng.root) if path.startswith(eng.root) else path
    emit("tune_done", message=f"saved {shown}", profile=profile, path=path)
    return profile


def seed(eng, label, args, choices=None, machine_id=None, note=""):
    """Record known-good flags as a profile without tuning (marked 'seeded')."""
    m = eng.model(label)
    sv = eng.serving(m)
    profile = {"args": list(args), "choices": choices or {}, "measured": {},
               "meta": {"method": "seeded", "date": profiles.now(), "machine": sv.machine.summary,
                        "server_version": eng.server_version(m),
                        "model_file": os.path.basename(m["model"]), "model_bytes": sv.identity.get("model_bytes"),
                        "notes": [note] if note else []}}
    path = profiles.save(machine_id or sv.machine.id, label, profile, eng.tuning_dir)
    eng._serving.clear()
    return path
