"""Find the largest context a model can be served with on this machine, by trying it, and save it
in tuning/<machine-id>/<label>.ctx.toml. Runs then use it in place of the fit check's estimate.

    tuieval tune MODEL --ctx [--max N] [--start N] [--step N] [--max-starts N]

It works the same for every server tuieval starts, because it only uses what they all have:

  the context   set through {ctx} in the server's cmd, or its ctx_flag
  the API       one chat request per try, and the token counts the server reports
  the machine   the server's memory (its processes' resident memory; VRAM on discrete GPUs),
                macOS memory pressure and swapping, memory available on Linux

One try ("probe") at a context: start the server with it, send a document that fills it (sized
from the server's own token counts, so any tokenizer works) with one fact near the start, and ask
for that fact. The probe passes when the answer is right, the server read the whole document (a
server that cuts long prompts short reports fewer tokens, or loses the fact), and the machine
stayed healthy. Filling matters: some servers only take memory as the context fills.

The search starts small and doubles. Memory grows about linearly with the context, so two probes
that passed predict the largest context that fits under the machine's limit; no probe goes above
that prediction, and a failed probe is bisected against the largest that passed. A probe is
stopped at once when memory pressure rises, the machine starts swapping, or (Linux) memory runs
out. Typically 3 to 5 server starts.

A server tuieval doesn't start, or can't set the context of, is only verified at the context its
model declares (max_context): one probe, nothing saved.
"""
import dataclasses
import os
import threading

from . import engine as engine_mod
from . import machines
from . import profiles
from . import tune

ANSWER_TOKENS = 1024       # room for the answer (and any thinking) in a probe
STEP = 8192                # how close to the true maximum the search gets
FLOOR = machines.MIN_CTX   # the smallest context tried
MAX_STARTS = 8
REQUEST_LIMIT_S = 3600     # reading a few hundred thousand tokens takes a while ([tune] ctx_request_timeout_s)
PRESSURE_WARNING = 2       # macOS memory pressure level that stops a probe (1 is normal)
SWAP_STOP_MB = 512         # swapped out while a probe ran: stop it
AVAILABLE_STOP_GB = 1.0    # Linux: memory available under this stops a probe
SAMPLE_CHARS = 12000       # the request that measures the server's characters per token
UNKNOWN_CHARS_PER_TOKEN = 3.0   # for a server that reports no token counts: short of full rather than over
NOT_ABOUT_CONTEXT = ("already in use", "isn't installed", "not found", "is not set", "skipped by user")
CODE = 481739              # six digits: no section number or step count in the document can equal it
WORDS = ("river", "station", "ledger", "orchard", "signal", "harbour", "copper", "meadow", "lantern",
         "quarry", "window", "valley", "engine", "garden", "bridge", "tower")


@dataclasses.dataclass
class Probe:
    ctx: int
    ok: bool = False
    error: str = ""
    mem_gb: float | None = None       # the server's peak memory with the context full
    prompt_tokens: int | None = None  # as the server counted them
    pp_tps: float | None = None
    tg_tps: float | None = None
    total_s: float | None = None
    load_s: float | None = None
    filled: bool = False              # the server's token count confirms the context was full

    def summary(self):
        if not self.ok:
            return f"failed: {self.error}"
        parts = ["ok"]
        if self.mem_gb is not None:
            parts.append(f"{self.mem_gb:.1f} GB")
        if self.prompt_tokens:
            parts.append(f"{self.prompt_tokens:,} prompt tokens" + ("" if self.filled else " (fill not confirmed)"))
        if self.pp_tps:
            parts.append(f"prompt {self.pp_tps:.0f} tok/s")
        if self.tg_tps:
            parts.append(f"gen {self.tg_tps:.1f} tok/s")
        return ", ".join(parts)


# ------------------------------------------------------------------ the document
def document(chars):
    """A ~chars-long synthetic document with one fact near its start and the question at its end
    (the same text every time). A server that drops the start of a long prompt loses the fact."""
    paras, size, i = [], 0, 0
    while size < chars:
        w = [WORDS[(i * 7 + k * 3) % len(WORDS)] for k in range(12)]
        p = (f"Section {i + 1}. The {w[0]} near the {w[1]} kept a {w[2]} of every {w[3]}, and the {w[4]} by the "
             f"{w[5]} was painted {w[6]} after the {w[7]} flooded. Visitors to the {w[8]} counted "
             f"{(i * 37) % 900 + 100} steps to the {w[9]}, past a {w[10]} facing the {w[11]}.")
        paras.append(p)
        size += len(p) + 2
        i += 1
    paras.insert(max(1, len(paras) // 20), f"Note: the archive room code is {CODE}.")
    return "\n\n".join(paras) + "\n\nWhat is the archive room code? Answer with the number only."


def _ask(eng, m, base_url, chars, max_tokens, name, emit, limit):
    sampling = engine_mod.effective_sampling(eng.cfg, m)
    request = eng.cfg["servers"][m["server"]].get("request", {})
    body = tune._body(m, sampling, [{"role": "user", "content": document(chars)}], max_tokens, request)
    body.pop("ignore_eos", None)   # the answer should end where the model ends it
    return tune._stream_request(eng, base_url, body, name, emit, limit)


def chars_per_token(eng, m, base_url, emit, limit):
    """Characters of the document per token, by this server's count (None if it reports none)."""
    from . import client
    res = _ask(eng, m, base_url, SAMPLE_CHARS, 8, "sizing", emit, limit)
    if res.get("error"):
        raise engine_mod.ModelFailed(f"sizing request: {res['error']}")
    tokens = client.metrics(res)["prompt_tokens"]
    return SAMPLE_CHARS / tokens if tokens else None


def too_long(error):
    e = (error or "").lower()
    return "http 4" in e and any(w in e for w in ("context", "too long", "too large", "exceed", "maximum"))


class Guard:
    """Watches the machine while a probe runs and stops the request when it gets unhealthy."""

    def __init__(self, eng, every_s=2.0):
        self.eng, self.every_s, self.reason = eng, every_s, ""
        self.pressure, self._stop = machines.memory_pressure_level(), threading.Event()
        self._swap0 = machines.swapped_out_bytes()
        self._linux = self.pressure is None and not eng.machine().discrete

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def check(self):
        lvl = machines.memory_pressure_level()
        if lvl is not None:
            self.pressure = max(self.pressure or 0, lvl)
            if lvl >= PRESSURE_WARNING:
                return f"macOS memory pressure rose to level {lvl}"
        swap = machines.swapped_out_bytes()
        if None not in (swap, self._swap0) and (swap - self._swap0) / 2**20 > SWAP_STOP_MB:
            return f"the machine swapped out {(swap - self._swap0) / 2**20:.0f} MB"
        if self._linux:
            free = machines.available_memory_gb()
            if free is not None and free < AVAILABLE_STOP_GB:
                return f"only {free:.1f} GB of memory was left"
        return ""

    def _run(self):
        while not self._stop.wait(self.every_s):
            self.reason = self.reason or self.check()
            if self.reason:
                stream = getattr(self.eng, "_stream", None)
                if stream is not None:
                    stream.close()
                return

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(self.every_s + 1)
        self.reason = self.reason or self.check()


def probe(eng, m, ctx, emit=lambda *a, **k: None, limit=REQUEST_LIMIT_S):
    """Serve the model with this context, fill it and ask for the fact. Returns a Probe."""
    from . import client
    out = Probe(ctx)
    reserve = ANSWER_TOKENS + max(256, ctx // 50)
    target = ctx - reserve
    if target < 1024:
        out.error = f"a context of {ctx:,} is too small to test"
        return out
    try:
        with eng.serve(m, log_name=f"{m['label']}.ctx", ctx=ctx) as (url, info):
            out.load_s = round(info["load_s"], 1) if info.get("load_s") else None
            with Guard(eng) as guard:
                ratio = chars_per_token(eng, m, url, emit, limit)
                chars = int(target * (ratio or UNKNOWN_CHARS_PER_TOKEN) * 0.98)
                emit("tune_progress", message=f"    filling the context: ~{target:,} tokens, {chars:,} characters"
                                              + ("" if ratio else " (the server reports no token counts)"))
                res = _ask(eng, m, url, chars, ANSWER_TOKENS, "fill", emit, limit)
                if too_long(res.get("error")) and not guard.reason:
                    emit("tune_progress", message="    the server found that too long; trying 7% shorter")
                    res = _ask(eng, m, url, int(chars * 0.93), ANSWER_TOKENS, "fill", emit, limit)
            met = client.metrics(res) if not res.get("error") else {}
            out.prompt_tokens, out.pp_tps, out.tg_tps = met.get("prompt_tokens"), met.get("prompt_tps"), met.get("gen_tps")
            out.total_s = round(res["total_s"], 1) if res.get("total_s") else None
            peak = info.get("vram_peak_gb") if eng.machine().discrete else \
                (info["peak_mb"]() or 0) / 1024 if info.get("peak_mb") else None
            out.mem_gb = round(peak, 2) if peak else None
            text = (res.get("answer") or "") + (res.get("reasoning") or "")
            if guard.reason:
                out.error = guard.reason
            elif res.get("error"):
                out.error = res["error"]
            elif not info["alive"]():
                out.error = "the server stopped"
            elif str(CODE) not in text:
                out.error = "it didn't find a fact near the start of the document (the context isn't all used)"
            elif out.prompt_tokens and out.prompt_tokens < 0.85 * target:
                out.error = (f"the server read {out.prompt_tokens:,} of ~{target:,} tokens "
                             "(it cuts long prompts short)")
            else:
                out.ok = True
                out.filled = bool(ratio and out.prompt_tokens and out.prompt_tokens >= 0.85 * target)
    except engine_mod.ModelFailed as e:
        out.error = str(e).splitlines()[0]
        if any(w in out.error for w in NOT_ABOUT_CONTEXT):
            raise   # a smaller context wouldn't help
    return out


# ------------------------------------------------------------------ the search
def predict(passed, limit_gb):
    """(base_gb, gb_per_token, largest context predicted to fit) from the two largest contexts that
    passed with a memory reading. The third is None when memory doesn't grow with the context
    (nothing to predict) or there aren't two readings."""
    pts = sorted((p.ctx, p.mem_gb) for p in passed if p.mem_gb)
    if len(pts) < 2:
        return None, None, None
    (c1, g1), (c2, g2) = pts[-2:]
    per = (g2 - g1) / (c2 - c1)
    base = g2 - per * c2
    if per <= 0 or limit_gb is None:
        return base, max(per, 0.0), None
    return base, per, int((limit_gb - base) / per)


def _grid(n):
    return int(n) // 1024 * 1024


def search(try_ctx, start, upper, limit_gb=None, step=STEP, floor=FLOOR, max_starts=MAX_STARTS,
           emit=lambda *a, **k: None):
    """Probe contexts until the largest that works is known to within `step`. try_ctx(ctx) -> Probe.
    Returns (probes in the order tried, why it stopped)."""
    tried, lo, hi, c = [], None, None, min(start, upper)
    why = f"{max_starts} server starts used"
    while len(tried) < max_starts:
        emit("tune_step", message=f"[{len(tried) + 1}] context {c:,}", start=len(tried) + 1, max_starts=max_starts)
        r = try_ctx(c)
        tried.append(r)
        emit("tune_result", message="    " + r.summary(), result=r)
        if r.ok:
            lo = max(lo or 0, c)
        else:
            hi = min(hi or c, c)
        if lo is None:   # nothing has worked yet: halve
            c = _grid(hi // 2)
            if c < floor:
                why = f"nothing from {floor:,} up worked"
                break
            continue
        if lo >= upper:
            why = f"reached the upper bound ({upper:,})"
            break
        if hi is not None and hi - lo <= step:
            why = f"{hi:,} failed: {next(p.error for p in tried if p.ctx == hi)}"
            break
        _, per, fit = predict([p for p in tried if p.ok], limit_gb)
        nxt = min(upper, lo * 2) if hi is None else lo + _grid((hi - lo) / 2)
        if fit is not None and fit < nxt:
            if _grid(fit) - lo < min(step, upper - lo):
                at = min(upper, lo + step)
                why = (f"memory: {at:,} is predicted to need more than the {limit_gb:.1f} GB this machine has "
                       f"for it ({per * 1000:.3f} GB per 1,000 tokens)")
                break
            nxt = _grid(fit)
        if nxt <= lo:
            why = f"{hi:,} failed: {next(p.error for p in tried if p.ctx == hi)}" if hi else why
            break
        c = nxt
    return tried, why


# ------------------------------------------------------------------ driver
def model_gb(m):
    """The model's size on disk (a file with its other shards beside it, or a folder), or None."""
    path = engine_mod.expand(str(m.get("model", "")))
    try:
        if os.path.isfile(path):
            return os.path.getsize(path) / machines.GB
        if os.path.isdir(path):
            return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs) / machines.GB
    except OSError:
        pass
    return None


def room_gb(eng):
    """(memory this machine has for a model now, a reason not to load one at all or "")."""
    mc = eng.machine()
    headroom = eng.headroom_gb(mc)
    if mc.discrete:
        used = machines.vram_used_gb()
        return mc.gpu_limit_gb - (used or 0) - headroom, ""
    level = machines.memory_pressure_level()
    if level is not None and level > 1:
        return None, f"macOS memory pressure is already elevated (level {level}); close other apps or models first"
    free = machines.available_memory_gb()
    limit = mc.gpu_limit_gb - headroom
    return (limit if free is None else min(limit, free - headroom)), ""


def upper_bound(eng, m, server, how, asked=None):
    """(the largest context worth trying, where that number comes from)."""
    if asked:
        return asked, "--max"
    caps = [(server.get("max_ctx"), f"max_ctx in [servers.{m['server']}]"), (m.get("ctx"), "the model's ctx"),
            (m.get("max_context"), "the model's max_context")]
    caps = [(v, src) for v, src in caps if v]
    if how == "flag":
        sv = eng.serving(m)
        written = engine_mod.read_context(list(m.get("server_args", [])) + list(sv.perf_args), server["ctx_flag"])
        if written:
            caps.append((written, f"{server['ctx_flag']} in models.toml"))
    if caps:
        return min(caps)
    path = engine_mod.expand(str(m.get("model", "")))
    if os.path.isfile(path):
        try:
            n = machines.read_gguf(path).get("context_length")
            if n:
                return n, "the model file"
        except (OSError, ValueError):
            pass
    raise engine_mod.ModelFailed("no upper bound for its context is known: pass --max N (the model's own limit)")


def bench(eng, label, emit=lambda *a, **k: None, max_ctx=None, start=None, step=STEP, max_starts=MAX_STARTS):
    """Measure the largest context one model can be served with here and save it. Returns the saved
    measurement, or {"verified": bool, ...} for a server whose context tuieval can't set."""
    m = eng.model(label)
    server = eng.cfg["servers"][m["server"]]
    if eng.hosted(m):
        raise engine_mod.ModelFailed("it's a hosted model: its context doesn't depend on this machine")
    how = engine_mod.takes_context(server)
    limit_s = eng.cfg.get("tune", {}).get("ctx_request_timeout_s", REQUEST_LIMIT_S)
    sv = eng.serving(m)
    if not how:
        ctx = m.get("max_context") or sv.ctx
        if not ctx:
            raise engine_mod.ModelFailed("tuieval can't set this server's context ({ctx} in its cmd, or ctx_flag), and "
                                         "the model has no max_context to verify")
        emit("tune_step", message=f"tuieval can't set this server's context: checking the {ctx:,} it declares")
        r = probe(eng, m, ctx, emit, limit_s)
        emit("tune_result", message="    " + r.summary(), result=r)
        return {"verified": r.ok, "ctx": ctx, "probe": r}
    upper, source = upper_bound(eng, m, server, how, max_ctx)
    upper = max(upper, FLOOR)
    room, refuse = room_gb(eng)
    if refuse:
        raise engine_mod.ModelFailed(refuse)
    size = model_gb(m)
    if room is not None and size is not None and size > room:
        raise engine_mod.ModelFailed(f"the model is {size:.1f} GB and {room:.1f} GB is free for it now: another "
                                     "model or app is holding memory")
    first = min(upper, max(FLOOR, _grid(start or upper // 4)))
    emit("tune_step", message=f"largest context up to {upper:,} ({source})"
                              + (f"; {room:.1f} GB of memory to work with" if room is not None else ""))

    def try_ctx(ctx):
        eng._check()
        return probe(eng, m, ctx, emit, limit_s)

    tried, why = search(try_ctx, first, upper, room, step, min(FLOOR, first), max_starts, emit)
    passed = [p for p in tried if p.ok]
    if not passed:
        raise engine_mod.ModelFailed(f"no context worked ({why}); the last try: {tried[-1].error}")
    best = max(passed, key=lambda p: p.ctx)
    base, per, _ = predict(passed, room)
    measured = {
        "max_ctx": best.ctx,
        "measured": {"memory_gb": best.mem_gb, "base_gb": round(base, 2) if base is not None else None,
                     "gb_per_1k_tokens": round(per * 1000, 4) if per is not None else None,
                     "prompt_tokens": best.prompt_tokens, "prompt_tps": best.pp_tps, "gen_tps": best.tg_tps,
                     "fill_s": best.total_s, "load_s": best.load_s, "limit_gb": round(room, 1) if room is not None else None,
                     "stopped": why},
        "probes": {str(p.ctx): p.summary() for p in tried},
        "meta": {"date": profiles.now(), "machine": sv.machine.summary, "server_version": eng.server_version(m),
                 "model_file": os.path.basename(str(m.get("model", ""))), "model_bytes": sv.identity.get("model_bytes"),
                 "kv_type": sv.kv_type, "upper_bound": upper, "upper_bound_from": source,
                 "fill_confirmed": best.filled, "server_starts": len(tried)},
    }
    path = profiles.save_context(sv.machine.id, label, measured, eng.tuning_dir)
    eng._serving.clear()
    shown = os.path.relpath(path, eng.root) if path.startswith(eng.root) else path
    emit("tune_done", message=f"saved {shown}", profile=measured, path=path)
    return measured
