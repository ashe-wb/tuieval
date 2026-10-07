"""Which machine this is, what a model needs, and whether it fits.

    detect()              -> Machine (chip, cores, RAM, GPU memory limit, id like "m3max-64gb";
                             with Nvidia/AMD cards on Linux, their VRAM and an id like "rtx4090-128gb")
    read_gguf(path)       -> GGUF header facts (architecture, layers, KV heads, context length, …)
    fit(model_path, …)    -> Fit (largest context that fits this machine, estimated memory)

Nothing here changes outputs: the fit check only chooses a context size. It never quantizes
the KV cache on its own (that would change answers); a quantized-KV variant is a separate model.
"""
import dataclasses
import functools
import json
import os
import platform
import re
import struct
import subprocess

GB = 1024 ** 3
KV_BYTES = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32,
            "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32}
MIN_CTX = 8192           # below this a model isn't useful for these evals
COMPUTE_OVERHEAD = 1.0   # GB for compute buffers, graph, runtime (conservative)


# ---------------------------------------------------------------- the machine
@dataclasses.dataclass
class Machine:
    id: str
    chip: str
    ram_gb: float
    gpu_limit_gb: float      # what Metal may wire for the GPU; with discrete GPUs, their VRAM (all cards)
    p_cores: int
    e_cores: int
    gpu_cores: int | None
    os: str
    gpu_name: str | None = None   # discrete GPUs, e.g. "2× NVIDIA GeForce RTX 4090"
    discrete: bool = False        # the GPU has its own memory (VRAM), separate from RAM

    @property
    def summary(self):
        cores = f"{self.p_cores}P+{self.e_cores}E" if self.e_cores else f"{self.p_cores} cores"
        if self.discrete:
            return f"{self.chip} · {self.ram_gb:.0f} GB RAM · {self.gpu_name} · {self.gpu_limit_gb:.0f} GB VRAM · {cores}"
        gpu = f", {self.gpu_cores}-core GPU" if self.gpu_cores else ""
        return f"{self.chip} · {self.ram_gb:.0f} GB (GPU up to {self.gpu_limit_gb:.0f} GB) · {cores}{gpu}"


def _sysctl(name):
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def machine_id(chip, ram_gb):
    name = re.sub(r"^apple\s+", "", chip.strip(), flags=re.I)
    return re.sub(r"[^a-z0-9]+", "", name.lower()) + f"-{round(ram_gb)}gb"


def gpu_id(gpus, ram_gb):
    """A machine id from its discrete GPUs: "rtx4090-128gb", "2xrtx4090-128gb", "rx7900xtx+rtx3090-64gb"."""
    names = []
    for name in sorted({n for n, _ in gpus}):
        short = re.sub(r"\b(nvidia|geforce|amd|radeon|ati|graphics|tm)\b", "", name.lower())
        short = re.sub(r"[^a-z0-9]+", "", short) or "gpu"
        count = sum(n == name for n, _ in gpus)
        names.append(f"{count}x{short}" if count > 1 else short)
    return "+".join(names) + f"-{round(ram_gb)}gb"


# ---------------------------------------------------------------- discrete GPUs (Linux: Nvidia, AMD)
# Read from the vendors' own tools. Anything unexpected counts as "not detected", never an error.
def _run(cmd, timeout=10):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return p.stdout if p.returncode == 0 else None


def _nvidia_gpus():
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    gpus = []
    for line in (out or "").splitlines():
        name, _, mib = line.rpartition(",")
        try:
            gpus.append((name.strip(), float(mib) / 1024))
        except ValueError:
            pass
    return [g for g in gpus if g[0] and g[1] > 0]


def _walk(obj):
    """Every dict inside a JSON value."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def _bytes(value, unit=None):
    """VRAM sizes as amd-smi/rocm-smi print them: 25753026560, "24560 MB", {"value": 24560, "unit": "MB"}."""
    if isinstance(value, dict):
        return _bytes(value.get("value"), value.get("unit"))
    m = re.match(r"\s*([\d.]+)\s*([KMGT]i?B|B)?\s*$", str(value), re.I)
    if not m:
        return None
    scale = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30, "TB": 2**40}
    u = (m.group(2) or unit or "B").upper().replace("I", "")
    return float(m.group(1)) * scale.get(u, 1)


def _amd_gpus():
    out = _run(["amd-smi", "static", "--asic", "--vram", "--json"])
    gpus = []
    try:
        for d in _walk(json.loads(out)) if out else []:
            if "vram" in d and isinstance(d["vram"], dict):
                size = _bytes(d["vram"].get("size"))
                name = (d.get("asic") or {}).get("market_name") or "AMD GPU"
                if size:
                    gpus.append((str(name), size / GB))
    except ValueError:
        gpus = []
    if gpus:
        return gpus
    out = _run(["rocm-smi", "--showmeminfo", "vram", "--showproductname", "--json"])
    try:
        cards = json.loads(out) if out else {}
    except ValueError:
        return []
    for key, d in sorted(cards.items()) if isinstance(cards, dict) else []:
        if not key.startswith("card") or not isinstance(d, dict):
            continue
        size = _bytes(d.get("VRAM Total Memory (B)"))
        name = d.get("Card Series") or d.get("Card series") or d.get("Card SKU") or d.get("Card model") or "AMD GPU"
        if size:
            gpus.append((str(name), size / GB))
    return gpus


def discrete_gpus():
    """[(name, VRAM GB)] for the Nvidia or AMD cards this machine has, or [] (none, or tools missing)."""
    return _nvidia_gpus() or _amd_gpus()


def vram_used_gb(pids=None):
    """VRAM in use now (GB): by these processes on Nvidia (all of it when pids is None), or on the
    whole machine on AMD, whose tools don't report it reliably per process. None if unknown."""
    if pids is not None:
        out = _run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
        if out is not None:
            used = 0.0
            for line in out.splitlines():
                pid, _, mib = line.partition(",")
                if pid.strip() in {str(p) for p in pids}:
                    try:
                        used += float(mib) / 1024
                    except ValueError:
                        pass
            return used
    out = _run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    if out is not None:
        try:
            return sum(float(x) for x in out.split()) / 1024
        except ValueError:
            return None
    out = _run(["rocm-smi", "--showmeminfo", "vram", "--json"])
    try:
        cards = json.loads(out) if out else None
    except ValueError:
        return None
    if not isinstance(cards, dict):
        return None
    used = [_bytes(d.get("VRAM Total Used Memory (B)")) for k, d in cards.items()
            if k.startswith("card") and isinstance(d, dict)]
    return sum(u for u in used if u) / GB if any(used) else None


@functools.lru_cache(maxsize=1)
def detect():
    """This machine. Override the id with EVALS_MACHINE if two machines would collide."""
    if platform.system() == "Darwin":
        chip = _sysctl("machdep.cpu.brand_string") or platform.processor() or "unknown"
        ram = int(_sysctl("hw.memsize") or 0) / GB
        p = int(_sysctl("hw.perflevel0.physicalcpu") or _sysctl("hw.physicalcpu") or 0)
        e = int(_sysctl("hw.perflevel1.physicalcpu") or 0)
        wired = int(_sysctl("iogpu.wired_limit_mb") or 0)
        # macOS default when unset: about 2/3 of RAM up to 36 GB, 3/4 above.
        limit = wired / 1024 if wired > 0 else ram * (0.75 if ram > 36 else 2 / 3)
        gpu_cores = None
        try:
            out = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True,
                                 timeout=15).stdout
            m = re.search(r"Total Number of Cores:\s*(\d+)", out)
            gpu_cores = int(m.group(1)) if m else None
        except (OSError, subprocess.TimeoutExpired):
            pass
        os_name = "macOS " + platform.mac_ver()[0]
    else:  # Linux and others: the CPU side, plus Nvidia/AMD cards if their tools are installed
        chip = platform.processor() or platform.machine()
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GB
        p, e, gpu_cores, limit = os.cpu_count() or 1, 0, None, ram * 0.9
        os_name = platform.system()
        gpus = discrete_gpus()
        if gpus:
            names = sorted({n for n, _ in gpus})
            gpu_name = " + ".join(f"{sum(n == x for n, _ in gpus)}× {x}" if sum(n == x for n, _ in gpus) > 1 else x
                                  for x in names)
            mid = os.environ.get("EVALS_MACHINE") or gpu_id(gpus, ram)
            return Machine(mid, chip, round(ram, 1), round(sum(g for _, g in gpus), 1), p, e, None, os_name,
                           gpu_name=gpu_name, discrete=True)
    mid = os.environ.get("EVALS_MACHINE") or machine_id(chip, ram)
    return Machine(mid, chip, round(ram, 1), round(limit, 1), p, e, gpu_cores, os_name)


# ---------------------------------------------------------------- GGUF header
_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def _read(f, fmt):
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) != size:
        raise ValueError("truncated GGUF header")
    return struct.unpack(fmt, data)[0]


def _string(f):
    n = _read(f, "<Q")
    return f.read(n).decode("utf-8", "replace")


def _value(f, vtype, keep_arrays):
    if vtype in _SCALARS:
        return _read(f, _SCALARS[vtype])
    if vtype == 8:
        return _string(f)
    if vtype == 9:
        itype, count = _read(f, "<I"), _read(f, "<Q")
        if itype in _SCALARS and not keep_arrays:  # skip big numeric arrays quickly
            f.seek(struct.calcsize(_SCALARS[itype]) * count, os.SEEK_CUR)
            return None
        items = [_value(f, itype, True) for _ in range(count)]
        return items if keep_arrays else None
    raise ValueError(f"unknown GGUF value type {vtype}")


@functools.lru_cache(maxsize=64)
def read_gguf(path):
    """Metadata from a GGUF file's header (no tensors are read). Arrays are kept only for the
    per-layer attention keys; tokenizer arrays are skipped."""
    path = os.path.expanduser(path)
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        version = _read(f, "<I")
        if version < 2:
            raise ValueError(f"GGUF version {version} is too old")
        _read(f, "<Q")                    # tensor count
        n_kv = _read(f, "<Q")
        kv = {}
        for _ in range(n_kv):
            key = _string(f)
            vtype = _read(f, "<I")
            keep = vtype == 9 and (".attention." in key or key.endswith(".block_count"))
            val = _value(f, vtype, keep)
            if val is not None:
                kv[key] = val
    arch = kv.get("general.architecture", "")
    get = lambda k, d=None: kv.get(f"{arch}.{k}", d)  # noqa: E731
    layers = int(get("block_count", 0) or 0)
    heads = get("attention.head_count", 0)
    kv_heads = get("attention.head_count_kv", heads)
    embd = int(get("embedding_length", 0) or 0)
    heads0 = max(heads) if isinstance(heads, list) else int(heads or 0)
    head_dim_k = int(get("attention.key_length", 0) or (embd // heads0 if heads0 else 0))
    head_dim_v = int(get("attention.value_length", 0) or head_dim_k)
    if isinstance(kv_heads, list):
        per_layer = [int(x) for x in kv_heads]
    else:
        per_layer = [int(kv_heads or 0)] * layers
    # Hybrid models (linear attention / SSM layers keep no KV cache): an explicit per-layer flag,
    # or "every Nth layer is full attention".
    recurrent = get("attention.recurrent_layers")
    interval = int(get("full_attention_interval", 0) or 0)
    if isinstance(recurrent, list) and len(recurrent) == len(per_layer):
        per_layer = [0 if r else h for h, r in zip(per_layer, recurrent)]
    elif interval > 1:
        per_layer = [h if (i + 1) % interval == 0 else 0 for i, h in enumerate(per_layer)]
    return {
        "path": path, "bytes": os.path.getsize(path), "architecture": arch, "name": kv.get("general.name", ""),
        "layers": layers, "context_length": int(get("context_length", 0) or 0), "embedding": embd,
        "kv_heads_per_layer": per_layer, "head_dim_k": head_dim_k, "head_dim_v": head_dim_v,
        "experts": int(get("expert_count", 0) or 0), "experts_used": int(get("expert_used_count", 0) or 0),
        "sliding_window": int(get("attention.sliding_window", 0) or 0),
        "mtp_layers": int(get("nextn_predict_layers", 0) or 0),   # built-in draft layers (draft-mtp)
    }


def kv_bytes_per_token(info, kv_type="f16"):
    """KV cache bytes per token of context. Layers with no KV heads (e.g. linear/SSM layers in
    hybrid models) take none. Sliding-window layers are counted as full attention, which
    overestimates memory: the fit check errs on the safe side."""
    per = KV_BYTES.get(kv_type, 2.0)
    return sum(h * (info["head_dim_k"] + info["head_dim_v"]) * per for h in info["kv_heads_per_layer"])


# ---------------------------------------------------------------- fit
@dataclasses.dataclass
class Fit:
    fits: bool
    max_ctx: int              # largest context that fits (capped at the model's own maximum)
    need_gb: float            # estimated memory at max_ctx (or at MIN_CTX when it doesn't fit)
    available_gb: float
    note: str


def fit(model_path, machine=None, kv_type="f16", headroom_gb=4.0, extra_bytes=0, want_ctx=None):
    """Largest context this model can be served with on this machine, without swapping."""
    machine = machine or detect()
    info = read_gguf(model_path)
    per_tok = kv_bytes_per_token(info, kv_type) or 1
    available = machine.gpu_limit_gb - headroom_gb
    fixed = (info["bytes"] + extra_bytes) / GB + COMPUTE_OVERHEAD
    room = (available - fixed) * GB
    model_max = info["context_length"] or 131072
    max_ctx = int(min(room / per_tok, model_max, want_ctx or model_max)) if room > 0 else 0
    max_ctx = max_ctx // 1024 * 1024
    if max_ctx < MIN_CTX:
        need = fixed + MIN_CTX * per_tok / GB
        return Fit(False, max_ctx, round(need, 1), round(available, 1),
                   f"doesn't fit on {machine.id}: needs ~{need:.0f} GB at {MIN_CTX // 1024}k context, "
                   f"{available:.0f} GB available")
    need = fixed + max_ctx * per_tok / GB
    return Fit(True, max_ctx, round(need, 1), round(available, 1),
               f"fits on {machine.id} up to {max_ctx // 1024}k context (~{need:.0f} of {available:.0f} GB)")


# ---------------------------------------------------------------- memory pressure
def swapped_out_bytes():
    """Bytes swapped out since boot (macOS vm_stat Swapouts x page size), or None elsewhere.
    The tuner compares this before and after a candidate to reject settings that swap."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    page = re.search(r"page size of (\d+)", out)
    swaps = re.search(r"Swapouts:\s+(\d+)", out)
    return int(page.group(1)) * int(swaps.group(1)) if page and swaps else None


# ---------------------------------------------------------------- GPU residency
# On Apple Silicon Macs, once the system's GPU allocations exceed about half of RAM, the GPU driver evicts and re-maps memory on every command submission (70-95% of the
# server's CPU in the kernel, GPU idle), whatever iogpu.wired_limit_mb says. Servers that submit
# many small GPU jobs collapse; see the stall guard in docs/models.md.
RESIDENCY_FRACTION = 0.5


def memory_pressure_level():
    """macOS memory pressure: 1 normal, 2 warning, 4 critical (the system starts ending processes).
    None elsewhere."""
    try:
        out = subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], capture_output=True,
                             text=True, timeout=5).stdout
        return int(out.strip())
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


def gpu_residency_gb(machine, override=None):
    """GPU memory the driver keeps resident without churning (models.toml
    [machines.<id>] gpu_residency_gb overrides the half-of-RAM default)."""
    return float(override) if override else round(machine.ram_gb * RESIDENCY_FRACTION, 1)


def gpu_allocated_gb():
    """System-wide GPU allocation (IOAccelerator 'Alloc system memory'), or None."""
    try:
        out = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], capture_output=True,
                             text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(r'"Alloc system memory"=(\d+)', out)
    return int(m.group(1)) / GB if m else None


def _cpu_seconds(text):
    """ps time ('1:02:03.45', '10:39.90', '0:21.12') -> seconds."""
    parts = [float(p) for p in text.replace("-", ":").split(":")]
    total = 0.0
    for p in parts:
        total = total * 60 + p
    return total


def tree_cpu_seconds(pid):
    """(user, kernel) CPU seconds used so far by a process and all its descendants, or None.
    Some servers run their worker in its own process group, so a process-group sum would miss it."""
    try:
        out = subprocess.run(["ps", "-axo", "pid=,ppid=,utime=,stime="], capture_output=True, text=True,
                             timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    rows = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 4:
            rows[parts[0]] = (parts[1], _cpu_seconds(parts[2]), _cpu_seconds(parts[3]))
    if str(pid) not in rows:
        return None
    tree, frontier = {str(pid)}, [str(pid)]
    while frontier:
        parent = frontier.pop()
        for child, (ppid, _, _) in rows.items():
            if ppid == parent and child not in tree:
                tree.add(child)
                frontier.append(child)
    return (sum(rows[p][1] for p in tree), sum(rows[p][2] for p in tree))


def tree_pids(pid):
    """A process and all its descendants (pids), or [pid] if ps can't be read."""
    try:
        out = subprocess.run(["ps", "-axo", "pid=,ppid="], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return [pid]
    children = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2:
            children.setdefault(parts[1], []).append(parts[0])
    tree, frontier = {str(pid)}, [str(pid)]
    while frontier:
        for child in children.get(frontier.pop(), []):
            if child not in tree:
                tree.add(child)
                frontier.append(child)
    return sorted(int(p) for p in tree)


def group_cpu_seconds(pgid):
    """(user, kernel) CPU seconds used so far by a process group, or None."""
    try:
        out = subprocess.run(["ps", "-axo", "pgid=,utime=,stime="], capture_output=True, text=True,
                             timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    user = kernel = 0.0
    found = False
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == str(pgid):
            user, kernel, found = user + _cpu_seconds(parts[1]), kernel + _cpu_seconds(parts[2]), True
    return (user, kernel) if found else None
