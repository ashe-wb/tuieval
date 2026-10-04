"""Getting started: find the model servers already running on this machine, add their models,
and (tuieval init in a terminal) walk a new workspace to its first Smoke check."""
import os
import sys

from . import engine
from . import workspace


def ask(question, default="", yes=False):
    """input() with a default; yes (or no terminal) takes the default without asking."""
    if yes or not sys.stdin.isatty():
        return default
    try:
        answer = input(f"{question} ").strip()
    except EOFError:
        return default
    return answer or default


def unregistered(cfg, found):
    """[(model id, detected server)] for models on running servers that models.toml doesn't have yet."""
    out = []
    for f in found:
        name = engine.server_for_url(cfg, f["url"])
        have = {m["served_name"] for m in cfg["models"] if m["server"] == name}
        out += [(mid, f) for mid in f["models"] if mid and mid not in have]
    return out


def where(f):
    return f["url"] + (f" ({f['usual']})" if f["usual"] else "")


def no_servers_help(cfg):
    """What to do when nothing is running: the three ways to get a model."""
    llama = next((s for s in cfg["servers"].values() if s.get("model_is_path") and s.get("cmd")), None)
    missing = engine.missing_program([str(a) for a in llama["cmd"]]) if llama else "llama-server"
    key = next((s.get("api_key_env") for s in cfg["servers"].values() if s.get("any_model")), None)
    ports = ", ".join(map(str, engine.detect_ports(cfg))) or "none (detect_ports is empty)"
    return [
        f"No model server is running (looked on ports {ports}). Three ways to get a model:",
        "  • An app that serves models (LM Studio, Ollama, vLLM): start its server, load a model, then run",
        "    tuieval add   (it finds the running server and lists its models)",
        "  • A GGUF file: tuieval add ~/path/to/model.gguf   (tuieval starts llama-server for it; "
        + ("installed ✓)" if not missing else f"{engine.install_hint(missing)})"),
        "  • A hosted model on OpenRouter: tuieval run --only openrouter:<model id>   "
        + (f"({key} is set ✓)" if key and os.environ.get(key) else f"(export {key or 'OPENROUTER_API_KEY'} first)"),
    ]


def add_found(models_path, picks):
    """Add (model id, detected server) picks to models.toml; returns the labels added."""
    labels = []
    for mid, f in picks:
        try:
            server = engine.ensure_server(models_path, f["url"], f["port"])
            label, _ = engine.add_model(models_path, mid, server)
            labels.append(label)
            print(f"  added {label}  ({mid} on {where(f)}, server [servers.{server}])")
        except engine.ConfigError as e:
            print(f"  not added: {mid}: {e}")
    return labels


def choose(items, yes=False, default="all"):
    """Numbered pick from items (strings): returns the chosen indexes. Accepts 'all', 'none' or '1,3'."""
    for i, text in enumerate(items, 1):
        print(f"  {i}) {text}")
    answer = ask(f"Add which? numbers like 1,3 · all · none [{default}]:", default, yes).lower()
    if answer in ("all", "a", "y", "yes"):
        return list(range(len(items)))
    if answer in ("none", "n", "no", ""):
        return []
    picked = []
    for part in answer.replace(" ", "").split(","):
        if part.isdigit() and 1 <= int(part) <= len(items):
            picked.append(int(part) - 1)
    return picked


def add_detected(models_path, yes=False):
    """tuieval add without a model: list what's running and add the chosen models. Returns labels."""
    cfg = engine.load_config(models_path)
    found = engine.detect_servers(engine.detect_ports(cfg))
    if not found:
        print("\n".join(no_servers_help(cfg)))
        return []
    items = unregistered(cfg, found)
    if not items:
        print("Every model on the running servers is already added: " + ", ".join(where(f) for f in found))
        return []
    print(f"Models on running servers ({len(items)}):")
    picks = choose([f"{mid}  on {where(f)}" for mid, f in items], yes,
                   default="all" if len(items) <= 3 else "none")
    return add_found(models_path, [items[i] for i in picks])


def guided_init(folder, yes=False):
    """After `tuieval init` in a terminal: add running models, offer a starter pack, run Smoke."""
    from . import scaffold
    os.environ[workspace.ENV] = folder
    models_path = os.path.join(folder, "models.toml")
    cfg = engine.load_config(models_path)
    labels = []
    if not cfg["models"]:
        print("\nLooking for model servers running on this machine…")
        labels = add_detected(models_path, yes)
    cfg = engine.load_config(models_path)
    packs_dir = os.path.join(folder, "packs")
    if not any(os.path.isfile(os.path.join(packs_dir, d, "pack.toml")) for d in os.listdir(packs_dir)):
        if ask("\nCreate a starter pack of example questions to try? [Y/n]", "y", yes).lower().startswith("y"):
            scaffold.new_pack("starter", packs_dir=packs_dir)
            print("  created packs/starter/ (example questions: replace them with your own later)")
    ready = [m["label"] for m in cfg["models"]]
    has_pack = any(os.path.isfile(os.path.join(packs_dir, d, "pack.toml")) for d in os.listdir(packs_dir))
    shown = os.path.relpath(folder) if folder.startswith(os.getcwd()) else folder
    cd = f"cd {shown} && " if os.path.abspath(folder) != os.getcwd() else ""
    if ready and has_pack and ask(f"\nRun a Smoke check now (3 questions per pack, {', '.join(ready)})? [Y/n]",
                                  "y", yes).lower().startswith("y"):
        from . import run_evals
        try:
            run_evals.main(["--tier", "smoke", "--brief", "--only", ",".join(ready)])
        except SystemExit as e:
            if e.code not in (None, 0):
                print(e.code if isinstance(e.code, str) else "the Smoke check didn't finish")
        print(f"\nNext: {cd}tuieval   (opens the TUI: r shows results, ? explains any screen)")
    elif not ready:
        print(f"\nNext: add a model (see above), then {cd}tuieval")
    else:
        print(f"\nNext: {cd}tuieval   (opens the TUI; ? explains any screen)")
    return labels
