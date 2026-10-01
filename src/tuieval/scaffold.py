"""tuieval init (a new workspace) and tuieval new-pack (a new pack from a starter template)."""
import argparse
import os
import re
import shutil
import sys

from . import workspace

TEMPLATES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
PACK_TEMPLATES = os.path.join(TEMPLATES, "packs")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

WORKSPACE_GITIGNORE = """\
# Written by tuieval. Keep results/ and tuning/ if you want to sync them between machines.
logs/
reports/
results/smoke/
*.partial.jsonl
*.sittings
"""


def graders_available():
    return sorted(d for d in os.listdir(PACK_TEMPLATES) if os.path.isdir(os.path.join(PACK_TEMPLATES, d)))


def init(folder):
    """Create a workspace in folder. Returns the files created (existing files are left alone)."""
    folder = os.path.abspath(os.path.expanduser(folder))
    os.makedirs(folder, exist_ok=True)
    created = []
    for name, src in (("models.toml", os.path.join(TEMPLATES, "models.toml")),):
        dst = os.path.join(folder, name)
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
            created.append(name)
    gitignore = os.path.join(folder, ".gitignore")
    if not os.path.exists(gitignore):
        with open(gitignore, "w") as f:
            f.write(WORKSPACE_GITIGNORE)
        created.append(".gitignore")
    for d in ("packs", "graders"):
        if not os.path.isdir(os.path.join(folder, d)):
            os.makedirs(os.path.join(folder, d))
            created.append(d + "/")
    return created


def new_pack(name, grader="answer", label=None, group=None, packs_dir=None):
    """Copy the starter template for a grader to packs/<name>/. Returns its path. Raises ValueError."""
    if not NAME_RE.match(name):
        raise ValueError(f"pack name {name!r} must be lowercase letters, digits, '.', '_' or '-'")
    if grader not in graders_available():
        raise ValueError(f"no starter template for grader {grader!r}; templates: {', '.join(graders_available())}")
    packs_dir = packs_dir or workspace.path("packs")
    dst = os.path.join(packs_dir, name)
    if os.path.exists(dst):
        raise ValueError(f"{dst} already exists")
    shutil.copytree(os.path.join(PACK_TEMPLATES, grader), dst)
    toml = os.path.join(dst, "pack.toml")
    with open(toml) as f:
        text = f.read()
    title = label or name.replace("-", " ").replace("_", " ").capitalize()
    text = text.replace("__LABEL__", title.replace('"', "'")).replace("__GROUP__", (group or title).replace('"', "'"))
    with open(toml, "w") as f:
        f.write(text)
    return dst


def cmd_init(argv):
    p = argparse.ArgumentParser(prog="tuieval init", description="Create a workspace: models.toml, packs/, graders/.")
    p.add_argument("folder", nargs="?", default=None, help="default: the workspace (current folder or --workspace)")
    a = p.parse_args(argv)
    folder = a.folder or workspace.root()
    created = init(folder)
    shown = os.path.relpath(folder) if os.path.abspath(folder).startswith(os.getcwd()) else folder
    if created:
        print(f"workspace ready in {shown}: created {', '.join(created)}")
    else:
        print(f"{shown} is already a workspace; nothing changed")
    cd = "" if os.path.abspath(folder) == os.getcwd() else f"cd {shown}\n  "
    print(f"next:\n  {cd}tuieval new-pack my-first-pack       # a pack of example questions to edit\n"
          "  tuieval add ~/models/Some-Model.gguf   # or press a in the TUI\n"
          "  tuieval                                # open the TUI")


def cmd_new_pack(argv):
    p = argparse.ArgumentParser(prog="tuieval new-pack",
                                description="Create packs/<name>/ from a starter template with example tests.")
    p.add_argument("name", help="folder name, e.g. support-bot")
    p.add_argument("--grader", default="answer", choices=graders_available(),
                   help="how answers are scored (default: answer)")
    p.add_argument("--label", help="name shown in the TUI (default: from the folder name)")
    p.add_argument("--group", help="use case it counts toward (default: the label)")
    p.add_argument("--packs-dir", help="default: the workspace's packs/")
    a = p.parse_args(argv)
    try:
        path = new_pack(a.name, a.grader, a.label, a.group, a.packs_dir)
    except ValueError as e:
        sys.exit(str(e))
    shown = os.path.relpath(path)
    print(f"created {shown}/ with example tests for the {a.grader} grader")
    print(f"next: replace the examples in {shown}/tests.yaml with your own questions, then\n"
          f"  tuieval selftest {a.name}     # checks every reference passes and every wrong answer fails\n"
          "guide: docs/writing-packs.md")
