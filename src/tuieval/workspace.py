"""The workspace: the folder that holds your evals, apart from the tuieval code.

    my-evals/
      models.toml       your servers and models (marks the folder as a workspace)
      packs/            your eval packs (see docs/writing-packs.md)
      graders/          optional: your own graders, one .py file each
      results/ logs/ tuning/ reports/ presets.toml    written by tuieval

tuieval uses the folder named by --workspace or $TUIEVAL_HOME, otherwise the current folder.
`tuieval init` creates one. Resolved on every call (not at import), so the CLI can set it first.
"""
import os

ENV = "TUIEVAL_HOME"
MARKER = "models.toml"


def root():
    return os.path.abspath(os.path.expanduser(os.environ.get(ENV) or os.getcwd()))


def path(*parts):
    return os.path.join(root(), *parts)


def is_workspace(folder=None):
    return os.path.isfile(os.path.join(folder or root(), MARKER))
