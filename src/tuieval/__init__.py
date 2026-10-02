"""tuieval: evaluate local and frontier models on your own eval packs, in the terminal."""
try:
    from ._version import __version__   # written by the build from the git tag (RELEASING.md)
except ImportError:                      # a source tree that was never built or installed
    __version__ = "0+unknown"
