# Releasing

1. Bump `__version__` in `src/tuieval/__init__.py` and add a `CHANGELOG.md` entry.
2. Commit, push, and wait for CI to pass.
3. Build from a clean checkout and check: `python -m pip install build twine && python -m build && twine check dist/*`
4. Publish to PyPI: `twine upload dist/*` (username `__token__`, password: a PyPI API token).
5. Check a fresh install: `pipx install --force tuieval==<version> && tuieval --version`
