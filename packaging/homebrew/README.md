# Releasing, and installing with Homebrew

tuieval is a normal Python package, so it installs with `pipx install tuieval` or `pip install tuieval` anywhere, and with Homebrew through a tap.

## Release

1. Bump `__version__` in `src/tuieval/__init__.py` and add a `CHANGELOG.md` entry.
2. Build and check: `python -m pip install build twine && python -m build && twine check dist/*`
3. Publish to PyPI: `twine upload dist/*` (or a trusted-publishing GitHub Action), and tag the release: `git tag v0.1.0 && git push --tags`.

## Homebrew tap

1. Create a repository named `homebrew-tap` under your GitHub account, and copy `tuieval.rb` to `Formula/tuieval.rb` in it.
2. Point `url` at the release's sdist (`https://files.pythonhosted.org/packages/source/t/tuieval/tuieval-<version>.tar.gz`) and set `sha256` (`shasum -a 256 dist/tuieval-<version>.tar.gz`).
3. `brew update-python-resources Formula/tuieval.rb` fills in the dependency resources.
4. Test: `brew install --build-from-source ./Formula/tuieval.rb && brew test tuieval && brew audit --strict --online tuieval`.
5. Commit and push. Users install with `brew install ashe-wb/tap/tuieval`.

For every later release, update `url` and `sha256` (and rerun step 3 if dependencies changed).
