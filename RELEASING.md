# Releasing

Versions come from git (hatch-vcs). Nobody edits a version number by hand.

| What | Version | Who gets it |
|---|---|---|
| A tag `vX.Y.Z` | `X.Y.Z` | `pip install tuieval` |
| Every other commit on `main` | `X.(Y+1).0.devN`, N = commits since the last tag | `pip install --pre tuieval` |

CI (`.github/workflows/test.yml`) runs the tests on every push. When they pass, the `publish` job builds the package, checks it with `twine check`, and uploads it to PyPI with trusted publishing (no token is stored anywhere): every push to `main` as a dev build, every `v*` tag as a release.

## Day to day

1. Commit and push to `main`. Add a line under `## Unreleased` in `CHANGELOG.md` for anything users would notice.
2. CI publishes the dev build. Check it: `pipx install --force --pip-args=--pre tuieval && tuieval --version`.

## A release

Cut one when there's something worth announcing, not for every change. Semantic versioning: PATCH for fixes, MINOR for features (and anything that changes behaviour while the major version is 0), MAJOR for breaking changes after 1.0.

1. In `CHANGELOG.md`, rename `## Unreleased` to `## X.Y.Z` and add a new empty `## Unreleased` above it. Commit and push.
2. Tag that commit and push the tag: `git tag -a vX.Y.Z -m "X.Y.Z" && git push origin vX.Y.Z`.
3. CI publishes `X.Y.Z`. Check a fresh install: `pipx install --force tuieval==X.Y.Z && tuieval --version`.

## One-time setup (done once per PyPI project)

On PyPI, under the project's Publishing settings, add a trusted publisher: owner `ashe-wb`, repository `tuieval`, workflow `test.yml`, environment `pypi`.

## Building by hand

`python -m pip install build twine && python -m build && twine check dist/*`. A build needs the git history and tags (a full clone), or it can't tell its version.
