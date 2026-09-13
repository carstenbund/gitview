# Publishing GitView to PyPI

GitView is published manually: the maintainer builds the distributions locally
and uploads them with `twine`. Nothing in CI touches PyPI, and no publishing
credentials live in the repository or in GitHub.

## Prerequisites

1. **PyPI account and API token** — https://pypi.org/manage/account/token/.
   A TestPyPI account and token are useful for rehearsing:
   https://test.pypi.org/manage/account/token/
2. **Build tools**:
   ```bash
   pip install --upgrade build twine
   ```

## Release checklist

### 1. Land the work and check CI

Everything to be released must be merged into `main`, and the test matrix on
`main` must be green. A release is cut from `main`, never from a branch.

### 2. Update the version

The version is defined in **one** place:

```
gitview/__init__.py:    __version__ = "0.7.1"
```

`pyproject.toml` reads it dynamically (`[tool.setuptools.dynamic]`) and
`setup.py` imports it, so nothing else needs editing. Verify:

```bash
python -m gitview.cli --version
```

### 3. Update the changelog

Add a section for the new version to [`CHANGELOG.md`](CHANGELOG.md), grouped
into *Added*, *Changed* and *Fixed*. Describe behaviour, not commits.

### 4. Commit and tag

```bash
git commit -am "chore(release): 0.7.1"
git tag -a v0.7.1 -m "GitView 0.7.1"
git push origin main
git push origin v0.7.1
```

The tag is what maps a PyPI release back to the history; every published
version should have one.

### 5. Build

```bash
./scripts/build.sh
```

This cleans `build/`, `dist/` and `*.egg-info`, runs `python -m build`, and
verifies the result with `twine check`. It produces:

- `dist/gitview-<version>-py3-none-any.whl`
- `dist/gitview-<version>.tar.gz`

Sanity-check the wheel before uploading anything:

```bash
python -m venv /tmp/gv && /tmp/gv/bin/pip install -q dist/gitview-*.whl
/tmp/gv/bin/gitview --version && /tmp/gv/bin/gitview --help
```

### 6. Rehearse on TestPyPI (optional but advised for a minor bump)

```bash
./scripts/publish-test.sh          # prompts for the TestPyPI token
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ gitview
```

The extra index is needed because TestPyPI does not carry GitView's
dependencies.

### 7. Publish

```bash
./scripts/publish.sh               # asks for confirmation, then the PyPI token
```

When `twine` prompts for a username, use `__token__` and paste the API token as
the password.

> **A version on PyPI cannot be replaced or reused.** If a release is broken,
> yank it on PyPI and publish a new patch version.

### 8. Verify

```bash
pip install --upgrade gitview
gitview --version
```

## Version numbering

- **Patch** (`0.7.0` → `0.7.1`): fixes, no new command or option.
- **Minor** (`0.7.x` → `0.8.0`): a new command, a new pipeline stage, or a
  change in what an existing command writes.
- **Major**: reserved for a `1.0` that declares the CLI stable.

## If a release goes wrong

1. Yank the version on PyPI (project → Manage → Releases → Yank). Yanking hides
   it from new installs while leaving pinned installs working.
2. Fix `main`, bump the patch version, tag it, and publish again.
3. Delete the bad tag only if it was never pushed; otherwise leave it and note
   the problem in the changelog.
