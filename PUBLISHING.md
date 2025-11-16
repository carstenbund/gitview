# Publishing GitView to PyPI

This guide explains how to publish GitView to PyPI so users can install it with `pip install gitview`.

## Prerequisites

1. **PyPI Account**: Create accounts on both:
   - TestPyPI (for testing): https://test.pypi.org/account/register/
   - PyPI (production): https://pypi.org/account/register/

2. **API Tokens**: Generate API tokens for uploading:
   - TestPyPI: https://test.pypi.org/manage/account/token/
   - PyPI: https://pypi.org/manage/account/token/

3. **Build Tools**: Install required tools:
   ```bash
   pip install --upgrade pip setuptools wheel build twine
   ```

## Publishing Workflow

### Step 1: Update Version

Update the version number in:
- `setup.py` (line 16)
- `pyproject.toml` (line 7)
- `gitview/__init__.py` (line 3)

```python
__version__ = "0.1.0"  # Increment for new releases
```

### Step 2: Build the Package

```bash
./scripts/build.sh
```

This creates:
- `dist/gitview-0.1.0-py3-none-any.whl` (wheel distribution)
- `dist/gitview-0.1.0.tar.gz` (source distribution)

The script also:
- Cleans previous builds
- Installs build tools
- Verifies the package with `twine check`

### Step 3: Test on TestPyPI

**Always test on TestPyPI first!**

```bash
./scripts/publish-test.sh
```

You'll be prompted for your TestPyPI API token.

After uploading, test the installation:

```bash
# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ gitview

# Test it works
gitview --version
gitview --help
```

View your test package at: https://test.pypi.org/project/gitview/

### Step 4: Publish to PyPI (Production)

Once tested, publish to the real PyPI:

```bash
./scripts/publish.sh
```

⚠️ **Warning**: You cannot overwrite versions on PyPI. Once published, you cannot delete or replace a version.

### Step 5: Verify

Check your package is live:
- Package page: https://pypi.org/project/gitview/
- Install it: `pip install gitview`

## Manual Publishing

If you prefer manual steps:

### Build
```bash
# Clean previous builds
rm -rf build/ dist/ *.egg-info

# Build distributions
python -m build

# Verify package
twine check dist/*
```

### Upload to TestPyPI
```bash
twine upload --repository testpypi dist/*
```

### Upload to PyPI
```bash
twine upload dist/*
```

## Authentication

### Using API Tokens (Recommended)

When prompted for username, use `__token__`

When prompted for password, use your API token (starts with `pypi-`)

### Using ~/.pypirc

Create `~/.pypirc` to avoid entering credentials:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR_PYPI_API_TOKEN_HERE

[testpypi]
username = __token__
password = pypi-YOUR_TESTPYPI_API_TOKEN_HERE
repository = https://test.pypi.org/legacy/
```

Secure the file:
```bash
chmod 600 ~/.pypirc
```

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):
- **MAJOR.MINOR.PATCH** (e.g., 1.2.3)
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

Examples:
- `0.1.0` - Initial alpha release
- `0.2.0` - Add new features
- `0.2.1` - Bug fix
- `1.0.0` - First stable release

## Pre-release Versions

For alpha/beta releases:
- `0.1.0a1` - Alpha 1
- `0.1.0b1` - Beta 1
- `0.1.0rc1` - Release candidate 1

## Package Naming

The package name on PyPI is **gitview**. Users will install with:

```bash
pip install gitview
```

This is defined in:
- `setup.py`: `name="gitview"`
- `pyproject.toml`: `name = "gitview"`

## What Gets Included

Files included in the distribution are controlled by:

1. **`MANIFEST.in`**: Specifies additional files to include
2. **`setup.py`**: `find_packages()` determines Python packages
3. **`.gitignore`**: Prevents unwanted files

Included:
- All Python packages (`gitview/`)
- `README.md`, `LICENSE`, `INSTALL.md`
- `requirements.txt`
- `bin/gitview` executable wrapper

Excluded:
- Tests, examples
- Git files, cache files
- Output directories

## Testing Your Package

After publishing to TestPyPI:

```bash
# Create a new virtual environment
python -m venv test-env
source test-env/bin/activate  # or `test-env\Scripts\activate` on Windows

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ gitview

# Test it
gitview --version
gitview analyze --help
python verify_installation.py
```

Note: `--extra-index-url` is needed because dependencies (anthropic, openai, etc.) are on the real PyPI.

## Troubleshooting

### "File already exists" error
- You cannot upload the same version twice
- Increment the version number
- Or delete the package on TestPyPI and try again (only possible on TestPyPI)

### "Invalid distribution" error
- Run `twine check dist/*` to see what's wrong
- Common issues:
  - Missing README.md
  - Invalid RST in long_description
  - Missing required metadata

### Dependencies not installing
- Make sure `requirements.txt` lists all dependencies
- Check `install_requires` in `setup.py`
- Verify `dependencies` in `pyproject.toml`

### Command not found after install
- Verify entry point in `pyproject.toml`: `gitview = "gitview.cli:main"`
- Check `console_scripts` in `setup.py`
- Try reinstalling: `pip install --force-reinstall gitview`

## After Publishing

1. **Create a Git tag** for the release:
   ```bash
   git tag -a v0.1.0 -m "Release version 0.1.0"
   git push origin v0.1.0
   ```

2. **Create a GitHub Release**:
   - Go to: https://github.com/carstenbund/gitview/releases
   - Click "Create a new release"
   - Select the tag
   - Add release notes

3. **Update documentation**:
   - Update README.md with installation instructions
   - Add release notes
   - Update CHANGELOG.md (if you have one)

## Resources

- **PyPI**: https://pypi.org
- **TestPyPI**: https://test.pypi.org
- **Python Packaging Guide**: https://packaging.python.org/
- **Twine Documentation**: https://twine.readthedocs.io/
- **Semantic Versioning**: https://semver.org/
