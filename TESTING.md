# Testing GitView on Multiple Platforms

## Why Test on Multiple Platforms?

Even though GitView is a pure Python package (works everywhere), you should test on different platforms to ensure:
- Dependencies install correctly
- File paths work (Windows uses `\`, Unix uses `/`)
- Command-line tool works
- No platform-specific bugs

## Automated Testing with GitHub Actions

The `.github/workflows/test.yml` file automatically tests on:
- **Linux** (ubuntu-latest)
- **macOS** (macos-latest) - Intel and Apple Silicon
- **Windows** (windows-latest)

With Python versions: 3.8, 3.9, 3.10, 3.11, 3.12

## Manual Testing

### Linux (x86_64)
```bash
docker run -it python:3.11 bash
pip install gitview
gitview --version
```

### Linux (ARM64)
```bash
docker run --platform linux/arm64 -it python:3.11 bash
pip install gitview
gitview --version
```

### macOS
```bash
# Works on both Intel and Apple Silicon
pip install gitview
gitview --version
```

### Windows
```powershell
pip install gitview
gitview --version
```

## Testing from TestPyPI

Before publishing to the real PyPI, test installation from TestPyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    gitview
```

Note: `--extra-index-url` is needed because dependencies are on the real PyPI.

## Platform-Specific Considerations

### Windows
- Command is `gitview.exe` (but `gitview` also works)
- Paths use backslashes (`\`)
- Line endings are CRLF

### macOS
- Both Intel and Apple Silicon are supported
- M1/M2/M3 run natively (no Rosetta needed)

### Linux
- Works on x86_64, ARM64, and other architectures
- Tested on Ubuntu, Debian, Fedora, Alpine, etc.

## What You DON'T Need

Since GitView is pure Python, you DON'T need:
- ❌ Separate builds for each platform
- ❌ Platform-specific code
- ❌ Compiled binaries
- ❌ cibuildwheel or similar tools

The single `py3-none-any` wheel works everywhere!
