# Installation Guide

## Three Ways to Use GitView

### 1. System Installation (Recommended)

**What happens:** Creates a `gitview` command available system-wide

```bash
pip install -e .
```

This reads the configuration from `pyproject.toml` and `setup.py`:
- Installs all dependencies from `requirements.txt`
- Creates entry point: `gitview` → `gitview.cli:main()`
- Puts executable wrapper in `/usr/local/bin/gitview` (or platform equivalent)

**Verify:**
```bash
which gitview  # Shows: /usr/local/bin/gitview
gitview --version
```

**Files involved:**
- `pyproject.toml` - Modern Python packaging config
- `setup.py` - Traditional setup script (compatible with pyproject.toml)
- `requirements.txt` - Dependencies list

---

### 2. Direct Execution (No Installation)

**What happens:** Run directly from the repo using the wrapper script

```bash
pip install -r requirements.txt  # Dependencies only
./bin/gitview analyze
```

**Files involved:**
- `bin/gitview` - Executable Python script that imports `gitview.cli:main()`

**Useful for:**
- Development without polluting system PATH
- Testing changes without reinstalling
- Running from a git clone without pip install

---

### 3. Python Module Mode

**What happens:** Run as a Python module

```bash
pip install -r requirements.txt  # Dependencies only
python -m gitview.cli analyze
```

**Files involved:**
- `gitview/cli.py` - Contains `main()` function and CLI definition

**Useful for:**
- Debugging with Python debugger
- Running in environments without executable permissions
- Scripting where you want explicit Python invocation

---

## Understanding the Entry Point

All three methods ultimately call the same function: `gitview.cli:main()`

### Method 1: System Installation
```
gitview
  ↓
/usr/local/bin/gitview (auto-generated wrapper)
  ↓
gitview.cli:main()
```

### Method 2: Direct Execution
```
./bin/gitview (explicit wrapper script)
  ↓
gitview.cli:main()
```

### Method 3: Module Mode
```
python -m gitview.cli
  ↓
gitview.cli:main()
```

---

## Files Explained

### `setup.py`
Traditional Python setup script. Defines:
- Package metadata (name, version, author)
- Dependencies from `requirements.txt`
- Entry points (console scripts)

### `pyproject.toml`
Modern Python packaging standard (PEP 518). Defines:
- Build system requirements
- Project metadata
- Dependencies
- Entry points via `[project.scripts]`

### `bin/gitview`
Simple executable Python script:
```python
#!/usr/bin/env python3
if __name__ == "__main__":
    from gitview.cli import main
    main()
```

### How `/usr/local/bin/gitview` Gets Created

When you run `pip install -e .`:

1. Pip reads `[project.scripts]` from `pyproject.toml`:
   ```toml
   [project.scripts]
   gitview = "gitview.cli:main"
   ```

2. Pip generates a wrapper script at `/usr/local/bin/gitview`:
   ```python
   #!/usr/bin/env python
   # -*- coding: utf-8 -*-
   import re
   import sys
   from gitview.cli import main
   if __name__ == '__main__':
       sys.exit(main())
   ```

3. Makes it executable (`chmod +x`)

4. Now `gitview` command works from anywhere!

---

## Verification

Run the verification script to check your installation:

```bash
python verify_installation.py
```

This checks:
- ✓ Python version (3.8+)
- ✓ All dependencies installed
- ✓ `gitview` command availability
- ✓ LLM backends configured
