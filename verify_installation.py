#!/usr/bin/env python3
"""Verify GitView installation."""

import sys
import subprocess


def check_module(module_name):
    """Check if a Python module can be imported."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def check_command(command):
    """Check if a command exists."""
    try:
        result = subprocess.run(
            [command, '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def main():
    """Run installation verification."""
    print("GitView Installation Verification")
    print("=" * 50)

    issues = []

    # Check Python version
    print(f"\n✓ Python version: {sys.version.split()[0]}")
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ required")

    # Check required modules
    print("\nChecking required dependencies:")
    required = [
        ('gitview', 'gitview'),
        ('GitPython', 'git'),
        ('anthropic', 'anthropic'),
        ('openai', 'openai'),
        ('click', 'click'),
        ('rich', 'rich'),
        ('pydantic', 'pydantic'),
        ('requests', 'requests'),
    ]

    for display_name, import_name in required:
        if check_module(import_name):
            print(f"  ✓ {display_name}")
        else:
            print(f"  ✗ {display_name} (missing)")
            issues.append(f"Missing module: {display_name}")

    # Check gitview command
    print("\nChecking gitview command:")
    if check_command('gitview'):
        print("  ✓ gitview command available")

        # Get version
        result = subprocess.run(
            ['gitview', '--version'],
            capture_output=True,
            text=True
        )
        print(f"  ✓ {result.stdout.strip()}")
    else:
        print("  ✗ gitview command not found")
        print("  → Try: python -m gitview.cli --help")
        issues.append("gitview command not in PATH")

    # Check backend availability
    print("\nChecking LLM backend availability:")
    import os

    if os.environ.get('ANTHROPIC_API_KEY'):
        print("  ✓ ANTHROPIC_API_KEY set")
    else:
        print("  - ANTHROPIC_API_KEY not set")

    if os.environ.get('OPENAI_API_KEY'):
        print("  ✓ OPENAI_API_KEY set")
    else:
        print("  - OPENAI_API_KEY not set")

    # Check if Ollama is available
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=2)
        if response.status_code == 200:
            print("  ✓ Ollama server running at http://localhost:11434")
        else:
            print("  - Ollama server not responding")
    except:
        print("  - Ollama server not available at http://localhost:11434")

    # Summary
    print("\n" + "=" * 50)
    if issues:
        print("\n⚠ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nInstallation incomplete. Run: pip install -e .")
        return 1
    else:
        print("\n✓ Installation verified successfully!")
        print("\nYou can now use: gitview analyze")
        return 0


if __name__ == '__main__':
    sys.exit(main())
