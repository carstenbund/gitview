#!/bin/bash
# Build GitView distribution packages for PyPI
# Creates wheel (.whl) and source distribution (.tar.gz)

set -e  # Exit on error

echo "====================================="
echo "Building GitView Distribution"
echo "====================================="

# Clean previous builds
echo ""
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info

# Install/upgrade build tools
echo ""
echo "Installing build tools..."
pip install --upgrade pip setuptools wheel build twine

# Build the package
echo ""
echo "Building source and wheel distributions..."
python -m build

# List built files
echo ""
echo "====================================="
echo "Build complete! Files created:"
echo "====================================="
ls -lh dist/

# Verify the package
echo ""
echo "Verifying package..."
twine check dist/*

echo ""
echo "====================================="
echo "✓ Build successful!"
echo "====================================="
echo ""
echo "Next steps:"
echo "  1. Test on TestPyPI: ./scripts/publish-test.sh"
echo "  2. Publish to PyPI: ./scripts/publish.sh"
echo ""
