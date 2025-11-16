#!/bin/bash
# Publish GitView to PyPI (production)
# WARNING: This publishes to the real PyPI - test on TestPyPI first!

set -e  # Exit on error

echo "====================================="
echo "Publishing to PyPI (PRODUCTION)"
echo "====================================="
echo ""
echo "⚠️  WARNING: This will publish to the REAL PyPI"
echo "⚠️  Make sure you have tested on TestPyPI first!"
echo ""
read -p "Are you sure you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 1
fi

# Check if dist/ exists
if [ ! -d "dist" ]; then
    echo "Error: No dist/ directory found. Run ./scripts/build.sh first"
    exit 1
fi

# Check version doesn't already exist on PyPI
VERSION=$(python -c "import setup; print(setup.__version__)" 2>/dev/null || echo "0.1.0")
echo ""
echo "Publishing version: $VERSION"
echo ""

# Upload to PyPI
echo "Uploading to PyPI..."
echo "You will be prompted for your PyPI credentials"
echo "(Get API token from: https://pypi.org/manage/account/token/)"
echo ""

twine upload dist/*

echo ""
echo "====================================="
echo "✓ Published to PyPI!"
echo "====================================="
echo ""
echo "Your package is now available at:"
echo "  https://pypi.org/project/gitview/"
echo ""
echo "Users can install it with:"
echo "  pip install gitview"
echo ""
