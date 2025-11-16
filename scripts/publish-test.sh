#!/bin/bash
# Publish GitView to TestPyPI for testing
# Use this to test your package before publishing to the real PyPI

set -e  # Exit on error

echo "====================================="
echo "Publishing to TestPyPI"
echo "====================================="

# Check if dist/ exists
if [ ! -d "dist" ]; then
    echo "Error: No dist/ directory found. Run ./scripts/build.sh first"
    exit 1
fi

# Upload to TestPyPI
echo ""
echo "Uploading to TestPyPI..."
echo "You will be prompted for your TestPyPI credentials"
echo "(Get API token from: https://test.pypi.org/manage/account/token/)"
echo ""

twine upload --repository testpypi dist/*

echo ""
echo "====================================="
echo "✓ Published to TestPyPI!"
echo "====================================="
echo ""
echo "To test the installation:"
echo "  pip install --index-url https://test.pypi.org/simple/ gitview"
echo ""
echo "View your package at:"
echo "  https://test.pypi.org/project/gitview/"
echo ""
