#!/bin/bash
# Example script demonstrating critical examination mode usage

set -e

echo "GitView Critical Examination Mode Examples"
echo "=========================================="
echo ""

# Check if ANTHROPIC_API_KEY is set
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Error: ANTHROPIC_API_KEY not set"
    echo "Please set it with: export ANTHROPIC_API_KEY='your-key'"
    exit 1
fi

# Example 1: Basic critical mode
echo "Example 1: Basic critical mode (no goals file)"
echo "------------------------------------------------"
echo "Command: gitview analyze --critical --output ./output/critical-basic"
echo ""
# Uncomment to run:
# gitview analyze --critical --output ./output/critical-basic

# Example 2: Critical mode with project goals
echo "Example 2: Critical mode with project goals file"
echo "-------------------------------------------------"
echo "Command: gitview analyze --critical --todo examples/PROJECT_GOALS.md --output ./output/critical-goals"
echo ""
# Uncomment to run:
# gitview analyze --critical --todo examples/PROJECT_GOALS.md --output ./output/critical-goals

# Example 3: Critical mode with custom directives
echo "Example 3: Critical mode with custom analysis directives"
echo "---------------------------------------------------------"
echo "Command: gitview analyze --critical \\"
echo "  --directives 'Focus on security vulnerabilities and performance issues' \\"
echo "  --output ./output/critical-security"
echo ""
# Uncomment to run:
# gitview analyze --critical \
#   --directives "Focus on security vulnerabilities and performance issues" \
#   --output ./output/critical-security

# Example 4: Combined - goals + directives
echo "Example 4: Combined - goals file + custom directives"
echo "-----------------------------------------------------"
echo "Command: gitview analyze --critical \\"
echo "  --todo examples/PROJECT_GOALS.md \\"
echo "  --directives 'Emphasize testing gaps and code quality issues' \\"
echo "  --output ./output/critical-combined"
echo ""
# Uncomment to run:
# gitview analyze --critical \
#   --todo examples/PROJECT_GOALS.md \
#   --directives "Emphasize testing gaps and code quality issues" \
#   --output ./output/critical-combined

# Example 5: Critical mode for specific branch
echo "Example 5: Critical mode for specific feature branch"
echo "----------------------------------------------------"
echo "Command: gitview analyze --critical \\"
echo "  --branch feature/new-feature \\"
echo "  --todo examples/PROJECT_GOALS.md \\"
echo "  --output ./output/critical-feature-branch"
echo ""
# Uncomment to run:
# gitview analyze --critical \
#   --branch feature/new-feature \
#   --todo examples/PROJECT_GOALS.md \
#   --output ./output/critical-feature-branch

# Example 6: Critical incremental analysis
echo "Example 6: Critical incremental analysis (update existing analysis)"
echo "-------------------------------------------------------------------"
echo "Command: gitview analyze --critical \\"
echo "  --incremental \\"
echo "  --todo examples/PROJECT_GOALS.md \\"
echo "  --output ./output/critical-incremental"
echo ""
# Uncomment to run (after running a full analysis first):
# gitview analyze --critical \
#   --incremental \
#   --todo examples/PROJECT_GOALS.md \
#   --output ./output/critical-incremental

echo ""
echo "=========================================="
echo "To run any example, uncomment the corresponding command in this script"
echo "or copy the command and run it directly"
echo ""
echo "Output files will contain:"
echo "  - history_story.md    (Critical assessment report)"
echo "  - timeline.md         (Critical timeline)"
echo "  - history_data.json   (Structured data)"
echo "  - repo_history.jsonl  (Raw git history)"
