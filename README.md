# GitView

[![Github-CI][github-ci]][github-link]
[![Coverage Status][codecov-badge]][codecov-link]
[![PyPI][pypi-badge]][pypi-link]
[![PyPI - Downloads][install-badge]][install-link]


**Git history analyzer with LLM-powered narrative generation**

GitView extracts your repository's git history and uses AI to generate compelling narratives about how your codebase evolved. Instead of manually reading through thousands of commits, get a comprehensive story of your project's journey.

Example run on this repository: 

[[(https://github.com/carstenbund/gitview/blob/main/output/history_story.md)] 
](https://github.com/carstenbund/gitview/blob/main/output/history_story.md)

## Features

- **Comprehensive History Extraction**: Extracts commit metadata, LOC changes, language breakdown, README evolution, comment analysis, and more
- **GitHub PR/Review Enrichment**: Optionally enrich commits with Pull Request context, review comments, and collaboration data via GitHub's GraphQL API
- **Smart Chunking**: Automatically divides history into meaningful "phases" or "epochs" based on significant changes
- **LLM-Powered Summaries**: Uses Claude to generate narrative summaries for each phase
- **Global Story Generation**: Combines phase summaries into executive summaries, timelines, technical retrospectives, and deletion stories
- **Storyline Tracking**: Track narrative threads (features, refactoring efforts, bug campaigns) across phases with automatic detection and lifecycle management
- **Multiple Output Formats**: Generates markdown reports, JSON data, and timelines
- **Critical Examination Mode**: Objective assessment focused on gaps, technical debt, and alignment with project goals (perfect for project leads)

## Installation

### Option 1: Install from PyPI (recommended)

Install the published package directly from PyPI to add the `gitview` command to your PATH:

```bash
pip3 install gitview

# Confirm the CLI is available
gitview --version
gitview --help
```

### Option 2: Install from source (editable)

This installs directly from the repository in editable mode so local changes take effect immediately:

```bash
# Clone the repository
git clone https://github.com/yourusername/gitview.git
cd gitview

# Install in editable mode with dependencies
pip install -e .

# The gitview command is now available system-wide
gitview --version
gitview --help
```

**How it works:** The `pip install -e .` command reads `pyproject.toml` and `setup.py`, which define an entry point that creates `/usr/local/bin/gitview` (or similar on Windows) that calls `gitview.cli:main`.

### Option 3: Run directly from repo (no installation)

Use the executable wrapper in `bin/`:

```bash
# Clone the repository
git clone https://github.com/yourusername/gitview.git
cd gitview

# Install dependencies only
pip install -r requirements.txt

# Run directly from the repo
./bin/gitview --version
./bin/gitview analyze

# Or add bin/ to your PATH
export PATH="$PWD/bin:$PATH"
gitview analyze
```

### Option 4: Run as Python module

```bash
# Install dependencies
pip install -r requirements.txt

# Run as a module
python -m gitview.cli --help
python -m gitview.cli analyze
```

### Verify Installation

Run the verification script to check everything is set up correctly:

```bash
python verify_installation.py
```

This will check:
- Python version (3.8+ required)
- All required dependencies
- `gitview` command availability
- LLM backend configuration (API keys, Ollama server)

### Troubleshooting Installation

If `gitview` command is not found after installation:

```bash
# Option 1: Use full path to module
python -m gitview.cli analyze

# Option 2: Reinstall in editable mode
pip uninstall gitview -y
pip install -e .

# Option 3: Check if it's in your PATH
which gitview  # Unix/Linux/Mac
where gitview  # Windows
```

## Quick Start

```bash
# Using Anthropic Claude (default)
export ANTHROPIC_API_KEY="your-api-key-here"
gitview analyze

# Using OpenAI GPT
export OPENAI_API_KEY="your-api-key-here"
gitview analyze --backend openai

# Using local Ollama (no API key needed)
gitview analyze --backend ollama --model llama3

# With GitHub PR/review enrichment (richer narratives!)
export GITHUB_TOKEN="ghp_your_token_here"
gitview analyze --repo owner/repo --github-token $GITHUB_TOKEN

# Critical examination mode (for project leads)
gitview analyze --critical --todo GOALS.md

# Skip LLM summarization (just extract and chunk)
gitview analyze --skip-llm
```

## Usage

### Full Analysis Pipeline

The main command runs the complete pipeline: extract → chunk → summarize → story → output

```bash
gitview analyze [OPTIONS]

Options:
  -r, --repo PATH              Path to git repository (default: current directory)
  -o, --output PATH            Output directory (default: "output")
  -s, --strategy STRATEGY      Chunking strategy: fixed, time, or adaptive (default: adaptive)
  --chunk-size INTEGER         Chunk size for fixed strategy (default: 50)
  --max-commits INTEGER        Maximum commits to analyze
  --branch TEXT                Branch to analyze (default: HEAD)
  -b, --backend BACKEND        LLM backend: anthropic, openai, or ollama (auto-detected)
  -m, --model TEXT             Model identifier (uses backend defaults if not specified)
  --api-key TEXT               API key for the backend (defaults to env var)
  --ollama-url TEXT            Ollama API URL (default: http://localhost:11434)
  --repo-name TEXT             Repository name for output
  --skip-llm                   Skip LLM summarization (extract and chunk only)
  --todo PATH                  Path to goals/todo file for critical examination mode
  --critical                   Enable critical examination mode (focus on gaps and issues)
  --directives TEXT            Additional plain text directives for LLM analysis
  --github-token TEXT          GitHub token for PR/review enrichment (or GITHUB_TOKEN env var)
```

### Extract Only

Extract git history to JSONL file without LLM processing:

```bash
gitview extract --repo /path/to/repo --output history.jsonl
```

### Chunk Only

Chunk an extracted JSONL file into phases:

```bash
gitview chunk history.jsonl --output ./phases --strategy adaptive
```

## File History Tracking & Header Injection

GitView provides powerful file-level change tracking with AI-powered summaries and the ability to inject change histories directly into source files as header comments. This is ideal for deep code analysis, debugging, accountability, and understanding individual file evolution.

### Features

- **Per-File Change Tracking**: Track detailed change history for every file in your repository
- **AI-Powered Summaries**: Generate intelligent summaries of changes using LLMs (with caching for cost optimization)
- **Multi-Language Header Injection**: Inject history as comments into source files (18+ languages supported)
- **Branch Comparison**: Compare file histories between branches with divergence analysis
- **Incremental Processing**: Checkpoint-based system only processes new commits
- **Cost Optimization**: 99.9% cost reduction through hash-based caching

### Quick Start - File Tracking

```bash
# Track all file changes in your repository
gitview track-files

# Track with AI-powered summaries
gitview track-files --with-ai

# View history for a specific file
gitview file-history gitview/cli.py

# Inject history as header comment into a file
gitview inject-history gitview/cli.py

# Remove injected header
gitview remove-history gitview/cli.py

# Compare file histories between branches
gitview compare-branches main feature-branch
```

### Phase 1: Core File Tracking

Track detailed change history for every file in your repository with incremental processing.

```bash
# Basic file tracking
gitview track-files

# Track specific file patterns
gitview track-files --pattern "*.py"

# Limit commits per file
gitview track-files --max-commits 50
```

**Output Structure:**
```
output/file_histories/
├── checkpoint.json              # Resume tracking from last processed commit
├── files/
│   ├── gitview_cli.py.history   # Human-readable history
│   └── gitview_cli.py.json      # Machine-readable JSON
└── index.json                   # Index of all tracked files
```

**File History Content:**
- Commit-by-commit changes with full metadata
- Lines added/removed per change
- Author information and timestamps
- Diff snippets for each change
- AI summaries (if enabled)

### Phase 2: AI-Powered Summaries

Generate intelligent summaries of file changes using LLMs with cost optimization.

```bash
# Track with AI summaries (uses caching)
gitview track-files --with-ai

# Cost estimate before running
gitview track-files --with-ai --dry-run
```

**Supported LLM Backends:**
- **Anthropic Claude** (claude-sonnet-4-5, claude-haiku)
- **OpenAI GPT** (gpt-4o, gpt-4o-mini)
- **Ollama** (llama3, mistral, codellama - runs locally, free)

**Cost Optimization:**
- Hash-based caching prevents duplicate summaries
- Incremental processing only analyzes new commits
- Cache hit rate typically >95% on reruns
- Estimated cost: $0.10-0.50 per 1000 files (using gpt-4o-mini)

**Cache Location:**
```
output/file_histories/summaries_cache.json
```

**Example AI Summary:**
```
Modified FileHistoryTracker.get_file_history() to support incremental
processing with checkpoint system. Added since_commit parameter to
iter_commits() to only process new commits after last checkpoint.
Breaking change: requires checkpoint.json for resume functionality.
```

### Phase 3: Header Injection

Inject file change histories as header comments into source files for debugging and accountability.

```bash
# Inject history into a Python file
gitview inject-history gitview/file_tracker.py

# Inject with limited entries
gitview inject-history gitview/file_tracker.py --max-entries 5

# Preview without writing (dry-run)
gitview inject-history gitview/file_tracker.py --dry-run

# Inject into multiple files
gitview inject-history gitview/*.py

# Remove injected headers
gitview remove-history gitview/file_tracker.py
```

**Supported Languages (18+):**
- Python, JavaScript, TypeScript, Java, Go, Rust, C/C++, C#
- Ruby, PHP, Swift, Kotlin, Scala, Shell, SQL, R, Perl, Lua, YAML

**Header Format Example (Python):**
```python
# ==============================================================================
# FILE CHANGE HISTORY
# ==============================================================================
# File: gitview/file_tracker.py
# Total changes: 15 commits
# Authors: John Doe (10), Jane Smith (5)
#
# [Recent Changes - Last 10 of 15]
#
# 2026-01-22 | abc123f | John Doe
#   Add incremental checkpoint system for resumable tracking
#   Changes: +45 -12 lines
#
#   Summary: Implemented checkpoint-based resume functionality...
#
# 2026-01-20 | def456a | Jane Smith
#   Fix diff parsing for binary files
#   Changes: +8 -3 lines
#
# ==============================================================================
# END FILE CHANGE HISTORY
# ==============================================================================

# Your actual code starts here...
```

**Use Cases:**
- Add accountability headers to critical files
- Include change context in code reviews
- Debug issues by understanding file evolution
- Onboarding - new developers see file history inline
- Compliance and audit trails

### Phase 4: Branch Comparison

Compare file histories between branches with divergence analysis and AI-powered comparison.

```bash
# Compare two branches
gitview compare-branches main feature-branch

# Compare with AI analysis of divergences
gitview compare-branches main feature-branch --with-ai

# Custom output location
gitview compare-branches main dev --output ./branch-analysis
```

**Output Structure:**
```
output/branch_comparisons/
├── branches/
│   ├── main/
│   │   ├── files/
│   │   │   ├── gitview_cli.py.json
│   │   │   └── gitview_tracker.py.json
│   │   └── branch_metadata.json
│   └── feature_branch/
│       ├── files/
│       └── branch_metadata.json
└── comparisons/
    └── main_vs_feature_branch/
        ├── summary.json
        ├── divergences.json
        ├── report.txt
        └── ai_analysis.json (if --with-ai used)
```

**Divergence Analysis:**
- Files unique to each branch
- Commits that diverged between branches
- Line change differences
- Divergence score (0-100) for each file

**Example Report:**
```
Branch Comparison: main vs feature-branch
==========================================

Summary:
  Files in main: 45
  Files in feature-branch: 47
  Files in both: 43
  Files only in main: 2
  Files only in feature-branch: 4

Top Divergent Files (score 0-100):
  1. gitview/file_tracker.py        Score: 87.5
     - 12 commits only in main
     - 8 commits only in feature-branch
     - +234 -156 lines difference

  2. gitview/cli.py                 Score: 65.3
     - 5 commits only in main
     - 3 commits only in feature-branch
     - +89 -45 lines difference
```

**AI Analysis (with --with-ai):**
- Semantic comparison of divergent changes
- Identifies conflicting implementations
- Suggests merge strategies
- Highlights breaking changes

### Complete Workflow Example

```bash
# 1. Initial file tracking with AI summaries
gitview track-files --with-ai

# 2. View history for a specific file
gitview file-history gitview/file_tracker.py

# 3. Inject history into critical files
gitview inject-history gitview/file_tracker.py gitview/cli.py

# 4. Switch to feature branch and track
git checkout feature-branch
gitview track-files --with-ai

# 5. Compare branches
gitview compare-branches main feature-branch --with-ai

# 6. Review divergences and make decisions
cat output/branch_comparisons/comparisons/main_vs_feature_branch/report.txt

# 7. After merge, update file histories
git checkout main
git merge feature-branch
gitview track-files --with-ai

# 8. Update injected headers
gitview inject-history gitview/file_tracker.py
```

### Configuration

File tracking uses the same LLM backend configuration as the main analysis:

```bash
# Use Anthropic Claude (default)
export ANTHROPIC_API_KEY="your-key"
gitview track-files --with-ai

# Use OpenAI GPT
export OPENAI_API_KEY="your-key"
gitview track-files --with-ai --backend openai --model gpt-4o-mini

# Use Ollama (local, free)
ollama serve
gitview track-files --with-ai --backend ollama --model llama3
```

### Performance & Costs

**Incremental Processing:**
- First run: Processes entire git history
- Subsequent runs: Only processes new commits since last checkpoint
- 10,000 commit repo: ~5 minutes first run, ~10 seconds for updates

**AI Summary Costs (estimated):**
- 1,000 files with gpt-4o-mini: ~$0.15-0.30
- 1,000 files with claude-haiku: ~$0.40-0.80
- 1,000 files with Ollama: $0 (local)
- Cache hit rate >95% on reruns = virtually free

**Storage:**
- Each file history: ~5-50KB depending on commit count
- AI cache: ~1KB per cached summary
- 1,000 files: ~50-100MB total

## Critical Examination Mode

For project leads who need objective assessment rather than celebratory narratives, GitView offers a critical examination mode that focuses on gaps, technical debt, and alignment with project goals.

### What Changes in Critical Mode?

**Tone & Focus:**
- Removes flowery, achievement-focused language
- Emphasizes objective assessment over celebration
- Focuses on gaps, issues, and misalignments
- Identifies what's missing or incomplete

**Analysis:**
- Evaluates progress against stated objectives
- Highlights incomplete features and technical debt
- Questions architectural decisions objectively
- Identifies delays and resource misalignment
- Notes concerning patterns and risks

### Usage

**Basic Critical Mode:**
```bash
gitview analyze --critical
```

**With Project Goals/TODO File:**
```bash
# Create a goals file (e.g., GOALS.md)
cat > GOALS.md <<EOF
# Project Goals Q1 2025
- Implement user authentication system
- Add API rate limiting
- Improve test coverage to 80%
- Migrate from SQLite to PostgreSQL
- Complete API documentation
EOF

# Analyze against goals
gitview analyze --critical --todo GOALS.md
```

**With Custom Directives:**
```bash
# Add specific analysis focus
gitview analyze --critical \
  --todo GOALS.md \
  --directives "Focus on security vulnerabilities and performance bottlenecks"
```

**Combined Example:**
```bash
# Critical assessment with all options
gitview analyze \
  --critical \
  --todo PROJECT_ROADMAP.md \
  --directives "Emphasize testing gaps and code quality issues" \
  --output ./critical-review
```

### Output in Critical Mode

The LLM will generate:

1. **Critical Executive Summary** - Assesses progress against goals, identifies gaps and delays
2. **Critical Timeline** - Highlights goal alignment/misalignment per phase
3. **Critical Technical Assessment** - Identifies architectural flaws and technical debt
4. **Critical Deletion Analysis** - Notes incomplete cleanup and lingering technical debt
5. **Comprehensive Critical Assessment** - Full project review with actionable insights

### When to Use Critical Mode

- **Project Reviews**: Objective assessment of development progress
- **Technical Audits**: Identify technical debt and architectural issues
- **Goal Alignment**: Measure actual work against stated objectives
- **Resource Planning**: Understand where effort was spent vs. planned
- **Risk Assessment**: Identify concerning patterns and project risks
- **Leadership Reports**: Provide factual assessment to stakeholders

## Storyline Tracking

GitView tracks "storylines" - narrative threads that span multiple phases of development. Instead of just seeing isolated phase summaries, you can follow the arc of features, refactoring efforts, bug fix campaigns, and other initiatives across your project's history.

### What Are Storylines?

Storylines are development threads that GitView automatically detects and tracks:

- **Features**: New functionality being built across multiple phases
- **Refactoring**: Code cleanup and restructuring efforts
- **Bug Fixes**: Bug fix campaigns and stability improvements
- **Tech Debt**: Debt reduction initiatives
- **Infrastructure**: CI/CD, tooling, and deployment improvements
- **Documentation**: Documentation efforts
- **Migrations**: Database or framework migrations
- **Performance**: Performance optimization work
- **Security**: Security hardening initiatives

### Multi-Signal Detection

Storylines are detected from multiple sources with confidence scoring:

| Source | Confidence | Description |
|--------|------------|-------------|
| PR Labels | 0.9 | GitHub PR labels (feature, bug, refactor) |
| PR Title Patterns | 0.8 | Patterns like "feat:", "fix:", "[WIP]" |
| File Clusters | 0.7 | Related files changing together |
| Commit Messages | 0.6 | Conventional commit patterns |
| LLM Extraction | 0.5 | AI-detected storylines from summaries |

### Storyline Lifecycle

Storylines progress through states automatically:

```
EMERGING → ACTIVE → PROGRESSING → COMPLETED
                  ↘ STALLED → ABANDONED
```

- **Emerging**: New storyline detected, building confidence
- **Active**: Confirmed storyline with ongoing work
- **Progressing**: Active work continues phase-over-phase
- **Completed**: Storyline reached completion (explicit or inferred)
- **Stalled**: No activity for 3+ phases
- **Abandoned**: Stalled for 6+ phases

### CLI Commands

GitView provides a `storyline` command group for exploring tracked storylines:

```bash
# List all storylines
gitview storyline list

# Filter by status
gitview storyline list --status active
gitview storyline list --status completed

# Filter by category
gitview storyline list --category feature
gitview storyline list --category refactor

# Show details for a specific storyline
gitview storyline show <storyline-id>
gitview storyline show oauth-impl  # partial ID match works

# Generate comprehensive report
gitview storyline report
gitview storyline report --save storyline-report.md

# View ASCII timeline visualization
gitview storyline timeline

# Export to JSON or CSV
gitview storyline export --format json
gitview storyline export --format csv --dest storylines.csv
```

### Example Output

**Storyline List:**
```
┌────────┬──────────────────────────┬────────────┬────────┬──────┬──────────────┐
│ Status │ Title                    │ Category   │ Phases │ Conf │ ID           │
├────────┼──────────────────────────┼────────────┼────────┼──────┼──────────────┤
│ ✓      │ OAuth Implementation     │ feature    │ 1→3    │ 90%  │ oauth-impl.. │
│ ●      │ API Rate Limiting        │ feature    │ 2→4    │ 85%  │ api-rate-l.. │
│ ▶      │ Test Coverage Expansion  │ tech_debt  │ 3→4    │ 75%  │ test-cover.. │
│ ◌      │ Legacy Migration         │ migration  │ 1→2    │ 70%  │ legacy-mig.. │
└────────┴──────────────────────────┴────────────┴────────┴──────┴──────────────┘
```

**ASCII Timeline:**
```
Storyline                                1  2  3  4  5
────────────────────────────────────────────────────────
OAuth Implementation                     ┌────┘
API Rate Limiting                           ┌────→
Test Coverage Expansion                        ┌──→
Legacy Migration                         ┌──╳

Legend: ┌─ start, ─┘ completed, ─→ ongoing, ─╳ stalled
```

### Output in Reports

When you run `gitview analyze`, storylines are automatically:

1. **Detected** from commits, PRs, and file patterns
2. **Tracked** across phases with state transitions
3. **Included** in the main `history_story.md` report
4. **Persisted** to `output/phases/storylines.json` for incremental analysis

The storyline section in your report includes:
- Summary of completed, active, and stalled storylines
- Timeline visualization
- Cross-phase theme analysis
- Category breakdown

### Persistence & Incremental Analysis

Storyline data is persisted to `output/phases/storylines.json`, enabling:

- **Incremental updates**: New phases add to existing storyline data
- **State continuity**: Storyline states persist across runs
- **Export capabilities**: Use the data in other tools

## GitHub Enrichment (PR & Review Context)

GitView can enrich commit history with Pull Request context from GitHub's GraphQL API, providing richer narratives based on actual PR descriptions and review feedback rather than just commit messages.

### What GitHub Enrichment Provides

- **PR Titles & Descriptions**: Use the "why" from PR descriptions instead of terse commit messages
- **Review Comments**: Include reviewer feedback and discussion context
- **Reviewer Attribution**: Track who reviewed and approved changes
- **PR Labels**: Categorize work by type (feature, bug, refactor, etc.)
- **Branch Information**: Understand feature branch to main branch flow

### Getting Started

1. **Generate a GitHub Token**:
   - Go to https://github.com/settings/tokens
   - Create "Personal access token (classic)"
   - Select `repo` scope for private repos (or `public_repo` for public only)
   - Copy the token

2. **Use with GitView**:

```bash
# Set as environment variable
export GITHUB_TOKEN="ghp_your_token_here"
gitview analyze --repo owner/repo --github-token $GITHUB_TOKEN

# Or pass directly
gitview analyze --repo owner/repo --github-token "ghp_your_token_here"
```

### Example with GitHub Enrichment

```bash
# Analyze a GitHub repository with PR context
export GITHUB_TOKEN="ghp_..."
gitview analyze \
  --repo carstenbund/gitview \
  --github-token $GITHUB_TOKEN \
  --output ./enriched-analysis

# The resulting narratives will include:
# - PR descriptions explaining WHY changes were made
# - Review feedback providing context on design decisions
# - Labels helping categorize types of work
```

### How It Improves Narratives

**Without GitHub Enrichment:**
> "Commit abc123: Fix bug in login flow"

**With GitHub Enrichment:**
> "PR #42 'Fix authentication race condition' addressed a critical issue where users could be logged out during token refresh. The fix was reviewed by @alice who suggested the retry mechanism that was ultimately implemented. Labels: [bug, security, priority-high]"

### Caching

GitHub API responses are cached locally in `~/.gitview/cache/github` for 24 hours to:
- Reduce API calls on repeated runs
- Improve performance for large repositories
- Stay within GitHub's rate limits

### Rate Limits

GitHub GraphQL API has a points-based rate limit (5,000 points/hour). GitView:
- Uses efficient batched queries
- Caches responses to minimize API calls
- Gracefully falls back if rate limited

## Chunking Strategies

GitView supports three chunking strategies:

### 1. **Adaptive** (Recommended)

Automatically splits history when significant changes occur:
- LOC changes by >30%
- Large deletions/additions detected
- README rewrites
- Major refactorings

```bash
gitview analyze --strategy adaptive
```

### 2. **Fixed Size**

Splits history into fixed-size chunks (e.g., 50 commits per phase):

```bash
gitview analyze --strategy fixed --chunk-size 50
```

### 3. **Time-Based**

Splits by time periods (week, month, quarter, year):

```bash
gitview analyze --strategy time --period quarter
```

## Output Files

GitView generates several output files:

```
output/
├── repo_history.jsonl           # Raw commit data
├── phases/                       # Phase data
│   ├── phase_01.json
│   ├── phase_02.json
│   ├── phase_index.json
│   └── storylines.json          # Storyline tracking data
├── history_story.md              # Main narrative report
├── timeline.md                   # Simple timeline
└── history_data.json             # Complete data in JSON
```

### Main Report (`history_story.md`)

Contains:
- **Executive Summary**: High-level overview for stakeholders
- **Timeline**: Chronological phases with descriptive headings
- **Full Narrative**: Complete story of the codebase evolution
- **Technical Evolution**: Architectural journey and key decisions
- **Story of Deletions**: What was removed and why
- **Storylines**: Cross-phase narrative threads with timeline visualization
- **Phase Details**: Detailed breakdown of each phase
- **Statistics**: Comprehensive metrics

## How It Works

### Phase 1: Extract Raw History

Analyzes git commits and extracts:
- Commit metadata (hash, author, date, message)
- Lines of code changes (insertions/deletions)
- File statistics
- Language breakdown
- README state and changes
- Code comments and density
- Detection of large changes, refactors, etc.

### Phase 2: Chunk into Epochs

Divides history into meaningful phases based on:
- Significant LOC changes
- Large deletions or additions
- Language mix changes
- README rewrites
- Major refactorings

### Phase 3: Summarize Each Phase

Uses Claude to generate narrative summaries for each phase, answering:
- What were the main activities?
- Why were changes made?
- What was deleted/added and why?
- How did documentation evolve?
- What do commit messages reveal?

### Phase 4: Generate Global Story

Combines phase summaries to create:
- Executive summary for non-technical readers
- Chronological timeline with meaningful headings
- Technical retrospective
- Story of code deletions and cleanups
- Full detailed narrative

## Examples

### Analyze a Large Open Source Project

```bash
gitview analyze \
  --repo /path/to/large-project \
  --output ./project-analysis \
  --strategy adaptive \
  --repo-name "My Project"
```

### Quick Analysis Without LLM

Perfect for quick exploration or when you don't have an API key:

```bash
gitview analyze --skip-llm --output ./quick-analysis
```

### Extract and Process Later

```bash
# Extract once
gitview extract --repo /path/to/repo --output history.jsonl

# Experiment with different chunking strategies
gitview chunk history.jsonl --strategy adaptive --output ./adaptive-phases
gitview chunk history.jsonl --strategy fixed --chunk-size 25 --output ./fixed-phases
```

### Critical Project Assessment

```bash
# Create a goals file for your project
cat > PROJECT_GOALS.md <<EOF
# Q1 2025 Objectives
- Complete user authentication with OAuth2
- Implement API rate limiting (1000 req/hour)
- Achieve 80% test coverage
- Migrate database to PostgreSQL
- Document all public APIs
EOF

# Run critical analysis
gitview analyze \
  --critical \
  --todo PROJECT_GOALS.md \
  --directives "Focus on security issues and incomplete features" \
  --output ./project-review-q1

# Review the critical assessment
cat ./project-review-q1/history_story.md
```

## Architecture

```
┌─────────────────────┐
│   Git Repository    │
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│  Extractor          │  Analyzes commits, extracts metadata
│  (extractor.py)     │  Output: repo_history.jsonl
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│  Chunker            │  Splits into meaningful phases
│  (chunker.py)       │  Strategies: adaptive, fixed, time
└──────────┬──────────┘
           │
           v
┌─────────────────────┐     ┌─────────────────────┐
│  Summarizer         │────>│  Storyline Tracker  │
│  (summarizer.py)    │     │  (storyline/)       │
└──────────┬──────────┘     │  Multi-signal       │
           │                │  detection & state  │
           v                │  machine lifecycle  │
┌─────────────────────┐     └──────────┬──────────┘
│  StoryTeller        │<───────────────┘
│  (storyteller.py)   │  Generates global narratives
└──────────┬──────────┘  with storyline context
           │
           v
┌─────────────────────┐
│  Writer             │  Outputs markdown, JSON, etc.
│  (writer.py)        │  Includes storyline reports
└─────────────────────┘
```

## Requirements

- Python 3.8+
- Git repository with commit history
- **One of the following LLM backends:**
  - **Anthropic Claude** (requires API key)
  - **OpenAI GPT** (requires API key)
  - **Ollama** (runs locally, no API key needed)
- Dependencies: gitpython, anthropic, openai, requests, click, rich, pydantic

## LLM Backend Configuration

GitView supports three LLM backends with automatic detection based on environment variables:

### Anthropic Claude (Default)

Get an API key from [Anthropic](https://www.anthropic.com/)

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
gitview analyze
```

Default models:
- `claude-sonnet-4-5-20250929` (default)
- `claude-3-opus-20240229` (more powerful)
- `claude-3-haiku-20240307` (faster)

### OpenAI GPT

Get an API key from [OpenAI](https://platform.openai.com/)

```bash
export OPENAI_API_KEY="your-api-key-here"
gitview analyze --backend openai
```

Default models:
- `gpt-4` (default)
- `gpt-4-turbo-preview` (faster)
- `gpt-3.5-turbo` (cheaper)

### Ollama (Local)

Install [Ollama](https://ollama.ai/) and pull a model:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3

# Start Ollama server
ollama serve

# Use with GitView (no API key needed)
gitview analyze --backend ollama --model llama3
```

Popular Ollama models:
- `llama3` (default, balanced)
- `mistral` (fast, good quality)
- `codellama` (optimized for code)
- `mixtral` (large, powerful)

### Custom Configuration

```bash
# Specify custom model
gitview analyze --backend anthropic --model claude-3-opus-20240229

# Use custom Ollama URL
gitview analyze --backend ollama --ollama-url http://192.168.1.100:11434

# Pass API key directly (instead of env var)
gitview analyze --backend openai --api-key "your-key"
```

## Use Cases

### Standard Mode (Celebratory Narrative)
- **Technical Documentation**: Automatically generate project history documentation
- **Onboarding**: Help new developers understand codebase evolution
- **Retrospectives**: Review what worked and what didn't
- **Project Reports**: Create compelling narratives for stakeholders
- **Code Archaeology**: Understand why code evolved the way it did
- **Cleanup Planning**: Identify what to remove based on deletion history

### Critical Examination Mode
- **Project Leadership**: Objective assessment for project leads and managers
- **Technical Audits**: Identify technical debt and architectural issues
- **Goal Tracking**: Measure actual progress against roadmap objectives
- **Resource Analysis**: Understand where development effort was spent
- **Risk Management**: Identify concerning patterns and project risks
- **Stakeholder Reports**: Provide factual, critical assessment to executives

### Storyline Tracking
- **Feature Tracking**: Follow features from inception to completion across phases
- **Refactoring Visibility**: Track long-running refactoring efforts and their progress
- **Stalled Work Detection**: Identify initiatives that have stalled or been abandoned
- **Cross-Phase Analysis**: Understand how work threads connect across time
- **Project Health**: See the balance of active, completed, and stalled storylines
- **Timeline Visualization**: ASCII timeline showing storyline arcs

### File Tracking & Header Injection
- **Deep Code Analysis**: Inject complete change history into files for debugging complex issues
- **Compliance & Accountability**: Track who changed what and when with inline headers
- **Code Reviews**: Include file evolution context directly in reviewed files
- **Developer Onboarding**: New team members see file history without leaving their editor
- **Branch Divergence Analysis**: Identify conflicts before merging feature branches
- **Refactoring Decisions**: Understand file evolution patterns to guide architecture changes
- **Bug Investigation**: Trace file changes to identify when bugs were introduced
- **Technical Debt Tracking**: Compare branch histories to assess divergence costs
- **Documentation**: Generate per-file change logs for critical components
- **AI-Powered Insights**: Get intelligent summaries of complex code changes

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## License

MIT License - see LICENSE file for details



[github-ci]: https://github.com/carstenbund/gitview/actions/workflows/test.yml/badge.svg?branch=main
[github-link]: https://github.com/carstenbund/gitview
[pypi-badge]: https://img.shields.io/pypi/v/gitview.svg
[pypi-link]: https://pypi.org/project/gitview
[codecov-badge]: https://codecov.io/gh/carstenbund/gitview/branch/master/graph/badge.svg
[codecov-link]: https://codecov.io/gh/carstenbund/gitview
[install-badge]: https://img.shields.io/pypi/dw/gitview?label=pypi%20installs
[install-link]: https://pypistats.org/packages/gitview
