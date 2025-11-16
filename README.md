# GitView

**Git history analyzer with LLM-powered narrative generation**

GitView extracts your repository's git history and uses AI to generate compelling narratives about how your codebase evolved. Instead of manually reading through thousands of commits, get a comprehensive story of your project's journey.

## Features

- **📊 Comprehensive History Extraction**: Extracts commit metadata, LOC changes, language breakdown, README evolution, comment analysis, and more
- **🧩 Smart Chunking**: Automatically divides history into meaningful "phases" or "epochs" based on significant changes
- **🤖 LLM-Powered Summaries**: Uses Claude to generate narrative summaries for each phase
- **📖 Global Story Generation**: Combines phase summaries into executive summaries, timelines, technical retrospectives, and deletion stories
- **📝 Multiple Output Formats**: Generates markdown reports, JSON data, and timelines

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gitview.git
cd gitview

# Install dependencies
pip install -e .
```

## Quick Start

```bash
# Set your Anthropic API key
export ANTHROPIC_API_KEY="your-api-key-here"

# Analyze a repository (full pipeline)
gitview analyze --repo /path/to/repo --output ./output

# Analyze current directory
gitview analyze

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
  --api-key TEXT               Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
  --model TEXT                 Claude model to use (default: claude-sonnet-4-5-20250929)
  --repo-name TEXT             Repository name for output
  --skip-llm                   Skip LLM summarization (extract and chunk only)
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
│   └── phase_index.json
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
┌─────────────────────┐
│  Summarizer         │  LLM summarizes each phase
│  (summarizer.py)    │  Uses Claude API
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│  StoryTeller        │  Generates global narratives
│  (storyteller.py)   │  Multiple story formats
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│  Writer             │  Outputs markdown, JSON, etc.
│  (writer.py)        │
└─────────────────────┘
```

## Requirements

- Python 3.8+
- Git repository with commit history
- Anthropic API key (for LLM features)
- Dependencies: gitpython, anthropic, click, rich, pydantic

## API Key

GitView uses the Anthropic Claude API for summarization. You need an API key from [Anthropic](https://www.anthropic.com/).

Set it as an environment variable:

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

Or pass it directly:

```bash
gitview analyze --api-key "your-api-key-here"
```

## Use Cases

- **Technical Documentation**: Automatically generate project history documentation
- **Onboarding**: Help new developers understand codebase evolution
- **Retrospectives**: Review what worked and what didn't
- **Project Reports**: Create compelling narratives for stakeholders
- **Code Archaeology**: Understand why code evolved the way it did
- **Cleanup Planning**: Identify what to remove based on deletion history

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## License

MIT License - see LICENSE file for details