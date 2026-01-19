# File Change History Generator - Proposal

## Vision

Generate detailed change history for **every file** in a repository that can be:
1. **Stored as companion `.history` files** alongside source files
2. **Injected as header comments** into source files on command
3. **Used for deep code analysis**, debugging, and accountability

## Use Cases

### 1. Code Understanding
```python
# FILE CHANGE HISTORY
# Last updated: 2026-01-18
# Total changes: 47 commits
# Primary authors: Alice (23), Bob (15), Carol (9)
#
# Recent changes:
# 2026-01-15 | abc1234 | Alice Chen
#   Added retry logic with exponential backoff
#   Modified: process_request(), handle_error()
#   +45 lines, -12 lines
#   Summary: Implemented robust retry mechanism to handle transient
#            network failures. Exponential backoff with max 5 attempts.
#
# 2025-12-10 | xyz5678 | Bob Smith
#   Refactored database connection pooling
#   Modified: init_db(), get_connection()
#   +23 lines, -8 lines
#   Summary: Replaced manual connection with pool for better performance.
# ========================================================================

import asyncio
import logging
from typing import Optional
...
```

### 2. Debugging & Accountability
- "Who introduced this bug?" → Check file history header
- "Why was this function refactored?" → See AI summary in header
- "What changed in the last 3 commits?" → Recent history at top of file

### 3. Code Review Context
Reviewers can see the evolution of a file without leaving their editor

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Git Repository                           │
│                     (All source files)                          │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              FileHistoryTracker (Orchestrator)                  │
│  • Tracks ALL files (or filtered by pattern)                    │
│  • Incremental: only process changed files since last run       │
│  • Generates companion .history files                           │
└────────────────┬────────────────────────────────────────────────┘
                 │
    ┌────────────┴─────────────┐
    │                          │
    ▼                          ▼
┌──────────────────┐    ┌─────────────────────────────────────────┐
│  Checkpoint      │    │      FileChangeExtractor                │
│  ───────────     │    │      ──────────────────                 │
│  Last run:       │    │  For each commit:                       │
│    commit_hash   │    │  1. Get changed files                   │
│  Files tracked:  │    │  2. Extract function/class changes      │
│    2,347         │    │  3. Compute line diffs                  │
│  AI summaries:   │    │  4. Parse code structure (AST)          │
│    1,823         │    │  5. Build FileChange record             │
└──────────────────┘    └──────────────┬──────────────────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────────────────┐
                        │       FileChange Object                  │
                        │       ─────────────────                  │
                        │  • Commit metadata                       │
                        │  • Functions/classes modified            │
                        │  • Lines added/removed                   │
                        │  • Code diff snippet                     │
                        │  • AST changes (optional)                │
                        └──────────────┬───────────────────────────┘
                                       │
                          ┌────────────┴────────────┐
                          │                         │
                          ▼                         ▼
                ┌──────────────────┐      ┌────────────────────────┐
                │  Summary Cache   │◄─────┤  AI Summarizer         │
                │  ──────────────  │      │  ─────────────         │
                │  Hash-based      │      │  (EXPENSIVE)           │
                │  Per-file/commit │      │                        │
                │  100% hit rate   │      │  Prompt:               │
                └──────────────────┘      │  "What changed and     │
                                          │   why in this file?"   │
                                          └────────────┬───────────┘
                                                       │
                                                       ▼
                        ┌──────────────────────────────────────────┐
                        │   FileChange (enriched with AI)          │
                        │   + ai_summary: "Implemented robust..."  │
                        └──────────────┬───────────────────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────────────────┐
                        │   Per-File History Aggregator            │
                        │   Group changes by file path             │
                        │   Sort chronologically (newest first)    │
                        │   Calculate statistics                   │
                        └──────────────┬───────────────────────────┘
                                       │
                          ┌────────────┴─────────────┐
                          │                          │
                          ▼                          ▼
            ┌──────────────────────┐    ┌───────────────────────────┐
            │  .history Files      │    │  Header Injector          │
            │  ────────────────    │    │  ───────────────          │
            │  src/main.py.history │    │  Command:                 │
            │  lib/util.js.history │    │  inject-history           │
            │  config.json.history │    │                           │
            │                      │    │  Reads .history file      │
            │  Human-readable      │    │  Formats as comment       │
            │  Separate files      │    │  Injects at top of file   │
            │  Can be committed    │    │  Preserves original code  │
            └──────────────────────┘    └───────────────────────────┘
```

## Data Model

### FileChange (Single commit change)
```python
@dataclass
class FileChange:
    """Represents a single commit's change to one file"""
    file_path: str
    commit_hash: str
    commit_date: str
    author_name: str
    author_email: str
    commit_message: str

    # Change details
    lines_added: int
    lines_removed: int

    # Code structure changes (parsed from AST)
    functions_added: List[str]
    functions_removed: List[str]
    functions_modified: List[str]
    classes_added: List[str]
    classes_modified: List[str]

    # Diff information
    diff_snippet: str  # Truncated or key sections
    full_diff_available: bool

    # AI summary (expensive, cached)
    ai_summary: Optional[str] = None
    ai_summary_model: Optional[str] = None
    ai_summary_generated_at: Optional[str] = None
```

### FileHistory (Complete file evolution)
```python
@dataclass
class FileHistory:
    """Complete change history for a single file"""
    file_path: str
    current_path: str  # May differ if renamed

    # Timeline
    first_commit: str
    first_commit_date: str
    last_commit: str
    last_commit_date: str

    # Statistics
    total_commits: int
    total_lines_added: int
    total_lines_removed: int

    # Contributors
    authors: List[Dict[str, int]]  # [{"name": "Alice", "commits": 23}, ...]

    # Rename history
    previous_paths: List[str]

    # Chronological changes (newest first)
    changes: List[FileChange]

    def format_as_comment(self,
                         language: str,
                         max_entries: int = 10,
                         include_diffs: bool = False) -> str:
        """Format history as header comment for injection"""
        pass

    def to_markdown(self) -> str:
        """Export as standalone markdown file"""
        pass
```

## Core Components

### 1. FileHistoryTracker (Main Orchestrator)

```python
class FileHistoryTracker:
    """
    Tracks detailed change history for all files in repository
    """

    def __init__(self,
                 repo_path: str,
                 output_dir: str,
                 llm_router,
                 file_patterns: List[str] = None,
                 exclude_patterns: List[str] = None):
        """
        Args:
            file_patterns: Files to track (e.g., ['*.py', '*.js', '*.json'])
            exclude_patterns: Files to ignore (e.g., ['*.min.js', '*-lock.json'])
        """
        self.repo = Repo(repo_path)
        self.output_dir = Path(output_dir) / "file_histories"
        self.change_extractor = FileChangeExtractor(repo_path)
        self.ai_summarizer = ChangeAISummarizer(llm_router)
        self.checkpoint = self._load_checkpoint()

    def track_all_files(self,
                       since_commit: Optional[str] = None,
                       with_ai_summaries: bool = False,
                       max_history_entries: int = 50):
        """
        Main tracking method

        Steps:
        1. Load checkpoint (resume from last run)
        2. Get all commits since checkpoint
        3. For each commit:
           - Get changed files matching patterns
           - Extract change details (diff, AST, functions)
           - Optionally generate AI summary
           - Update file history
        4. Save .history files for each tracked file
        5. Update checkpoint

        Args:
            since_commit: Override checkpoint, start from specific commit
            with_ai_summaries: Generate AI summaries (expensive!)
            max_history_entries: Limit entries per file (keep recent)
        """
        pass

    def get_file_history(self, file_path: str) -> FileHistory:
        """Retrieve complete history for a specific file"""
        pass

    def generate_history_files(self, format: str = "compact"):
        """
        Generate .history companion files for all tracked files

        Formats:
        - compact: Recent 10 changes, one-line summaries
        - detailed: Recent 25 changes, full summaries
        - full: All changes with diffs
        """
        pass
```

### 2. FileChangeExtractor (Diff & AST Analysis)

```python
class FileChangeExtractor:
    """
    Extracts detailed change information for a file in a commit
    """

    def extract_change(self,
                      commit,
                      file_path: str) -> FileChange:
        """
        Extract comprehensive change details

        Steps:
        1. Get git diff for this file
        2. Count lines added/removed
        3. Parse code structure (if supported language):
           - Python: ast module
           - JavaScript: esprima/babel parser
           - Go: go/parser
           - etc.
        4. Identify function/class changes
        5. Generate diff snippet (key sections only)

        Returns: FileChange object
        """
        pass

    def parse_code_structure(self,
                            code: str,
                            language: str) -> Dict:
        """
        Parse code to extract structure using AST

        Returns:
        {
            'functions': ['func1', 'func2'],
            'classes': ['ClassA', 'ClassB'],
            'imports': ['os', 'sys'],
            'top_level_vars': ['CONFIG', 'VERSION']
        }
        """
        pass

    def compare_structures(self,
                          old_structure: Dict,
                          new_structure: Dict) -> Dict:
        """
        Compare two code structures to find what changed

        Returns:
        {
            'functions_added': ['new_func'],
            'functions_removed': ['old_func'],
            'functions_modified': ['existing_func'],  # by diff
            'classes_added': ['NewClass'],
            ...
        }
        """
        pass

    def generate_diff_snippet(self,
                             diff: str,
                             max_lines: int = 20) -> str:
        """
        Extract key sections from diff

        Strategy:
        - Include function/class definition changes
        - Show first few lines of large blocks
        - Truncate repetitive changes
        """
        pass
```

### 3. ChangeAISummarizer (LLM Integration)

```python
class ChangeAISummarizer:
    """
    Generates concise AI summaries of file changes
    """

    def __init__(self, llm_router, cache_path: str):
        self.llm = llm_router
        self.cache = self._load_cache(cache_path)

    def summarize_change(self,
                        change: FileChange,
                        code_context: Optional[str] = None) -> str:
        """
        Generate AI summary for a file change

        Prompt template:
        '''
        Analyze this code change from commit {hash}:

        File: {file_path}
        Author: {author}
        Date: {date}
        Commit message: {message}

        Changes:
        - Lines: +{added} -{removed}
        - Functions modified: {functions_modified}
        - Functions added: {functions_added}
        - Classes modified: {classes_modified}

        Diff:
        {diff_snippet}

        {code_context if provided}

        Provide a 1-2 sentence summary explaining:
        1. What changed (be specific about code elements)
        2. Why (infer from commit message and context)
        3. Impact (breaking changes, new features, bug fixes)
        '''

        Cache key: hash(commit_hash + file_path + diff)
        """
        pass

    def batch_summarize_file_history(self,
                                    changes: List[FileChange],
                                    batch_size: int = 10) -> Dict[str, str]:
        """
        Batch summarize multiple changes for cost efficiency
        Group changes to same file in single LLM call
        """
        pass
```

### 4. HistoryInjector (Comment Header Generator)

```python
class HistoryInjector:
    """
    Injects file history as header comments into source files
    """

    # Comment styles by language
    COMMENT_STYLES = {
        'python': {'single': '#', 'block_start': '"""', 'block_end': '"""'},
        'javascript': {'single': '//', 'block_start': '/*', 'block_end': '*/'},
        'java': {'single': '//', 'block_start': '/*', 'block_end': '*/'},
        'go': {'single': '//', 'block_start': '/*', 'block_end': '*/'},
        'rust': {'single': '//', 'block_start': '/*', 'block_end': '*/'},
        'ruby': {'single': '#', 'block_start': '=begin', 'block_end': '=end'},
        # ... more languages
    }

    def inject_history(self,
                      file_path: str,
                      history: FileHistory,
                      max_entries: int = 10,
                      include_diffs: bool = False,
                      dry_run: bool = False) -> str:
        """
        Inject history header into source file

        Steps:
        1. Detect file language
        2. Read current file content
        3. Check if history header already exists
        4. Generate new history header
        5. Replace old header or prepend to file
        6. Write updated file (or return if dry_run)

        Returns: Path to modified file
        """
        pass

    def remove_history_header(self, file_path: str):
        """Remove injected history header from file"""
        pass

    def format_header(self,
                     history: FileHistory,
                     language: str,
                     max_entries: int,
                     include_diffs: bool) -> str:
        """
        Format history as comment header

        Example output for Python:
        '''
        # ==============================================================================
        # FILE CHANGE HISTORY
        # ==============================================================================
        # Generated: 2026-01-18 15:30:00
        # Total changes: 47 commits
        # Primary authors: Alice Chen (23), Bob Smith (15), Carol Davis (9)
        #
        # [Recent Changes - Last 10 of 47]
        #
        # 2026-01-15 10:23:45 | abc1234 | Alice Chen
        #   Added retry logic with exponential backoff
        #   Modified: process_request(), handle_error()
        #   Changes: +45 lines, -12 lines
        #   Summary: Implemented robust retry mechanism to handle transient network
        #            failures. Added exponential backoff with max 5 retry attempts.
        #
        # 2025-12-10 08:15:32 | xyz5678 | Bob Smith
        #   Refactored database connection pooling
        #   Modified: init_db(), get_connection()
        #   Changes: +23 lines, -8 lines
        #   Summary: Replaced manual connection management with connection pool to
        #            improve performance under high load conditions.
        #
        # ... (8 more entries)
        #
        # Full history: file_histories/src/main.py.history
        # ==============================================================================
        '''
        """
        pass
```

## File Organization

```
project/
├── src/
│   ├── main.py                    # Original source file
│   ├── main.py.history            # Companion history (human-readable)
│   ├── utils.py
│   └── utils.py.history
│
├── .gitview/
│   └── file_histories/
│       ├── checkpoint.json         # Last processed commit
│       ├── summary_cache.json      # AI summaries cache
│       ├── src/
│       │   ├── main.py.json       # Machine-readable history
│       │   └── utils.py.json
│       └── index.json             # Master file index
│
└── output/
    └── file_history_report.md     # Global report (all files)
```

## CLI Commands

### Generate History Files

```bash
# Generate .history files for all tracked files (no AI)
gitview track-files . --generate-histories

# With AI summaries (expensive)
gitview track-files . --generate-histories --with-ai

# Specific file patterns only
gitview track-files . --patterns "*.py,*.js" --exclude "*_test.py"

# Incremental update (only process new commits)
gitview track-files . --generate-histories --incremental
```

### Inject History Headers

```bash
# Inject history into all tracked files
gitview inject-history . --max-entries 10

# Dry run (show what would be injected)
gitview inject-history . --dry-run

# Specific files only
gitview inject-history src/main.py src/utils.py

# Include diff snippets (verbose)
gitview inject-history . --include-diffs --max-entries 5

# Remove injected headers
gitview inject-history . --remove
```

### Query File History

```bash
# Show history for specific file
gitview file-history src/main.py

# Export to markdown
gitview file-history src/main.py --format markdown > main_history.md

# Show recent N changes
gitview file-history src/main.py --recent 5

# With AI summaries
gitview file-history src/main.py --with-ai
```

## Usage Workflow

### Initial Setup

```bash
# 1. Generate histories for entire repo (fast, no AI)
gitview track-files /path/to/repo --patterns "*.py,*.js,*.go"

# Output:
# Processing 1,247 commits...
# Tracking 487 files
# Generated 487 .history files
# Saved checkpoint: commit xyz789
# Time: 45s, Cost: $0.00
```

### Regular Updates

```bash
# 2. Incremental update with AI summaries (only new commits)
gitview track-files . --incremental --with-ai

# Output:
# Loading checkpoint from 2026-01-15 (commit xyz789)
# Found 23 new commits
# 15 files changed
# Generating AI summaries...
# Estimated cost: $0.35
# Continue? [y/N]: y
# ✓ Updated 15 file histories
# Time: 8s, Cost: $0.32
```

### Inject Headers for Code Review

```bash
# 3. Inject history headers into files for review
gitview inject-history src/ --max-entries 5

# Output:
# Injecting histories into 23 files...
# ✓ src/main.py (5 entries injected)
# ✓ src/utils.py (5 entries injected)
# ...
#
# Files modified: 23
# Run 'gitview inject-history --remove' to undo
```

### Review Specific File

```bash
# 4. Check detailed history
gitview file-history src/critical_module.py --with-ai

# Output shows:
# ========================================
# FILE: src/critical_module.py
# ========================================
#
# First commit: 2024-06-12 (def456)
# Last modified: 2026-01-18 (abc123)
# Total changes: 34 commits
# Authors: Alice (19), Bob (10), Carol (5)
#
# [Change History]
#
# 2026-01-18 | abc123 | Alice
#   Modified: validate_input(), sanitize_data()
#   +12 lines, -3 lines
#   Summary: Added input validation to prevent SQL injection
#            vulnerabilities. Implemented whitelist-based sanitization
#            for user-provided data.
# ...
```

### Remove Headers After Review

```bash
# 5. Clean up injected headers
gitview inject-history . --remove

# Output:
# Removing history headers from 23 files...
# ✓ All headers removed
```

## Advanced Features

### 1. Smart History Truncation

```python
# Keep recent changes + significant milestones
history_config = {
    'recent_entries': 10,  # Always show last 10
    'include_major_refactors': True,  # Keep breaking changes
    'include_security_fixes': True,  # Keep security commits
    'max_total_entries': 50  # Hard limit
}
```

### 2. Contextual AI Summaries

```python
# Pass surrounding code context to AI for better summaries
summarizer.summarize_change(
    change=file_change,
    code_context={
        'file_purpose': 'Database connection manager',
        'key_functions': ['connect()', 'execute_query()'],
        'dependencies': ['psycopg2', 'asyncpg']
    }
)
```

### 3. Multi-Language Support

```python
# Automatic language detection
SUPPORTED_LANGUAGES = [
    'python', 'javascript', 'typescript', 'java', 'go',
    'rust', 'ruby', 'php', 'c', 'cpp', 'csharp', 'swift',
    'kotlin', 'scala', 'shell', 'sql', 'json', 'yaml'
]
```

### 4. Rename Tracking

```python
# Track file across renames
FileHistory(
    current_path='src/new_module.py',
    previous_paths=[
        'src/old_module.py',
        'lib/legacy_module.py'
    ],
    changes=[...]  # Combined history from all paths
)
```

## Cost Optimization

### Strategy 1: Incremental Processing
```
First run: 1,000 commits × 5 files = 5,000 changes → No AI
Incremental: 10 commits × 3 files = 30 changes → $0.30 with AI
```

### Strategy 2: Smart Caching
```
Cache key: hash(commit_hash + file_path + diff)
Re-run same commit range → 0 API calls
```

### Strategy 3: Selective AI Summarization
```
# Only summarize important files
--ai-for-patterns "src/*.py,lib/*.js"
--skip-ai-for "tests/*,*_generated.py"
```

### Strategy 4: Batch Processing
```
# 10 changes per API call instead of 10 separate calls
Cost reduction: ~40%
```

### Cost Estimate

**Typical Repository (500 files, 1,000 commits):**

| Operation | Files × Changes | With AI | Cost |
|-----------|----------------|---------|------|
| Initial scan | 500 × 5 = 2,500 | No | $0.00 |
| Add AI summaries | 2,500 changes | Yes | $2.50 |
| Incremental update | 20 changes | Yes | $0.02 |
| Re-run (cached) | 2,500 changes | Yes | $0.00 |

**Model: GPT-4o-mini @ $0.001/change**

## Implementation Phases

### Phase 1: Core Infrastructure ✓
- [ ] FileChange and FileHistory data models
- [ ] FileChangeExtractor (git diff parsing)
- [ ] Basic AST parsing (Python, JavaScript)
- [ ] Checkpoint system
- [ ] Generate .history companion files

**Deliverable:** Fast file tracking without AI

### Phase 2: AI Integration ✓
- [ ] ChangeAISummarizer with caching
- [ ] Cost estimation
- [ ] Batch processing
- [ ] Integration with LLMRouter

**Deliverable:** Optional AI-powered summaries

### Phase 3: Header Injection ✓
- [ ] HistoryInjector with multi-language support
- [ ] Comment formatting by language
- [ ] Dry-run mode
- [ ] Header removal

**Deliverable:** Inject/remove headers on command

### Phase 4: Advanced Features
- [ ] Rename tracking across file moves
- [ ] More language parsers (Go, Rust, Java)
- [ ] Interactive HTML reports
- [ ] Diff visualization
- [ ] Security/breaking change detection

## Security Considerations

1. **Sensitive file exclusion**: Skip files with secrets (`.env`, `credentials.json`)
2. **Diff sanitization**: Option to redact sensitive values in diffs
3. **Size limits**: Skip files > 1MB (likely binary or generated)
4. **Injection safety**: Validate comment syntax to prevent code injection

## Success Metrics

- **Speed**: Process 1,000 commits in < 60 seconds (no AI)
- **Cost**: < $0.05 per 50 changes with AI (GPT-4o-mini)
- **Accuracy**: 100% cache hit rate on re-runs
- **Usability**: Single command to inject/remove headers

---

## Questions for Next Steps

1. **Default file patterns**: Which file types to track by default?
   - Suggested: `*.py,*.js,*.ts,*.go,*.java,*.rs,*.rb,*.php,*.c,*.cpp,*.h`

2. **History header size**: How many entries in injected headers?
   - Suggested: 10 recent changes (configurable)

3. **AI summarization**: Always opt-in or offer presets?
   - Suggested: Opt-in with `--with-ai` flag

4. **Storage location**: Where to store .history files?
   - Option A: Alongside source files (easy to find)
   - Option B: Separate `.gitview/histories/` directory (cleaner)
   - Option C: Both (companion + centralized JSON)

5. **Git integration**: Should .history files be committed to repo?
   - Pros: Shared across team
   - Cons: Repo bloat
   - Suggested: Add to `.gitignore` by default, opt-in to commit

Ready to implement Phase 1?
