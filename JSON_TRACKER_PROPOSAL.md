# JSON File Change Tracker - Technical Proposal

## Overview
Add intelligent JSON file change tracking to GitView with AI-powered diff summarization, caching, and incremental processing.

## Architecture

### 1. Data Model

```python
@dataclass
class JsonFileChange:
    """Represents a single change to a JSON file in a commit"""
    file_path: str
    commit_hash: str
    timestamp: str
    author: str
    commit_message: str

    # Diff information
    diff_type: str  # 'added', 'modified', 'deleted', 'renamed'
    old_path: Optional[str]  # For renames

    # Structured diff
    keys_added: List[str]
    keys_removed: List[str]
    keys_modified: List[str]
    structural_changes: Dict[str, Any]  # Nested structure changes

    # Raw diff (for AI analysis)
    raw_diff: str

    # AI-generated summary (expensive field)
    ai_summary: Optional[str] = None
    ai_summary_generated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON"""
        pass

@dataclass
class JsonFileHistory:
    """Complete history for a single JSON file"""
    file_path: str
    first_seen_commit: str
    last_seen_commit: str
    total_changes: int

    # Aggregated statistics
    total_keys_added: int
    total_keys_removed: int
    total_modifications: int

    # Authors who touched this file
    contributors: List[Dict[str, int]]  # {author: change_count}

    # Chronological changes
    changes: List[JsonFileChange]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON"""
        pass
```

### 2. Cache & Checkpoint System

```python
@dataclass
class JsonTrackerCheckpoint:
    """Tracks last processed state for incremental runs"""
    last_processed_commit: str
    last_processed_timestamp: str
    total_commits_processed: int
    total_json_files_tracked: int
    total_ai_summaries_generated: int

    # Cost tracking
    total_api_calls: int
    estimated_cost_usd: float

    # Next run should start after this commit
    resume_from_commit: str

    cache_version: str = "1.0"

    def save(self, path: str):
        """Save checkpoint to JSON"""
        pass

    @classmethod
    def load(cls, path: str) -> Optional['JsonTrackerCheckpoint']:
        """Load checkpoint from JSON"""
        pass
```

**Cache Location:**
```
output/
├── json_tracking/
│   ├── checkpoint.json          # Last run metadata
│   ├── files/
│   │   ├── config.json.history  # Per-file change history
│   │   ├── package.json.history
│   │   └── data/settings.json.history
│   ├── summaries_cache.json     # AI summaries cache (separate for quick lookup)
│   └── tracking_index.json      # Master index of all tracked files
```

### 3. JSON Diff Analyzer

```python
class JsonDiffAnalyzer:
    """Extracts structured diffs from JSON file changes"""

    def analyze_commit(self, commit, file_path: str) -> JsonFileChange:
        """
        Extract structured diff for a JSON file in a commit

        Steps:
        1. Get before/after content
        2. Parse JSON (handle parse errors gracefully)
        3. Compare structures recursively
        4. Identify added/removed/modified keys
        5. Generate raw diff for AI analysis
        """
        pass

    def extract_structural_changes(self, old_json, new_json) -> Dict:
        """
        Deep comparison of JSON structures

        Returns:
        {
            'keys_added': ['new.nested.key', 'another.key'],
            'keys_removed': ['old.deprecated.key'],
            'keys_modified': ['config.timeout', 'settings.enabled'],
            'type_changes': [
                {'key': 'version', 'old_type': 'str', 'new_type': 'int'}
            ],
            'array_length_changes': [
                {'key': 'items', 'old_length': 5, 'new_length': 10}
            ]
        }
        """
        pass

    def should_skip_file(self, file_path: str) -> bool:
        """
        Skip certain JSON files (configurable)
        - package-lock.json (too noisy)
        - Large auto-generated files
        """
        pass
```

### 4. AI Summary Generator (Expensive Operation)

```python
class JsonChangeAISummarizer:
    """Generates AI summaries for JSON diffs"""

    def __init__(self, llm_router, cache_path: str):
        self.llm = llm_router
        self.cache = self._load_cache(cache_path)

    def summarize_change(self, change: JsonFileChange, force: bool = False) -> str:
        """
        Generate AI summary for a JSON file change

        Caching strategy:
        - Cache key: hash(commit_hash + file_path + diff)
        - Skip if already summarized (unless force=True)
        - Batch multiple changes for cost efficiency

        Prompt:
        "Analyze this JSON file change from commit {hash}:

        File: {file_path}
        Author: {author}
        Commit: {commit_message}

        Structured changes:
        - Keys added: {keys_added}
        - Keys removed: {keys_removed}
        - Keys modified: {keys_modified}

        Raw diff:
        {raw_diff}

        Provide a concise 2-3 sentence summary of:
        1. What changed and why (infer from context)
        2. Impact on functionality or configuration
        3. Any breaking changes or deprecations"

        Returns: AI-generated summary
        """
        pass

    def batch_summarize(self, changes: List[JsonFileChange],
                       max_batch_size: int = 10) -> Dict[str, str]:
        """
        Batch summarize multiple changes for cost efficiency
        Uses single LLM call with multiple diffs
        """
        pass

    def estimate_cost(self, num_changes: int) -> float:
        """Estimate API cost before running"""
        pass
```

### 5. Main Orchestrator

```python
class JsonFileTracker:
    """Main orchestrator for JSON file change tracking"""

    def __init__(self, repo_path: str, output_dir: str, llm_router):
        self.repo = Repo(repo_path)
        self.output_dir = Path(output_dir) / "json_tracking"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.diff_analyzer = JsonDiffAnalyzer()
        self.ai_summarizer = JsonChangeAISummarizer(
            llm_router,
            self.output_dir / "summaries_cache.json"
        )

        self.checkpoint = JsonTrackerCheckpoint.load(
            self.output_dir / "checkpoint.json"
        )

    def track_changes(self,
                     since_commit: Optional[str] = None,
                     since_date: Optional[str] = None,
                     with_ai_summaries: bool = False,
                     file_patterns: List[str] = ["*.json"],
                     exclude_patterns: List[str] = ["*lock.json"]):
        """
        Main tracking method

        Steps:
        1. Load checkpoint (resume from last run)
        2. Get commits since checkpoint/since_commit/since_date
        3. Filter commits with JSON file changes
        4. Extract structured diffs
        5. Optionally generate AI summaries (expensive!)
        6. Update per-file histories
        7. Save checkpoint

        Args:
            since_commit: Start from specific commit (overrides checkpoint)
            since_date: Start from date (overrides checkpoint)
            with_ai_summaries: Generate AI summaries (expensive)
            file_patterns: JSON files to track
            exclude_patterns: Patterns to ignore
        """
        pass

    def get_file_history(self, file_path: str) -> JsonFileHistory:
        """Retrieve complete history for a file"""
        pass

    def generate_report(self, output_format: str = "markdown") -> str:
        """
        Generate human-readable report

        Formats:
        - markdown: Timeline-style report per file
        - json: Machine-readable export
        - html: Interactive web report
        """
        pass
```

### 6. CLI Integration

```python
# In gitview/cli.py

@click.command()
@click.option('--track-json', is_flag=True,
              help='Enable JSON file change tracking')
@click.option('--json-ai-summaries', is_flag=True,
              help='Generate AI summaries for JSON changes (expensive!)')
@click.option('--json-since-commit',
              help='Track JSON changes since specific commit')
@click.option('--json-file-patterns', default='*.json',
              help='Patterns for JSON files to track')
@click.option('--json-exclude', default='*lock.json',
              help='Patterns to exclude (package-lock, etc.)')
def analyze(..., track_json, json_ai_summaries, ...):
    """Enhanced analyze command with JSON tracking"""

    if track_json:
        json_tracker = JsonFileTracker(repo_path, output_dir, llm_router)

        # Incremental by default
        since_commit = json_ai_summaries or checkpoint.resume_from_commit

        if json_ai_summaries:
            # Estimate cost first
            num_changes = json_tracker.estimate_changes(since_commit)
            cost = json_tracker.ai_summarizer.estimate_cost(num_changes)

            if not click.confirm(f"AI summarization will cost ~${cost:.2f}. Continue?"):
                return

        json_tracker.track_changes(
            since_commit=since_commit,
            with_ai_summaries=json_ai_summaries
        )

        # Generate report
        report_path = json_tracker.generate_report()
        click.echo(f"JSON tracking complete: {report_path}")
```

## Usage Examples

### 1. Initial Full Scan (No AI)
```bash
# Fast: Just extract structural diffs, no AI
gitview analyze . --track-json

# Output:
# ✓ Processed 1,247 commits
# ✓ Found 342 JSON file changes across 15 files
# ✓ Saved checkpoint: output/json_tracking/checkpoint.json
# ✓ Report: output/json_tracking/report.md
```

### 2. Incremental Update with AI Summaries
```bash
# Expensive: Only process new commits, generate AI summaries
gitview analyze . --track-json --json-ai-summaries

# Output:
# ✓ Resuming from commit abc1234 (last run: 2026-01-15)
# ✓ Found 23 new commits with JSON changes
# ⚠ AI summarization will process 23 diffs (~$0.45)
# Continue? [y/N]: y
# ✓ Generated 23 AI summaries
# ✓ Updated checkpoint
```

### 3. Track Specific Files Only
```bash
# Only track config files
gitview analyze . --track-json --json-file-patterns "config/*.json,settings.json"
```

### 4. Backfill AI Summaries
```bash
# Already ran structural analysis, now add AI summaries
gitview analyze . --track-json --json-ai-summaries --json-since-commit HEAD~100
```

## Output Examples

### Per-File History (JSON)
```json
{
  "file_path": "config/database.json",
  "first_seen_commit": "abc1234",
  "last_seen_commit": "xyz9876",
  "total_changes": 12,
  "contributors": [
    {"author": "Alice", "change_count": 8},
    {"author": "Bob", "change_count": 4}
  ],
  "changes": [
    {
      "commit_hash": "abc1234",
      "timestamp": "2025-01-15T10:30:00Z",
      "author": "Alice",
      "commit_message": "Update database timeout settings",
      "diff_type": "modified",
      "keys_added": ["connection_pool.max_size"],
      "keys_removed": ["deprecated_option"],
      "keys_modified": ["timeout", "retry_attempts"],
      "ai_summary": "Increased database timeout from 30s to 60s and added connection pooling with max 100 connections. Removed deprecated 'use_legacy_driver' option. This change improves stability under high load but may slightly increase memory usage."
    }
  ]
}
```

### Summary Report (Markdown)
```markdown
# JSON File Change History

Generated: 2026-01-18
Commits analyzed: 1,247
JSON files tracked: 15
Total changes: 342

---

## config/database.json

**First seen:** 2024-03-15 (commit abc1234)
**Last modified:** 2026-01-18 (commit xyz9876)
**Total changes:** 12
**Contributors:** Alice (8), Bob (4)

### Timeline

#### 2026-01-18 | xyz9876 | Alice
**Update database timeout settings**

- **Modified keys:** `timeout`, `retry_attempts`
- **Added keys:** `connection_pool.max_size`
- **Removed keys:** `deprecated_option`

**AI Summary:**
Increased database timeout from 30s to 60s and added connection pooling with max 100 connections. Removed deprecated 'use_legacy_driver' option. This change improves stability under high load but may slightly increase memory usage.

---

#### 2025-12-10 | def5678 | Bob
...
```

## Cost Optimization Strategies

### 1. Incremental Processing
- **Checkpoint system:** Only process commits since last run
- **Smart resumption:** Detect last processed commit automatically
- **Skip unchanged files:** If file didn't change between runs

### 2. Selective AI Summarization
- **Flag-based:** `--json-ai-summaries` opt-in (expensive)
- **Tiered approach:**
  - Free tier: Structural diffs only
  - Paid tier: AI summaries
- **File filtering:** Only summarize important files (exclude lock files, etc.)

### 3. Batch Processing
- **Batch LLM calls:** Send 10-20 diffs per API call
- **Parallel processing:** Use async/concurrent requests
- **Model selection:** Use cheaper models (gpt-4o-mini, claude-haiku) for summaries

### 4. Caching
- **Summary cache:** Never regenerate same diff summary
- **Cache key:** `hash(commit + file_path + diff)`
- **Invalidation:** Only on force flag

### 5. Smart Diff Generation
- **Skip trivial changes:** Version bumps, formatting-only changes
- **Threshold filtering:** Only summarize diffs > N lines
- **Semantic grouping:** Batch related changes (e.g., all config changes in one commit)

## Implementation Plan

### Phase 1: Core Infrastructure (No AI)
1. ✅ Create `JsonFileChange` and `JsonFileHistory` data models
2. ✅ Implement `JsonDiffAnalyzer` for structural diff extraction
3. ✅ Add checkpoint system for incremental processing
4. ✅ Build file-based cache storage
5. ✅ Integrate with existing `GitHistoryExtractor`

**Deliverable:** Fast JSON change tracking without AI costs

### Phase 2: AI Integration
1. ✅ Implement `JsonChangeAISummarizer` with caching
2. ✅ Add cost estimation before running
3. ✅ Integrate with existing `LLMRouter`
4. ✅ Add `--json-ai-summaries` CLI flag

**Deliverable:** Optional AI-powered summaries

### Phase 3: Reporting & Polish
1. ✅ Generate markdown reports per file
2. ✅ Create master index of all tracked files
3. ✅ Add filtering/search capabilities
4. ✅ Optional: HTML interactive report
5. ✅ Optional: Export to separate branch (as requested)

**Deliverable:** Human-readable change histories

### Phase 4: Advanced Features (Future)
- Semantic change detection (breaking changes, deprecations)
- Diff visualization (before/after JSON trees)
- Integration with PR context (why JSON changed)
- Change impact analysis (which code depends on this config?)

## Alternative: Separate Branch Storage

Instead of `output/json_tracking/`, store histories in orphan branch:

```bash
# Create orphan branch for metadata
git checkout --orphan json-tracking-data
git rm -rf .

# Structure:
# files/config/database.json.history
# files/package.json.history
# checkpoint.json
# index.json

# Commit each update
git add .
git commit -m "Update JSON tracking (processed commits abc..xyz)"
git push origin json-tracking-data
```

**Pros:**
- Version-controlled change histories
- Separate from working tree
- Easy to share across team

**Cons:**
- More complex git operations
- Requires branch management
- Harder to merge conflicts

**Recommendation:** Start with file-based storage, add branch storage as Phase 4 feature.

## Security & Privacy

- **Sensitive data:** Add `--json-exclude` for secrets files
- **Diff sanitization:** Option to redact values, keep only keys
- **API key safety:** Never log API keys from JSON configs
- **PII detection:** Warn if tracking files with potential PII

## Testing Strategy

1. **Unit tests:**
   - JSON diff extraction accuracy
   - Cache invalidation logic
   - Checkpoint save/load

2. **Integration tests:**
   - Full workflow on sample repo
   - Incremental processing correctness
   - AI summarization with mock LLM

3. **Cost tests:**
   - Verify cost estimation accuracy
   - Ensure caching prevents duplicate API calls

## Success Metrics

- **Performance:** Process 1,000 commits in < 30 seconds (without AI)
- **Cost:** < $0.10 per 100 JSON changes with AI summaries (using GPT-4o-mini)
- **Accuracy:** 100% cache hit rate on re-runs (no duplicate summaries)
- **Usability:** One-command incremental updates (`gitview analyze . --track-json`)

---

## Questions for Discussion

1. **Storage preference:** File-based or separate branch?
2. **AI summarization:** Always on, opt-in, or separate command?
3. **File filtering:** Default patterns to include/exclude?
4. **Report format:** Markdown only or add HTML/interactive?
5. **Batch size:** How many diffs per AI call for cost optimization?

Let me know your preferences and I can start implementation!
