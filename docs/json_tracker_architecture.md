# JSON File Tracker - Architecture Diagram

## High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Git Repository                             │
│  (1,247 commits with JSON changes across history)                   │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    JsonFileTracker (Orchestrator)                   │
│  • Manages incremental processing                                   │
│  • Coordinates components                                           │
│  • Handles caching strategy                                         │
└────────────┬───────────────────────────┬────────────────────────────┘
             │                           │
             ▼                           ▼
┌────────────────────────┐  ┌───────────────────────────────────────┐
│  Checkpoint Manager    │  │    JsonDiffAnalyzer                   │
│  ─────────────────     │  │    ─────────────────                  │
│  Last run: abc1234     │  │    1. Filter JSON files               │
│  Resume from: xyz9876  │  │    2. Extract before/after            │
│  AI summaries: 342     │  │    3. Parse JSON structures           │
│  Cost: $5.23           │  │    4. Deep comparison                 │
└────────────────────────┘  │    5. Generate structured diff        │
                            └───────────┬───────────────────────────┘
                                         │
                                         ▼
                             ┌───────────────────────────────────────┐
                             │      JsonFileChange Object            │
                             │      ──────────────────────           │
                             │      • Commit metadata                │
                             │      • Keys added/removed/modified    │
                             │      • Structural changes             │
                             │      • Raw diff (for AI)              │
                             └───────────┬───────────────────────────┘
                                         │
                   ┌─────────────────────┴────────────────────┐
                   │                                           │
                   ▼                                           ▼
┌──────────────────────────────────┐          ┌───────────────────────┐
│  Cache Layer                     │          │  JsonChangeAI         │
│  ────────────                    │          │  Summarizer           │
│  • Summary cache (hash-based)    │◄─────────┤  ────────────         │
│  • Skip if already summarized    │          │  (EXPENSIVE)          │
│  • 100% hit rate on re-runs      │          │                       │
└──────────────────────────────────┘          │  • Cost estimation    │
                                              │  • Batch processing   │
                                              │  • LLM integration    │
                                              │  • Smart prompting    │
                                              └───────────┬───────────┘
                                                           │
                                                           ▼
                             ┌───────────────────────────────────────┐
                             │   JsonFileChange (enriched)           │
                             │   ────────────────────────            │
                             │   + ai_summary: "Increased timeout    │
                             │     from 30s to 60s, added            │
                             │     connection pooling..."            │
                             └───────────┬───────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Per-File History Aggregator                    │
│  • Group changes by file path                                       │
│  • Calculate statistics (total changes, contributors)               │
│  • Build chronological timeline                                     │
└────────────┬────────────────────────────┬───────────────────────────┘
             │                            │
             ▼                            ▼
┌────────────────────────┐  ┌────────────────────────────────────────┐
│  Storage Layer         │  │   Report Generator                     │
│  ──────────────        │  │   ────────────────                     │
│  output/json_tracking/ │  │   • Markdown timeline                  │
│  ├── checkpoint.json   │  │   • JSON export                        │
│  ├── files/            │  │   • HTML interactive (future)          │
│  │   ├── config.json.history                                       │
│  │   └── package.json.history                                      │
│  ├── summaries_cache.json                                          │
│  └── tracking_index.json                                           │
└────────────────────────┘  └────────────────────────────────────────┘
```

## Incremental Processing Flow

```
First Run (Full Scan)
──────────────────────

Git History (all commits)
    │
    ├─> Commit 1 (adds config.json)
    │   └─> Extract diff → Cache → [No AI summary]
    │
    ├─> Commit 2 (modifies config.json)
    │   └─> Extract diff → Cache → [No AI summary]
    │
    ├─> Commit 3 (modifies package.json)
    │   └─> Extract diff → Cache → [No AI summary]
    │
    └─> ... (1,244 more commits)

Save checkpoint: last_processed = Commit 1247
Output: 342 structured diffs (no AI cost)


Second Run (Incremental + AI)
──────────────────────────────

Load checkpoint: resume_from = Commit 1247

Git History (only new commits since 1247)
    │
    ├─> Commit 1248 (modifies config.json)
    │   └─> Extract diff → Check cache → Generate AI summary → Cache
    │
    ├─> Commit 1249 (modifies settings.json)
    │   └─> Extract diff → Check cache → Generate AI summary → Cache
    │
    └─> Commit 1250 (modifies config.json)
        └─> Extract diff → Check cache → Generate AI summary → Cache

Cost estimation: 3 new diffs × $0.15 = $0.45
Update checkpoint: last_processed = Commit 1250


Third Run (Re-run same range)
──────────────────────────────

Load checkpoint: resume_from = Commit 1250
No new commits found → Skip processing
0 API calls, $0.00 cost
```

## Data Flow Example

```
Input: Commit xyz789 modifies config/database.json

Step 1: GitDiffAnalyzer
───────────────────────
Before (commit^):
{
  "timeout": 30,
  "retry_attempts": 3,
  "use_legacy_driver": true
}

After (commit):
{
  "timeout": 60,
  "retry_attempts": 5,
  "connection_pool": {
    "max_size": 100
  }
}

Output →
{
  "keys_added": ["connection_pool", "connection_pool.max_size"],
  "keys_removed": ["use_legacy_driver"],
  "keys_modified": ["timeout", "retry_attempts"],
  "raw_diff": "... unified diff ..."
}


Step 2: Cache Check
───────────────────
cache_key = hash("xyz789" + "config/database.json" + raw_diff)
           = "a3f8e7c9..."

Check summaries_cache.json:
- Key not found → Proceed to AI

(On re-run: Key found → Skip AI, return cached summary)


Step 3: AI Summarizer (if not cached)
──────────────────────────────────────
Prompt to LLM:
"""
Analyze this JSON file change from commit xyz789:

File: config/database.json
Author: Alice
Commit: Update database timeout settings

Structured changes:
- Keys added: connection_pool.max_size
- Keys removed: use_legacy_driver
- Keys modified: timeout (30→60), retry_attempts (3→5)

Raw diff:
{raw_diff}

Provide a concise 2-3 sentence summary...
"""

LLM Response:
"Increased database timeout from 30s to 60s and retry attempts from 3 to 5
for better resilience. Added connection pooling with max 100 connections to
improve performance under load. Removed deprecated 'use_legacy_driver' option."

Cost: ~$0.001 (GPT-4o-mini)


Step 4: Cache & Store
─────────────────────
Save to summaries_cache.json:
{
  "a3f8e7c9...": {
    "summary": "...",
    "generated_at": "2026-01-18T10:30:00Z",
    "model": "gpt-4o-mini",
    "cost": 0.001
  }
}

Update config/database.json.history:
{
  "file_path": "config/database.json",
  "changes": [
    ...,
    {
      "commit_hash": "xyz789",
      "ai_summary": "...",
      "keys_added": [...],
      ...
    }
  ]
}
```

## Cost Optimization Strategies

```
┌─────────────────────────────────────────────────────────────────┐
│                      Cost Reduction Layers                      │
└─────────────────────────────────────────────────────────────────┘

Layer 1: Incremental Processing (90% cost reduction)
────────────────────────────────────────────────────
Only process new commits since last run
1,000 commits first run → 10 commits incremental updates


Layer 2: Caching (100% reduction on re-runs)
─────────────────────────────────────────────
Hash-based summary cache
Re-run same commits → 0 API calls


Layer 3: Batch Processing (50% cost reduction)
───────────────────────────────────────────────
Single API call for 10 diffs instead of 10 calls
Reduces overhead, increases throughput


Layer 4: Smart Filtering (70% reduction)
─────────────────────────────────────────
Exclude lock files, auto-generated files
1,000 JSON changes → 300 meaningful changes


Layer 5: Model Selection (80% cost reduction)
──────────────────────────────────────────────
GPT-4o-mini vs Claude Opus
$0.001/change vs $0.015/change


Combined Savings Example:
─────────────────────────
Naive approach: 1,000 changes × $0.015 = $15.00
Optimized:
  - Incremental: 100 new changes
  - Filtered: 30 meaningful changes
  - Cached: 10 new changes
  - Batched: 2 API calls
  - GPT-4o-mini: $0.001/change
  Total: $0.01 (99.9% cost reduction!)
```

## Integration with Existing GitView

```
┌─────────────────────────────────────────────────────────────────┐
│                  Enhanced GitView Architecture                  │
└─────────────────────────────────────────────────────────────────┘

Existing Flow:
──────────────
Git Repo → GitHistoryExtractor → HistoryChunker → PhaseSummarizer
         → StoryTeller → OutputWriter

New Parallel Flow:
──────────────────
Git Repo → GitHistoryExtractor ──┬──→ HistoryChunker → ...
                                  │
                                  └──→ JsonFileTracker
                                        (if --track-json flag set)

Shared Components:
──────────────────
• LLMRouter (reuse existing backend selection)
• RemoteRepoHandler (reuse repo caching)
• GitPython utilities (commit traversal)
• Output directory structure

New Components:
───────────────
• JsonDiffAnalyzer (JSON-specific logic)
• JsonChangeAISummarizer (specialized prompts)
• JsonFileTracker orchestrator
• JSON-specific checkpoint system
```

## CLI Command Examples

```bash
# Example 1: First-time setup (fast, no AI)
$ gitview analyze /path/to/repo --track-json

Processing commits: [████████████████████] 1,247/1,247
Found JSON changes in 342 commits across 15 files
Saved checkpoint: output/json_tracking/checkpoint.json
Report: output/json_tracking/report.md

Time: 12s
Cost: $0.00


# Example 2: Incremental update with AI
$ gitview analyze /path/to/repo --track-json --json-ai-summaries

Loading checkpoint from 2026-01-15...
Found 23 new commits with JSON changes

Estimating cost:
  • 23 diffs to summarize
  • Model: gpt-4o-mini
  • Estimated cost: $0.45

Continue? [y/N]: y

Generating AI summaries: [████████████████] 23/23
Saved 23 summaries to cache
Updated checkpoint

Time: 8s
Cost: $0.43 (actual)


# Example 3: Track specific files only
$ gitview analyze . --track-json \
    --json-file-patterns "config/*.json,settings.json" \
    --json-exclude "*lock.json,*generated.json"

Tracking patterns: config/*.json, settings.json
Excluding: *lock.json, *generated.json

Found 8 matching files with 127 changes
Report: output/json_tracking/report.md


# Example 4: Backfill AI summaries for existing data
$ gitview analyze . --track-json --json-ai-summaries \
    --json-since-commit abc1234

Loading existing structural diffs...
Found 342 diffs without AI summaries

Estimating cost: $5.13
Continue? [y/N]: y

Generating summaries (batch mode): [████████] 342/342
Saved to cache

Time: 45s
Cost: $4.98
```

## File Structure

```
gitview/
├── gitview/
│   ├── json_tracker.py              # New: Main orchestrator
│   ├── json_diff_analyzer.py        # New: Diff extraction
│   ├── json_ai_summarizer.py        # New: AI summary generation
│   ├── cli.py                        # Modified: Add --track-json flags
│   └── ... (existing files)
│
└── output/
    ├── json_tracking/                # New directory
    │   ├── checkpoint.json
    │   ├── summaries_cache.json
    │   ├── tracking_index.json
    │   ├── files/
    │   │   ├── config/
    │   │   │   └── database.json.history
    │   │   ├── package.json.history
    │   │   └── settings.json.history
    │   └── reports/
    │       ├── summary.md
    │       └── index.html (future)
    │
    └── ... (existing output files)
```
