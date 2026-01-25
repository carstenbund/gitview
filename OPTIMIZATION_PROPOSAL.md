# GitView Optimization Proposal
**Date:** 2026-01-25
**Status:** Draft for Review
**Version:** 1.0

---

## Executive Summary

This proposal addresses critical fragmentation, duplication, and maintainability issues in the GitView codebase. The analysis reveals:

1. **CLI Bloat**: `cli.py` has grown to **2,877 lines** with 15+ commands embedded directly, creating maintenance burden
2. **Competing Systems**: Two incompatible storyline implementations exist side-by-side without integration
3. **Storage Duplication**: Multiple overlapping JSON storage trees and caching mechanisms
4. **Missed Modularization**: Existing modular structure (backends/, storyline/) is underutilized by CLI

**Impact**: Increased technical debt, code duplication, confusing user experience, and difficult feature maintenance.

**Recommendation**: Implement a 3-phase refactoring plan to consolidate systems, modularize CLI, and unify storage mechanisms.

---

## Table of Contents

1. [Problem Analysis](#problem-analysis)
   - [CLI.py Bloat](#1-clipy-bloat-2877-lines)
   - [Competing Storyline Systems](#2-competing-storyline-systems)
   - [Storage Tree Duplication](#3-storage-tree-duplication)
   - [Multiple JSON Extraction Processes](#4-multiple-json-extraction-processes)
2. [Detailed Findings](#detailed-findings)
3. [Optimization Proposal](#optimization-proposal)
4. [Implementation Plan](#implementation-plan)
5. [Migration Strategy](#migration-strategy)
6. [Risk Assessment](#risk-assessment)

---

## Problem Analysis

### 1. CLI.py Bloat (2,877 Lines)

#### Current State
The `cli.py` file contains:
- 1 main `@cli.group()` entry point
- 15+ command functions embedded directly:
  - `analyze()` (largest, ~500+ lines)
  - `extract()`
  - `chunk()`
  - `track_files()`
  - `file_history()`
  - `inject_history()`
  - `remove_history()`
  - `compare_branches()`
  - `storyline_group()` with 5 sub-commands:
    - `storyline_list()`
    - `storyline_show()`
    - `storyline_report()`
    - `storyline_export()`
    - `storyline_timeline()`
- Helper functions: `_estimate_analysis_cost()`, `_load_cached_analysis()`, `_analyze_single_branch()`, `_display_phase_overview()`
- 290+ lines of help text constants

#### Problems
1. **Single Responsibility Violation**: CLI should delegate to command modules, not implement logic
2. **Difficult Testing**: CLI commands are hard to unit test in isolation
3. **High Cognitive Load**: Developers must navigate 2,877 lines to find/modify commands
4. **Merge Conflicts**: High likelihood of conflicts with multiple developers working on different commands
5. **Code Reuse**: Logic like phase loading, summarization orchestration cannot be easily reused

#### Example of Bloat
The `analyze()` function alone spans **~500 lines** (lines 820-1320+) handling:
- Argument parsing and validation
- Repository cloning/detection
- Branch management
- Cache loading
- Extraction orchestration
- GitHub enrichment
- Phase chunking
- LLM summarization
- Story generation
- Output writing
- Multi-branch support
- Cost estimation

This should be delegated to an `AnalyzeCommand` class in `commands/analyze.py`.

---

### 2. Competing Storyline Systems

#### The Problem
**Two incompatible storyline implementations exist without integration:**

#### System 1: Old/Simple Storyline System (Summarizer)
**Location**: `gitview/summarizer.py`
**Created by**: `analyze` command during phase summarization
**Storage**: `output/phases/storylines.json` (simple format)
**Detection**: Regex parsing from LLM markdown summaries
**Format**:
```json
{
  "storylines": [
    {
      "title": "OAuth Implementation",
      "status": "active",
      "category": "feature",
      "description": "...",
      "first_phase": 1,
      "last_phase": 3,
      "phases_involved": [1, 2, 3]
    }
  ],
  "summary": {
    "total": 5,
    "active": 3,
    "completed": 2,
    "stalled": 0
  }
}
```

**Code References**:
- Parsing: `summarizer.py:13-47` (`_parse_storylines()`)
- Updating: `summarizer.py:49-92` (`_update_storyline_tracker()`)
- Saving: `summarizer.py:208-224` (`_save_storyline_tracker()`)
- Usage: `summarizer.py:155-206` (`summarize_all_phases()`)

**Characteristics**:
- ✅ Simple, lightweight
- ✅ Integrated into analyze workflow
- ❌ Single signal (LLM only)
- ❌ No confidence scoring
- ❌ No state machine lifecycle
- ❌ Regex-based parsing (brittle)
- ❌ Limited metadata

#### System 2: New/Advanced Storyline System (StorylineTracker)
**Location**: `gitview/storyline/` module
**Created by**: **NONE** (imported but not used in analyze)
**Storage**: `output/phases/storylines.json` (StorylineDatabase format)
**Detection**: Multi-signal detection with confidence scoring
**Format**:
```json
{
  "storylines": {
    "oauth-impl-phase1-abc123": {
      "id": "oauth-impl-phase1-abc123",
      "title": "OAuth Implementation",
      "category": "feature",
      "status": "progressing",
      "confidence": 0.85,
      "first_phase": 1,
      "last_phase": 3,
      "phases_involved": [1, 2, 3],
      "description": "...",
      "current_summary": "...",
      "signals": [
        {
          "source": "pr_label",
          "confidence": 0.9,
          "phase": 1,
          "category": "feature",
          "metadata": {"label": "feature", "pr_number": 42}
        },
        {
          "source": "file_cluster",
          "confidence": 0.7,
          "phase": 2,
          "files": ["auth/oauth.py", "auth/providers.py"]
        }
      ],
      "phase_updates": [...],
      "key_files": [...],
      "related_prs": [...]
    }
  },
  "indexes": {
    "title_index": {...},
    "phase_index": {...},
    "file_index": {...},
    "pr_index": {...}
  },
  "metadata": {
    "last_phase_analyzed": 5,
    "version": "1.0"
  }
}
```

**Code References**:
- Models: `storyline/models.py` (Storyline, StorylineDatabase, StorylineSignal)
- Detector: `storyline/detector.py` (multi-signal detection)
- Parser: `storyline/parser.py` (LLM response parsing with fallbacks)
- Tracker: `storyline/tracker.py` (StorylineTracker.process_phase)
- State Machine: `storyline/state_machine.py` (lifecycle management)
- Reporter: `storyline/reporter.py` (visualization and reporting)

**Multi-Signal Detection Sources**:
| Source | Confidence | Location |
|--------|------------|----------|
| PR Labels | 0.9 | `detector.py:32-65` |
| PR Title Patterns | 0.8 | `detector.py:67-98` |
| File Clusters | 0.7 | `detector.py:100-132` |
| Commit Messages | 0.6 | `detector.py:134-158` |
| LLM Extraction | 0.5 | `detector.py:160-184` |

**Characteristics**:
- ✅ Multi-signal detection
- ✅ Confidence scoring and weighting
- ✅ State machine lifecycle (emerging → active → progressing → completed/stalled/abandoned)
- ✅ Rich metadata (signals, updates, files, PRs)
- ✅ Indexed for fast queries
- ✅ Multiple JSON parsing strategies (fallback support)
- ❌ **NOT INTEGRATED** with analyze command
- ❌ Dead code (imported but unused)
- ❌ Duplicate effort

#### The Disconnect

**Current Flow**:
```
analyze command
    ↓
PhaseSummarizer.summarize_all_phases()
    ↓
_parse_storylines(summary)  ← OLD SYSTEM
    ↓
_update_storyline_tracker()
    ↓
_save_storyline_tracker() → phases/storylines.json (simple format)
```

**Expected Flow** (not happening):
```
analyze command
    ↓
StorylineTracker.process_phase()  ← NEW SYSTEM (NOT USED!)
    ↓
Multi-signal detection (PR labels, file clusters, commit msgs, LLM)
    ↓
State machine transitions
    ↓
StorylineDatabase.save() → phases/storylines.json (rich format)
```

**Impact on CLI Commands**:
- `gitview storyline list` expects `StorylineDatabase` format
- `analyze` command produces simple format
- **Incompatibility handled by conversion logic in CLI** (lines 2494-2542 in `cli.py`)
- Fragile, error-prone fallback system

#### Why This Happened
Based on code archaeology:
1. **Phase 1**: Simple storyline tracking was added to summarizer for basic narrative continuity
2. **Phase 2**: Advanced storyline system was built in `storyline/` module with multi-signal detection
3. **Phase 3**: CLI commands were added for storyline exploration (`storyline list`, `show`, etc.)
4. **Missing Step**: Integration of StorylineTracker into analyze command was never completed
5. **Result**: Two systems coexist, user confusion about which is authoritative

#### Evidence
- `StorylineTracker` is imported in `cli.py:27` but **never instantiated**
- `analyze` command never calls `StorylineTracker.process_phase()`
- Conversion logic exists to translate old format to new (`cli.py:2494-2542`)
- User documentation refers to advanced features (multi-signal, confidence) that don't work in practice

---

### 3. Storage Tree Duplication

#### Problem Overview
Multiple overlapping storage trees create confusion about data authority, increase disk usage, and complicate incremental analysis.

#### Duplication 1: Phase Data (3 Locations)

**Location 1**: Individual phase files
```
output/phases/
├── phase_01.json
├── phase_02.json
├── phase_03.json
└── phase_index.json
```
- **Format**: One JSON file per phase with full `Phase` object
- **Size**: ~5-50KB per phase
- **Created by**: `HistoryChunker.save_phases()` (`chunker.py:305-333`)
- **Purpose**: Modular storage, easy to load individual phases

**Location 2**: Complete phases array in history_data.json
```json
{
  "metadata": {...},
  "phases": [
    {...phase 1 full object...},
    {...phase 2 full object...},
    {...phase 3 full object...}
  ]
}
```
- **Format**: Array of full Phase objects
- **Size**: Entire array duplicated
- **Created by**: `OutputWriter.write_json()` (`writer.py:195-244`)
- **Purpose**: Complete snapshot for external tools

**Location 3**: Raw commit data (can reconstruct phases)
```
output/repo_history.jsonl
```
- **Format**: JSONL with one commit per line
- **Size**: ~1-5KB per commit
- **Created by**: `GitHistoryExtractor.save_to_jsonl()` (`extractor.py:485-492`)
- **Purpose**: Raw data, allows re-chunking with different strategies

**Problem**:
- Phase data exists in 3 serialized forms
- Updating a phase requires updating multiple files
- Incremental analysis must track which files are stale
- No single source of truth

#### Duplication 2: Storyline Data (3 Representations)

**Location 1**: Full database (new system)
```
output/phases/storylines.json  (StorylineDatabase format)
```
- Complete database with signals, indexes, metadata
- Created by: `StorylineDatabase.save()` (should be, but isn't currently)

**Location 2**: Legacy format (old system)
```
output/phases/storylines.json  (simple format)
```
- Basic storyline list with summary
- Created by: `PhaseSummarizer._save_storyline_tracker()`
- **Same filename, different format!**

**Location 3**: Embedded in history_data.json
```json
{
  "storylines": {
    "storylines": [...],
    "summary": {...}
  }
}
```
- Duplicate of whatever format was produced
- Created by: `OutputWriter.write_json()` (`writer.py:239-241`)

**Problem**:
- Same filename used for incompatible formats (overwrites)
- Embedded copy in history_data.json may be stale
- No versioning to distinguish formats

#### Duplication 3: Summary/Narrative Data

**Location 1**: Markdown report
```
output/history_story.md
```
- Human-readable narrative
- Sections: executive summary, timeline, technical evolution, deletion story, full narrative, storylines

**Location 2**: JSON stories object
```json
{
  "stories": {
    "executive_summary": "...",
    "timeline": "...",
    "technical_evolution": "...",
    "deletion_story": "...",
    "full_narrative": "...",
    "storylines": "..."
  }
}
```
- Embedded in `history_data.json`
- Exact duplicate of markdown content (but in JSON)

**Problem**:
- Narrative text stored twice
- Updates require modifying both files
- Unclear which is authoritative for programmatic access

#### Cache Overlap Analysis

**Multiple caching mechanisms with unclear coordination**:

| Cache | Location | Format | Invalidation | Purpose |
|-------|----------|--------|--------------|---------|
| Story Cache | `output/cached_story.json` | Hash-based | Phase content hash | Skip story regeneration |
| GitHub Cache | `~/.gitview/cache/github/` | GraphQL response | 24-hour TTL | Reduce GitHub API calls |
| Repo Cache | `~/.gitview/cache/repos/` | Git clone | 24-hour TTL | Avoid re-cloning |
| AI Summary Cache | `output/file_histories/summaries_cache.json` | Content hash | Content change | File tracking LLM costs |
| Checkpoint | `output/file_histories/checkpoint.json` | Commit hash | Manual reset | Resume file tracking |
| Phase Cache | Reusing `output/phases/` | File existence | Manual delete | Incremental analysis |

**Problems**:
1. **No unified cache manager**: Each cache has its own invalidation logic
2. **Inconsistent TTLs**: 24 hours vs. hash-based vs. manual
3. **Orphaned caches**: No cleanup mechanism for stale caches
4. **Cache interactions undefined**: What happens if story cache is valid but phase cache is invalid?

#### Recommendations

**Consolidation Strategy**:
1. **Phases**: Keep individual `phase_NN.json` files, reference them in `history_data.json` instead of embedding
2. **Storylines**: Single authoritative `phases/storylines.json` in `StorylineDatabase` format, backward-compat view in `history_data.json`
3. **Stories**: Keep markdown as primary, generate JSON view on demand
4. **Cache**: Unified cache manager with consistent invalidation rules

---

### 4. Multiple JSON Extraction Processes

#### Problem Overview
Three different mechanisms parse storyline data from LLM responses, leading to code duplication, inconsistent handling, and maintenance burden.

#### Extractor 1: Simple Regex in Summarizer
**Location**: `summarizer.py:13-47`
**Function**: `_parse_storylines(summary: str)`

**Method**:
- Regex pattern: `r'##\s*Storylines\s*\n(.*?)(?=\n##|\Z)'`
- Looks for markdown section `## Storylines`
- Extracts bullet points: `- [STATUS:category] title: description`
- Regex per line: `r'\[(\w+):(\w+)\]\s*([^:]+):\s*(.+)'`

**Returns**:
```python
[
  {
    "status": "active",
    "category": "feature",
    "title": "OAuth Implementation",
    "description": "Building OAuth2 support..."
  }
]
```

**Pros**:
- Simple, fast
- Integrated into summarizer

**Cons**:
- Brittle (exact markdown format required)
- No JSON support
- Limited error handling
- No validation

**Code**:
```python
def _parse_storylines(summary: str) -> List[Dict[str, Any]]:
    storyline_match = re.search(r'##\s*Storylines\s*\n(.*?)(?=\n##|\Z)', summary, re.DOTALL)
    if not storyline_match:
        return []

    storyline_text = storyline_match.group(1).strip()
    storylines = []

    for line in storyline_text.split('\n'):
        line = line.strip()
        if not line or not line.startswith('-'):
            continue

        match = re.match(r'-\s*\[(\w+):(\w+)\]\s*([^:]+):\s*(.+)', line)
        if match:
            status, category, title, description = match.groups()
            storylines.append({
                'status': status.lower(),
                'category': category.lower(),
                'title': title.strip(),
                'description': description.strip()
            })

    return storylines
```

#### Extractor 2: Signal Conversion in Tracker
**Location**: `storyline/tracker.py:134-164`
**Function**: `_convert_llm_storylines_to_signals()`

**Method**:
- Takes output from Extractor 1 (simple dict)
- Converts to `StorylineSignal` objects
- Adds metadata: source="llm_extraction", confidence=0.5

**Returns**:
```python
[
  StorylineSignal(
    source=SignalSource.LLM_EXTRACTION,
    confidence=0.5,
    phase_number=1,
    title="OAuth Implementation",
    category=StorylineCategory.FEATURE,
    description="...",
    metadata={"status": "active"}
  )
]
```

**Pros**:
- Bridges old and new systems
- Adds confidence scoring

**Cons**:
- Duplicate parsing (relies on Extractor 1)
- Assumes Extractor 1 format is correct
- No independent validation

**Code**:
```python
def _convert_llm_storylines_to_signals(
    llm_storylines: List[Dict[str, Any]],
    phase_number: int
) -> List[StorylineSignal]:
    signals = []
    for storyline in llm_storylines:
        signal = StorylineSignal(
            source=SignalSource.LLM_EXTRACTION,
            confidence=0.5,
            phase_number=phase_number,
            title=storyline['title'],
            category=StorylineCategory.from_string(storyline.get('category', 'feature')),
            description=storyline.get('description', ''),
            metadata={'status': storyline.get('status', 'active')}
        )
        signals.append(signal)
    return signals
```

#### Extractor 3: Robust Parser with Fallbacks
**Location**: `storyline/parser.py:27-104`
**Class**: `StorylineParser`

**Method** (Multi-Strategy):
1. **JSON Code Block Extraction**: Looks for ```json...```
2. **Raw JSON Object Extraction**: Looks for `{...}` in response
3. **Regex Markdown Extraction**: Similar to Extractor 1 but more robust

**Returns**: Same format as Extractor 1 but with validation

**Pros**:
- Robust with multiple fallback strategies
- Handles JSON and markdown
- Validates output
- Status normalization (continued → active)

**Cons**:
- Duplicate regex logic with Extractor 1
- Not used by analyze command (dead code)
- Adds complexity without benefit if not integrated

**Code** (simplified):
```python
class StorylineParser:
    def parse(self, response: str) -> Tuple[str, List[Dict[str, Any]]]:
        # Try JSON extraction first
        storylines = self._try_json_extraction(response)

        # Fallback to regex
        if not storylines:
            storylines = self._try_regex_extraction(response)

        # Validate
        validated = []
        for sl in storylines:
            if self._validate_storyline(sl):
                validated.append(self._normalize_storyline(sl))

        return response, validated

    def _try_json_extraction(self, response: str) -> List[Dict]:
        # Look for ```json code block
        match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Look for raw JSON object
        match = re.search(r'\{.*"storylines".*\}', response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                return data.get('storylines', [])
            except json.JSONDecodeError:
                pass

        return []

    def _try_regex_extraction(self, response: str) -> List[Dict]:
        # Similar to Extractor 1 but more robust
        ...
```

#### The Problem

**Three parsers doing the same job**:
1. `_parse_storylines()` in summarizer - **actually used**
2. `_convert_llm_storylines_to_signals()` in tracker - **bridges formats**
3. `StorylineParser.parse()` - **dead code, most robust but unused**

**Why This Happened**:
1. Extractor 1 was built first for simple use case
2. Advanced storyline system was added later
3. Extractor 2 was created to bridge old and new systems
4. Extractor 3 was built as "proper" implementation but never integrated
5. Extractor 1 is still in use because analyze command never switched

**Impact**:
- **Code duplication**: 3x regex patterns for same task
- **Maintenance burden**: Bug fixes need to be applied in 3 places
- **Dead code**: Extractor 3 is sophisticated but unused
- **Fragility**: Production uses simplest, least robust parser

#### Recommendations

**Unification Strategy**:
1. **Promote Extractor 3** to production (most robust)
2. **Replace Extractor 1** in summarizer with call to Extractor 3
3. **Remove Extractor 2** (signal conversion happens in parser)
4. **Single source of truth** for LLM response parsing

---

## Detailed Findings

### Finding 1: CLI Architecture Anti-Pattern

**Current Architecture**:
```
cli.py (2,877 lines)
├── 15+ @cli.command() decorators
├── Inline command logic (500+ lines per command)
├── Helper functions mixed with commands
└── Direct calls to extractor, chunker, summarizer, etc.
```

**Problems**:
- ❌ Single Responsibility Principle violation
- ❌ Difficult to unit test
- ❌ Hard to navigate and maintain
- ❌ Merge conflict hotspot
- ❌ Cannot reuse command logic programmatically

**Recommended Architecture**:
```
cli.py (main entry point, ~300 lines)
├── Command registration
├── Error handling
└── Delegates to commands/

commands/
├── __init__.py
├── analyze.py (AnalyzeCommand class)
├── extract.py (ExtractCommand class)
├── chunk.py (ChunkCommand class)
├── track_files.py (TrackFilesCommand class)
├── file_history.py (FileHistoryCommand class)
├── inject_history.py (InjectHistoryCommand class)
├── compare_branches.py (CompareBranchesCommand class)
└── storyline/
    ├── __init__.py
    ├── list.py (ListCommand class)
    ├── show.py (ShowCommand class)
    ├── report.py (ReportCommand class)
    ├── export.py (ExportCommand class)
    └── timeline.py (TimelineCommand class)
```

**Benefits**:
- ✅ Each command is a testable class
- ✅ Easy to navigate and maintain
- ✅ Logic can be reused (e.g., from Python API)
- ✅ Parallel development without conflicts
- ✅ Clear separation of concerns

**Example Refactoring**:

**Before** (`cli.py`):
```python
@cli.command()
@click.option('--repo', '-r', default=".")
@click.option('--output', '-o', default=None)
# ... 20+ more options ...
def analyze(repo, output, strategy, ...):
    """500+ lines of inline logic"""
    console.print(...)
    repo_handler = RemoteRepoHandler(repo)
    # ... extract, chunk, summarize, write ...
```

**After** (`cli.py`):
```python
from gitview.commands.analyze import AnalyzeCommand

@cli.command()
@click.option('--repo', '-r', default=".")
@click.option('--output', '-o', default=None)
# ... 20+ more options ...
def analyze(**kwargs):
    """Delegate to AnalyzeCommand"""
    cmd = AnalyzeCommand(**kwargs)
    cmd.run()
```

**After** (`commands/analyze.py`):
```python
class AnalyzeCommand:
    def __init__(self, repo, output, strategy, ...):
        self.repo = repo
        self.output = output
        self.strategy = strategy
        # ... store all options ...

    def run(self):
        """Main execution logic"""
        self.validate_options()
        self.setup_repository()
        self.extract_history()
        self.enrich_with_github()
        self.chunk_phases()
        self.summarize_phases()
        self.track_storylines()  # NEW: Actually use StorylineTracker
        self.generate_story()
        self.write_output()

    def validate_options(self): ...
    def setup_repository(self): ...
    def extract_history(self): ...
    # ... each step is a testable method ...
```

### Finding 2: Storyline Integration Failure

**What Was Intended** (based on code structure):
```
analyze command
    ↓
Extract commits → Chunk phases
    ↓
PhaseSummarizer.summarize_phase() for each phase
    ↓
StorylineTracker.process_phase() for each phase
    ├── Detect from PR labels (0.9 confidence)
    ├── Detect from PR titles (0.8 confidence)
    ├── Detect from file clusters (0.7 confidence)
    ├── Detect from commit messages (0.6 confidence)
    └── Detect from LLM summary (0.5 confidence)
    ↓
State machine updates (emerging → active → progressing → completed)
    ↓
StorylineDatabase.save() → phases/storylines.json
    ↓
StoryTeller uses storylines for narrative continuity
```

**What Actually Happens**:
```
analyze command
    ↓
Extract commits → Chunk phases
    ↓
PhaseSummarizer.summarize_phase() for each phase
    ↓
_parse_storylines(summary)  ← Regex extraction from LLM only
    ↓
_update_storyline_tracker()  ← Simple dict, no state machine
    ↓
_save_storyline_tracker() → phases/storylines.json (simple format)
    ↓
StoryTeller uses storylines for narrative continuity

[ StorylineTracker is imported but NEVER CALLED ]
```

**Evidence**:
- `StorylineTracker` imported at `cli.py:27`
- Never instantiated in `analyze()` function
- Never called in `_analyze_single_branch()` helper
- Not called in `PhaseSummarizer.summarize_all_phases()`

**Impact**:
- Multi-signal detection: **NOT WORKING**
- Confidence scoring: **NOT WORKING**
- State machine lifecycle: **NOT WORKING**
- Advanced storyline features advertised in README: **NOT WORKING**

**User Confusion**:
From README.md:
> ### Multi-Signal Detection
> Storylines are detected from multiple sources with confidence scoring:
> - PR Labels (0.9 confidence)
> - PR Title Patterns (0.8 confidence)
> - File Clusters (0.7 confidence)
> - Commit Messages (0.6 confidence)
> - LLM Extraction (0.5 confidence)

**Reality**: Only LLM extraction is used, always 0.5 confidence assumed.

### Finding 3: Cache Strategy Chaos

**Six Different Caching Mechanisms**:

1. **Story Cache** (`output/cached_story.json`)
   - Hash: SHA256 of phase summaries
   - Invalidation: Phase content change
   - Saves: 5 LLM calls (story sections)
   - Cost savings: ~$0.10-0.50 per unchanged run

2. **GitHub Cache** (`~/.gitview/cache/github/`)
   - Hash: GraphQL query hash
   - Invalidation: 24-hour TTL
   - Saves: GitHub API calls
   - Rate limit protection

3. **Repository Clone Cache** (`~/.gitview/cache/repos/`)
   - Key: {host}/{org}/{repo}
   - Invalidation: 24-hour TTL, then fetch
   - Saves: Re-cloning time (minutes to hours)

4. **AI Summary Cache** (`output/file_histories/summaries_cache.json`)
   - Hash: File change content hash
   - Invalidation: Content change
   - Saves: LLM calls for file summaries
   - Hit rate: >95% on reruns

5. **Checkpoint Cache** (`output/file_histories/checkpoint.json`)
   - Value: Last processed commit hash
   - Invalidation: Manual reset
   - Purpose: Resume file tracking

6. **Phase Cache** (reusing `output/phases/`)
   - Detection: File existence
   - Invalidation: Manual deletion or incremental mode
   - Purpose: Skip re-extraction and chunking

**Problems**:

1. **No Coordination**:
   - What if story cache is valid but phase cache is invalid?
   - What if GitHub cache is stale but repo cache is fresh?
   - No unified cache invalidation

2. **Inconsistent TTLs**:
   - 24 hours for GitHub and repo
   - Content-hash for story and AI summaries
   - Manual for checkpoint and phase
   - No configuration

3. **No Cleanup**:
   - Caches grow indefinitely
   - No LRU eviction
   - No max size limit
   - `~/.gitview/cache/` can grow to GBs

4. **Unclear Authority**:
   - If incremental mode is enabled, which caches are used?
   - If a cache is corrupted, how to recover?
   - No cache versioning

**Recommendation**:
- Unified `CacheManager` class
- Consistent TTL configuration
- Cache versioning and migration
- Automatic cleanup (LRU with max size)
- Clear invalidation rules

### Finding 4: Output File Redundancy

**Current Output Structure**:
```
output/
├── repo_history.jsonl           # 1-5KB per commit
├── phases/
│   ├── phase_01.json            # 5-50KB per phase
│   ├── phase_02.json
│   ├── phase_03.json
│   ├── phase_index.json         # Phase index
│   ├── cached_story.json        # Story cache
│   └── storylines.json          # Storyline data
├── history_story.md             # Markdown narrative (50-500KB)
├── history_data.json            # Complete JSON (50-500KB)
└── timeline.md                  # Simple timeline (10-50KB)
```

**Redundancy Analysis**:

| Data | Location 1 | Location 2 | Location 3 |
|------|------------|------------|------------|
| Phase objects | `phases/phase_NN.json` | `history_data.json` "phases" array | Can reconstruct from `repo_history.jsonl` |
| Storylines | `phases/storylines.json` | `history_data.json` "storylines" | None |
| Narratives | `history_story.md` | `history_data.json` "stories" | None |
| Timeline | `timeline.md` | `history_data.json` "stories.timeline" | `history_story.md` includes timeline |

**Why This Matters**:

**Scenario**: User runs `gitview analyze` on a 10,000 commit repository
- `repo_history.jsonl`: 50MB
- `phases/*.json`: 30MB (100 phases × 300KB average)
- `history_data.json`: 80MB (includes full copy of all phases!)
- `history_story.md`: 2MB
- **Total**: 162MB
- **Actual unique data**: ~82MB
- **Redundancy**: 80MB (49%)

**Incremental Analysis Impact**:
- When updating analysis, must update:
  - `repo_history.jsonl` (append new commits)
  - `phases/phase_NN.json` (update modified phases)
  - `history_data.json` (rewrite entire file with all phases)
  - `history_story.md` (regenerate)
  - `timeline.md` (regenerate)
- Risk of inconsistency if process is interrupted

**Recommendation**:
1. **Primary Storage**: Individual phase files and JSONL
2. **Views**: `history_data.json` should reference phases, not embed them
3. **Generated**: Markdown files generated on demand or from JSON view

**Proposed history_data.json**:
```json
{
  "metadata": {...},
  "phases": {
    "count": 100,
    "directory": "./phases",
    "files": ["phase_01.json", "phase_02.json", ...]
  },
  "storylines_file": "./phases/storylines.json",
  "stories": {
    "executive_summary": "...",
    "timeline": "...",
    ...
  }
}
```

**Benefits**:
- Reduce disk usage by 40-50%
- Faster incremental updates
- Clear data authority
- Easier to manage large repositories

---

## Optimization Proposal

### Phase 1: Storyline System Consolidation (High Priority)

**Goal**: Integrate `StorylineTracker` into analyze command, retire old simple system.

**Changes**:

1. **Modify `analyze` command** (`cli.py` or `commands/analyze.py` after refactoring):
   ```python
   # After phase summarization
   storyline_tracker = StorylineTracker(
       github_token=github_token,
       confidence_threshold=0.6
   )

   for phase in phases:
       # Process with multi-signal detection
       storyline_tracker.process_phase(phase)

   # Save database
   db_path = Path(output) / "phases" / "storylines.json"
   storyline_tracker.save_database(str(db_path))
   ```

2. **Remove old system** from `PhaseSummarizer`:
   - Delete `_parse_storylines()`
   - Delete `_update_storyline_tracker()`
   - Delete `_save_storyline_tracker()`
   - Remove `storyline_tracker` dict from `summarize_all_phases()`

3. **Unify JSON Extraction**:
   - Create `gitview/storyline/extraction.py` with `StorylineExtractor` class
   - Migrate logic from `StorylineParser` (most robust)
   - Use in `StorylineDetector` for LLM signal
   - Delete redundant parsers

4. **Update `StoryTeller`**:
   - Accept `StorylineDatabase` instead of simple list
   - Use rich metadata (confidence, signals) in narratives

5. **Backward Compatibility**:
   - `OutputWriter.write_json()` includes both formats temporarily:
     ```json
     {
       "storylines_v2": {...StorylineDatabase...},
       "storylines": {...legacy simple format for old tools...}
     }
     ```
   - Deprecation warning in v2.0, remove in v3.0

**Testing**:
- Unit tests for `StorylineExtractor`
- Integration test: analyze → verify StorylineDatabase format
- Regression test: compare narratives before/after (should improve)

**Timeline**: 2-3 weeks

**Impact**:
- ✅ Multi-signal detection enabled
- ✅ Confidence scoring working
- ✅ State machine lifecycle functional
- ✅ Advanced features match documentation
- ❌ Breaking change for tools that parse old format (hence backward compat)

---

### Phase 2: CLI Modularization (Medium Priority)

**Goal**: Extract command logic from `cli.py` into `commands/` module.

**Structure**:
```
gitview/
├── cli.py                  # Main entry, command registration (~300 lines)
├── commands/
│   ├── __init__.py         # Command registry
│   ├── base.py             # BaseCommand class
│   ├── analyze.py          # AnalyzeCommand
│   ├── extract.py          # ExtractCommand
│   ├── chunk.py            # ChunkCommand
│   ├── track_files.py      # TrackFilesCommand
│   ├── file_history.py     # FileHistoryCommand
│   ├── inject_history.py   # InjectHistoryCommand
│   ├── compare_branches.py # CompareBranchesCommand
│   └── storyline/
│       ├── __init__.py
│       ├── list.py         # ListStorylineCommand
│       ├── show.py         # ShowStorylineCommand
│       ├── report.py       # ReportStorylineCommand
│       ├── export.py       # ExportStorylineCommand
│       └── timeline.py     # TimelineStorylineCommand
└── ...existing modules...
```

**Base Command Pattern**:
```python
# commands/base.py
from abc import ABC, abstractmethod
from rich.console import Console

class BaseCommand(ABC):
    """Base class for all CLI commands."""

    def __init__(self, **kwargs):
        self.options = kwargs
        self.console = Console()

    @abstractmethod
    def validate(self):
        """Validate command options."""
        pass

    @abstractmethod
    def execute(self):
        """Execute command logic."""
        pass

    def run(self):
        """Template method: validate then execute."""
        self.validate()
        return self.execute()
```

**Example Command**:
```python
# commands/analyze.py
from .base import BaseCommand
from ..extractor import GitHistoryExtractor
from ..chunker import HistoryChunker
from ..summarizer import PhaseSummarizer
from ..storyteller import StoryTeller
from ..storyline import StorylineTracker
from ..writer import OutputWriter

class AnalyzeCommand(BaseCommand):
    """Analyze git repository and generate narrative history."""

    def validate(self):
        """Validate options."""
        if self.options.get('critical') and not self.options.get('todo'):
            self.console.print("[yellow]Warning: --critical without --todo[/yellow]")
        # ... more validation ...

    def execute(self):
        """Main analysis pipeline."""
        self.console.print("[bold]GitView - Repository History Analyzer[/bold]")

        # Step 1: Setup
        self._setup_repository()

        # Step 2: Extract
        self.console.print("[bold]Step 1: Extracting git history...[/bold]")
        records = self._extract_history()

        # Step 3: Enrich
        if self.options.get('github_token'):
            records = self._enrich_with_github(records)

        # Step 4: Chunk
        self.console.print("[bold]Step 2: Chunking into phases...[/bold]")
        phases = self._chunk_phases(records)

        # Step 5: Summarize
        self.console.print("[bold]Step 3: Summarizing phases...[/bold]")
        phases = self._summarize_phases(phases)

        # Step 6: Track storylines
        self.console.print("[bold]Step 4: Tracking storylines...[/bold]")
        storyline_db = self._track_storylines(phases)

        # Step 7: Generate story
        self.console.print("[bold]Step 5: Generating global narrative...[/bold]")
        stories = self._generate_story(phases, storyline_db)

        # Step 8: Write output
        self.console.print("[bold]Step 6: Writing output files...[/bold]")
        self._write_output(stories, phases, storyline_db)

        self.console.print("[green]Analysis complete![/green]")

    def _setup_repository(self):
        """Setup repository path and cloning."""
        # ... extracted from current analyze() ...

    def _extract_history(self):
        """Extract git history."""
        extractor = GitHistoryExtractor(self.repo_path)
        return extractor.extract_history(
            max_commits=self.options.get('max_commits'),
            branch=self.options.get('branch')
        )

    # ... other methods ...

    def _track_storylines(self, phases):
        """Track storylines using multi-signal detection."""
        tracker = StorylineTracker(
            github_token=self.options.get('github_token'),
            confidence_threshold=0.6
        )

        for phase in phases:
            tracker.process_phase(phase)

        # Save database
        db_path = self.output_path / "phases" / "storylines.json"
        tracker.save_database(str(db_path))

        return tracker.database
```

**CLI Registration**:
```python
# cli.py (simplified)
import click
from gitview.commands.analyze import AnalyzeCommand
from gitview.commands.extract import ExtractCommand
# ... other imports ...

@click.group()
@click.version_option(version=__version__)
def cli():
    """GitView - Git history analyzer with LLM-powered narrative generation."""
    pass

@cli.command()
@click.option('--repo', '-r', default=".")
@click.option('--output', '-o', default=None)
# ... all options ...
def analyze(**kwargs):
    """Analyze git repository and generate narrative history."""
    cmd = AnalyzeCommand(**kwargs)
    return cmd.run()

@cli.command()
@click.option('--repo', default=".")
@click.option('--output', required=True)
# ... options ...
def extract(**kwargs):
    """Extract git history to JSONL file."""
    cmd = ExtractCommand(**kwargs)
    return cmd.run()

# ... similar for all commands ...

def main():
    cli()
```

**Benefits**:
- Each command is independently testable
- Clear separation of concerns
- Can reuse command logic programmatically:
  ```python
  from gitview.commands.analyze import AnalyzeCommand

  cmd = AnalyzeCommand(repo=".", output="output")
  result = cmd.run()
  ```
- Parallel development without conflicts
- Easier onboarding (one file per command)

**Migration Strategy**:
1. Create `commands/` module and `BaseCommand`
2. Extract one command at a time (start with smallest: `extract`, `chunk`)
3. Update `cli.py` to delegate
4. Add unit tests for each command
5. Repeat for remaining commands
6. Remove old inline implementations

**Timeline**: 4-6 weeks

**Risk**: Low (backward compatible, CLI interface unchanged)

---

### Phase 3: Storage Consolidation (Low Priority)

**Goal**: Reduce redundancy, establish clear data authority.

**Changes**:

1. **Phases Storage**:
   - **Keep**: Individual `phases/phase_NN.json` files
   - **Change**: `history_data.json` references phases instead of embedding
   ```json
   {
     "metadata": {...},
     "phases": {
       "count": 100,
       "directory": "./phases",
       "index_file": "./phases/phase_index.json"
     },
     "stories": {...}
   }
   ```

2. **Storylines Storage**:
   - **Primary**: `phases/storylines.json` (StorylineDatabase format)
   - **Deprecated**: Remove from `history_data.json` (or keep reference only)

3. **Cache Consolidation**:
   - Create `CacheManager` class:
     ```python
     class CacheManager:
         """Unified cache management."""

         def __init__(self, cache_dir: str = "~/.gitview/cache"):
             self.cache_dir = Path(cache_dir).expanduser()
             self.config = self._load_config()

         def get(self, key: str, namespace: str = "default") -> Optional[Any]:
             """Get cached value."""
             cache_file = self._get_cache_file(key, namespace)
             if not cache_file.exists():
                 return None

             # Check TTL
             if self._is_expired(cache_file, namespace):
                 cache_file.unlink()
                 return None

             with open(cache_file) as f:
                 return json.load(f)

         def set(self, key: str, value: Any, namespace: str = "default"):
             """Set cached value."""
             cache_file = self._get_cache_file(key, namespace)
             cache_file.parent.mkdir(parents=True, exist_ok=True)

             with open(cache_file, 'w') as f:
                 json.dump({
                     'timestamp': datetime.now().isoformat(),
                     'value': value
                 }, f)

         def invalidate(self, key: str = None, namespace: str = None):
             """Invalidate cache."""
             if key and namespace:
                 # Invalidate specific key
                 cache_file = self._get_cache_file(key, namespace)
                 if cache_file.exists():
                     cache_file.unlink()
             elif namespace:
                 # Invalidate entire namespace
                 namespace_dir = self.cache_dir / namespace
                 if namespace_dir.exists():
                     shutil.rmtree(namespace_dir)
             else:
                 # Invalidate all
                 if self.cache_dir.exists():
                     shutil.rmtree(self.cache_dir)

         def cleanup(self, max_size_mb: int = 1000):
             """LRU cleanup if cache exceeds max size."""
             total_size = self._get_cache_size()
             if total_size > max_size_mb * 1024 * 1024:
                 self._lru_evict(total_size - max_size_mb * 1024 * 1024)

         def _is_expired(self, cache_file: Path, namespace: str) -> bool:
             """Check if cache entry is expired based on namespace TTL."""
             ttl = self.config.get('ttls', {}).get(namespace, 86400)  # 24h default
             mtime = cache_file.stat().st_mtime
             return (time.time() - mtime) > ttl

         # ... other methods ...
     ```

   - Use across all caching needs:
     ```python
     cache = CacheManager()

     # GitHub cache
     cache.set(query_hash, response, namespace="github")

     # Story cache
     cache.set(phase_hash, stories, namespace="stories")

     # AI summaries
     cache.set(content_hash, summary, namespace="ai_summaries")
     ```

   - Configuration file `~/.gitview/cache/config.json`:
     ```json
     {
       "ttls": {
         "github": 86400,
         "repos": 86400,
         "stories": 0,
         "ai_summaries": 0
       },
       "max_size_mb": 1000,
       "cleanup_policy": "lru"
     }
     ```

**Benefits**:
- Reduce disk usage by 40-50%
- Faster incremental updates
- Consistent caching behavior
- Configurable TTLs
- Automatic cleanup

**Timeline**: 3-4 weeks

**Risk**: Medium (data migration required, potential for data loss if not careful)

---

## Implementation Plan

### Recommended Sequence

**Phase 1: Storyline Consolidation** (Weeks 1-3)
- Week 1: Integrate `StorylineTracker` into analyze command
- Week 2: Create `StorylineExtractor`, migrate parsing logic
- Week 3: Remove old system, add backward compatibility, testing

**Phase 2: CLI Modularization** (Weeks 4-9)
- Week 4: Create `commands/` structure, `BaseCommand`, extract simple commands
- Week 5-6: Extract `analyze` command (largest, most complex)
- Week 7: Extract file tracking commands
- Week 8: Extract storyline subcommands
- Week 9: Final cleanup, documentation, testing

**Phase 3: Storage Consolidation** (Weeks 10-13)
- Week 10: Implement `CacheManager`, migrate GitHub cache
- Week 11: Migrate other caches, add cleanup
- Week 12: Modify `history_data.json` to use references
- Week 13: Testing, data migration script, documentation

### Milestones

- **M1** (End of Week 3): Storyline system unified, multi-signal detection working
- **M2** (End of Week 9): CLI fully modularized, all commands extracted
- **M3** (End of Week 13): Storage consolidated, cache unified

### Success Criteria

**Phase 1**:
- ✅ `analyze` command produces `StorylineDatabase` format
- ✅ Multi-signal detection tests pass (PR labels, file clusters, etc.)
- ✅ State machine transitions work correctly
- ✅ Backward compatibility maintained for old format consumers

**Phase 2**:
- ✅ `cli.py` reduced to <500 lines
- ✅ Each command has >80% test coverage
- ✅ Command classes can be instantiated and run programmatically
- ✅ All existing CLI functionality preserved (no regressions)

**Phase 3**:
- ✅ `history_data.json` uses phase references, not embeddings
- ✅ Disk usage reduced by >40% on test repositories
- ✅ Cache cleanup runs automatically, respects max size
- ✅ Cache TTLs configurable via config file

---

## Migration Strategy

### For Users

**Phase 1 (Storyline Consolidation)**:
- **Before**: `phases/storylines.json` in simple format
- **After**: `phases/storylines.json` in StorylineDatabase format
- **Migration**: Automatic conversion on first `analyze` run after upgrade
- **Backward Compat**: Old tools can still read legacy format from `history_data.json` (temporary)

**Phase 2 (CLI Modularization)**:
- **No Changes**: CLI interface remains identical
- **Benefit**: Faster command execution, better error messages

**Phase 3 (Storage Consolidation)**:
- **Before**: `history_data.json` includes full phase objects
- **After**: `history_data.json` references phase files
- **Migration**: Script to regenerate `history_data.json` from existing phases
- **Risk**: Tools that parse `history_data.json` directly need update

### For Developers

**Phase 1**:
```python
# Old way (still works but deprecated)
from gitview.summarizer import _parse_storylines
storylines = _parse_storylines(summary)  # Simple list

# New way
from gitview.storyline import StorylineTracker
tracker = StorylineTracker()
tracker.process_phase(phase)
storylines = tracker.database  # StorylineDatabase object
```

**Phase 2**:
```python
# Old way (no longer possible)
# Had to use CLI only

# New way (programmatic access)
from gitview.commands.analyze import AnalyzeCommand

cmd = AnalyzeCommand(
    repo="/path/to/repo",
    output="./output",
    backend="openai"
)
result = cmd.run()
```

**Phase 3**:
```python
# Old way
with open("output/history_data.json") as f:
    data = json.load(f)
    phases = [Phase.from_dict(p) for p in data['phases']]  # Full objects

# New way
with open("output/history_data.json") as f:
    data = json.load(f)
    phase_dir = data['phases']['directory']
    phases = HistoryChunker.load_phases(phase_dir)  # Load from files
```

### Versioning

**Semantic Versioning**:
- **Phase 1**: Minor version bump (v2.1.0) - new features, backward compatible
- **Phase 2**: Minor version bump (v2.2.0) - internal refactoring, no API changes
- **Phase 3**: Major version bump (v3.0.0) - storage format changes, migration required

**Deprecation Timeline**:
- v2.1.0: Old storyline format deprecated, warning issued
- v2.2.0: Old format still readable for 6 months
- v3.0.0: Old format support removed

---

## Risk Assessment

### Phase 1: Storyline Consolidation

**Risks**:
1. **Breaking Change**: Old tools that parse `phases/storylines.json` will fail
   - **Mitigation**: Maintain legacy format in `history_data.json` for 6 months
   - **Detection**: Version field in JSON indicates format

2. **Confidence Threshold Tuning**: Multi-signal detection may produce too many/few storylines
   - **Mitigation**: Make threshold configurable (default 0.6)
   - **Testing**: Run on 10+ diverse repos, validate output quality

3. **Performance**: Multi-signal detection is slower than simple regex
   - **Mitigation**: Parallelize PR label fetching, cache file cluster analysis
   - **Benchmark**: Should be <20% slower than current implementation

**Likelihood**: Medium
**Impact**: Medium
**Overall Risk**: Medium

### Phase 2: CLI Modularization

**Risks**:
1. **Regression**: Command behavior changes during extraction
   - **Mitigation**: Comprehensive integration tests before/after
   - **Testing**: CLI smoke tests for all commands

2. **Import Cycles**: Commands depend on each other
   - **Mitigation**: Use dependency injection, avoid circular imports
   - **Design**: BaseCommand should not import other commands

3. **Merge Conflicts**: Refactoring during active development
   - **Mitigation**: Feature freeze during extraction, communicate broadly
   - **Timeline**: Schedule during low-activity period

**Likelihood**: Low
**Impact**: High
**Overall Risk**: Medium

### Phase 3: Storage Consolidation

**Risks**:
1. **Data Loss**: Migration script fails, corrupts data
   - **Mitigation**: Backup before migration, dry-run mode
   - **Testing**: Test on diverse output directories

2. **Tool Breakage**: External tools parsing `history_data.json` fail
   - **Mitigation**: Deprecation period, migration guide
   - **Communication**: Announce breaking change in release notes

3. **Cache Invalidation Bugs**: Wrong data returned from cache
   - **Mitigation**: Conservative TTLs, version cache entries
   - **Testing**: Unit tests for cache expiration logic

**Likelihood**: Medium
**Impact**: High
**Overall Risk**: High

### Overall Project Risk

**Total Timeline**: 13 weeks
**Effort**: ~2-3 engineer-months
**Complexity**: Medium-High

**Risk Ranking**:
1. **Phase 3** (High Risk) - Storage changes are most dangerous
2. **Phase 1** (Medium Risk) - Feature integration, some backward compat concerns
3. **Phase 2** (Medium Risk) - Internal refactoring, low user impact

**Recommendation**:
- Start with Phase 2 (CLI modularization) - lowest risk, foundational
- Then Phase 1 (storyline consolidation) - builds on modular CLI
- Finally Phase 3 (storage) - most risky, benefits from previous work

**Alternative Sequence** (lower risk):
1. Phase 2 (Weeks 1-6): CLI modularization
2. Phase 1 (Weeks 7-9): Storyline consolidation
3. Phase 3 (Weeks 10-13): Storage consolidation

---

## Appendix: Code Size Reduction Estimates

### Current State
```
cli.py:                    2,877 lines
Total CLI code:            2,877 lines
```

### After Phase 2 (CLI Modularization)
```
cli.py:                      300 lines  (-89.6%)
commands/__init__.py:         50 lines
commands/base.py:            100 lines
commands/analyze.py:         600 lines
commands/extract.py:         150 lines
commands/chunk.py:           100 lines
commands/track_files.py:     200 lines
commands/file_history.py:    120 lines
commands/inject_history.py:  150 lines
commands/compare_branches.py: 180 lines
commands/storyline/*.py:     400 lines (5 files × 80 avg)
---------------------------------------------
Total CLI code:            2,350 lines  (-18.3% overall)
Largest file:                600 lines  (-79.1% from 2,877)
Average command file:        ~150 lines
```

**Benefits**:
- ✅ Main `cli.py` reduced by 90%
- ✅ No single file >600 lines
- ✅ Each command independently maintainable
- ✅ Total lines reduced 18% (removal of redundant code)

### After Phase 1 (Storyline Consolidation)
```
Removed from summarizer.py:
  _parse_storylines():        ~35 lines
  _update_storyline_tracker(): ~45 lines
  _save_storyline_tracker():  ~17 lines

Removed from tracker.py:
  _convert_llm_storylines_to_signals(): ~30 lines

Added to storyline/extraction.py:
  StorylineExtractor class:    ~150 lines (unified, robust)

Net change:                   +23 lines (but removes duplication)
Duplicate code removed:       ~127 lines
```

**Benefits**:
- ✅ Three parsers → one unified parser
- ✅ More robust (JSON fallback support)
- ✅ Easier to maintain (single source of truth)

### After Phase 3 (Storage Consolidation)
```
Added CacheManager:          ~300 lines
Simplified OutputWriter:     -50 lines
Simplified cache logic:      -100 lines (scattered)
---------------------------------------------
Net change:                  +150 lines
```

**Benefits**:
- ✅ Unified cache management
- ✅ Configurable, maintainable
- ✅ Automatic cleanup (new functionality)

### Total Impact
```
Before:  cli.py (2,877 lines) + existing modules
After:   cli.py (300 lines) + commands/ (2,050 lines) + new modules (+450 lines)

Largest file before: 2,877 lines
Largest file after:    600 lines (-79%)

Code duplication eliminated: ~130 lines
New functionality added:     ~200 lines (CacheManager, robust parsing)
```

---

## Conclusion

This optimization proposal addresses critical technical debt in GitView:

1. **CLI Bloat**: 2,877-line `cli.py` will be reduced to 300 lines with clear command separation
2. **Competing Systems**: Storyline duplication will be eliminated, enabling advertised features
3. **Storage Chaos**: Multiple overlapping caches and data stores will be unified
4. **Code Duplication**: Three JSON parsers will become one robust implementation

**Expected Outcomes**:
- ✅ Faster development velocity (parallel work on isolated commands)
- ✅ Better testing (unit tests for command classes)
- ✅ User-facing features actually work (multi-signal storyline detection)
- ✅ Reduced disk usage (40-50% savings)
- ✅ Easier maintenance (clear module boundaries)

**Timeline**: 13 weeks (3 months)
**Effort**: 2-3 engineer-months
**Risk**: Medium (mitigated by phased approach)

**Recommendation**: **Proceed with implementation**, starting with **Phase 2 (CLI modularization)** as the lowest-risk foundation, followed by Phase 1 and Phase 3.

---

**Next Steps**:
1. Review and approve this proposal
2. Create GitHub issues for each phase
3. Set up feature branch for Phase 2
4. Begin CLI modularization with smallest commands
5. Iterate based on feedback

**Questions or Concerns**: Please comment on this proposal document or reach out to the development team.

---

*End of Optimization Proposal*
