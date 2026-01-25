# Storyline Architecture Implementation Plan

## Current State (Quick Wins - Implemented)

The following has been implemented as quick wins:
- Phase summaries request `## Storylines` section with `[STATUS:category]` format
- Basic storyline parsing from LLM output via regex
- Simple storyline tracker that accumulates across phases
- Active storylines injected into phase context (up to 8 storylines)
- Storylines passed to storyteller for global narratives
- Context window increased: 6 phases (was 3), 400 chars (was 200)
- Story summaries: 600 chars (was 400)

**Limitations of current approach:**
- Relies entirely on LLM to identify storylines (unreliable)
- Simple regex parsing prone to extraction failures
- No confidence scoring or signal combination
- No formal state machine for storyline lifecycle
- Limited cross-phase theme detection

---

## Target Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    STORYLINE DETECTION LAYER                     │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────┤
│ PR Labels   │ PR Titles   │ Commit Msg  │ File        │ LLM     │
│ (0.9 conf)  │ (0.8 conf)  │ (0.6 conf)  │ Clusters    │ Extract │
│             │             │             │ (0.7 conf)  │ (0.5)   │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┴────┬────┘
       │             │             │             │           │
       └─────────────┴─────────────┴─────────────┴───────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │    SIGNAL MERGER         │
                    │  (Semantic Similarity)   │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   STORYLINE TRACKER      │
                    │  - Match to existing     │
                    │  - Create new            │
                    │  - State transitions     │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │  STORYLINE DATABASE      │
                    │  (JSON persistence)      │
                    └────────────┬─────────────┘
                                 │
                                 ▼
              ┌──────────────────┴──────────────────┐
              │                                     │
              ▼                                     ▼
┌─────────────────────────┐         ┌─────────────────────────┐
│   Phase Summarizer      │         │   Storyline Reporter    │
│   (injects context)     │         │   (storyline-centric)   │
└─────────────────────────┘         └─────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Core Data Model (Foundation)

**Create new module: `gitview/storyline/`**

```
gitview/storyline/
    __init__.py
    models.py           # Core data classes
    state_machine.py    # Storyline state transitions
    detector.py         # Multi-signal storyline detection
    tracker.py          # Cross-phase tracking and persistence
    reporter.py         # Storyline-centric output generation
```

**Key data structures:**

```python
class StorylineStatus(Enum):
    EMERGING = "emerging"      # First signal, not confirmed
    ACTIVE = "active"          # Confirmed ongoing
    PROGRESSING = "progressing" # Active with recent updates
    STALLED = "stalled"        # No updates in N phases
    COMPLETED = "completed"    # Explicitly resolved
    ABANDONED = "abandoned"    # No activity, never completed

class StorylineCategory(Enum):
    FEATURE = "feature"
    REFACTOR = "refactor"
    BUGFIX = "bugfix"
    TECH_DEBT = "tech_debt"
    INFRASTRUCTURE = "infrastructure"
    DOCUMENTATION = "documentation"
    MIGRATION = "migration"
    PERFORMANCE = "performance"
    SECURITY = "security"

@dataclass
class StorylineSignal:
    source: str              # 'pr_label', 'commit_pattern', 'llm_extraction'
    confidence: float        # 0.0 to 1.0
    phase_number: int
    commit_hashes: List[str]
    data: Dict[str, Any]

@dataclass
class Storyline:
    id: str
    title: str
    category: StorylineCategory
    status: StorylineStatus
    confidence: float

    first_phase: int
    last_phase: int
    phases_involved: List[int]

    description: str
    current_summary: str
    updates: List[StorylineUpdate]
    signals: List[StorylineSignal]

    related_storylines: List[str]
    key_files: Set[str]
    key_authors: Set[str]
    pr_numbers: List[int]
```

**Tasks:**
1. Create `gitview/storyline/` package structure
2. Implement `models.py` with all data classes
3. Implement `state_machine.py` with transition logic
4. Write unit tests for models and state machine

---

### Phase 2: Multi-Signal Detection

**Detector hierarchy:**

| Detector | Confidence | Source |
|----------|------------|--------|
| PRLabelDetector | 0.9 | GitHub PR labels |
| PRTitlePatternDetector | 0.8 | PR title keywords |
| FileClusterDetector | 0.7 | Files changing together |
| CommitMessagePatternDetector | 0.6 | Commit message patterns |
| LLMExtractionDetector | 0.5 | LLM analysis (fallback) |

**PRLabelDetector mapping:**
```python
LABEL_MAPPINGS = {
    'feature': StorylineCategory.FEATURE,
    'enhancement': StorylineCategory.FEATURE,
    'bug': StorylineCategory.BUGFIX,
    'fix': StorylineCategory.BUGFIX,
    'refactor': StorylineCategory.REFACTOR,
    'tech-debt': StorylineCategory.TECH_DEBT,
    'documentation': StorylineCategory.DOCUMENTATION,
    'infrastructure': StorylineCategory.INFRASTRUCTURE,
    'performance': StorylineCategory.PERFORMANCE,
    'security': StorylineCategory.SECURITY,
}
```

**Signal merging:**
- Group signals by semantic similarity (title, files, commits)
- Calculate combined confidence using weighted average
- Merge into candidate storylines

**Tasks:**
1. Implement `PRLabelDetector`
2. Implement `PRTitlePatternDetector`
3. Implement `CommitMessagePatternDetector`
4. Implement `FileClusterDetector`
5. Update `LLMExtractionDetector` with structured prompts
6. Implement signal merging in `StorylineDetector`
7. Write integration tests for detectors

---

### Phase 3: State Machine & Tracking

**State transitions:**
```
EMERGING ──▶ ACTIVE ──▶ PROGRESSING ──▶ COMPLETED
    │           │           │
    │           ▼           │
    │       STALLED ◀───────┘
    │           │
    ▼           ▼
ABANDONED ◀─────┘
```

**Transition rules:**
- `EMERGING → ACTIVE`: confidence > 0.6, multiple signals
- `ACTIVE → STALLED`: no updates for 3 phases
- `STALLED → ABANDONED`: no updates for 6 phases
- `* → COMPLETED`: explicit completion signal

**Tracker responsibilities:**
1. Match new signals to existing storylines
2. Create new storylines from unmatched signals
3. Update existing storylines with new data
4. Evaluate and apply state transitions
5. Persist database after each phase

**Matching criteria (priority order):**
1. Exact title match (normalized)
2. PR number match
3. High file overlap (>60%)
4. Semantic similarity (keyword overlap)

**Tasks:**
1. Implement `StorylineTracker` core logic
2. Implement signal-to-storyline matching
3. Implement JSON persistence layer
4. Implement incremental analysis support
5. Add database version migration support
6. Write tests for persistence and incremental updates

---

### Phase 4: Enhanced Prompting

**Improved phase summary prompt:**
```
Analyze this phase and provide a structured response.

**Active Storylines to Track:**
- [ONGOING] OAuth2 Implementation (feature): Token refresh work
- [STALLED] Mobile Support (feature): Waiting on dependencies

---

Provide your analysis in the following EXACT format:

## Summary
[2-3 paragraph narrative summary]

## Storyline Updates
```json
{
  "storylines": [
    {
      "title": "Exact storyline title",
      "status": "new|continued|completed|stalled",
      "category": "feature|refactor|bugfix|...",
      "confidence": 0.0-1.0,
      "summary": "1-2 sentence update",
      "key_commits": ["abc123"],
      "key_files": ["path/to/file.py"]
    }
  ]
}
```

## Cross-Phase Themes
- Theme 1: description
- Theme 2: description
```

**Robust response parsing:**
1. Look for JSON code block
2. Look for raw JSON object
3. Fall back to regex parsing

**Tasks:**
1. Update `summarizer.py` prompts with structured format
2. Implement `StorylineResponseParser` with fallbacks
3. Add JSON schema validation
4. Write parser tests with edge cases

---

### Phase 5: Storyline-Centric Reports

**New report types:**

1. **Storyline Index** - Table of all storylines with status
2. **Storyline Narratives** - Detailed story per storyline
3. **Cross-Phase Themes** - Patterns spanning storylines
4. **Storyline Timeline** - Visual arc representation

**Report generator:**
```python
class StorylineReporter:
    def generate_storyline_report(self, storyline_id: str) -> str:
        """Generate narrative for single storyline."""

    def generate_storyline_index(self) -> str:
        """Generate index of all storylines."""

    def generate_cross_cutting_themes(self) -> str:
        """Identify themes spanning storylines."""

    def generate_storyline_timeline(self) -> str:
        """ASCII timeline of storyline arcs."""
```

**Tasks:**
1. Implement `StorylineReporter`
2. Update `writer.py` for storyline outputs
3. Add storyline section to markdown reports
4. Create ASCII timeline visualization

---

### Phase 6: CLI & Integration

**New CLI options:**
```bash
gitview analyze --storyline-mode auto|enhanced|disabled
gitview analyze --storyline-report
gitview storyline show <storyline-id>
gitview storyline list [--status active|completed|all]
gitview storyline export [--format json|csv]
```

**GitHub enricher integration:**
- Extract storyline signals during enrichment phase
- Pass signals to storyline tracker
- Group multi-commit PRs as single storyline unit

**Tasks:**
1. Update CLI with storyline options
2. Add `gitview storyline` command group
3. Integrate with GitHub enricher
4. Update help documentation

---

### Phase 7: Incremental Analysis

**Support for incremental runs:**
- Load existing storyline database
- Process only new phases
- Update storyline states based on inactivity
- Handle database merging (multi-branch)

**Tasks:**
1. Implement `process_incremental()` in tracker
2. Implement `merge_databases()` for branch merging
3. Add database backup before updates
4. Test incremental scenarios

---

## File Changes Summary

| File | Changes |
|------|---------|
| `gitview/storyline/__init__.py` | New - package init |
| `gitview/storyline/models.py` | New - data classes |
| `gitview/storyline/state_machine.py` | New - state transitions |
| `gitview/storyline/detector.py` | New - multi-signal detection |
| `gitview/storyline/tracker.py` | New - cross-phase tracking |
| `gitview/storyline/reporter.py` | New - storyline reports |
| `gitview/summarizer.py` | Update prompts, integrate tracker |
| `gitview/storyteller.py` | Consume richer storyline data |
| `gitview/extractor.py` | Add `get_pr_number()` method |
| `gitview/github_enricher.py` | Extract storyline signals |
| `gitview/writer.py` | Add storyline output sections |
| `gitview/cli.py` | Add storyline CLI options |

---

## Configuration

```yaml
# Future: gitview.yaml
storyline:
  detection:
    confidence_threshold: 0.6
    min_signals_for_confirmation: 2
    file_cluster_overlap_threshold: 0.6

  state_machine:
    stall_threshold_phases: 3
    abandon_threshold_phases: 6

  persistence:
    path: "output/storylines.json"
    backup_count: 3

  reporting:
    include_completed: true
    max_storylines_in_summary: 20
```

---

## Database Schema

```json
{
  "version": "1.0.0",
  "metadata": {
    "last_phase_analyzed": 15,
    "last_commit_hash": "abc123",
    "last_updated": "2026-01-25T10:30:00Z",
    "total_storylines": 42
  },
  "storylines": [
    {
      "id": "sl_abc123",
      "title": "OAuth2 Authentication",
      "category": "feature",
      "status": "completed",
      "confidence": 0.95,
      "first_phase": 2,
      "last_phase": 7,
      "phases_involved": [2, 4, 5, 7],
      "updates": [...],
      "signals": [...],
      "key_files": ["auth/oauth2.py"],
      "pr_numbers": [42, 58, 71]
    }
  ],
  "indexes": {
    "title_index": {"oauth2 authentication": "sl_abc123"},
    "pr_index": {"42": ["sl_abc123"]}
  }
}
```

---

## Success Metrics

1. **Detection reliability**: >80% of PRs with labels correctly mapped to storylines
2. **Continuity accuracy**: >90% of multi-phase features tracked as single storyline
3. **State accuracy**: <5% false positive completions/abandonments
4. **Narrative quality**: Storyline-centric narratives rated more useful than phase-centric

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| LLM extraction unreliable | Multi-signal approach with data-driven detectors as primary |
| Performance on large repos | Batch processing, incremental analysis, caching |
| Database corruption | Versioned schema, backups before updates |
| Over-fragmentation | Aggressive signal merging, configurable thresholds |
| Under-detection | Multiple detector types, low initial thresholds |
