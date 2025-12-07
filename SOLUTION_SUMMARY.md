# Solution: Hierarchical Commit Significance Preservation for Large Repos

## Problem Statement

Your LLM strategy for large repositories was losing critical details, resulting in generic timelines like:

```
## Phase 1: Initial Development (Nov 21, 2025)
Project commences with a massive code addition by Carsten Bund, resulting in a
7581.7% increase in LOC from 208 to 15,978.
```

Instead of detailed insights like:

```
## Phase 1: Foundation of Lens Design System (Nov 21, 2025)
- Created mklens_json script with lens geometry calculations for aspheric corneal lenses
- Implemented template-based zone generation supporting 6-zone microlens designs (PR #1)
- Added Gauss-Jordan solver and lens code extraction modules
- Established data parsers for bestel orders and product definitions
```

### Root Causes

1. **Only first 20 commits processed** - Rest ignored (summarizer.py:100)
2. **Aggressive 400-char truncation** - Phase summaries compressed (storyteller.py:155)
3. **Generic timeline prompts** - Only LOC changes and basic metadata sent to LLM

## Solution: Hierarchical Summarization Strategy

I've implemented a new strategy that:

✅ **Processes ALL commits** (not just first 20)
✅ **Groups commits semantically** (features, fixes, refactors, docs)
✅ **Preserves PR context** (titles, descriptions, labels)
✅ **Maintains technical details** (file changes, decisions)
✅ **Avoids truncation** (full context to timeline generator)

### New Architecture

```
Raw Commits
    ↓
Semantic Clustering (ALL commits)
    ↓
Cluster Summarization (1-2 sentences each)
    ↓
Phase Narrative (2-3 paragraphs)
    ↓
Timeline Generation (with full context)
```

## Implemented Components

### 1. `gitview/significance_analyzer.py`

**Purpose:** Group commits into semantic clusters

**Key Classes:**
- `CommitCluster`: Represents a group of related commits
- `SignificanceAnalyzer`: Classifies and clusters commits

**Classification Logic:**
- PR labels (most reliable)
- Commit message keywords
- File patterns
- Refactor flags

**Clustering Strategy:**
- Groups consecutive commits by PR
- Groups by semantic type
- Max 10 commits per cluster

**Example Output:**
```
Cluster 1: feature (5 commits)
  Key: Add verified mklens5 vault lens templates
  PR #9: Verify mklens5 calculations against template
  Files: templates/lens_zone_templates.json, docs/mklens5_verification.md

Cluster 2: refactor (8 commits)
  Key: Refactor mklens_json into reusable modules
  PR #5: Comprehensive module reorganization
  Files: mklens/*.py, mklens_json.py
```

### 2. `gitview/hierarchical_summarizer.py`

**Purpose:** Generate hierarchical summaries preserving details

**Key Classes:**
- `HierarchicalPhaseSummarizer`: Replaces `PhaseSummarizer`

**Process:**
1. Cluster all commits in phase
2. Generate mini-summary per cluster (1-2 sentences)
3. Combine into full phase narrative (2-3 paragraphs)
4. Create timeline context with highlights

**Output Structure:**
```json
{
  "full_summary": "Complete narrative (NOT truncated)",
  "clusters": [
    {
      "type": "feature",
      "summary": "Added comprehensive lens template documentation...",
      "commit_count": 5,
      "key_commit": "Add comprehensive lens template documentation"
    }
  ],
  "timeline_context": {
    "highlights": [
      {
        "type": "feature",
        "count": 5,
        "summary": "...",
        "pr_title": "Add comprehensive lens template documentation"
      }
    ],
    "cluster_count": 3
  }
}
```

### 3. `gitview/hierarchical_storyteller.py`

**Purpose:** Generate timeline using hierarchical context

**Key Classes:**
- `HierarchicalStoryTeller`: Replaces timeline generation in `StoryTeller`

**Differences from Old Approach:**
- Uses full summaries (not 400-char truncated)
- Includes cluster highlights
- Preserves PR titles
- Provides detailed context to LLM

## Usage

### Method 1: Direct Python API

```python
from gitview.extractor import load_commit_records
from gitview.chunker import HistoryChunker
from gitview.hierarchical_summarizer import summarize_phases_hierarchical
from gitview.hierarchical_storyteller import generate_hierarchical_timeline

# Load commits
commits = load_commit_records("commits.json")

# Create phases
chunker = HistoryChunker(strategy="adaptive")
phases = chunker.chunk(commits, min_chunk_size=5, max_chunk_size=30)

# Summarize with hierarchical approach
phases = summarize_phases_hierarchical(
    phases,
    output_dir="output/phases",
    backend='anthropic',
)

# Generate timeline
timeline = generate_hierarchical_timeline(
    phases,
    repo_name="MyRepo",
    backend='anthropic',
)

print(timeline)
```

### Method 2: Comparison Script

```bash
# Extract commits
python -m gitview.cli extract /path/to/repo -o output/commits.json

# Compare old vs new strategies
python compare_strategies.py output/commits.json
```

This will show:
- Commits processed vs ignored
- Clustering analysis
- Summary quality differences
- API call estimates

### Method 3: CLI Integration (After Applying Patch)

See `CLI_INTEGRATION.md` for detailed integration steps.

Once integrated:

```bash
# Use hierarchical strategy
gitview analyze /path/to/repo --summarization-strategy hierarchical

# Or with a simple flag
gitview analyze /path/to/repo --hierarchical
```

## Comparison: Old vs New

### Commit Coverage

| Aspect | Old Strategy | New Strategy |
|--------|-------------|--------------|
| Commits processed | First 20 (~40%) | ALL (100%) |
| Ignored commits | 60% | 0% |
| Processing | Linear | Clustered |

### Detail Preservation

| Aspect | Old Strategy | New Strategy |
|--------|-------------|--------------|
| Summary length | 400 chars (truncated) | Full (2-3 paragraphs) |
| PR context | Lost | Preserved |
| File changes | Lost | Top files per cluster |
| Technical decisions | Lost | Preserved |

### API Calls

| Aspect | Old Strategy | New Strategy |
|--------|-------------|--------------|
| Per phase | 1 call | 3-6 calls |
| For 10 phases | ~10 calls | ~40-60 calls |

**Trade-off:** More API calls, but MUCH better quality.

For repositories where quality matters, this is essential.

## Expected Improvements

### Before (Old Strategy):
```markdown
## Phase 5: Database Development (Nov 22, 2025)
Carsten Bund and Claude drive extensive documentation work and bug fixing,
causing a 22.1% increase in LOC.
```

**Lost Details:**
- What database? What schema?
- Which bugs were fixed?
- What documentation was added?
- Why was this work done?

### After (New Strategy):
```markdown
## Phase 5: Database Schema and Conversion Tools (Nov 22, 2025)

**Feature Development (8 commits, PR #15):**
- Designed database schema for lens orders with temporal tracking
- Created automated migration from legacy bestel.bes format
- Added validation layer for data integrity checks

**Bug Fixes (3 commits, PR #16):**
- Fixed mklens test script path resolution issues (mklens/template_interpreter.py:14-18)
- Corrected data-driven test scenario generation edge cases

**Documentation (5 commits):**
- Added comprehensive TESTING.md covering data-driven test framework
- Documented product code mapping for all lens templates
- Created troubleshooting guide for CI/CD integration
```

**Preserved Details:**
- Specific features: database schema, migration tools, validation
- Specific bugs: path resolution, test scenarios
- Specific docs: TESTING.md, product code mapping
- Technical context: file paths, PR numbers

## Performance Optimization Options

If API calls are a concern:

### Option 1: Hybrid Strategy
```python
if len(phase.commits) > 20:
    # Use hierarchical for large phases
    use_hierarchical = True
else:
    # Use simple for small phases
    use_hierarchical = False
```

### Option 2: Adjust Cluster Size
```python
# In SignificanceAnalyzer
max_cluster_size = 15  # Default: 10
# Fewer clusters = fewer API calls
```

### Option 3: Skip Minor Clusters
```python
# Only summarize clusters with 3+ commits
if len(cluster.commits) >= 3:
    cluster.summary = summarizer._summarize_cluster(cluster)
else:
    # Use key commit message directly
    cluster.summary = cluster.key_commit.commit_subject
```

## Testing & Validation

### 1. Run Comparison
```bash
python compare_strategies.py output/commits.json
```

### 2. Review Output
Check `output/comparison/new_strategy_result.json` for:
- Meaningful cluster groupings
- Preserved PR titles
- Technical context
- File changes

### 3. Validate Timeline Quality
Compare timelines:
```bash
# Old strategy
gitview analyze /path/to/repo -o output/old

# New strategy (after integration)
gitview analyze /path/to/repo -o output/new --hierarchical

# Compare
diff output/old/timeline.md output/new/timeline.md
```

## Files Created

1. **`gitview/significance_analyzer.py`** - Semantic clustering
2. **`gitview/hierarchical_summarizer.py`** - Hierarchical phase summarization
3. **`gitview/hierarchical_storyteller.py`** - Timeline with preserved details
4. **`compare_strategies.py`** - Comparison tool
5. **`examples/hierarchical_example.py`** - Usage example
6. **`docs/HIERARCHICAL_STRATEGY.md`** - Complete documentation
7. **`CLI_INTEGRATION.md`** - Integration guide
8. **`SOLUTION_SUMMARY.md`** - This file

## Next Steps

1. **Test the comparison script:**
   ```bash
   python compare_strategies.py output/commits.json
   ```

2. **Review the results:**
   - Check if clusters are meaningful
   - Verify PR context is preserved
   - Confirm technical details are retained

3. **Integrate into CLI** (optional):
   - Follow `CLI_INTEGRATION.md`
   - Add `--hierarchical` flag
   - Test with real repositories

4. **Optimize if needed:**
   - Adjust cluster sizes
   - Implement hybrid strategy
   - Fine-tune API call frequency

## Conclusion

The hierarchical strategy solves your large repository problem by:

✅ Processing ALL commits (not just first 20)
✅ Organizing commits semantically (features, fixes, etc.)
✅ Preserving PR context and technical details
✅ Avoiding aggressive truncation
✅ Generating meaningful, detailed timelines

**The trade-off:** More API calls vs. significantly better quality.

For large repositories where understanding the evolution matters, this is the right approach.
