# Hierarchical Summarization Strategy

## Problem with Original Strategy

The original LLM strategy for large repositories had critical information loss:

### Issue 1: Only First 20 Commits Processed
```python
for commit in phase.commits[:20]:  # Rest are ignored!
```
For phases with 50+ commits, **60% of commits were discarded**.

### Issue 2: Aggressive Truncation
```python
'summary': self._truncate_content(phase.summary, 400)
```
Rich 3-5 paragraph summaries compressed to **400 characters** before timeline generation.

### Issue 3: Timeline Sees Only Generic Metadata
```python
Phase 1 (2025-11-21 to 2025-11-21)
- LOC D: +9,616 (+581.7%)
- Summary: Project commences with a massive code addition...
```

**Lost Details:**
- Specific commit messages
- PR titles and descriptions
- Technical context
- File changes
- WHY changes were made

## New Hierarchical Strategy

### Overview

Instead of:
1. Take first 20 commits
2. Create one summary
3. Truncate to 400 chars

We:
1. **Cluster ALL commits semantically**
2. **Create mini-summaries for each cluster**
3. **Combine into hierarchical phase summary**
4. **Preserve key details separately** (PR titles, file changes)

### Architecture

```
Commits → Semantic Clustering → Cluster Summaries → Phase Narrative
                ↓                      ↓                    ↓
         All commits          Per-cluster context    Full timeline
         grouped by:          - PR titles            context
         - Feature            - File changes         (NOT truncated)
         - Bugfix             - Technical decisions
         - Refactor
         - Docs
         - Infrastructure
```

### Key Components

#### 1. SignificanceAnalyzer (`significance_analyzer.py`)

Groups commits into semantic clusters:

```python
class CommitCluster:
    cluster_type: str  # 'feature', 'bugfix', 'refactor', 'docs', 'infrastructure'
    commits: List[CommitRecord]
    key_commit: CommitRecord  # Most significant commit
    summary: Optional[str] = None
```

**Classification Logic:**
- Checks PR labels first (most reliable)
- Falls back to commit message keywords
- Analyzes file patterns (*.md files → docs)

**Clustering Strategy:**
- Groups consecutive commits by PR
- Groups by semantic type
- Limits cluster size to 10 commits max

#### 2. HierarchicalPhaseSummarizer (`hierarchical_summarizer.py`)

Generates hierarchical summaries:

```python
result = summarizer.summarize_phase(phase)
# Returns:
# {
#   'full_summary': Complete narrative (NOT truncated),
#   'clusters': List of cluster summaries,
#   'timeline_context': {
#     'highlights': [...],  # Grouped by type
#     'cluster_count': N,
#   },
#   'cluster_details': [...]  # Full cluster info
# }
```

**Process:**
1. Cluster ALL commits (not just first 20)
2. Generate mini-summary per cluster (1-2 sentences)
3. Combine into full phase narrative (2-3 paragraphs)
4. Create timeline context with highlights

#### 3. HierarchicalStoryTeller (`hierarchical_storyteller.py`)

Generates timeline using hierarchical context:

```python
timeline = generate_hierarchical_timeline(phases, repo_name="MyRepo")
```

**Timeline Includes:**
- Descriptive phase headings (not just "Phase 1")
- Detailed highlights with PR titles
- Technical context preserved
- File changes noted

## Usage

### Method 1: Full Pipeline

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
    backend='anthropic',  # or 'openai', 'ollama'
)

# Generate timeline
timeline = generate_hierarchical_timeline(
    phases,
    repo_name="MyRepo",
    backend='anthropic',
)

print(timeline)
```

### Method 2: Compare Strategies

```bash
# Extract commits first
python -m gitview.cli extract /path/to/repo -o output/commits.json

# Compare old vs new
python compare_strategies.py output/commits.json
```

This will show:
- How many commits were ignored vs processed
- Clustering analysis
- Summary quality difference
- Timeline context comparison

## Expected Improvements

### Commit Coverage
- **Old:** First 20 commits only (~40% for large phases)
- **New:** ALL commits (100%)

### Semantic Organization
- **Old:** Chronological dump
- **New:** Grouped by purpose (features, fixes, refactors)

### Context Preservation
- **Old:** 400 char truncation
- **New:** Full summaries + structured highlights

### Detail Retention
- **Old:** Generic "added features, fixed bugs"
- **New:** Specific PR titles, file changes, technical decisions

### Example Output Comparison

#### Old Strategy:
```
## Phase 1: Initial Development (Nov 21, 2025)
Project commences with a massive code addition by Carsten Bund, resulting in a
7581.7% increase in LOC from 208 to 15,978.
```

#### New Strategy:
```
## Phase 1: Foundation of Lens Design System (Nov 21, 2025)

- Created mklens_json script with lens geometry calculations for aspheric corneal lenses
- Implemented template-based zone generation supporting 6-zone microlens designs (PR #1: "Align JFL writer with legacy format")
- Added Gauss-Jordan solver (gauss_jordan.py) and lens code extraction modules
- Established data parsers for bestel orders and product definitions
- Added comprehensive documentation for mklens3.c reference implementation
```

## Performance Considerations

### API Calls
- **Old:** 1 call per phase
- **New:** N calls per phase (where N = number of clusters + 1 for narrative)

For a phase with 30 commits:
- Old: 1 API call
- New: ~4-6 API calls (3-5 clusters + 1 narrative)

### Cost vs. Quality Trade-off
- More API calls = higher cost
- BUT: Significantly better output quality
- For large repos where the report matters, this is worthwhile

### Optimization Options

1. **Adjust cluster size:**
   ```python
   # In SignificanceAnalyzer._cluster_commits()
   elif len(current_cluster) >= 15:  # Larger clusters = fewer API calls
   ```

2. **Skip cluster summarization for small phases:**
   ```python
   if len(phase.commits) <= 10:
       # Use old strategy
   else:
       # Use hierarchical strategy
   ```

3. **Batch cluster summarization:**
   ```python
   # Summarize multiple small clusters in one call
   ```

## Migration Path

### Option 1: Keep Both Strategies
Add a flag to CLI:
```bash
gitview story --strategy hierarchical  # New approach
gitview story --strategy simple        # Old approach
```

### Option 2: Automatic Selection
```python
if len(phase.commits) > 20:
    # Use hierarchical for complex phases
else:
    # Use simple for small phases
```

### Option 3: Hybrid Approach
Use hierarchical for phase summarization, but keep old timeline generation.

## Testing

Run comparison on your repository:

```bash
# Extract commits
python -m gitview.cli extract /path/to/large-repo -o commits.json

# Compare strategies
python compare_strategies.py commits.json

# Review output/comparison/new_strategy_result.json
```

Look for:
- Are clusters meaningful?
- Are summaries more detailed?
- Are PR titles preserved?
- Do you see technical context that was missing before?

## Future Enhancements

1. **Smart clustering:** Use embeddings to group semantically similar commits
2. **Cross-phase analysis:** Identify themes across multiple phases
3. **Automatic section headings:** Generate descriptive phase names automatically
4. **Importance scoring:** Weight commits by significance for adaptive summarization
5. **Interactive refinement:** Allow users to adjust clustering thresholds

## Conclusion

The hierarchical strategy solves the information loss problem for large repositories by:

1. ✅ Processing ALL commits (not just first 20)
2. ✅ Organizing commits semantically
3. ✅ Preserving PR context and technical details
4. ✅ Avoiding aggressive truncation
5. ✅ Generating meaningful, detailed timelines

The trade-off is more API calls, but for repositories where quality matters, this is essential.
