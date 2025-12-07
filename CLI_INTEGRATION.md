# CLI Integration for Hierarchical Strategy

## How to Add Hierarchical Strategy to CLI

### Step 1: Add Import Statements

In `gitview/cli.py`, add these imports at the top:

```python
from .hierarchical_summarizer import HierarchicalPhaseSummarizer
from .hierarchical_storyteller import HierarchicalStoryTeller
```

### Step 2: Add CLI Option

Find the `@click.option` decorators before the `analyze` function (around line 599).

Add this new option:

```python
@click.option('--summarization-strategy', type=click.Choice(['simple', 'hierarchical']),
              default='simple',
              help="Summarization strategy: 'simple' (faster, less detailed) or "
                   "'hierarchical' (slower, preserves more details for large repos)")
```

### Step 3: Add Parameter to analyze() Function

Update the `analyze` function signature (line 654) to include the new parameter:

```python
def analyze(repo, output, strategy, chunk_size, max_commits, branch, list_branches,
           branches, all_branches, exclude_branches, backend, model, api_key, ollama_url,
           repo_name, skip_llm, incremental, since_commit, since_date, keep_clone,
           todo, critical, directives, github_token, summarization_strategy):  # <-- ADD THIS
```

### Step 4: Update _analyze_single_branch() Call

Find where `_analyze_single_branch` is called (around line 815) and add the new parameter:

```python
_analyze_single_branch(
    str(repo_path),
    branch_to_use,
    output,
    repo_name,
    strategy,
    chunk_size,
    max_commits,
    backend,
    model,
    api_key,
    ollama_url,
    skip_llm,
    incremental,
    since_commit,
    since_date,
    todo_content,
    critical,
    directives,
    github_token,
    summarization_strategy,  # <-- ADD THIS
)
```

### Step 5: Update _analyze_single_branch() Definition

Find the `_analyze_single_branch` function definition (around line 284) and add the parameter:

```python
def _analyze_single_branch(
    repo_path: str,
    branch: str,
    output: str,
    repo_name: str,
    strategy: str,
    chunk_size: int,
    max_commits: Optional[int],
    backend: Optional[str],
    model: Optional[str],
    api_key: Optional[str],
    ollama_url: str,
    skip_llm: bool,
    incremental: bool,
    since_commit: Optional[str],
    since_date: Optional[str],
    todo_content: Optional[str],
    critical_mode: bool,
    directives: Optional[str],
    github_token: Optional[str],
    summarization_strategy: str = 'simple',  # <-- ADD THIS
):
```

### Step 6: Update Summarization Logic

Find where `PhaseSummarizer` is instantiated (around line 497). Replace with:

```python
# Step 3: Summarize phases with LLM
console.print("[bold]Step 3: Summarizing phases with LLM...[/bold]")

if summarization_strategy == 'hierarchical':
    console.print("[cyan]Using hierarchical summarization strategy[/cyan]")
    console.print("[yellow]Note: This makes more API calls but preserves more details[/yellow]\n")

    summarizer = HierarchicalPhaseSummarizer(
        backend=backend,
        model=model,
        api_key=api_key,
        ollama_url=ollama_url,
    )

    # Hierarchical summarizer returns phases with metadata
    phases = summarizer.summarize_all_phases(phases, output_dir=str(phases_dir))

else:
    console.print("[cyan]Using simple summarization strategy[/cyan]\n")

    summarizer = PhaseSummarizer(
        backend=backend,
        model=model,
        api_key=api_key,
        ollama_url=ollama_url,
        todo_content=todo_content,
        critical_mode=critical_mode,
        directives=directives
    )

    # Identify phases that need summarization (no summary)
    phases_to_summarize = [p for p in phases if p.summary is None]

    if phases_to_summarize:
        console.print(f"Summarizing {len(phases_to_summarize)} phases...")
        phases = summarizer.summarize_all_phases(phases, output_dir=str(phases_dir))
    else:
        console.print("[green]All phases already have summaries (using cached)[/green]")
```

### Step 7: Update Storytelling Logic

Find where `StoryTeller` is instantiated (around line 515). Replace with:

```python
# Step 4: Generate global story
if not skip_llm:
    console.print("[bold]Step 4: Generating global repository story...[/bold]")

    if summarization_strategy == 'hierarchical':
        # Use hierarchical storyteller
        storyteller = HierarchicalStoryTeller(
            backend=backend,
            model=model,
            api_key=api_key,
            ollama_url=ollama_url,
        )

        # Generate timeline with preserved details
        timeline = storyteller.generate_timeline(phases, repo_name=repo_name)

        # For hierarchical strategy, we just use timeline as the main story
        stories = {
            'timeline': timeline,
            'executive_summary': 'See timeline for detailed evolution',
            'technical_evolution': 'See timeline for technical details',
            'deletion_story': '',
            'full_narrative': timeline,
        }

    else:
        # Use traditional storyteller
        storyteller = StoryTeller(
            backend=backend,
            model=model,
            api_key=api_key,
            ollama_url=ollama_url,
            todo_content=todo_content,
            critical_mode=critical_mode,
            directives=directives
        )

        stories = storyteller.generate_global_story(phases, repo_name=repo_name)
```

## Usage Examples

After integrating, users can use:

```bash
# Simple strategy (current behavior, default)
gitview analyze /path/to/repo

# Hierarchical strategy (for large repos)
gitview analyze /path/to/repo --summarization-strategy hierarchical

# Hierarchical with specific backend
gitview analyze /path/to/repo --summarization-strategy hierarchical --backend anthropic
```

## Testing the Integration

1. **Small repo test:**
   ```bash
   gitview analyze /path/to/small-repo --summarization-strategy hierarchical
   ```

2. **Large repo test:**
   ```bash
   gitview analyze /path/to/large-repo --summarization-strategy hierarchical
   ```

3. **Compare outputs:**
   ```bash
   # Generate with simple strategy
   gitview analyze /path/to/repo -o output/simple

   # Generate with hierarchical strategy
   gitview analyze /path/to/repo -o output/hierarchical --summarization-strategy hierarchical

   # Compare timelines
   diff output/simple/timeline.md output/hierarchical/timeline.md
   ```

## Alternative: Simpler Integration (Just Add Flag)

If you want a minimal change, just add a `--hierarchical` flag:

```python
@click.option('--hierarchical', is_flag=True,
              help="Use hierarchical summarization (better for large repos, more API calls)")
```

Then use:
```bash
gitview analyze /path/to/repo --hierarchical
```

## Backward Compatibility

The default is `'simple'`, so existing users won't see any changes unless they explicitly opt-in to `--summarization-strategy hierarchical`.

## Performance Considerations

When using hierarchical strategy, warn users:

```python
if summarization_strategy == 'hierarchical':
    total_commits = sum(len(p.commits) for p in phases)
    estimated_clusters = total_commits // 5  # Rough estimate
    estimated_api_calls = estimated_clusters + len(phases)

    console.print(f"[yellow]Hierarchical strategy will make approximately {estimated_api_calls} API calls[/yellow]")
    console.print(f"[yellow](vs ~{len(phases)} calls with simple strategy)[/yellow]\n")
```

This helps users understand the cost trade-off.
