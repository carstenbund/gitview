"""Command-line interface for GitView."""

import os
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from . import __version__
from .extractor import GitHistoryExtractor
from .chunker import HistoryChunker
from .summarizer import PhaseSummarizer
from .storyteller import StoryTeller
from .hierarchical_summarizer import HierarchicalPhaseSummarizer
from .hierarchical_storyteller import HierarchicalStoryTeller
from .writer import OutputWriter
from .remote import RemoteRepoHandler
from .branches import BranchManager, parse_branch_spec
from .index_writer import IndexWriter
from .github_enricher import GitHubEnricher, enrich_commits_with_github
from .file_tracker import FileHistoryTracker

console = Console()


@click.group()
@click.version_option(version=__version__)
def cli():
    """GitView - Git history analyzer with LLM-powered narrative generation.

    \b
    Extract, chunk, and use LLMs to generate compelling narratives from your
    git repository's history.

    \b
    Quick Start:
      # Using Anthropic Claude (default)
      export ANTHROPIC_API_KEY="your-key"
      gitview analyze

      # Using OpenAI GPT
      export OPENAI_API_KEY="your-key"
      gitview analyze --backend openai

      # Using local Ollama (no API key needed)
      gitview analyze --backend ollama --model llama3

    \b
    See 'gitview analyze --help' for detailed LLM configuration options.
    """
    pass


ANALYZE_HELP = """Analyze git repository and generate narrative history.

\b
This command runs the full pipeline:
  1. Extract git history with detailed metadata
  2. Enrich with GitHub PR/review context (optional, with --github-token)
  3. Chunk commits into meaningful phases/epochs
  4. Summarize each phase using LLM
  5. Generate global narrative stories
  6. Write markdown reports and JSON data

\b
REPOSITORY SOURCES:

GitView supports both local and remote repositories:

  Local repository:
    gitview analyze --repo /path/to/repo

  GitHub shortcut (automatically clones):
    gitview analyze --repo org/repo

  Full HTTPS URL:
    gitview analyze --repo https://github.com/org/repo.git

  SSH URL:
    gitview analyze --repo git@github.com:org/repo.git

For remote repositories:
  - Automatically clones to cached directory (~/.gitview/cache/repos/{host}/{org}/{repo})
  - Reuses cached clones for 24 hours and fetches updates to avoid full re-downloads
  - Cached clones are preserved for reuse; use --keep-clone to skip cleanup of additional temp copies
  - Outputs to ~/Documents/gitview/github.com/org/repo/ by default
  - Warns if repository is very large (>100MB)

\b
LLM BACKEND CONFIGURATION:

GitView supports three LLM backends:

\b
1. OpenAI GPT (default for cost efficiency, requires API key):
   export OPENAI_API_KEY="your-key"
   gitview analyze

   Or: gitview analyze --backend openai --api-key "your-key"

   Default model: gpt-4o-mini (most cost-effective, ~$0.15-0.60 per 1000 commits)
   Other models: gpt-4o, gpt-4-turbo-preview

\b
2. Anthropic Claude (premium quality, requires API key):
   export ANTHROPIC_API_KEY="your-key"
   gitview analyze --backend anthropic

   Default model: claude-sonnet-4-5-20250929 (~$2-10 per 1000 commits)
   Budget option: claude-haiku-3-5-20241022 (~$0.20-1.00 per 1000 commits)

\b
3. Ollama (local, FREE, no API key needed):
   # Start Ollama server first: ollama serve
   # Pull a model: ollama pull llama3
   gitview analyze --backend ollama --model llama3

   Popular models: llama3, mistral, codellama, mixtral
   Default URL: http://localhost:11434

\b
Backend auto-detection (cost-optimized priority):
  If no --backend is specified, GitView checks environment variables:
  - If OPENAI_API_KEY is set -> uses OpenAI (most cost-effective)
  - If ANTHROPIC_API_KEY is set -> uses Anthropic (premium quality)
  - Otherwise -> uses Ollama (local, free)

\b
EXAMPLES:

  # Analyze current directory with Claude (auto-detected)
  export ANTHROPIC_API_KEY="sk-ant-..."
  gitview analyze

  # Use OpenAI GPT-4 with custom model
  gitview analyze --backend openai --model gpt-4-turbo-preview

  # Use local Ollama (no API costs!)
  gitview analyze --backend ollama --model llama3

  # Analyze specific repository
  gitview analyze --repo /path/to/repo --output ./analysis

  # Quick analysis without LLM (just extract and chunk)
  gitview analyze --skip-llm

  # Hierarchical summarization (preserves more details for large repos)
  gitview analyze --summarization-strategy hierarchical

  # Shortcut flag for hierarchical mode
  gitview analyze --hierarchical

  # Analyze last 100 commits only
  gitview analyze --max-commits 100

  # Adaptive chunking (default, splits on significant changes)
  gitview analyze --strategy adaptive

  # Fixed-size chunks (50 commits per phase)
  gitview analyze --strategy fixed --chunk-size 50

  # Use custom Ollama server
  gitview analyze --backend ollama --ollama-url http://192.168.1.100:11434

\b
SMART CACHING & COST OPTIMIZATION:

  GitView now includes intelligent caching to minimize LLM costs:

  1. AUTO-INCREMENTAL MODE (NEW):
     When you re-run analysis on a repo, GitView automatically detects cached
     data and only analyzes NEW commits. This is perfect for daily/weekly usage
     across multiple projects.

     First run:  gitview analyze              # Analyzes all history (~$0.50-2.00)
     Second run: gitview analyze              # Auto-incremental, only new commits (~$0.05-0.15)

  2. STORY CACHING:
     Generated narratives are cached. If phase summaries haven't changed,
     story generation is skipped entirely (saves ~$0.10-0.50 per run).

  3. COST ESTIMATION:
     Before starting analysis, GitView shows estimated cost and suggests
     cheaper alternatives if cost is high.

  For managers analyzing multiple projects on an ongoing basis, these features
  dramatically reduce costs by reusing previous LLM work.

  # Initial full analysis
  gitview analyze --output reports/myproject

  # Later: incremental update (only analyzes new commits)
  gitview analyze --output reports/myproject --incremental

  # Manual incremental from specific commit
  gitview analyze --since-commit abc123def

  # Incremental from date
  gitview analyze --since-date 2025-11-01

  How it works:
  - Detects previous analysis in output directory
  - Extracts only commits since last run
  - Reuses existing phase summaries (no LLM calls!)
  - Only summarizes new/modified phases
  - Updates JSON with new metadata

  Benefits:
  - Massive API cost savings for ongoing monitoring
  - Much faster analysis (only processes new commits)
  - Perfect for CI/CD integration or periodic reviews

\b
CRITICAL EXAMINATION MODE (Project Leadership):

  For project leads who need objective assessment rather than celebratory
  narratives. Critical mode focuses on gaps, technical debt, and alignment
  with project goals.

  # Basic critical mode
  gitview analyze --critical

  # Critical mode with project goals/TODO file
  gitview analyze --critical --todo PROJECT_GOALS.md

  # Critical mode with custom analysis directives
  gitview analyze --critical --directives "Focus on security vulnerabilities"

  # Combined: goals + directives
  gitview analyze --critical --todo ROADMAP.md \\
    --directives "Emphasize testing gaps and code quality issues"

  What changes in critical mode:
  - Removes flowery, achievement-focused language
  - Evaluates progress against stated objectives
  - Identifies incomplete features and technical debt
  - Questions architectural decisions objectively
  - Highlights gaps, delays, and misalignments
  - Focuses on what's missing or needs improvement

  Use cases:
  - Project reviews and technical audits
  - Goal alignment and resource planning
  - Risk assessment and leadership reports
  - Stakeholder communication requiring objectivity

\b
GITHUB ENRICHMENT (PR & Review Context):

  Enhance narratives with Pull Request context, review comments, and
  collaboration data from GitHub's GraphQL API.

  # Basic usage with GitHub token
  export GITHUB_TOKEN="ghp_..."
  gitview analyze --repo org/repo --github-token $GITHUB_TOKEN

  # Or inline
  gitview analyze --repo org/repo --github-token "ghp_..."

  What GitHub enrichment provides:
  - PR titles and descriptions for context
  - Review comments and feedback
  - Reviewer attribution
  - PR labels for categorization
  - Merge/branch information

  How to get a GitHub token:
  1. Go to https://github.com/settings/tokens
  2. Generate new token (classic)
  3. Select 'repo' scope for private repos (or 'public_repo' for public only)
  4. Copy the token and use with --github-token

  Benefits:
  - Stories use PR descriptions instead of just commit messages
  - Review feedback provides context on why changes were made
  - Better understanding of team collaboration patterns
  - Labels help categorize types of changes

  Note: Results are cached locally (~/.gitview/cache/github) for 24 hours
  to reduce API calls and improve performance on repeated runs.
"""


def _estimate_analysis_cost(commit_count: int, avg_msg_length: int, backend: str, model: str) -> dict:
    """
    Estimate LLM API cost before analysis.

    Args:
        commit_count: Number of commits to analyze
        avg_msg_length: Average commit message length in characters
        backend: LLM backend name
        model: Model name

    Returns:
        Dict with cost_usd (estimated cost), num_phases, and explanation
    """
    # Estimate number of phases (adaptive strategy default: 5-60 commits per phase)
    avg_commits_per_phase = 40
    num_phases = max(1, commit_count // avg_commits_per_phase)

    # Estimate tokens per phase summarization
    # Base: commit metadata (hash, author, date) ~50 tokens per commit
    # Message content varies significantly
    tokens_per_commit = 50 + (avg_msg_length // 4)  # ~4 chars per token
    commits_shown_per_phase = min(20, commit_count // num_phases)  # First 20 commits shown

    input_tokens_per_phase = commits_shown_per_phase * tokens_per_commit + 500  # +500 for prompt
    output_tokens_per_phase = 600  # Phase summaries are ~400-600 tokens

    phase_summarization_input = num_phases * input_tokens_per_phase
    phase_summarization_output = num_phases * output_tokens_per_phase

    # Estimate story generation tokens (5 sections)
    story_input_tokens = num_phases * 400 + 2000  # Truncated phase summaries + prompts
    story_output_tokens = 10000  # Total across 5 sections

    total_input_tokens = phase_summarization_input + story_input_tokens
    total_output_tokens = phase_summarization_output + story_output_tokens

    # Cost per million tokens (as of 2025)
    cost_table = {
        ('openai', 'gpt-4o-mini'): (0.150, 0.600),
        ('openai', 'gpt-4o'): (2.50, 10.00),
        ('anthropic', 'claude-sonnet-4-5-20250929'): (3.00, 15.00),
        ('anthropic', 'claude-sonnet-3-5-20240229'): (3.00, 15.00),
        ('anthropic', 'claude-haiku-3-5-20241022'): (0.25, 1.25),
        ('ollama', None): (0, 0),  # Local, free
    }

    # Find matching cost
    input_cost_per_m, output_cost_per_m = cost_table.get((backend, model), (1.0, 5.0))

    estimated_cost = (
        (total_input_tokens / 1_000_000) * input_cost_per_m +
        (total_output_tokens / 1_000_000) * output_cost_per_m
    )

    return {
        'cost_usd': estimated_cost,
        'num_phases': num_phases,
        'input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens,
        'backend': backend,
        'model': model
    }


def _load_cached_analysis(output_dir: str):
    """Load cached commit history and phases if available."""

    history_file = Path(output_dir) / "repo_history.jsonl"
    phases_dir = Path(output_dir) / "phases"

    cached_records = None
    cached_phases = None

    if history_file.exists():
        try:
            cached_records = GitHistoryExtractor.load_from_jsonl(str(history_file))
        except Exception as exc:  # pragma: no cover - defensive logging
            console.print(f"[yellow]Warning: Failed to load cached commit history: {exc}[/yellow]")

    if phases_dir.exists():
        try:
            cached_phases = HistoryChunker.load_phases(str(phases_dir))
        except Exception as exc:  # pragma: no cover - defensive logging
            console.print(f"[yellow]Warning: Failed to load cached phases: {exc}[/yellow]")
            console.print("[yellow]This usually means the cache directory only has summaries "
                          "from an interrupted or older run; delete the phases folder to "
                          "recompute commit details.[/yellow]")

    return cached_records, cached_phases


def _analyze_single_branch(
    repo_path: str,
    branch: str,
    output: str,
    repo_name: str,
    strategy: str,
    chunk_size: int = 50,
    summarization_strategy: str = 'simple',
    max_commits: Optional[int] = None,
    backend: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    ollama_url: str = 'http://localhost:11434',
    skip_llm: bool = False,
    incremental: bool = False,
    since_commit: Optional[str] = None,
    since_date: Optional[str] = None,
    todo_content: Optional[str] = None,
    critical_mode: bool = False,
    directives: Optional[str] = None,
    github_token: Optional[str] = None,
    github_repo_url: Optional[str] = None
):
    """Analyze a single branch (helper function for multi-branch support)."""
    from typing import Optional

    # Check for incremental analysis
    previous_analysis = None
    existing_phases = []
    starting_loc = 0
    cached_records = None
    cached_phases = None
    auto_incremental = False

    # Smart incremental-by-default: if cache exists and user didn't specify otherwise,
    # automatically enable incremental mode to save costs
    if not incremental and not since_commit and not since_date:
        cached_records, cached_phases = _load_cached_analysis(output)
        if cached_records and cached_phases:
            previous_analysis = OutputWriter.load_previous_analysis(output)
            if previous_analysis:
                # Automatically enable incremental mode
                auto_incremental = True
                incremental = True
                metadata = previous_analysis.get('metadata', {})
                since_commit = metadata.get('last_commit_hash')
                console.print(f"[cyan]Smart cache detected:[/cyan] Auto-enabling incremental mode to save costs")
                console.print(f"[cyan]Analyzing commits since:[/cyan] {since_commit[:8] if since_commit else 'last analysis'}")
                console.print(f"[cyan]Last analysis:[/cyan] {metadata.get('generated_at', 'unknown')}\n")

    if incremental or since_commit or since_date:
        # Load previous analysis if not already loaded
        if not previous_analysis:
            previous_analysis = OutputWriter.load_previous_analysis(output)

        if incremental and not previous_analysis:
            if auto_incremental:
                # Auto-incremental failed, fall back to full analysis
                console.print("[yellow]Cache exists but incomplete. Running full analysis...[/yellow]\n")
                incremental = False
                auto_incremental = False
                since_commit = None
            else:
                console.print("[yellow]Warning: --incremental specified but no previous analysis found.[/yellow]")
                console.print("[yellow]Running full analysis instead...[/yellow]\n")
                incremental = False
        elif previous_analysis:
            metadata = previous_analysis.get('metadata', {})
            last_hash = metadata.get('last_commit_hash')
            last_date = metadata.get('last_commit_date')

            if incremental:
                since_commit = last_hash
                console.print(f"[cyan]Incremental mode:[/cyan] Analyzing commits since {last_hash[:8]}")
                console.print(f"[cyan]Last analysis:[/cyan] {metadata.get('generated_at', 'unknown')}\n")

            # Load existing phases
            from .chunker import Phase
            existing_phases = [Phase.from_dict(p) for p in previous_analysis.get('phases', [])]

            if existing_phases and existing_phases[-1].commits:
                starting_loc = existing_phases[-1].commits[-1].loc_total

    # Load cached data if not already loaded by auto-incremental logic
    if not since_commit and not since_date and not auto_incremental:
        cached_records, cached_phases = _load_cached_analysis(output)

    # Step 1: Extract git history
    console.print("[bold]Step 1: Extracting git history...[/bold]")
    extractor = GitHistoryExtractor(repo_path)

    if cached_records is not None and not since_commit and not since_date:
        records = cached_records
        console.print("[cyan]Found cached commit history; reusing repo_history.jsonl from previous run.[/cyan]\n")
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Extracting commits...", total=None)

            # Use incremental extraction if requested
            if since_commit or since_date:
                records = extractor.extract_incremental(
                    since_commit=since_commit,
                    since_date=since_date,
                    branch=branch
                )
                # Adjust LOC to continue from previous analysis
                if starting_loc > 0:
                    extractor._calculate_cumulative_loc(records, starting_loc)
            else:
                records = extractor.extract_history(max_commits=max_commits, branch=branch)

            progress.update(task, completed=True)

    if since_commit or since_date:
        console.print(f"[green]Extracted {len(records)} new commits[/green]\n")

        # Exit early if no new commits
        if len(records) == 0:
            console.print("[yellow]No new commits found since last analysis.[/yellow]")
            console.print("[green]Branch is up to date![/green]\n")
            return
    else:
        console.print(f"[green]Extracted {len(records)} commits[/green]\n")

    # Save raw history
    history_file = Path(output) / "repo_history.jsonl"
    extractor.save_to_jsonl(records, str(history_file))

    # Step 1.5: Enrich with GitHub context (if token provided)
    if github_token and github_repo_url and records:
        console.print("[bold]Step 1.5: Enriching with GitHub context...[/bold]")
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Fetching GitHub PR/review data...", total=None)

                # Enrich commits with GitHub context
                github_contexts = enrich_commits_with_github(
                    commits=records,
                    github_token=github_token,
                    repo_url=github_repo_url,
                    branch=branch,
                )

                # Attach GitHub context to commit records
                enriched_count = 0
                for record in records:
                    if record.commit_hash in github_contexts:
                        ctx = github_contexts[record.commit_hash]
                        record.github_context = ctx.to_dict()
                        if ctx.pr_number:
                            enriched_count += 1

                progress.update(task, completed=True)

            console.print(f"[green]Enriched {enriched_count} commits with GitHub PR context[/green]\n")

            # Re-save with GitHub context
            extractor.save_to_jsonl(records, str(history_file))

        except Exception as e:
            console.print(f"[yellow]Warning: GitHub enrichment failed: {e}[/yellow]")
            console.print("[yellow]Continuing without GitHub context...[/yellow]\n")

    # Step 2: Chunk into phases
    console.print("[bold]Step 2: Chunking into phases...[/bold]")
    chunker = HistoryChunker(strategy)

    kwargs = {}
    if strategy == 'fixed':
        kwargs['chunk_size'] = chunk_size

    # Handle incremental phase management
    if existing_phases and len(records) > 0:
        # Incremental mode: merge new commits with existing phases
        merge_threshold = 10  # commits - merge if fewer, create new phase if more

        if len(records) < merge_threshold:
            # Append new commits to last phase
            console.print(f"[yellow]Merging {len(records)} new commits into last phase...[/yellow]")
            last_phase = existing_phases[-1]
            last_phase.commits.extend(records)

            # Recalculate phase stats
            from .chunker import Phase
            last_phase.commit_count = len(last_phase.commits)
            last_phase.end_date = records[-1].timestamp
            last_phase.total_insertions = sum(c.insertions for c in last_phase.commits)
            last_phase.total_deletions = sum(c.deletions for c in last_phase.commits)
            last_phase.loc_end = records[-1].loc_total
            last_phase.loc_delta = last_phase.loc_end - last_phase.loc_start
            if last_phase.loc_start > 0:
                last_phase.loc_delta_percent = (last_phase.loc_delta / last_phase.loc_start) * 100

            # Clear summary so it will be regenerated
            last_phase.summary = None

            phases = existing_phases
            console.print(f"[green]Updated last phase (now {last_phase.commit_count} commits)[/green]\n")
        else:
            # Create new phases for new commits
            new_phases = chunker.chunk(records, **kwargs)

            # Renumber new phases to continue from existing
            for phase in new_phases:
                phase.phase_number = len(existing_phases) + phase.phase_number

            phases = existing_phases + new_phases
            console.print(f"[green]Created {len(new_phases)} new phases (total: {len(phases)})[/green]\n")
    elif cached_phases is not None and not since_commit and not since_date:
        phases = cached_phases
        console.print(f"[cyan]Reusing {len(phases)} cached phases from previous run.[/cyan]\n")
    else:
        # Full analysis: chunk normally
        phases = chunker.chunk(records, **kwargs)
        console.print(f"[green]Created {len(phases)} phases[/green]\n")

    # Display phase overview
    _display_phase_overview(phases)

    # Save phases
    phases_dir = Path(output) / "phases"
    chunker.save_phases(phases, str(phases_dir))

    if skip_llm:
        console.print("\n[yellow]Skipping LLM summarization. Writing basic timeline...[/yellow]")
        timeline_file = Path(output) / "timeline.md"
        OutputWriter.write_simple_timeline(phases, str(timeline_file))
        console.print(f"[green]Wrote timeline to {timeline_file}[/green]\n")
        return

    # Estimate cost before running LLM analysis
    # Calculate average commit message length
    avg_msg_length = sum(len(r.message) for r in records) // max(1, len(records))

    # Determine which backend/model will be used
    from .backends.router import LLMRouter
    router_for_estimate = LLMRouter(backend=backend, model=model, api_key=api_key)
    estimate = _estimate_analysis_cost(
        commit_count=len(records),
        avg_msg_length=avg_msg_length,
        backend=router_for_estimate.backend_type.value,
        model=router_for_estimate.model
    )

    # Show cost estimate
    if estimate['cost_usd'] > 0:
        console.print(f"\n[bold]Cost Estimate:[/bold]")
        console.print(f"  Backend: {estimate['backend']} / {estimate['model']}")
        console.print(f"  Estimated tokens: ~{estimate['input_tokens']:,} input + ~{estimate['output_tokens']:,} output")
        console.print(f"  Estimated cost: [yellow]${estimate['cost_usd']:.2f}[/yellow]")
        console.print(f"  ({estimate['num_phases']} phases to summarize + story generation)")

        # Suggest alternatives if cost is high
        if estimate['cost_usd'] > 2.0:
            console.print(f"\n[cyan]💡 To reduce costs:[/cyan]")
            if estimate['backend'] == 'anthropic':
                console.print(f"  • Use OpenAI gpt-4o-mini: --backend openai (~4-10x cheaper)")
            console.print(f"  • Use larger chunks: --strategy fixed --chunk-size 100")
            console.print(f"  • Limit commits: --max-commits 500")
        console.print()

    # Step 3: Summarize phases with LLM
    console.print("[bold]Step 3: Summarizing phases with LLM...[/bold]")

    use_hierarchical = summarization_strategy == 'hierarchical'

    if use_hierarchical:
        console.print("[cyan]Using hierarchical summarization strategy[/cyan]")
        console.print("[yellow]Note: This makes more API calls but preserves more details\n[/yellow]")
        summarizer = HierarchicalPhaseSummarizer(
            backend=backend,
            model=model,
            api_key=api_key,
            ollama_url=ollama_url,
        )
        phases_to_summarize = [p for p in phases if p.summary is None or
                               not getattr(p, 'metadata', {}).get('hierarchical_summary')]
    else:
        summarizer = PhaseSummarizer(
            backend=backend,
            model=model,
            api_key=api_key,
            ollama_url=ollama_url,
            todo_content=todo_content,
            critical_mode=critical_mode,
            directives=directives
        )
        phases_to_summarize = [p for p in phases if p.summary is None]

    if previous_analysis and len(phases_to_summarize) < len(phases):
        console.print(f"[cyan]Incremental mode: {len(phases_to_summarize)} phases need summarization "
                     f"({len(phases) - len(phases_to_summarize)} already summarized)[/cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Summarizing phases...", total=len(phases_to_summarize))

        # Build previous summaries from all phases (including existing ones)
        previous_summaries = []
        for i, phase in enumerate(phases):
            progress.update(task, description=f"Processing phase {i+1}/{len(phases)}...")

            if phase.summary is None:
                if use_hierarchical:
                    result = summarizer.summarize_phase(phase)
                    phase.summary = result['full_summary']
                    if not hasattr(phase, 'metadata'):
                        phase.metadata = {}
                    phase.metadata['hierarchical_summary'] = result
                    summarizer._save_phase_summary(phase, result, str(phases_dir))  # type: ignore[attr-defined]
                else:
                    # Need to summarize this phase
                    context = summarizer._build_context(previous_summaries)
                    summary = summarizer.summarize_phase(phase, context)
                    phase.summary = summary
                    summarizer._save_phase_with_summary(phase, str(phases_dir))
                progress.update(task, advance=1)

            previous_summaries.append({
                'phase_number': phase.phase_number,
                'summary': phase.summary,
                'loc_delta': phase.loc_delta,
            })

    if len(phases_to_summarize) > 0:
        console.print(f"[green]Summarized {len(phases_to_summarize)} phase(s)[/green]\n")
    else:
        console.print(f"[green]All phases already summarized[/green]\n")

    # Step 4: Generate global story
    console.print("[bold]Step 4: Generating global narrative...[/bold]")

    if use_hierarchical:
        storyteller = HierarchicalStoryTeller(
            backend=backend,
            model=model,
            api_key=api_key,
            ollama_url=ollama_url,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating hierarchical timeline...", total=None)
            timeline = storyteller.generate_timeline(phases, repo_name=repo_name)
            progress.update(task, completed=True)

        stories = {
            'timeline': timeline,
            'executive_summary': 'See timeline for detailed evolution',
            'technical_evolution': 'See timeline for technical details',
            'deletion_story': '',
            'full_narrative': timeline,
        }

        console.print(f"[green]Generated hierarchical timeline[/green]\n")
    else:
        storyteller = StoryTeller(
            backend=backend,
            model=model,
            api_key=api_key,
            ollama_url=ollama_url,
            todo_content=todo_content,
            critical_mode=critical_mode,
            directives=directives
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating story...", total=None)
            stories = storyteller.generate_global_story(phases, repo_name, cache_dir=output)
            progress.update(task, completed=True)

        console.print(f"[green]Generated global narrative[/green]\n")

    # Step 5: Write output
    console.print("[bold]Step 5: Writing output files...[/bold]")
    output_path = Path(output)

    # Write markdown report
    markdown_path = output_path / "history_story.md"
    OutputWriter.write_markdown(stories, phases, str(markdown_path), repo_name)
    console.print(f"[green]Wrote {markdown_path}[/green]")

    # Write JSON data with metadata for incremental analysis
    json_path = output_path / "history_data.json"
    OutputWriter.write_json(stories, phases, str(json_path), repo_path=repo_path)
    console.print(f"[green]Wrote {json_path}[/green]")

    # Write timeline
    timeline_path = output_path / "timeline.md"
    if use_hierarchical:
        timeline_path.parent.mkdir(parents=True, exist_ok=True)
        timeline_path.write_text(stories['timeline'])
    else:
        OutputWriter.write_simple_timeline(phases, str(timeline_path))
    console.print(f"[green]Wrote {timeline_path}[/green]\n")

    # Success summary
    console.print("[bold green]Branch analysis complete![/bold green]\n")
    console.print(f"Analyzed {len(records)} commits across {len(phases)} phases")
    console.print(f"Output written to: {output_path.resolve()}\n")


@cli.command(help=ANALYZE_HELP)
@click.option('--repo', '-r', default=".",
              help="Repository: local path, GitHub shortcut (org/repo), or full URL")
@click.option('--output', '-o', default=None,
              help="Output directory (default: auto-generated based on repo)")
@click.option('--strategy', '-s', type=click.Choice(['fixed', 'time', 'adaptive']),
              default='adaptive',
              help="Chunking strategy: 'adaptive' (default, splits on significant changes), "
                   "'fixed' (N commits per phase), 'time' (by time period)")
@click.option('--chunk-size', type=int, default=50,
              help="Commits per chunk when using 'fixed' strategy")
@click.option('--max-commits', type=int,
              help="Maximum commits to analyze (default: all commits)")
@click.option('--branch', default='HEAD',
              help="Branch to analyze (default: HEAD/current branch). Use --branches for multiple.")
@click.option('--list-branches', is_flag=True,
              help="List all available branches and exit")
@click.option('--branches',
              help="Analyze specific branches (comma-separated or patterns like 'feature/*')")
@click.option('--all-branches', is_flag=True,
              help="Analyze all branches (local and remote)")
@click.option('--exclude-branches',
              help="Exclude branches matching patterns (comma-separated)")
@click.option('--backend', '-b', type=click.Choice(['anthropic', 'openai', 'ollama']),
              help="LLM backend: 'anthropic' (Claude), 'openai' (GPT), 'ollama' (local). "
                   "Auto-detected from env vars if not specified.")
@click.option('--model', '-m',
              help="Model identifier. Defaults: claude-sonnet-4-5-20250929 (Anthropic), "
                   "gpt-4 (OpenAI), llama3 (Ollama)")
@click.option('--api-key',
              help="API key for Anthropic/OpenAI. Defaults to ANTHROPIC_API_KEY or "
                   "OPENAI_API_KEY environment variable")
@click.option('--ollama-url', default='http://localhost:11434',
              help="Ollama server URL (only for --backend ollama)")
@click.option('--repo-name',
              help="Repository name for output (default: directory name)")
@click.option('--summarization-strategy',
              type=click.Choice(['simple', 'hierarchical']),
              default='simple',
              help="Summarization strategy: 'simple' (default) or 'hierarchical' "
                   "(better fidelity for large/complex histories)")
@click.option('--hierarchical', is_flag=True,
              help="Shortcut for --summarization-strategy hierarchical")
@click.option('--skip-llm', is_flag=True,
              help="Skip LLM summarization - only extract and chunk history "
                   "(useful for quick analysis without API costs)")
@click.option('--incremental', is_flag=True,
              help="Incremental analysis: only process commits since last run. "
                   "Automatically detects previous analysis in output directory")
@click.option('--since-commit',
              help="Extract commits since this commit hash (for manual incremental analysis)")
@click.option('--since-date',
              help="Extract commits since this date (ISO format: YYYY-MM-DD)")
@click.option('--keep-clone', is_flag=True,
              help="Keep any temporary clone copies (cached clones under ~/.gitview/cache/repos are preserved automatically)")
@click.option('--todo',
              help="Path to todo/goals file for critical examination mode. "
                   "Evaluates commits against objectives defined in this file.")
@click.option('--critical', is_flag=True,
              help="Enable critical examination mode: focus on gaps, issues, and goal alignment. "
                   "Removes flowery achievement-focused language.")
@click.option('--directives',
              help="Additional plain text directives to inject into LLM prompts for custom analysis focus.")
@click.option('--github-token',
              envvar='GITHUB_TOKEN',
              help="GitHub Personal Access Token for enriching commits with PR/review context. "
                   "Can also be set via GITHUB_TOKEN environment variable.")
def analyze(repo, output, strategy, chunk_size, max_commits, branch, list_branches,
           branches, all_branches, exclude_branches, backend, model, api_key, ollama_url,
           repo_name, summarization_strategy, hierarchical, skip_llm, incremental, since_commit,
           since_date, keep_clone, todo, critical, directives, github_token):
    """Analyze git repository and generate narrative history.

    This is the main command that runs the full pipeline:
    1. Extract git history
    2. Chunk into meaningful phases
    3. Summarize each phase with LLM
    4. Generate global narrative
    5. Write output files
    """
    console.print("\n[bold blue]GitView - Repository History Analyzer[/bold blue]\n")

    # Load todo/goals file for critical examination mode
    todo_content = None
    if todo:
        todo_path = Path(todo)
        if not todo_path.exists():
            console.print(f"[red]Error: Todo file not found: {todo}[/red]")
            sys.exit(1)

        with open(todo_path, 'r') as f:
            todo_content = f.read()

        console.print(f"[cyan]Loaded goals from:[/cyan] {todo}")
        if not critical:
            console.print("[yellow]Note: --todo specified without --critical. Consider using --critical for goal-focused analysis.[/yellow]")
        console.print()

    # Normalize summarization strategy
    if hierarchical:
        summarization_strategy = 'hierarchical'

    if summarization_strategy == 'hierarchical':
        console.print("[cyan]Using hierarchical summarization strategy[/cyan]")

    # Validate critical mode
    if critical and not todo and not directives:
        console.print("[yellow]Warning: --critical mode enabled without --todo or --directives.[/yellow]")
        console.print("[yellow]Critical mode works best with goals/directives to measure against.[/yellow]\n")

    # Handle remote repository detection and cloning
    repo_handler = RemoteRepoHandler(repo)
    cloned_repo_path = None

    try:
        if repo_handler.is_local:
            # Local repository
            repo_path = repo_handler.get_local_path()
            if not (repo_path / '.git').exists():
                console.print(f"[red]Error: {repo_path} is not a git repository[/red]")
                sys.exit(1)

            # Get repo name if not provided
            if not repo_name:
                repo_name = repo_path.name

            # Set default output if not provided
            if output is None:
                output = "output"

        else:
            # Remote repository - clone it
            console.print(f"[cyan]Remote repository detected:[/cyan] {repo_handler.repo_info.short_name}\n")

            cloned_repo_path = repo_handler.clone()
            repo_path = cloned_repo_path

            # Get repo name from repo info
            if not repo_name:
                repo_name = repo_handler.repo_info.repo

            # Set default output path for remote repos
            if output is None:
                output = str(repo_handler.get_default_output_path())
                console.print(f"[cyan]Output will be saved to:[/cyan] {output}\n")

        # Initialize branch manager
        branch_manager = BranchManager(str(repo_path))

        # Handle --list-branches flag
        if list_branches:
            console.print("[bold]Available Branches:[/bold]\n")
            all_branches = branch_manager.list_all_branches(include_remote=True)

            local_branches = [b for b in all_branches if not b.is_remote]
            remote_branches = [b for b in all_branches if b.is_remote]

            if local_branches:
                console.print("[cyan]Local Branches:[/cyan]")
                for b in sorted(local_branches, key=lambda x: x.name):
                    console.print(f"  - {b.name} ({b.commit_count:,} commits)")

            if remote_branches:
                console.print(f"\n[cyan]Remote Branches:[/cyan]")
                for b in sorted(remote_branches, key=lambda x: x.name):
                    console.print(f"  - {b.name} ({b.commit_count:,} commits)")

            console.print(f"\n[green]Total: {len(all_branches)} branches[/green]")
            return

        # Determine which branches to analyze
        branches_to_analyze = []

        if all_branches:
            # Analyze all branches
            branches_to_analyze = branch_manager.list_all_branches(include_remote=True)
            console.print(f"[cyan]Analyzing all branches:[/cyan] {len(branches_to_analyze)} branches")

        elif branches:
            # Analyze specific branches based on patterns
            patterns = parse_branch_spec(branches)
            all_available = branch_manager.list_all_branches(include_remote=True)
            branches_to_analyze = branch_manager.filter_branches(all_available, include_patterns=patterns)

            if not branches_to_analyze:
                console.print(f"[red]Error: No branches match pattern(s): {branches}[/red]")
                sys.exit(1)

            console.print(f"[cyan]Analyzing {len(branches_to_analyze)} branch(es) matching '{branches}'[/cyan]")

        else:
            # Single branch mode (backward compatible)
            branch_info = branch_manager.get_branch_by_name(branch)
            if branch_info:
                branches_to_analyze = [branch_info]
            else:
                # Fallback: treat as branch name even if not found
                console.print(f"[yellow]Warning: Branch '{branch}' not found in branch list, proceeding anyway...[/yellow]")
                branches_to_analyze = []  # Will use old single-branch logic

        # Apply exclusion patterns
        if exclude_branches and branches_to_analyze:
            exclude_patterns = parse_branch_spec(exclude_branches)
            before_count = len(branches_to_analyze)
            branches_to_analyze = branch_manager.filter_branches(
                branches_to_analyze,
                exclude_patterns=exclude_patterns
            )
            excluded_count = before_count - len(branches_to_analyze)
            if excluded_count > 0:
                console.print(f"[yellow]Excluded {excluded_count} branch(es) matching exclusion patterns[/yellow]")

        # Multi-branch mode
        is_multi_branch = len(branches_to_analyze) > 1

        if is_multi_branch:
            # Show warning about costs and get confirmation for large analyses
            stats = branch_manager.get_branch_statistics(branches_to_analyze)
            total_commits = stats['total_commits']

            console.print(f"\n[bold yellow]Multi-Branch Analysis[/bold yellow]")
            console.print(f"  Branches: {len(branches_to_analyze)}")
            console.print(f"  Estimated total commits: {total_commits:,}")

            if not skip_llm:
                # Rough estimate: ~1 phase per 50 commits, ~1 LLM call per phase
                estimated_llm_calls = (total_commits // 50) * len(branches_to_analyze) + len(branches_to_analyze) * 5
                console.print(f"  Estimated LLM calls: ~{estimated_llm_calls:,}")
                console.print(f"\n[yellow]This will incur API costs. Use --skip-llm to avoid LLM costs.[/yellow]")

            if total_commits > 10000 and not skip_llm:
                console.print(f"\n[bold red]WARNING: Large analysis detected ({total_commits:,} commits)[/bold red]")
                console.print("[red]This may take a long time and incur significant API costs.[/red]")
                console.print("[yellow]Consider using --skip-llm or limiting branches.[/yellow]\n")

                # Prompt for confirmation
                if not click.confirm("Do you want to continue?", default=False):
                    console.print("[yellow]Analysis cancelled.[/yellow]")
                    return

            console.print()

        console.print(f"[cyan]Repository:[/cyan] {repo_path}")
        console.print(f"[cyan]Output:[/cyan] {output}")
        console.print(f"[cyan]Strategy:[/cyan] {strategy}")

        if not skip_llm:
            # Determine backend for display
            from .backends import LLMRouter
            router = LLMRouter(backend=backend, model=model, api_key=api_key, ollama_url=ollama_url)
            console.print(f"[cyan]Backend:[/cyan] {router.backend_type.value}")
            console.print(f"[cyan]Model:[/cyan] {router.model}\n")
        else:
            console.print("[yellow]Skipping LLM summarization[/yellow]\n")

        # Multi-branch analysis mode
        if is_multi_branch:
            base_output_dir = Path(output)
            analyzed_branches = []

            for idx, branch_info in enumerate(branches_to_analyze, 1):
                console.print(f"\n[bold cyan]=== Analyzing Branch {idx}/{len(branches_to_analyze)}: {branch_info.name} ===[/bold cyan]\n")

                # Set branch-specific output directory
                branch_output = base_output_dir / branch_info.sanitized_name
                current_branch = branch_info.name

                # Determine GitHub repo URL for enrichment
                github_repo_url = None
                if github_token:
                    if not repo_handler.is_local and repo_handler.repo_info:
                        github_repo_url = f"{repo_handler.repo_info.org}/{repo_handler.repo_info.repo}"
                    elif repo_handler.is_local:
                        try:
                            from git import Repo as GitRepo
                            git_repo = GitRepo(str(repo_path))
                            for remote in git_repo.remotes:
                                for url in remote.urls:
                                    if 'github.com' in url:
                                        from .github_graphql import parse_github_url
                                        owner, repo_n = parse_github_url(url)
                                        github_repo_url = f"{owner}/{repo_n}"
                                        break
                                if github_repo_url:
                                    break
                        except Exception:
                            pass

                # Analyze this branch (call single-branch logic)
                _analyze_single_branch(
                    repo_path=str(repo_path),
                    branch=current_branch,
                    output=str(branch_output),
                    repo_name=f"{repo_name} ({branch_info.short_name})",
                    strategy=strategy,
                    summarization_strategy=summarization_strategy,
                    chunk_size=chunk_size,
                    max_commits=max_commits,
                    backend=backend,
                    model=model,
                    api_key=api_key,
                    ollama_url=ollama_url,
                    skip_llm=skip_llm,
                    incremental=incremental,
                    since_commit=since_commit,
                    since_date=since_date,
                    todo_content=todo_content,
                    critical_mode=critical,
                    directives=directives,
                    github_token=github_token,
                    github_repo_url=github_repo_url
                )

                analyzed_branches.append(branch_info)

            # Generate index report
            console.print(f"\n[bold]Generating multi-branch index...[/bold]")
            IndexWriter.write_branch_index(analyzed_branches, base_output_dir, repo_name)
            IndexWriter.write_branch_metadata(analyzed_branches, base_output_dir)
            IndexWriter.write_simple_branch_list(analyzed_branches, base_output_dir)

            console.print(f"\n[bold green]Multi-branch analysis complete![/bold green]")
            console.print(f"Analyzed {len(analyzed_branches)} branches")
            console.print(f"Index report: {base_output_dir / 'index.md'}\n")
            return

        # Single-branch analysis (backward compatible)
        # Use branch from branches_to_analyze if available, otherwise use original branch param
        if branches_to_analyze:
            branch = branches_to_analyze[0].name

        # Check for incremental analysis
        previous_analysis = None
        existing_phases = []
        starting_loc = 0
        cached_records = None
        cached_phases = None

        if incremental or since_commit or since_date:
            # Load previous analysis
            previous_analysis = OutputWriter.load_previous_analysis(output)

            if incremental and not previous_analysis:
                console.print("[yellow]Warning: --incremental specified but no previous analysis found.[/yellow]")
                console.print("[yellow]Running full analysis instead...[/yellow]\n")
                incremental = False
            elif previous_analysis:
                metadata = previous_analysis.get('metadata', {})
                last_hash = metadata.get('last_commit_hash')
                last_date = metadata.get('last_commit_date')

                if incremental:
                    since_commit = last_hash
                    console.print(f"[cyan]Incremental mode:[/cyan] Analyzing commits since {last_hash[:8]}")
                    console.print(f"[cyan]Last analysis:[/cyan] {metadata.get('generated_at', 'unknown')}\n")

                # Load existing phases
                from .chunker import Phase
                existing_phases = [Phase.from_dict(p) for p in previous_analysis.get('phases', [])]

            if existing_phases and existing_phases[-1].commits:
                starting_loc = existing_phases[-1].commits[-1].loc_total

        if not since_commit and not since_date:
            cached_records, cached_phases = _load_cached_analysis(output)

        # Step 1: Extract git history
        console.print("[bold]Step 1: Extracting git history...[/bold]")
        extractor = GitHistoryExtractor(str(repo_path))

        if cached_records is not None and not since_commit and not since_date:
            records = cached_records
            console.print("[cyan]Found cached commit history; reusing repo_history.jsonl from previous run.[/cyan]\n")
        else:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Extracting commits...", total=None)

                # Use incremental extraction if requested
                if since_commit or since_date:
                    records = extractor.extract_incremental(
                        since_commit=since_commit,
                        since_date=since_date,
                        branch=branch
                    )
                    # Adjust LOC to continue from previous analysis
                    if starting_loc > 0:
                        extractor._calculate_cumulative_loc(records, starting_loc)
                else:
                    records = extractor.extract_history(max_commits=max_commits, branch=branch)

                progress.update(task, completed=True)

        if since_commit or since_date:
            console.print(f"[green]Extracted {len(records)} new commits[/green]\n")

            # Exit early if no new commits
            if len(records) == 0:
                console.print("[yellow]No new commits found since last analysis.[/yellow]")
                console.print("[green]Repository is up to date![/green]\n")
                return
        else:
            console.print(f"[green]Extracted {len(records)} commits[/green]\n")

        # Save raw history
        history_file = Path(output) / "repo_history.jsonl"
        extractor.save_to_jsonl(records, str(history_file))

        # Step 1.5: Enrich with GitHub context (if token provided)
        if github_token and records:
            console.print("[bold]Step 1.5: Enriching with GitHub context...[/bold]")

            # Determine repo URL for GitHub API
            github_repo_url = None
            if not repo_handler.is_local and repo_handler.repo_info:
                github_repo_url = f"{repo_handler.repo_info.org}/{repo_handler.repo_info.repo}"
            elif repo_handler.is_local:
                # Try to detect GitHub remote from local repo
                try:
                    from git import Repo as GitRepo
                    git_repo = GitRepo(str(repo_path))
                    for remote in git_repo.remotes:
                        for url in remote.urls:
                            if 'github.com' in url:
                                from .github_graphql import parse_github_url
                                owner, repo_n = parse_github_url(url)
                                github_repo_url = f"{owner}/{repo_n}"
                                break
                        if github_repo_url:
                            break
                except Exception:
                    pass

            if github_repo_url:
                try:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console
                    ) as progress:
                        task = progress.add_task("Fetching GitHub PR/review data...", total=None)

                        # Enrich commits with GitHub context
                        github_contexts = enrich_commits_with_github(
                            commits=records,
                            github_token=github_token,
                            repo_url=github_repo_url,
                            branch=branch,
                        )

                        # Attach GitHub context to commit records
                        enriched_count = 0
                        for record in records:
                            if record.commit_hash in github_contexts:
                                ctx = github_contexts[record.commit_hash]
                                record.github_context = ctx.to_dict()
                                if ctx.pr_number:
                                    enriched_count += 1

                        progress.update(task, completed=True)

                    console.print(f"[green]Enriched {enriched_count} commits with GitHub PR context[/green]\n")

                    # Re-save with GitHub context
                    extractor.save_to_jsonl(records, str(history_file))

                except Exception as e:
                    console.print(f"[yellow]Warning: GitHub enrichment failed: {e}[/yellow]")
                    console.print("[yellow]Continuing without GitHub context...[/yellow]\n")
            else:
                console.print("[yellow]Warning: Could not determine GitHub repository URL[/yellow]")
                console.print("[yellow]GitHub enrichment requires a GitHub-hosted repository[/yellow]\n")

        # Step 2: Chunk into phases
        console.print("[bold]Step 2: Chunking into phases...[/bold]")
        chunker = HistoryChunker(strategy)

        kwargs = {}
        if strategy == 'fixed':
            kwargs['chunk_size'] = chunk_size

        # Handle incremental phase management
        if existing_phases and len(records) > 0:
            # Incremental mode: merge new commits with existing phases
            merge_threshold = 10  # commits - merge if fewer, create new phase if more

            if len(records) < merge_threshold:
                # Append new commits to last phase
                console.print(f"[yellow]Merging {len(records)} new commits into last phase...[/yellow]")
                last_phase = existing_phases[-1]
                last_phase.commits.extend(records)

                # Recalculate phase stats
                from .chunker import Phase
                last_phase.commit_count = len(last_phase.commits)
                last_phase.end_date = records[-1].timestamp
                last_phase.total_insertions = sum(c.insertions for c in last_phase.commits)
                last_phase.total_deletions = sum(c.deletions for c in last_phase.commits)
                last_phase.loc_end = records[-1].loc_total
                last_phase.loc_delta = last_phase.loc_end - last_phase.loc_start
                if last_phase.loc_start > 0:
                    last_phase.loc_delta_percent = (last_phase.loc_delta / last_phase.loc_start) * 100

                # Clear summary so it will be regenerated
                last_phase.summary = None

                phases = existing_phases
                console.print(f"[green]Updated last phase (now {last_phase.commit_count} commits)[/green]\n")
            else:
                # Create new phases for new commits
                new_phases = chunker.chunk(records, **kwargs)

                # Renumber new phases to continue from existing
                for phase in new_phases:
                    phase.phase_number = len(existing_phases) + phase.phase_number

                phases = existing_phases + new_phases
                console.print(f"[green]Created {len(new_phases)} new phases (total: {len(phases)})[/green]\n")
        elif cached_phases is not None and not since_commit and not since_date:
            phases = cached_phases
            console.print(f"[cyan]Reusing {len(phases)} cached phases from previous run.[/cyan]\n")
        else:
            # Full analysis: chunk normally
            phases = chunker.chunk(records, **kwargs)
            console.print(f"[green]Created {len(phases)} phases[/green]\n")

        # Display phase overview
        _display_phase_overview(phases)

        # Save phases
        phases_dir = Path(output) / "phases"
        chunker.save_phases(phases, str(phases_dir))

        if skip_llm:
            console.print("\n[yellow]Skipping LLM summarization. Writing basic timeline...[/yellow]")
            timeline_file = Path(output) / "timeline.md"
            OutputWriter.write_simple_timeline(phases, str(timeline_file))
            console.print(f"[green]Wrote timeline to {timeline_file}[/green]\n")
            return

        # Step 3: Summarize phases with LLM
        console.print("[bold]Step 3: Summarizing phases with LLM...[/bold]")

        use_hierarchical = summarization_strategy == 'hierarchical'

        if use_hierarchical:
            console.print("[cyan]Using hierarchical summarization strategy[/cyan]")
            console.print("[yellow]Note: This makes more API calls but preserves more details\n[/yellow]")
            summarizer = HierarchicalPhaseSummarizer(
                backend=backend,
                model=model,
                api_key=api_key,
                ollama_url=ollama_url,
            )
            phases_to_summarize = [p for p in phases if p.summary is None or
                                   not getattr(p, 'metadata', {}).get('hierarchical_summary')]
        else:
            summarizer = PhaseSummarizer(
                backend=backend,
                model=model,
                api_key=api_key,
                ollama_url=ollama_url,
                todo_content=todo_content,
                critical_mode=critical,
                directives=directives
            )

            # Identify phases that need summarization (no summary)
            phases_to_summarize = [p for p in phases if p.summary is None]

        if previous_analysis and len(phases_to_summarize) < len(phases):
            console.print(f"[cyan]Incremental mode: {len(phases_to_summarize)} phases need summarization "
                         f"({len(phases) - len(phases_to_summarize)} already summarized)[/cyan]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Summarizing phases...", total=len(phases_to_summarize))

            # Build previous summaries from all phases (including existing ones)
            previous_summaries = []
            for i, phase in enumerate(phases):
                progress.update(task, description=f"Processing phase {i+1}/{len(phases)}...")

                if phase.summary is None:
                    if use_hierarchical:
                        result = summarizer.summarize_phase(phase)
                        phase.summary = result['full_summary']
                        if not hasattr(phase, 'metadata'):
                            phase.metadata = {}
                        phase.metadata['hierarchical_summary'] = result
                        summarizer._save_phase_summary(phase, result, str(phases_dir))  # type: ignore[attr-defined]
                    else:
                        # Need to summarize this phase
                        context = summarizer._build_context(previous_summaries)
                        summary = summarizer.summarize_phase(phase, context)
                        phase.summary = summary
                        summarizer._save_phase_with_summary(phase, str(phases_dir))
                    progress.update(task, advance=1)

                previous_summaries.append({
                    'phase_number': phase.phase_number,
                    'summary': phase.summary,
                    'loc_delta': phase.loc_delta,
                })

        if len(phases_to_summarize) > 0:
            console.print(f"[green]Summarized {len(phases_to_summarize)} phase(s)[/green]\n")
        else:
            console.print(f"[green]All phases already summarized[/green]\n")

        # Step 4: Generate global story
        console.print("[bold]Step 4: Generating global narrative...[/bold]")

        if use_hierarchical:
            storyteller = HierarchicalStoryTeller(
                backend=backend,
                model=model,
                api_key=api_key,
                ollama_url=ollama_url,
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Generating hierarchical timeline...", total=None)
                timeline = storyteller.generate_timeline(phases, repo_name=repo_name)
                progress.update(task, completed=True)

            stories = {
                'timeline': timeline,
                'executive_summary': 'See timeline for detailed evolution',
                'technical_evolution': 'See timeline for technical details',
                'deletion_story': '',
                'full_narrative': timeline,
            }

            console.print(f"[green]Generated hierarchical timeline[/green]\n")
        else:
            storyteller = StoryTeller(
                backend=backend,
                model=model,
                api_key=api_key,
                ollama_url=ollama_url,
                todo_content=todo_content,
                critical_mode=critical,
                directives=directives
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Generating story...", total=None)
                stories = storyteller.generate_global_story(phases, repo_name)
                progress.update(task, completed=True)

            console.print(f"[green]Generated global narrative[/green]\n")

        # Step 5: Write output
        console.print("[bold]Step 5: Writing output files...[/bold]")
        output_path = Path(output)

        # Write markdown report
        markdown_path = output_path / "history_story.md"
        OutputWriter.write_markdown(stories, phases, str(markdown_path), repo_name)
        console.print(f"[green]Wrote {markdown_path}[/green]")

        # Write JSON data with metadata for incremental analysis
        json_path = output_path / "history_data.json"
        OutputWriter.write_json(stories, phases, str(json_path), repo_path=str(repo_path))
        console.print(f"[green]Wrote {json_path}[/green]")

        # Write timeline
        timeline_path = output_path / "timeline.md"
        if use_hierarchical:
            timeline_path.parent.mkdir(parents=True, exist_ok=True)
            timeline_path.write_text(stories['timeline'])
        else:
            OutputWriter.write_simple_timeline(phases, str(timeline_path))
        console.print(f"[green]Wrote {timeline_path}[/green]\n")

        # Success summary
        console.print("[bold green]Analysis complete![/bold green]\n")
        console.print(f"Analyzed {len(records)} commits across {len(phases)} phases")
        console.print(f"Output written to: {output_path.resolve()}\n")

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Cleanup temporary clone if it exists and --keep-clone not specified
        if cloned_repo_path and not keep_clone:
            repo_handler.cleanup()
        elif cloned_repo_path and keep_clone:
            console.print(f"\n[cyan]Temporary clone preserved at:[/cyan] {cloned_repo_path}")


EXTRACT_HELP = """Extract git history to JSONL file (no LLM needed).

\b
This command extracts detailed metadata from git commits without using an LLM.
Useful for:
  - Quick history extraction
  - Pre-processing for later analysis
  - Exploring repository metrics

\b
Extracted data includes:
  - Commit metadata (hash, author, date, message)
  - Lines of code changes (insertions/deletions)
  - Language breakdown per commit
  - README evolution
  - Comment density analysis
  - Detection of large changes and refactors

\b
EXAMPLES:

  # Extract full history to default location
  gitview extract

  # Extract to custom file
  gitview extract --output my_history.jsonl

  # Extract only last 100 commits
  gitview extract --max-commits 100

  # Extract from specific branch
  gitview extract --branch develop

  # Extract from different repository
  gitview extract --repo /path/to/repo --output repo_data.jsonl
"""


@cli.command(help=EXTRACT_HELP)
@click.option('--repo', '-r', default=".",
              help="Path to git repository (default: current directory)")
@click.option('--output', '-o', default="output/repo_history.jsonl",
              help="Output JSONL file path")
@click.option('--max-commits', type=int,
              help="Maximum commits to extract (default: all commits)")
@click.option('--branch', default='HEAD',
              help="Branch to extract from (default: HEAD/current branch)")
def extract(repo, output, max_commits, branch):
    console.print("\n[bold blue]Extracting Git History[/bold blue]\n")

    repo_path = Path(repo).resolve()
    if not (repo_path / '.git').exists():
        console.print(f"[red]Error: {repo_path} is not a git repository[/red]")
        sys.exit(1)

    try:
        extractor = GitHistoryExtractor(str(repo_path))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Extracting commits...", total=None)
            records = extractor.extract_history(max_commits=max_commits, branch=branch)
            progress.update(task, completed=True)

        extractor.save_to_jsonl(records, output)

        console.print(f"\n[green]Extracted {len(records)} commits to {output}[/green]\n")

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)


CHUNK_HELP = """Chunk extracted history into meaningful phases (no LLM needed).

\b
Takes a JSONL file from 'gitview extract' and splits it into phases/epochs
based on the chosen strategy. No LLM or API key required.

\b
CHUNKING STRATEGIES:

1. Adaptive (recommended):
   Automatically splits when significant changes occur:
   - LOC changes by >30%
   - Large deletions or additions (>1000 lines)
   - README rewrites
   - Major refactorings

2. Fixed:
   Split into fixed-size chunks (e.g., 50 commits per phase)

3. Time:
   Split by time periods (week, month, quarter, year)

\b
EXAMPLES:

  # Chunk with adaptive strategy (recommended)
  gitview chunk repo_history.jsonl

  # Chunk with fixed size (25 commits per phase)
  gitview chunk repo_history.jsonl --strategy fixed --chunk-size 25

  # Save phases to custom directory
  gitview chunk repo_history.jsonl --output ./my_phases

  # First extract, then chunk separately
  gitview extract --output data.jsonl
  gitview chunk data.jsonl --output phases/
"""


@cli.command(help=CHUNK_HELP)
@click.argument('history_file', type=click.Path(exists=True))
@click.option('--output', '-o', default="output/phases",
              help="Output directory for phase JSON files")
@click.option('--strategy', '-s', type=click.Choice(['fixed', 'time', 'adaptive']),
              default='adaptive',
              help="Chunking strategy: 'adaptive' (default), 'fixed', 'time'")
@click.option('--chunk-size', type=int, default=50,
              help="Commits per chunk when using 'fixed' strategy")
def chunk(history_file, output, strategy, chunk_size):
    console.print("\n[bold blue]Chunking History into Phases[/bold blue]\n")

    try:
        # Load history
        from .extractor import GitHistoryExtractor
        records = GitHistoryExtractor.load_from_jsonl(history_file)

        console.print(f"[cyan]Loaded {len(records)} commits[/cyan]")
        console.print(f"[cyan]Strategy: {strategy}[/cyan]\n")

        # Chunk
        chunker = HistoryChunker(strategy)
        kwargs = {}
        if strategy == 'fixed':
            kwargs['chunk_size'] = chunk_size

        phases = chunker.chunk(records, **kwargs)

        console.print(f"[green]Created {len(phases)} phases[/green]\n")
        _display_phase_overview(phases)

        # Save
        chunker.save_phases(phases, output)
        console.print(f"\n[green]Saved phases to {output}[/green]\n")

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)


def _display_phase_overview(phases):
    """Display phase overview table."""
    table = Table(title="Phase Overview")

    table.add_column("Phase", style="cyan", justify="right")
    table.add_column("Period", style="magenta")
    table.add_column("Commits", justify="right")
    table.add_column("LOC D", justify="right")
    table.add_column("Events", style="yellow")

    for phase in phases:
        events = []
        if phase.has_large_deletion:
            events.append("x")
        if phase.has_large_addition:
            events.append("+")
        if phase.has_refactor:
            events.append(">>")
        if phase.readme_changed:
            events.append(">")

        table.add_row(
            str(phase.phase_number),
            f"{phase.start_date[:10]} to {phase.end_date[:10]}",
            str(phase.commit_count),
            f"{phase.loc_delta:+,d}",
            " ".join(events)
        )

    console.print(table)


@cli.command('track-files')
@click.option('--repo', default='.', help='Path to repository (local or remote)')
@click.option('--output', default='output/file_histories',
              help='Output directory for file histories')
@click.option('--patterns', default=None,
              help='Comma-separated file patterns to track (e.g., "*.py,*.js,*.go")')
@click.option('--exclude', default=None,
              help='Comma-separated exclude patterns (e.g., "*_test.py,*.min.js")')
@click.option('--since-commit', default=None,
              help='Only track changes since this commit')
@click.option('--incremental/--no-incremental', default=True,
              help='Use checkpoint for incremental processing (default: True)')
@click.option('--max-entries', default=100, type=int,
              help='Maximum history entries per file (default: 100)')
def track_files(repo, output, patterns, exclude, since_commit, incremental, max_entries):
    """Track detailed change history for all files in repository.

    \b
    This command generates:
      - .history companion files for each tracked file (human-readable)
      - .json files with complete structured history (machine-readable)
      - index.json with summary of all tracked files
      - checkpoint.json for incremental processing

    \b
    Examples:

      # Track all files in current repository
      gitview track-files

      # Track only Python and JavaScript files
      gitview track-files --patterns "*.py,*.js"

      # Exclude test files and minified files
      gitview track-files --exclude "*_test.py,*.min.js,*-lock.json"

      # Incremental update (only process new commits)
      gitview track-files --incremental

      # Full re-scan from scratch
      gitview track-files --no-incremental

      # Track changes since specific commit
      gitview track-files --since-commit abc1234

    \b
    Output Structure:
      output/file_histories/
        ├── checkpoint.json           # Incremental processing state
        ├── index.json                # Summary of all files
        └── files/
            ├── src/
            │   ├── main.py.json      # Machine-readable history
            │   ├── main.py.history   # Human-readable history
            │   └── ...
            └── ...

    \b
    Next Steps:
      After tracking files, you can:
      - View file history: gitview file-history <path>
      - Inject headers: gitview inject-history <path>  [Phase 3]
      - Add AI summaries: --with-ai flag  [Phase 2]
    """
    console.print("\n[bold cyan]GitView File History Tracker[/bold cyan]")
    console.print("=" * 70)

    # Handle remote repositories (reuse existing logic)
    from .remote import RemoteRepoHandler
    remote_handler = None

    if not os.path.isdir(repo):
        console.print(f"\n[yellow]Remote repository detected: {repo}[/yellow]")
        remote_handler = RemoteRepoHandler(repo)

        try:
            with console.status("[bold green]Handling remote repository..."):
                repo = remote_handler.get_local_path()
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            sys.exit(1)

    # Verify repository exists
    if not os.path.isdir(os.path.join(repo, '.git')):
        console.print(f"[bold red]Error:[/bold red] Not a git repository: {repo}")
        sys.exit(1)

    console.print(f"\n[green]Repository:[/green] {repo}")
    console.print(f"[green]Output directory:[/green] {output}")

    # Parse patterns
    file_patterns = None
    if patterns:
        file_patterns = [p.strip() for p in patterns.split(',')]
        console.print(f"[green]Include patterns:[/green] {', '.join(file_patterns)}")

    exclude_patterns = None
    if exclude:
        exclude_patterns = [p.strip() for p in exclude.split(',')]
        console.print(f"[green]Exclude patterns:[/green] {', '.join(exclude_patterns)}")

    if since_commit:
        console.print(f"[green]Since commit:[/green] {since_commit[:7]}")

    console.print(f"[green]Incremental:[/green] {incremental}")
    console.print()

    # Initialize tracker
    try:
        tracker = FileHistoryTracker(repo_path=repo, output_dir=output)
    except Exception as e:
        console.print(f"[bold red]Error initializing tracker:[/bold red] {e}")
        sys.exit(1)

    # Track files
    try:
        with console.status("[bold green]Tracking file histories..."):
            summary = tracker.track_all_files(
                file_patterns=file_patterns,
                exclude_patterns=exclude_patterns,
                since_commit=since_commit,
                incremental=incremental,
                max_history_entries=max_entries
            )

        # Display results
        console.print("\n[bold green]✓ Tracking Complete[/bold green]")
        console.print("=" * 70)

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")

        table.add_row("Files tracked", str(summary['files_tracked']))
        table.add_row("Total changes", f"{summary['total_changes']:,}")
        table.add_row("Checkpoint commit", summary['checkpoint_commit'])
        table.add_row("Checkpoint date", summary['checkpoint_date'][:10])
        table.add_row("Output directory", summary['output_dir'])

        console.print(table)

        # Show some sample files
        console.print("\n[bold cyan]Generated Files:[/bold cyan]")
        console.print(f"  • checkpoint.json")
        console.print(f"  • index.json")
        console.print(f"  • files/*.history (human-readable)")
        console.print(f"  • files/*.json (machine-readable)")

        console.print("\n[bold cyan]Next Steps:[/bold cyan]")
        console.print("  • View file history: [green]gitview file-history <path>[/green]")
        console.print("  • Incremental update: [green]gitview track-files --incremental[/green]")
        console.print("  • Add AI summaries: [yellow]Coming in Phase 2[/yellow]")
        console.print()

    except Exception as e:
        console.print(f"\n[bold red]Error during tracking:[/bold red] {e}")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)

    finally:
        # Cleanup remote repo if needed
        if remote_handler:
            remote_handler.cleanup()


@cli.command('file-history')
@click.argument('file_path')
@click.option('--repo', default='.', help='Path to repository')
@click.option('--output', default='output/file_histories',
              help='Output directory where histories are stored')
@click.option('--format', 'output_format', default='text',
              type=click.Choice(['text', 'json'], case_sensitive=False),
              help='Output format (text or json)')
@click.option('--recent', default=None, type=int,
              help='Show only N most recent changes')
def file_history(file_path, repo, output, output_format, recent):
    """Display change history for a specific file.

    \b
    Shows detailed timeline of changes including:
      - Commit metadata (hash, date, author)
      - Lines added/removed
      - Commit messages
      - Function/class changes (when available)

    \b
    Examples:

      # Show history for a file
      gitview file-history src/main.py

      # Show only last 5 changes
      gitview file-history src/main.py --recent 5

      # Export to JSON
      gitview file-history src/main.py --format json > history.json
    """
    tracker = FileHistoryTracker(repo_path=repo, output_dir=output)

    try:
        history = tracker.get_file_history(file_path)

        if not history:
            console.print(f"[yellow]No history found for:[/yellow] {file_path}")
            console.print(f"\nMake sure you've run: [green]gitview track-files[/green]")
            sys.exit(1)

        if output_format == 'json':
            # JSON output
            import json
            print(json.dumps(history.to_dict(), indent=2))
        else:
            # Text output
            console.print()
            console.print("=" * 80)
            console.print(f"[bold cyan]FILE HISTORY: {file_path}[/bold cyan]")
            console.print("=" * 80)

            console.print(f"\n[green]First seen:[/green] {history.first_commit_date[:10]} (commit {history.first_commit[:7]})")
            console.print(f"[green]Last modified:[/green] {history.last_commit_date[:10]} (commit {history.last_commit[:7]})")
            console.print(f"[green]Total commits:[/green] {history.total_commits}")
            console.print(f"[green]Lines added:[/green] +{history.total_lines_added}")
            console.print(f"[green]Lines removed:[/green] -{history.total_lines_removed}")
            console.print(f"[green]Net change:[/green] {history.total_lines_added - history.total_lines_removed:+d} lines")

            # Contributors
            if history.authors:
                console.print(f"\n[bold cyan]Contributors:[/bold cyan]")
                for author in history.authors[:5]:
                    name = author.get('name', 'Unknown')
                    count = author.get('commits', 0)
                    pct = (count / history.total_commits * 100) if history.total_commits > 0 else 0
                    console.print(f"  {name:30} {count:3} commits ({pct:5.1f}%)")

            # Changes
            display_count = recent if recent else len(history.changes)
            display_count = min(display_count, len(history.changes))

            console.print(f"\n[bold cyan]Recent Changes - Last {display_count} of {history.total_commits}[/bold cyan]")
            console.print("-" * 80)

            for i, change in enumerate(history.changes[:display_count]):
                console.print(f"\n[yellow]{change.commit_date[:19]}[/yellow] | [cyan]{change.commit_hash[:7]}[/cyan] | {change.author_name}")

                # Commit message
                msg_lines = change.commit_message.strip().split('\n')
                console.print(f"  {msg_lines[0]}")

                # Stats
                console.print(f"  [green]+{change.lines_added}[/green] [red]-{change.lines_removed}[/red] lines")

            console.print("\n" + "=" * 80)

            # Show where full data is
            json_path = Path(output) / "files" / f"{file_path}.json"
            history_path = Path(output) / "files" / f"{file_path}.history"
            console.print(f"\n[dim]Full history available at:[/dim]")
            console.print(f"  [dim]• {json_path}[/dim]")
            console.print(f"  [dim]• {history_path}[/dim]")
            console.print()

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


def main():
    """Main entry point."""
    cli()


if __name__ == '__main__':
    main()
