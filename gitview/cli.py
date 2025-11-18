"""Command-line interface for GitView."""

import os
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .extractor import GitHistoryExtractor
from .chunker import HistoryChunker
from .summarizer import PhaseSummarizer
from .storyteller import StoryTeller
from .writer import OutputWriter
from .remote import RemoteRepoHandler
from .branches import BranchManager, parse_branch_spec
from .index_writer import IndexWriter

console = Console()


@click.group()
@click.version_option(version="0.1.0")
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
  2. Chunk commits into meaningful phases/epochs
  3. Summarize each phase using LLM
  4. Generate global narrative stories
  5. Write markdown reports and JSON data

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
  - Automatically clones to temporary directory
  - Cleans up temp clone after analysis (use --keep-clone to preserve)
  - Outputs to ~/Documents/gitview/github.com/org/repo/ by default
  - Warns if repository is very large (>100MB)

\b
LLM BACKEND CONFIGURATION:

GitView supports three LLM backends:

\b
1. Anthropic Claude (default, requires API key):
   export ANTHROPIC_API_KEY="your-key"
   gitview analyze

   Or: gitview analyze --backend anthropic --api-key "your-key"

   Default model: claude-sonnet-4-5-20250929
   Other models: claude-3-opus-20240229, claude-3-haiku-20240307

\b
2. OpenAI GPT (requires API key):
   export OPENAI_API_KEY="your-key"
   gitview analyze --backend openai

   Default model: gpt-4
   Other models: gpt-4-turbo-preview, gpt-3.5-turbo

\b
3. Ollama (local, FREE, no API key needed):
   # Start Ollama server first: ollama serve
   # Pull a model: ollama pull llama3
   gitview analyze --backend ollama --model llama3

   Popular models: llama3, mistral, codellama, mixtral
   Default URL: http://localhost:11434

\b
Backend auto-detection:
  If no --backend is specified, GitView checks environment variables:
  - If ANTHROPIC_API_KEY is set → uses Anthropic
  - If OPENAI_API_KEY is set → uses OpenAI
  - Otherwise → uses Ollama (local)

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

  # Analyze last 100 commits only
  gitview analyze --max-commits 100

  # Adaptive chunking (default, splits on significant changes)
  gitview analyze --strategy adaptive

  # Fixed-size chunks (50 commits per phase)
  gitview analyze --strategy fixed --chunk-size 50

  # Use custom Ollama server
  gitview analyze --backend ollama --ollama-url http://192.168.1.100:11434

\b
INCREMENTAL ANALYSIS (Cost-Efficient Ongoing Monitoring):

  For managers analyzing multiple projects on an ongoing basis, incremental
  analysis dramatically reduces costs by reusing previous LLM summaries.

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
"""


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
    since_date: Optional[str]
):
    """Analyze a single branch (helper function for multi-branch support)."""
    from typing import Optional

    # Check for incremental analysis
    previous_analysis = None
    existing_phases = []
    starting_loc = 0

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

    # Step 1: Extract git history
    console.print("[bold]Step 1: Extracting git history...[/bold]")
    extractor = GitHistoryExtractor(repo_path)

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
    summarizer = PhaseSummarizer(
        backend=backend,
        model=model,
        api_key=api_key,
        ollama_url=ollama_url
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
                # Need to summarize this phase
                context = summarizer._build_context(previous_summaries)
                summary = summarizer.summarize_phase(phase, context)
                phase.summary = summary
                progress.update(task, advance=1)

            previous_summaries.append({
                'phase_number': phase.phase_number,
                'summary': phase.summary,
                'loc_delta': phase.loc_delta,
            })

            summarizer._save_phase_with_summary(phase, str(phases_dir))

    if len(phases_to_summarize) > 0:
        console.print(f"[green]Summarized {len(phases_to_summarize)} phase(s)[/green]\n")
    else:
        console.print(f"[green]All phases already summarized[/green]\n")

    # Step 4: Generate global story
    console.print("[bold]Step 4: Generating global narrative...[/bold]")
    storyteller = StoryTeller(
        backend=backend,
        model=model,
        api_key=api_key,
        ollama_url=ollama_url
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
    OutputWriter.write_json(stories, phases, str(json_path), repo_path=repo_path)
    console.print(f"[green]Wrote {json_path}[/green]")

    # Write timeline
    timeline_path = output_path / "timeline.md"
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
              help="Keep temporary clone of remote repository (default: cleanup after analysis)")
def analyze(repo, output, strategy, chunk_size, max_commits, branch, list_branches,
           branches, all_branches, exclude_branches, backend, model, api_key, ollama_url,
           repo_name, skip_llm, incremental, since_commit, since_date, keep_clone):
    """Analyze git repository and generate narrative history.

    This is the main command that runs the full pipeline:
    1. Extract git history
    2. Chunk into meaningful phases
    3. Summarize each phase with LLM
    4. Generate global narrative
    5. Write output files
    """
    console.print("\n[bold blue]GitView - Repository History Analyzer[/bold blue]\n")

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
                console.print(f"\n[bold cyan]═══ Analyzing Branch {idx}/{len(branches_to_analyze)}: {branch_info.name} ═══[/bold cyan]\n")

                # Set branch-specific output directory
                branch_output = base_output_dir / branch_info.sanitized_name
                current_branch = branch_info.name

                # Analyze this branch (call single-branch logic)
                _analyze_single_branch(
                    repo_path=str(repo_path),
                    branch=current_branch,
                    output=str(branch_output),
                    repo_name=f"{repo_name} ({branch_info.short_name})",
                    strategy=strategy,
                    chunk_size=chunk_size,
                    max_commits=max_commits,
                    backend=backend,
                    model=model,
                    api_key=api_key,
                    ollama_url=ollama_url,
                    skip_llm=skip_llm,
                    incremental=incremental,
                    since_commit=since_commit,
                    since_date=since_date
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

        # Step 1: Extract git history
        console.print("[bold]Step 1: Extracting git history...[/bold]")
        extractor = GitHistoryExtractor(str(repo_path))

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
        summarizer = PhaseSummarizer(
            backend=backend,
            model=model,
            api_key=api_key,
            ollama_url=ollama_url
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
                    # Need to summarize this phase
                    context = summarizer._build_context(previous_summaries)
                    summary = summarizer.summarize_phase(phase, context)
                    phase.summary = summary
                    progress.update(task, advance=1)

                previous_summaries.append({
                    'phase_number': phase.phase_number,
                    'summary': phase.summary,
                    'loc_delta': phase.loc_delta,
                })

                summarizer._save_phase_with_summary(phase, str(phases_dir))

        if len(phases_to_summarize) > 0:
            console.print(f"[green]Summarized {len(phases_to_summarize)} phase(s)[/green]\n")
        else:
            console.print(f"[green]All phases already summarized[/green]\n")

        # Step 4: Generate global story
        console.print("[bold]Step 4: Generating global narrative...[/bold]")
        storyteller = StoryTeller(
            backend=backend,
            model=model,
            api_key=api_key,
            ollama_url=ollama_url
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
    table.add_column("LOC Δ", justify="right")
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


def main():
    """Main entry point."""
    cli()


if __name__ == '__main__':
    main()
