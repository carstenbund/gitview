"""Command-line interface for GitView."""

import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .extractor import GitHistoryExtractor
from .chunker import HistoryChunker
from .summarizer import PhaseSummarizer
from .storyteller import StoryTeller
from .writer import OutputWriter

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
"""


@cli.command(help=ANALYZE_HELP)
@click.option('--repo', '-r', default=".",
              help="Path to git repository (default: current directory)")
@click.option('--output', '-o', default="output",
              help="Output directory for reports and data")
@click.option('--strategy', '-s', type=click.Choice(['fixed', 'time', 'adaptive']),
              default='adaptive',
              help="Chunking strategy: 'adaptive' (default, splits on significant changes), "
                   "'fixed' (N commits per phase), 'time' (by time period)")
@click.option('--chunk-size', type=int, default=50,
              help="Commits per chunk when using 'fixed' strategy")
@click.option('--max-commits', type=int,
              help="Maximum commits to analyze (default: all commits)")
@click.option('--branch', default='HEAD',
              help="Branch to analyze (default: HEAD/current branch)")
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
def analyze(repo, output, strategy, chunk_size, max_commits, branch, backend,
           model, api_key, ollama_url, repo_name, skip_llm):
    """Analyze git repository and generate narrative history.

    This is the main command that runs the full pipeline:
    1. Extract git history
    2. Chunk into meaningful phases
    3. Summarize each phase with LLM
    4. Generate global narrative
    5. Write output files
    """
    console.print("\n[bold blue]GitView - Repository History Analyzer[/bold blue]\n")

    # Validate repository
    repo_path = Path(repo).resolve()
    if not (repo_path / '.git').exists():
        console.print(f"[red]Error: {repo_path} is not a git repository[/red]")
        sys.exit(1)

    # Get repo name if not provided
    if not repo_name:
        repo_name = repo_path.name

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

    try:
        # Step 1: Extract git history
        console.print("[bold]Step 1: Extracting git history...[/bold]")
        extractor = GitHistoryExtractor(str(repo_path))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Extracting commits...", total=None)
            records = extractor.extract_history(max_commits=max_commits, branch=branch)
            progress.update(task, completed=True)

        console.print(f"[green]✓ Extracted {len(records)} commits[/green]\n")

        # Save raw history
        history_file = Path(output) / "repo_history.jsonl"
        extractor.save_to_jsonl(records, str(history_file))

        # Step 2: Chunk into phases
        console.print("[bold]Step 2: Chunking into phases...[/bold]")
        chunker = HistoryChunker(strategy)

        kwargs = {}
        if strategy == 'fixed':
            kwargs['chunk_size'] = chunk_size

        phases = chunker.chunk(records, **kwargs)
        console.print(f"[green]✓ Created {len(phases)} phases[/green]\n")

        # Display phase overview
        _display_phase_overview(phases)

        # Save phases
        phases_dir = Path(output) / "phases"
        chunker.save_phases(phases, str(phases_dir))

        if skip_llm:
            console.print("\n[yellow]Skipping LLM summarization. Writing basic timeline...[/yellow]")
            timeline_file = Path(output) / "timeline.md"
            OutputWriter.write_simple_timeline(phases, str(timeline_file))
            console.print(f"[green]✓ Wrote timeline to {timeline_file}[/green]\n")
            return

        # Step 3: Summarize phases with LLM
        console.print("[bold]Step 3: Summarizing phases with LLM...[/bold]")
        summarizer = PhaseSummarizer(
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
            task = progress.add_task("Summarizing phases...", total=len(phases))

            previous_summaries = []
            for i, phase in enumerate(phases):
                progress.update(task, description=f"Summarizing phase {i+1}/{len(phases)}...")

                context = summarizer._build_context(previous_summaries)
                summary = summarizer.summarize_phase(phase, context)
                phase.summary = summary

                previous_summaries.append({
                    'phase_number': phase.phase_number,
                    'summary': summary,
                    'loc_delta': phase.loc_delta,
                })

                summarizer._save_phase_with_summary(phase, str(phases_dir))
                progress.update(task, advance=1)

        console.print(f"[green]✓ Summarized all phases[/green]\n")

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

        console.print(f"[green]✓ Generated global narrative[/green]\n")

        # Step 5: Write output
        console.print("[bold]Step 5: Writing output files...[/bold]")
        output_path = Path(output)

        # Write markdown report
        markdown_path = output_path / "history_story.md"
        OutputWriter.write_markdown(stories, phases, str(markdown_path), repo_name)
        console.print(f"[green]✓ Wrote {markdown_path}[/green]")

        # Write JSON data
        json_path = output_path / "history_data.json"
        OutputWriter.write_json(stories, phases, str(json_path))
        console.print(f"[green]✓ Wrote {json_path}[/green]")

        # Write timeline
        timeline_path = output_path / "timeline.md"
        OutputWriter.write_simple_timeline(phases, str(timeline_path))
        console.print(f"[green]✓ Wrote {timeline_path}[/green]\n")

        # Success summary
        console.print("[bold green]✓ Analysis complete![/bold green]\n")
        console.print(f"📊 Analyzed {len(records)} commits across {len(phases)} phases")
        console.print(f"📝 Output written to: {output_path.resolve()}\n")

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


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

        console.print(f"\n[green]✓ Extracted {len(records)} commits to {output}[/green]\n")

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

        console.print(f"[green]✓ Created {len(phases)} phases[/green]\n")
        _display_phase_overview(phases)

        # Save
        chunker.save_phases(phases, output)
        console.print(f"\n[green]✓ Saved phases to {output}[/green]\n")

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
