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

    Extract, analyze, and generate compelling narratives from your git repository's history.
    """
    pass


@cli.command()
@click.option('--repo', '-r', default=".", help="Path to git repository")
@click.option('--output', '-o', default="output", help="Output directory")
@click.option('--strategy', '-s', type=click.Choice(['fixed', 'time', 'adaptive']),
              default='adaptive', help="Chunking strategy")
@click.option('--chunk-size', type=int, default=50,
              help="Chunk size for fixed strategy")
@click.option('--max-commits', type=int, help="Maximum commits to analyze")
@click.option('--branch', default='HEAD', help="Branch to analyze")
@click.option('--backend', '-b', type=click.Choice(['anthropic', 'openai', 'ollama']),
              help="LLM backend (auto-detected from environment if not specified)")
@click.option('--model', '-m', help="Model identifier (uses backend defaults if not specified)")
@click.option('--api-key', help="API key for the backend (defaults to env var)")
@click.option('--ollama-url', default='http://localhost:11434', help="Ollama API URL")
@click.option('--repo-name', help="Repository name for output")
@click.option('--skip-llm', is_flag=True, help="Skip LLM summarization (extract and chunk only)")
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


@cli.command()
@click.option('--repo', '-r', default=".", help="Path to git repository")
@click.option('--output', '-o', default="output/repo_history.jsonl",
              help="Output JSONL file")
@click.option('--max-commits', type=int, help="Maximum commits to extract")
@click.option('--branch', default='HEAD', help="Branch to extract from")
def extract(repo, output, max_commits, branch):
    """Extract git history to JSONL file.

    This command only extracts the git history without chunking or summarization.
    """
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


@cli.command()
@click.argument('history_file', type=click.Path(exists=True))
@click.option('--output', '-o', default="output/phases", help="Output directory for phases")
@click.option('--strategy', '-s', type=click.Choice(['fixed', 'time', 'adaptive']),
              default='adaptive', help="Chunking strategy")
@click.option('--chunk-size', type=int, default=50, help="Chunk size for fixed strategy")
def chunk(history_file, output, strategy, chunk_size):
    """Chunk extracted history into phases.

    Takes a JSONL file from the extract command and chunks it into phases.
    """
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
            events.append("🗑️")
        if phase.has_large_addition:
            events.append("➕")
        if phase.has_refactor:
            events.append("♻️")
        if phase.readme_changed:
            events.append("📝")

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
