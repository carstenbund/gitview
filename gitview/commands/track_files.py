"""Track files command implementation."""

import os
import sys
from pathlib import Path

import click

from .base import BaseCommand
from ..file_tracker import FileHistoryTracker


class TrackFilesCommand(BaseCommand):
    """Track detailed change history for all files in repository.

    This command generates:
      - .history companion files for each tracked file (human-readable)
      - .json files with complete structured history (machine-readable)
      - index.json with summary of all tracked files
      - checkpoint.json for incremental processing
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._remote_handler = None
        self._llm_router = None
        self._tracker = None

    def validate(self) -> None:
        """Validate command options."""
        repo = self.get_option('repo', '.')

        # Handle remote repositories
        if not os.path.isdir(repo):
            from ..remote import RemoteRepoHandler
            self.print_warning(f"Remote repository detected: {repo}")
            self._remote_handler = RemoteRepoHandler(repo)

            try:
                with self.console.status("[bold green]Handling remote repository..."):
                    repo = str(self._remote_handler.get_local_path())
                    self.options['repo'] = repo
            except Exception as e:
                self.print_error(f"Error: {e}")
                sys.exit(1)

        # Verify repository exists
        if not os.path.isdir(os.path.join(repo, '.git')):
            self.print_error(f"Error: Not a git repository: {repo}")
            sys.exit(1)

    def execute(self):
        """Execute file tracking."""
        self.console.print("\n[bold cyan]GitView File History Tracker[/bold cyan]")
        self.console.print("=" * 70)

        repo = self.get_option('repo', '.')
        output = self.get_option('output', 'output/file_histories')
        patterns = self.get_option('patterns')
        exclude = self.get_option('exclude')
        since_commit = self.get_option('since_commit')
        incremental = self.get_option('incremental', True)
        max_entries = self.get_option('max_entries', 100)
        with_ai = self.get_option('with_ai', False)
        backend = self.get_option('backend')
        model = self.get_option('model')

        self.console.print(f"\n[green]Repository:[/green] {repo}")
        self.console.print(f"[green]Output directory:[/green] {output}")

        # Parse patterns
        file_patterns = None
        if patterns:
            file_patterns = [p.strip() for p in patterns.split(',')]
            self.console.print(f"[green]Include patterns:[/green] {', '.join(file_patterns)}")

        exclude_patterns = None
        if exclude:
            exclude_patterns = [p.strip() for p in exclude.split(',')]
            self.console.print(f"[green]Exclude patterns:[/green] {', '.join(exclude_patterns)}")

        if since_commit:
            self.console.print(f"[green]Since commit:[/green] {since_commit[:7]}")

        self.console.print(f"[green]Incremental:[/green] {incremental}")
        self.console.print()

        # Initialize LLM router if AI summaries requested
        if with_ai:
            self._setup_llm_router(backend, model)

        # Initialize tracker
        try:
            self._tracker = FileHistoryTracker(
                repo_path=repo,
                output_dir=output,
                llm_router=self._llm_router
            )
        except Exception as e:
            self.print_error(f"Error initializing tracker: {e}")
            sys.exit(1)

        # Estimate cost if AI summaries requested
        if with_ai and self._llm_router:
            if not self._estimate_and_confirm_cost(file_patterns, exclude_patterns, since_commit, repo, output):
                with_ai = False
                self._llm_router = None
                self._tracker = FileHistoryTracker(repo_path=repo, output_dir=output, llm_router=None)

        # Track files
        try:
            with self.console.status("[bold green]Tracking file histories..."):
                summary = self._tracker.track_all_files(
                    file_patterns=file_patterns,
                    exclude_patterns=exclude_patterns,
                    since_commit=since_commit,
                    incremental=incremental,
                    max_history_entries=max_entries,
                    with_ai_summaries=with_ai
                )

            self._display_results(summary, with_ai)

        except Exception as e:
            self.print_error(f"Error during tracking: {e}")
            import traceback
            self.console.print(traceback.format_exc())
            sys.exit(1)

        finally:
            if self._remote_handler:
                self._remote_handler.cleanup()

    def _setup_llm_router(self, backend, model):
        """Setup LLM router for AI summaries."""
        self.console.print(f"[yellow]AI Summaries:[/yellow] Enabled")
        try:
            from ..backends.router import LLMRouter
            self._llm_router = LLMRouter(backend=backend, model=model)
            self.console.print(f"[green]LLM Backend:[/green] {self._llm_router.backend_type.value}")
            self.console.print(f"[green]LLM Model:[/green] {self._llm_router.model}")
        except Exception as e:
            self.print_error(f"Error initializing LLM: {e}")
            self.print_warning("Hint: Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
            sys.exit(1)
        self.console.print()

    def _estimate_and_confirm_cost(self, file_patterns, exclude_patterns, since_commit, repo, output):
        """Estimate cost and get confirmation for AI summaries."""
        self.print_info("Estimating AI summary cost...")
        from ..file_tracker import FileChangeExtractor
        extractor = FileChangeExtractor(repo)
        tracked_files = extractor.get_tracked_files(
            patterns=file_patterns,
            exclude_patterns=exclude_patterns
        )

        # Estimate changes
        estimated_changes = len(tracked_files) * 5
        if since_commit:
            estimated_changes = min(estimated_changes, len(tracked_files) * 2)

        cost_estimate = self._tracker.ai_summarizer.estimate_cost(estimated_changes)

        self.console.print(f"\n[bold yellow]Cost Estimate:[/bold yellow]")
        self.console.print(f"  Estimated changes: ~{estimated_changes}")
        self.console.print(f"  Model: {cost_estimate['model']}")
        self.console.print(f"  Estimated cost: ${cost_estimate['total_cost_usd']:.3f}")
        self.console.print(f"  Cost per change: ${cost_estimate['cost_per_change']:.4f}")

        if cost_estimate['total_cost_usd'] > 1.0:
            self.print_error("⚠ Warning: Estimated cost is over $1.00")

        if not click.confirm("\nProceed with AI summary generation?", default=True):
            self.print_warning("Skipping AI summaries. Running without --with-ai flag.")
            return False

        self.console.print()
        return True

    def _display_results(self, summary, with_ai):
        """Display tracking results."""
        self.print_success("\n✓ Tracking Complete")
        self.console.print("=" * 70)

        table = self.create_table()
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")

        table.add_row("Files tracked", str(summary['files_tracked']))
        table.add_row("Total changes", f"{summary['total_changes']:,}")
        table.add_row("Checkpoint commit", summary['checkpoint_commit'])
        table.add_row("Checkpoint date", summary['checkpoint_date'][:10])
        table.add_row("Output directory", summary['output_dir'])

        self.console.print(table)

        self.console.print("\n[bold cyan]Generated Files:[/bold cyan]")
        self.console.print("  • checkpoint.json")
        self.console.print("  • index.json")
        self.console.print("  • files/*.history (human-readable)")
        self.console.print("  • files/*.json (machine-readable)")

        self.console.print("\n[bold cyan]Next Steps:[/bold cyan]")
        self.console.print("  • View file history: [green]gitview file-history <path>[/green]")
        self.console.print("  • Incremental update: [green]gitview track-files --incremental[/green]")
        if not with_ai:
            self.console.print("  • Add AI summaries: [green]gitview track-files --with-ai[/green]")
        self.console.print("  • Inject headers: [green]gitview inject-history <path>[/green]")
        self.console.print("  • Compare branches: [green]gitview compare-branches <branch1> <branch2>[/green]")
        self.console.print()
