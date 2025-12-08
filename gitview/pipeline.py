"""Analysis pipeline orchestrator.

This module provides the main AnalysisPipeline class that orchestrates
the complete analysis workflow by coordinating specialized handlers.
"""

import time
from typing import List, Optional
from rich.console import Console

from .analyzer import AnalysisConfig, AnalysisContext, AnalysisResult
from .handlers import (
    BaseHandler,
    ExtractionHandler,
    GitHubEnrichmentHandler,
    ChunkerHandler,
    SummarizerHandler,
    StorytellerHandler,
    OutputWriterHandler,
)


class AnalysisPipeline:
    """Orchestrates the complete analysis pipeline.

    This class coordinates all analysis handlers, managing the flow from
    git extraction through to final output generation. It handles:
    - Handler initialization and configuration
    - Context creation and management
    - Incremental analysis detection
    - Error handling and reporting
    - Progress tracking
    """

    def __init__(self, config: AnalysisConfig, console: Optional[Console] = None):
        """Initialize pipeline with configuration.

        Args:
            config: Analysis configuration
            console: Rich console for output (creates default if None)
        """
        self.config = config
        self.console = console or Console()

        # Initialize handlers
        self._handlers: List[BaseHandler] = []
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Initialize all handlers in execution order."""
        self._handlers = [
            ExtractionHandler(self.config, self.console),
            GitHubEnrichmentHandler(self.config, self.console),
            ChunkerHandler(self.config, self.console),
            SummarizerHandler(self.config, self.console),
            StorytellerHandler(self.config, self.console),
            OutputWriterHandler(self.config, self.console),
        ]

    def run(self) -> AnalysisResult:
        """Execute the complete analysis pipeline.

        This is the main entry point that runs all handlers in sequence,
        managing the shared context and handling errors gracefully.

        Returns:
            AnalysisResult with execution details and statistics
        """
        start_time = time.time()

        try:
            # 1. Validate configuration
            self.config.validate()

            # 2. Create context
            context = self._create_context()

            # 3. Load previous analysis (for incremental)
            self._load_previous_analysis(context)

            # 4. Execute handlers
            self._execute_handlers(context)

            # 5. Check for early exit
            if self._should_exit_early(context):
                return self._create_early_exit_result(context)

            # 6. Create success result
            duration = time.time() - start_time
            result = AnalysisResult.success_result(context, duration)

            # 7. Display success message
            self._display_success(context, result)

            return result

        except Exception as e:
            # Handle failure
            self.console.print(f"\n[red]Error: {e}[/red]")
            import traceback
            traceback.print_exc()
            return AnalysisResult.failure_result(e)

    def _create_context(self) -> AnalysisContext:
        """Create initial analysis context.

        Returns:
            Initialized AnalysisContext
        """
        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        return AnalysisContext(config=self.config)

    def _load_previous_analysis(self, context: AnalysisContext) -> None:
        """Load previous analysis for incremental runs.

        Args:
            context: Analysis context to populate with previous data
        """
        if not context.is_incremental():
            return

        from .writer import OutputWriter

        context.previous_analysis = OutputWriter.load_previous_analysis(
            str(self.config.output_dir)
        )

        if self.config.incremental and not context.previous_analysis:
            self.console.print(
                "[yellow]Warning: --incremental specified but no previous analysis found.[/yellow]"
            )
            self.console.print("[yellow]Running full analysis instead...[/yellow]\n")
            # Cannot modify frozen config, but context will handle it gracefully
            return

        if context.previous_analysis:
            self._load_existing_phases(context)

    def _load_existing_phases(self, context: AnalysisContext) -> None:
        """Load existing phases from previous analysis.

        Args:
            context: Analysis context to populate with existing phases
        """
        from .chunker import Phase

        metadata = context.previous_analysis.get('metadata', {})

        if self.config.incremental:
            last_hash = metadata.get('last_commit_hash')
            if last_hash:
                self.console.print(
                    f"[cyan]Incremental mode:[/cyan] Analyzing commits since {last_hash[:8]}"
                )
            self.console.print(
                f"[cyan]Last analysis:[/cyan] {metadata.get('generated_at', 'unknown')}\n"
            )

        # Load phases
        context.existing_phases = [
            Phase.from_dict(p)
            for p in context.previous_analysis.get('phases', [])
        ]

        # Calculate starting LOC
        if context.existing_phases and context.existing_phases[-1].commits:
            context.starting_loc = context.existing_phases[-1].commits[-1].loc_total

    def _execute_handlers(self, context: AnalysisContext) -> None:
        """Execute all handlers in sequence.

        Args:
            context: Shared analysis context
        """
        for handler in self._handlers:
            if handler.should_execute(context):
                handler.execute(context)

    def _should_exit_early(self, context: AnalysisContext) -> bool:
        """Check if we should exit early (e.g., no new commits in incremental).

        Args:
            context: Analysis context

        Returns:
            True if should exit early
        """
        if context.is_incremental() and len(context.records) == 0:
            self.console.print("[yellow]No new commits found since last analysis.[/yellow]")
            self.console.print("[green]Repository is up to date![/green]\n")
            return True
        return False

    def _create_early_exit_result(self, context: AnalysisContext) -> AnalysisResult:
        """Create result for early exit scenarios.

        Args:
            context: Analysis context

        Returns:
            AnalysisResult for early exit
        """
        return AnalysisResult.early_exit_result(context)

    def _display_success(self, context: AnalysisContext, result: AnalysisResult) -> None:
        """Display success message with summary.

        Args:
            context: Analysis context
            result: Analysis result
        """
        self.console.print("[bold green]Analysis complete![/bold green]\n")
        self.console.print(
            f"Analyzed {result.commits_analyzed} commits across {result.phases_created} phases"
        )
        self.console.print(f"Output written to: {self.config.output_dir.resolve()}\n")
