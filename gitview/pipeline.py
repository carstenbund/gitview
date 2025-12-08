"""Analysis pipeline orchestrator.

This module provides the main AnalysisPipeline class that orchestrates
the complete analysis workflow by coordinating specialized handlers.
"""

import time
from typing import List, Optional, Dict, Any
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


def estimate_analysis_cost(
    commit_count: int,
    avg_msg_length: int,
    backend: str,
    model: str
) -> Dict[str, Any]:
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
        for i, handler in enumerate(self._handlers):
            if handler.should_execute(context):
                handler.execute(context)

                # Display cost estimate after extraction, before other LLM work
                if i == 0 and not self.config.skip_llm and len(context.records) > 0:
                    self._display_cost_estimate(context)

    def _display_cost_estimate(self, context: AnalysisContext) -> None:
        """Display cost estimate after extraction.

        Args:
            context: Analysis context with extracted records
        """
        from .backends.router import LLMRouter

        # Calculate average message length
        if context.records:
            avg_msg_length = sum(len(r.commit_message) for r in context.records) // len(context.records)
        else:
            avg_msg_length = 100

        # Get backend and model
        router = LLMRouter(
            backend=self.config.backend,
            model=self.config.model,
            api_key=self.config.api_key,
            ollama_url=self.config.ollama_url
        )

        estimate = estimate_analysis_cost(
            commit_count=len(context.records),
            avg_msg_length=avg_msg_length,
            backend=router.backend_type.value,
            model=router.model
        )

        # Show cost estimate
        if estimate['cost_usd'] > 0:
            self.console.print(f"\n[bold]Cost Estimate:[/bold]")
            self.console.print(f"  Backend: {estimate['backend']} / {estimate['model']}")
            self.console.print(
                f"  Estimated tokens: ~{estimate['input_tokens']:,} input + "
                f"~{estimate['output_tokens']:,} output"
            )
            self.console.print(f"  Estimated cost: [yellow]${estimate['cost_usd']:.2f}[/yellow]")
            self.console.print(
                f"  ({estimate['num_phases']} phases to summarize + story generation)"
            )

            # Suggest alternatives if cost is high
            if estimate['cost_usd'] > 2.0:
                self.console.print(f"\n[cyan]💡 To reduce costs:[/cyan]")
                if estimate['backend'] == 'anthropic':
                    self.console.print(
                        f"  • Use OpenAI gpt-4o-mini: --backend openai (~4-10x cheaper)"
                    )
                self.console.print(f"  • Use larger chunks: --strategy fixed --chunk-size 100")
                self.console.print(f"  • Limit commits: --max-commits 500")
            self.console.print()

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
