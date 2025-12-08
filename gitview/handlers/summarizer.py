"""Phase summarization handler.

Handles summarizing phases using LLM with support for incremental
analysis and context building.
"""

from rich.progress import Progress, SpinnerColumn, TextColumn

from ..analyzer import AnalysisContext
from .base import BaseHandler, HandlerError


class SummarizerHandler(BaseHandler):
    """Handles phase summarization with LLM.

    This handler summarizes each phase using the configured LLM backend,
    building context from previous summaries to maintain narrative continuity.
    Supports incremental analysis by reusing existing summaries.
    """

    def should_execute(self, context: AnalysisContext) -> bool:
        """Only execute if LLM is not skipped.

        Args:
            context: Analysis context

        Returns:
            True if handler should execute
        """
        return not self.config.skip_llm

    def execute(self, context: AnalysisContext) -> None:
        """Summarize phases with LLM.

        Args:
            context: Analysis context with phases to summarize

        Raises:
            HandlerError: If summarization fails
        """
        self._log_bold("Step 3: Summarizing phases with LLM...")

        try:
            # 1. Initialize summarizer
            summarizer = self._create_summarizer()

            # 2. Identify phases needing summarization
            phases_to_summarize = [p for p in context.phases if p.summary is None]

            if len(phases_to_summarize) < len(context.phases):
                self._log_info(
                    f"{len(phases_to_summarize)} phases need summarization "
                    f"({len(context.phases) - len(phases_to_summarize)} already summarized)"
                )

            # 3. Summarize phases
            self._summarize_phases(context, summarizer, phases_to_summarize)

            # 4. Display summary
            if len(phases_to_summarize) > 0:
                self._log_success(f"Summarized {len(phases_to_summarize)} phase(s)\n")
            else:
                self._log_success("All phases already summarized\n")

            context.phases_summarized = len(phases_to_summarize)

        except Exception as e:
            raise HandlerError(f"Failed to summarize phases: {e}") from e

    def _create_summarizer(self):
        """Create and configure PhaseSummarizer.

        Returns:
            Configured PhaseSummarizer instance
        """
        from ..summarizer import PhaseSummarizer

        return PhaseSummarizer(
            backend=self.config.backend,
            model=self.config.model,
            api_key=self.config.api_key,
            ollama_url=self.config.ollama_url,
            todo_content=self.config.todo_content,
            critical_mode=self.config.critical_mode,
            directives=self.config.directives
        )

    def _summarize_phases(
        self,
        context: AnalysisContext,
        summarizer,
        phases_to_summarize: list
    ) -> None:
        """Summarize all phases, reusing existing summaries.

        Args:
            context: Analysis context with phases
            summarizer: Configured PhaseSummarizer
            phases_to_summarize: List of phases needing summarization
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task(
                "Summarizing phases...",
                total=len(phases_to_summarize)
            )

            previous_summaries = []

            for i, phase in enumerate(context.phases):
                progress.update(
                    task,
                    description=f"Processing phase {i+1}/{len(context.phases)}..."
                )

                if phase.summary is None:
                    # Summarize this phase
                    context_str = summarizer._build_context(previous_summaries)
                    summary = summarizer.summarize_phase(phase, context_str)
                    phase.summary = summary
                    progress.update(task, advance=1)

                # Build context for next phase
                previous_summaries.append({
                    'phase_number': phase.phase_number,
                    'summary': phase.summary,
                    'loc_delta': phase.loc_delta,
                })

                # Save phase with summary
                summarizer._save_phase_with_summary(phase, str(context.phases_dir))
