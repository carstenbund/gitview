"""Global narrative generation handler.

Handles generating the global narrative story from phase summaries
using LLM.
"""

from rich.progress import Progress, SpinnerColumn, TextColumn

from ..analyzer import AnalysisContext
from .base import BaseHandler, HandlerError


class StorytellerHandler(BaseHandler):
    """Handles global narrative generation.

    This handler generates the overarching narrative story that ties
    together all the phase summaries into a cohesive history.
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
        """Generate global narrative stories.

        Args:
            context: Analysis context with summarized phases

        Raises:
            HandlerError: If story generation fails
        """
        self._log_bold("Step 4: Generating global narrative...")

        try:
            # 1. Create storyteller
            storyteller = self._create_storyteller()

            # 2. Generate stories
            context.stories = self._generate_stories(context, storyteller)

            self._log_success("Generated global narrative\n")

        except Exception as e:
            raise HandlerError(f"Failed to generate global narrative: {e}") from e

    def _create_storyteller(self):
        """Create and configure StoryTeller.

        Returns:
            Configured StoryTeller instance
        """
        from ..storyteller import StoryTeller

        return StoryTeller(
            backend=self.config.backend,
            model=self.config.model,
            api_key=self.config.api_key,
            ollama_url=self.config.ollama_url,
            todo_content=self.config.todo_content,
            critical_mode=self.config.critical_mode,
            directives=self.config.directives
        )

    def _generate_stories(self, context: AnalysisContext, storyteller) -> dict:
        """Generate global stories with progress indicator.

        Args:
            context: Analysis context with phases
            storyteller: Configured StoryTeller

        Returns:
            Dictionary of generated stories
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Generating story...", total=None)

            stories = storyteller.generate_global_story(
                context.phases,
                self.config.repo_name
            )

            progress.update(task, completed=True)

        return stories
