"""Output file writing handler.

Handles writing all output files including markdown reports,
JSON data, and timelines.
"""

from ..analyzer import AnalysisContext
from .base import BaseHandler, HandlerError


class OutputWriterHandler(BaseHandler):
    """Handles writing output files.

    This handler writes all analysis outputs:
    - Markdown narrative report
    - JSON data file with metadata
    - Timeline markdown file
    """

    def execute(self, context: AnalysisContext) -> None:
        """Write all output files.

        Args:
            context: Analysis context with results to write

        Raises:
            HandlerError: If writing fails
        """
        self._log_bold("Step 5: Writing output files...")

        try:
            if self.config.skip_llm:
                self._write_timeline_only(context)
            else:
                self._write_all_outputs(context)

        except Exception as e:
            raise HandlerError(f"Failed to write output files: {e}") from e

    def _write_timeline_only(self, context: AnalysisContext) -> None:
        """Write only timeline (when LLM is skipped).

        Args:
            context: Analysis context with phases
        """
        from ..writer import OutputWriter

        self._log_info("Skipping LLM summarization. Writing basic timeline...")

        timeline_path = self.config.output_dir / "timeline.md"
        OutputWriter.write_simple_timeline(context.phases, str(timeline_path))

        self._log_success(f"Wrote timeline to {timeline_path}\n")

    def _write_all_outputs(self, context: AnalysisContext) -> None:
        """Write all output files (markdown, JSON, timeline).

        Args:
            context: Analysis context with complete results
        """
        from ..writer import OutputWriter

        # Write markdown report
        markdown_path = self.config.output_dir / "history_story.md"
        OutputWriter.write_markdown(
            context.stories,
            context.phases,
            str(markdown_path),
            self.config.repo_name
        )
        self._log_success(f"Wrote {markdown_path}")

        # Write JSON data
        json_path = self.config.output_dir / "history_data.json"
        OutputWriter.write_json(
            context.stories,
            context.phases,
            str(json_path),
            repo_path=str(self.config.repo_path)
        )
        self._log_success(f"Wrote {json_path}")

        # Write timeline
        timeline_path = self.config.output_dir / "timeline.md"
        OutputWriter.write_simple_timeline(context.phases, str(timeline_path))
        self._log_success(f"Wrote {timeline_path}\n")
