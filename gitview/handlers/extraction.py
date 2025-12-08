"""Git history extraction handler.

Handles extracting commit history from repositories with support
for caching and incremental analysis.
"""

from rich.progress import Progress, SpinnerColumn, TextColumn

from ..analyzer import AnalysisContext
from .base import BaseHandler, HandlerError


class ExtractionHandler(BaseHandler):
    """Handles git history extraction with caching and incremental support.

    This handler extracts commits from the repository, supporting:
    - Full history extraction
    - Incremental extraction (since commit/date)
    - Caching of previous extractions
    """

    def execute(self, context: AnalysisContext) -> None:
        """Extract git history into context.records.

        Args:
            context: Analysis context to populate with commit records

        Raises:
            HandlerError: If extraction fails
        """
        self._log_bold("Step 1: Extracting git history...")

        try:
            # 1. Try to load cached records (if not incremental)
            if not context.is_incremental():
                self._try_load_cached_records(context)

            # 2. If cached, use them; otherwise extract
            if context.cached_records is not None:
                context.records = context.cached_records
                self._log_info("Using cached commit history from previous run.\n")
            else:
                self._extract_commits(context)

            # 3. Save to JSONL
            self._save_history(context)

            # 4. Display summary
            self._display_summary(context)

        except Exception as e:
            raise HandlerError(f"Failed to extract git history: {e}") from e

    def _try_load_cached_records(self, context: AnalysisContext) -> None:
        """Attempt to load cached records from previous run.

        Args:
            context: Analysis context to populate with cached records
        """
        if context.history_file and context.history_file.exists():
            try:
                from ..extractor import GitHistoryExtractor
                context.cached_records = GitHistoryExtractor.load_from_jsonl(
                    str(context.history_file)
                )
            except Exception as e:
                self._log_warning(f"Failed to load cached history: {e}")

    def _extract_commits(self, context: AnalysisContext) -> None:
        """Extract commits from repository.

        Args:
            context: Analysis context to populate with extracted commits
        """
        from ..extractor import GitHistoryExtractor

        extractor = GitHistoryExtractor(str(self.config.repo_path))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Extracting commits...", total=None)

            if context.is_incremental():
                records = extractor.extract_incremental(
                    since_commit=self.config.since_commit,
                    since_date=self.config.since_date,
                    branch=self.config.branch
                )
                # Adjust LOC for continuation
                if context.starting_loc > 0:
                    extractor._calculate_cumulative_loc(records, context.starting_loc)
            else:
                records = extractor.extract_history(
                    max_commits=self.config.max_commits,
                    branch=self.config.branch
                )

            context.records = records
            progress.update(task, completed=True)

    def _save_history(self, context: AnalysisContext) -> None:
        """Save extracted history to JSONL file.

        Args:
            context: Analysis context with records to save
        """
        from ..extractor import GitHistoryExtractor

        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        extractor = GitHistoryExtractor(str(self.config.repo_path))
        extractor.save_to_jsonl(context.records, str(context.history_file))

    def _display_summary(self, context: AnalysisContext) -> None:
        """Display extraction summary.

        Args:
            context: Analysis context with extraction results
        """
        count = len(context.records)
        if context.is_incremental():
            self._log_success(f"Extracted {count} new commits\n")
            if count == 0:
                self._log_info("No new commits since last analysis.")
        else:
            self._log_success(f"Extracted {count} commits\n")
