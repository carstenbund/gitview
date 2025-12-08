"""Phase chunking handler.

Handles chunking commits into meaningful phases with support for
different strategies and incremental analysis.
"""

from rich.table import Table

from ..analyzer import AnalysisContext
from .base import BaseHandler, HandlerError


class ChunkerHandler(BaseHandler):
    """Handles phase chunking with incremental support.

    This handler chunks commits into phases based on the configured strategy:
    - Adaptive: Split on significant changes
    - Fixed: Fixed number of commits per phase
    - Time: Split by time periods

    Also supports incremental analysis by merging or appending to existing phases.
    """

    def execute(self, context: AnalysisContext) -> None:
        """Chunk commits into phases.

        Args:
            context: Analysis context with records to chunk

        Raises:
            HandlerError: If chunking fails
        """
        self._log_bold("Step 2: Chunking into phases...")

        try:
            # 1. Try to load cached phases
            if not context.is_incremental():
                self._try_load_cached_phases(context)

            # 2. Handle different chunking scenarios
            if context.cached_phases is not None:
                context.phases = context.cached_phases
                self._log_info(f"Reusing {len(context.phases)} cached phases.\n")
            elif context.existing_phases and len(context.records) > 0:
                self._handle_incremental_chunking(context)
            else:
                self._create_new_phases(context)

            # 3. Display overview
            self._display_phase_overview(context)

            # 4. Save phases
            self._save_phases(context)

        except Exception as e:
            raise HandlerError(f"Failed to chunk history into phases: {e}") from e

    def _try_load_cached_phases(self, context: AnalysisContext) -> None:
        """Try to load cached phases from previous run.

        Args:
            context: Analysis context to populate with cached phases
        """
        if context.phases_dir and context.phases_dir.exists():
            try:
                from ..chunker import HistoryChunker
                context.cached_phases = HistoryChunker.load_phases(
                    str(context.phases_dir)
                )
            except Exception as e:
                self._log_warning(f"Failed to load cached phases: {e}")
                self._log_warning(
                    "This usually means the cache directory only has summaries "
                    "from an interrupted or older run; delete the phases folder to "
                    "recompute commit details."
                )

    def _create_new_phases(self, context: AnalysisContext) -> None:
        """Create new phases from records.

        Args:
            context: Analysis context with records to chunk
        """
        from ..chunker import HistoryChunker

        chunker = HistoryChunker(self.config.strategy)

        kwargs = {}
        if self.config.strategy == 'fixed':
            kwargs['chunk_size'] = self.config.chunk_size

        context.phases = chunker.chunk(context.records, **kwargs)
        self._log_success(f"Created {len(context.phases)} phases\n")

    def _handle_incremental_chunking(self, context: AnalysisContext) -> None:
        """Handle incremental phase creation/merging.

        Args:
            context: Analysis context with existing phases and new records
        """
        merge_threshold = 10  # commits

        if len(context.records) < merge_threshold:
            # Merge into last phase
            self._merge_into_last_phase(context)
        else:
            # Create new phases
            self._create_and_append_phases(context)

    def _merge_into_last_phase(self, context: AnalysisContext) -> None:
        """Merge new commits into the last existing phase.

        Args:
            context: Analysis context with existing phases and new records
        """
        self._log_info(f"Merging {len(context.records)} commits into last phase...")

        last_phase = context.existing_phases[-1]
        last_phase.commits.extend(context.records)

        # Recalculate stats
        last_phase.commit_count = len(last_phase.commits)
        last_phase.end_date = context.records[-1].timestamp
        last_phase.total_insertions = sum(c.insertions for c in last_phase.commits)
        last_phase.total_deletions = sum(c.deletions for c in last_phase.commits)
        last_phase.loc_end = context.records[-1].loc_total
        last_phase.loc_delta = last_phase.loc_end - last_phase.loc_start
        if last_phase.loc_start > 0:
            last_phase.loc_delta_percent = (last_phase.loc_delta / last_phase.loc_start) * 100

        # Clear summary for regeneration
        last_phase.summary = None

        context.phases = context.existing_phases
        self._log_success(f"Updated last phase (now {last_phase.commit_count} commits)\n")

    def _create_and_append_phases(self, context: AnalysisContext) -> None:
        """Create new phases and append to existing.

        Args:
            context: Analysis context with existing phases and new records
        """
        from ..chunker import HistoryChunker

        chunker = HistoryChunker(self.config.strategy)

        kwargs = {}
        if self.config.strategy == 'fixed':
            kwargs['chunk_size'] = self.config.chunk_size

        new_phases = chunker.chunk(context.records, **kwargs)

        # Renumber to continue from existing
        for phase in new_phases:
            phase.phase_number = len(context.existing_phases) + phase.phase_number

        context.phases = context.existing_phases + new_phases
        self._log_success(
            f"Created {len(new_phases)} new phases (total: {len(context.phases)})\n"
        )

    def _display_phase_overview(self, context: AnalysisContext) -> None:
        """Display phase overview table.

        Args:
            context: Analysis context with phases to display
        """
        table = Table(title="Phase Overview")
        table.add_column("Phase", style="cyan", justify="right")
        table.add_column("Period", style="magenta")
        table.add_column("Commits", justify="right")
        table.add_column("LOC Δ", justify="right")
        table.add_column("Events", style="yellow")

        for phase in context.phases:
            events = []
            if phase.has_large_deletion:
                events.append("×")
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

        self.console.print(table)

    def _save_phases(self, context: AnalysisContext) -> None:
        """Save phases to disk.

        Args:
            context: Analysis context with phases to save
        """
        from ..chunker import HistoryChunker

        chunker = HistoryChunker(self.config.strategy)
        chunker.save_phases(context.phases, str(context.phases_dir))
