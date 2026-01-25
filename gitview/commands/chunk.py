"""Chunk command implementation."""

import sys
from pathlib import Path

from .base import BaseCommand
from ..extractor import GitHistoryExtractor
from ..chunker import HistoryChunker


class ChunkCommand(BaseCommand):
    """Chunk extracted history into meaningful phases.

    Takes a JSONL file from 'gitview extract' and splits it into phases/epochs
    based on the chosen strategy. No LLM or API key required.
    """

    def validate(self) -> None:
        """Validate command options."""
        history_file = self.get_option('history_file')
        if not history_file:
            self.print_error("Error: history_file is required")
            sys.exit(1)

        if not Path(history_file).exists():
            self.print_error(f"Error: History file not found: {history_file}")
            sys.exit(1)

    def execute(self):
        """Execute history chunking."""
        self.print_header("Chunking History into Phases")

        history_file = self.get_option('history_file')
        output = self.get_option('output', 'output/phases')
        strategy = self.get_option('strategy', 'adaptive')
        chunk_size = self.get_option('chunk_size', 50)

        try:
            # Load history
            records = GitHistoryExtractor.load_from_jsonl(history_file)

            self.print_info(f"Loaded {len(records)} commits")
            self.print_info(f"Strategy: {strategy}\n")

            # Chunk
            chunker = HistoryChunker(strategy)
            kwargs = {}
            if strategy == 'fixed':
                kwargs['chunk_size'] = chunk_size

            phases = chunker.chunk(records, **kwargs)

            self.print_success(f"Created {len(phases)} phases\n")
            self._display_phase_overview(phases)

            # Ensure output directory exists
            Path(output).mkdir(parents=True, exist_ok=True)

            # Save
            chunker.save_phases(phases, output)
            self.print_success(f"\nSaved phases to {output}\n")

            return phases

        except Exception as e:
            self.print_error(f"\nError: {e}")
            sys.exit(1)

    def _display_phase_overview(self, phases):
        """Display phase overview table."""
        table = self.create_table("Phase Overview")

        table.add_column("Phase", style="cyan", justify="right")
        table.add_column("Period", style="magenta")
        table.add_column("Commits", justify="right")
        table.add_column("LOC Δ", justify="right")
        table.add_column("Events", style="yellow")

        for phase in phases:
            events = []
            if phase.has_large_deletion:
                events.append("×")
            if phase.has_large_addition:
                events.append("+")
            if phase.has_refactor:
                events.append("»")
            if phase.readme_changed:
                events.append("▶")

            table.add_row(
                str(phase.phase_number),
                f"{phase.start_date[:10]} to {phase.end_date[:10]}",
                str(phase.commit_count),
                f"{phase.loc_delta:+,d}",
                " ".join(events)
            )

        self.console.print(table)
