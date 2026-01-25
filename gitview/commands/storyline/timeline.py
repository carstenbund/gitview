"""Timeline storyline command implementation."""

from pathlib import Path

from ..base import BaseCommand
from ...storyline import StorylineDatabase, StorylineReporter


class TimelineStorylineCommand(BaseCommand):
    """Display ASCII timeline of storyline arcs.

    Shows a visual representation of how storylines span across phases,
    with indicators for start, end, completion, and stalls.

    Legend:
      ┌─  start
      ─┘  completed
      ─→  ongoing
      ─╳  stalled
      ·   gap
    """

    def validate(self) -> None:
        """Validate command options."""
        pass  # No required validation

    def execute(self):
        """Display storyline timeline."""
        self.console.print("\n[bold cyan]Storyline Timeline[/bold cyan]")
        self.console.print("=" * 70)

        output = self.get_option('output', 'output')

        # Load database
        db_path = Path(output) / "phases" / "storylines.json"

        if not db_path.exists():
            self.print_error(f"Storyline database not found at {db_path}")
            return

        try:
            database = StorylineDatabase.load(str(db_path))
        except Exception as e:
            self.print_error(f"Error loading database: {e}")
            return

        if not database.storylines:
            self.print_warning("No storylines to display.")
            return

        reporter = StorylineReporter(database=database)
        timeline = reporter.generate_timeline_view()

        self.console.print()
        self.console.print(timeline)
