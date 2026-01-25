"""Show storyline command implementation."""

from pathlib import Path

from ..base import BaseCommand
from ...storyline import StorylineDatabase, StorylineReporter


class ShowStorylineCommand(BaseCommand):
    """Show detailed information about a specific storyline.

    Displays the full narrative, timeline of updates, key files,
    and related pull requests for the specified storyline.
    """

    def validate(self) -> None:
        """Validate command options."""
        storyline_id = self.get_option('storyline_id')
        if not storyline_id:
            self.print_error("Error: storyline_id is required")
            import sys
            sys.exit(1)

    def execute(self):
        """Show storyline details."""
        self.console.print("\n[bold cyan]Storyline Details[/bold cyan]")
        self.console.print("=" * 70)

        output = self.get_option('output', 'output')
        storyline_id = self.get_option('storyline_id')

        # Load database
        db_path = Path(output) / "phases" / "storylines.json"

        if not db_path.exists():
            self.print_error(f"Storyline database not found at {db_path}")
            self.console.print("Run 'gitview analyze' first to generate storylines.")
            return

        try:
            database = StorylineDatabase.load(str(db_path))
        except Exception as e:
            self.print_error(f"Error loading database: {e}")
            return

        # Find storyline by ID or partial match
        storyline = database.storylines.get(storyline_id)

        if not storyline:
            # Try partial ID match
            matches = [sl for sl in database.storylines.values()
                       if storyline_id.lower() in sl.id.lower() or storyline_id.lower() in sl.title.lower()]

            if len(matches) == 1:
                storyline = matches[0]
            elif len(matches) > 1:
                self.print_warning(f"Multiple matches found for '{storyline_id}':")
                for m in matches[:5]:
                    self.console.print(f"  - {m.id}: {m.title}")
                return
            else:
                self.print_error(f"Storyline '{storyline_id}' not found.")
                self.console.print("\nUse 'gitview storyline list' to see available storylines.")
                return

        # Generate detailed narrative
        reporter = StorylineReporter(database=database)
        narrative = reporter.generate_storyline_narrative(storyline.id)

        self.console.print(narrative)
