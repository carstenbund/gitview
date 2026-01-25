"""Report storyline command implementation."""

from pathlib import Path

from ..base import BaseCommand
from ...storyline import StorylineDatabase, StorylineReporter


class ReportStorylineCommand(BaseCommand):
    """Generate a comprehensive storyline report.

    Creates a detailed report including:
      - Storyline index with status summary
      - ASCII timeline visualization
      - Cross-phase theme analysis
      - Category breakdown
    """

    def validate(self) -> None:
        """Validate command options."""
        pass  # No required validation

    def execute(self):
        """Generate storyline report."""
        self.console.print("\n[bold cyan]Generating Storyline Report...[/bold cyan]")
        self.console.print("=" * 70)

        output = self.get_option('output', 'output')
        fmt = self.get_option('fmt', 'terminal')
        include_timeline = self.get_option('include_timeline', True)
        include_themes = self.get_option('include_themes', True)
        save_path = self.get_option('save')

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

        if not database.storylines:
            self.print_warning("No storylines found in database.")
            return

        # Generate report
        reporter = StorylineReporter(database=database)

        if fmt == 'markdown' or save_path:
            report = reporter.generate_full_report()
        else:
            # For terminal, generate each section
            sections = []
            sections.append(reporter.generate_storyline_index())
            if include_timeline:
                sections.append(reporter.generate_timeline_view())
            if include_themes:
                sections.append(reporter.generate_cross_phase_themes())
            report = '\n\n---\n\n'.join(sections)

        if save_path:
            save_file = Path(save_path)
            save_file.parent.mkdir(parents=True, exist_ok=True)
            with open(save_file, 'w') as f:
                f.write(report)
            self.print_success(f"\nReport saved to: {save_file}")
        else:
            self.console.print()
            self.console.print(report)
