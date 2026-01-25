"""Export storylines command implementation."""

import csv
import json
from datetime import datetime
from pathlib import Path

from ..base import BaseCommand
from ...storyline import StorylineDatabase, StorylineStatus


class ExportStorylineCommand(BaseCommand):
    """Export storylines to JSON or CSV format.

    Exports storyline data for use in other tools or analysis.
    """

    def validate(self) -> None:
        """Validate command options."""
        pass  # No required validation

    def execute(self):
        """Export storylines."""
        self.console.print("\n[bold cyan]Exporting Storylines...[/bold cyan]")

        output = self.get_option('output', 'output')
        fmt = self.get_option('fmt', 'json')
        dest = self.get_option('dest')
        status_filter = self.get_option('status', 'all')

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

        # Filter storylines
        storylines = list(database.storylines.values())

        if status_filter != 'all':
            status_map = {
                'active': [StorylineStatus.ACTIVE, StorylineStatus.PROGRESSING],
                'completed': [StorylineStatus.COMPLETED],
                'stalled': [StorylineStatus.STALLED],
            }
            target_statuses = status_map.get(status_filter, [])
            storylines = [sl for sl in storylines if sl.status in target_statuses]

        if not storylines:
            self.print_warning("No storylines to export.")
            return

        # Determine output path
        if dest:
            dest_path = Path(dest)
        else:
            dest_path = Path(output) / f"storylines_export.{fmt}"

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == 'json':
            export_data = {
                'exported_at': datetime.now().isoformat(),
                'total': len(storylines),
                'storylines': [sl.to_dict() for sl in storylines],
            }
            with open(dest_path, 'w') as f:
                json.dump(export_data, f, indent=2)
        else:  # CSV
            with open(dest_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'id', 'title', 'category', 'status', 'confidence',
                    'first_phase', 'last_phase', 'phase_count', 'description'
                ])
                for sl in storylines:
                    writer.writerow([
                        sl.id, sl.title, sl.category.value, sl.status.value,
                        f"{sl.confidence:.2f}", sl.first_phase, sl.last_phase,
                        len(sl.phases_involved), sl.description[:200]
                    ])

        self.print_success(f"\nExported {len(storylines)} storylines to: {dest_path}")
