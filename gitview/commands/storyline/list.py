"""List storylines command implementation."""

import json
from pathlib import Path

from ..base import BaseCommand
from ...storyline import (
    StorylineDatabase,
    StorylineStatus,
    StorylineCategory,
)


class ListStorylineCommand(BaseCommand):
    """List all tracked storylines.

    Shows a summary table of storylines with their status, category,
    phase range, and confidence score.
    """

    def validate(self) -> None:
        """Validate command options."""
        pass  # No required validation

    def execute(self):
        """List storylines."""
        self.console.print("\n[bold cyan]Storylines[/bold cyan]")
        self.console.print("=" * 70)

        output = self.get_option('output', 'output')
        status_filter = self.get_option('status', 'all')
        category_filter = self.get_option('category')
        limit = self.get_option('limit', 20)

        # Try to load storyline database
        db_path = Path(output) / "phases" / "storylines.json"
        json_path = Path(output) / "history_data.json"

        database = None

        if db_path.exists():
            try:
                database = StorylineDatabase.load(str(db_path))
            except Exception as e:
                self.print_warning(f"Warning: Could not load storyline database: {e}")

        if database is None and json_path.exists():
            database = self._load_from_legacy(json_path)

        if database is None or not database.storylines:
            self.print_warning("\nNo storylines found.")
            self.console.print("\nRun 'gitview analyze' first to generate storylines.")
            return

        # Filter storylines
        storylines = list(database.storylines.values())

        if status_filter != 'all':
            status_map = {
                'active': [StorylineStatus.ACTIVE, StorylineStatus.PROGRESSING],
                'completed': [StorylineStatus.COMPLETED],
                'stalled': [StorylineStatus.STALLED],
                'abandoned': [StorylineStatus.ABANDONED],
            }
            target_statuses = status_map.get(status_filter, [])
            storylines = [sl for sl in storylines if sl.status in target_statuses]

        if category_filter:
            target_category = StorylineCategory.from_string(category_filter)
            storylines = [sl for sl in storylines if sl.category == target_category]

        # Sort by last_phase descending
        storylines.sort(key=lambda x: (-x.last_phase, x.title))
        storylines = storylines[:limit]

        if not storylines:
            self.print_warning("\nNo storylines match the filters.")
            return

        # Create table
        table = self.create_table()
        table.add_column("Status", style="dim", width=8)
        table.add_column("Title", style="cyan", max_width=35)
        table.add_column("Category", style="magenta", width=12)
        table.add_column("Phases", style="green", width=10)
        table.add_column("Conf", style="yellow", width=5)
        table.add_column("ID", style="dim", width=12)

        status_icons = {
            'completed': '[green]✓[/green]',
            'active': '[blue]●[/blue]',
            'progressing': '[blue]▶[/blue]',
            'stalled': '[yellow]◌[/yellow]',
            'abandoned': '[red]✗[/red]',
            'emerging': '[dim]○[/dim]',
        }

        for sl in storylines:
            icon = status_icons.get(sl.status.value, '?')
            phases = f"{sl.first_phase}→{sl.last_phase}" if sl.first_phase != sl.last_phase else str(sl.first_phase)
            conf = f"{sl.confidence:.0%}"
            short_id = sl.id[:10] + ".." if len(sl.id) > 12 else sl.id

            table.add_row(
                icon,
                sl.title[:33] + ".." if len(sl.title) > 35 else sl.title,
                sl.category.value,
                phases,
                conf,
                short_id,
            )

        self.console.print(table)

        # Summary
        self.console.print(f"\n[dim]Showing {len(storylines)} of {len(database.storylines)} storylines[/dim]")
        self.console.print("[dim]Use 'gitview storyline show <ID>' for details[/dim]")

    def _load_from_legacy(self, json_path):
        """Load storylines from legacy history_data.json format."""
        try:
            with open(json_path) as f:
                data = json.load(f)
            if 'storylines' in data:
                self.print_info("Loading storylines from history_data.json...")
                storylines_data = data.get('storylines', {}).get('storylines', [])
                if storylines_data:
                    database = StorylineDatabase()
                    from ...storyline.models import Storyline as StorylineModel
                    for sl_data in storylines_data:
                        storyline = StorylineModel(
                            id=StorylineModel.generate_id(sl_data['title'], sl_data.get('first_phase', 1)),
                            title=sl_data['title'],
                            category=StorylineCategory.from_string(sl_data.get('category', 'feature')),
                            status=StorylineStatus[sl_data.get('status', 'active').upper()]
                                   if sl_data.get('status', '').upper() in StorylineStatus.__members__
                                   else StorylineStatus.ACTIVE,
                            confidence=sl_data.get('confidence', 0.7),
                            first_phase=sl_data.get('first_phase', 1),
                            last_phase=sl_data.get('last_phase', 1),
                            phases_involved=sl_data.get('phases_involved', []),
                            description=sl_data.get('description', ''),
                            current_summary=sl_data.get('last_update', sl_data.get('description', '')),
                        )
                        database.add_storyline(storyline)
                    return database
        except Exception as e:
            self.print_warning(f"Warning: Could not load from history_data.json: {e}")
        return None
