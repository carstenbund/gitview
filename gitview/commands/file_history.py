"""File history command implementation."""

import json
import sys
from pathlib import Path

from .base import BaseCommand
from ..file_tracker import FileHistoryTracker


class FileHistoryCommand(BaseCommand):
    """Display change history for a specific file.

    Shows detailed timeline of changes including:
      - Commit metadata (hash, date, author)
      - Lines added/removed
      - Commit messages
      - Function/class changes (when available)
    """

    def validate(self) -> None:
        """Validate command options."""
        file_path = self.get_option('file_path')
        if not file_path:
            self.print_error("Error: file_path is required")
            sys.exit(1)

    def execute(self):
        """Display file history."""
        file_path = self.get_option('file_path')
        repo = self.get_option('repo', '.')
        output = self.get_option('output', 'output/file_histories')
        output_format = self.get_option('output_format', 'text')
        recent = self.get_option('recent')

        tracker = FileHistoryTracker(repo_path=repo, output_dir=output)

        try:
            history = tracker.get_file_history(file_path)

            if not history:
                self.print_warning(f"No history found for: {file_path}")
                self.console.print(f"\nMake sure you've run: [green]gitview track-files[/green]")
                sys.exit(1)

            if output_format == 'json':
                print(json.dumps(history.to_dict(), indent=2))
            else:
                self._display_text_history(history, file_path, recent, output)

        except Exception as e:
            self.print_error(f"Error: {e}")
            import traceback
            self.console.print(traceback.format_exc())
            sys.exit(1)

    def _display_text_history(self, history, file_path, recent, output):
        """Display history in text format."""
        self.console.print()
        self.console.print("=" * 80)
        self.console.print(f"[bold cyan]FILE HISTORY: {file_path}[/bold cyan]")
        self.console.print("=" * 80)

        self.console.print(f"\n[green]First seen:[/green] {history.first_commit_date[:10]} (commit {history.first_commit[:7]})")
        self.console.print(f"[green]Last modified:[/green] {history.last_commit_date[:10]} (commit {history.last_commit[:7]})")
        self.console.print(f"[green]Total commits:[/green] {history.total_commits}")
        self.console.print(f"[green]Lines added:[/green] +{history.total_lines_added}")
        self.console.print(f"[green]Lines removed:[/green] -{history.total_lines_removed}")
        self.console.print(f"[green]Net change:[/green] {history.total_lines_added - history.total_lines_removed:+d} lines")

        # Contributors
        if history.authors:
            self.console.print(f"\n[bold cyan]Contributors:[/bold cyan]")
            for author in history.authors[:5]:
                name = author.get('name', 'Unknown')
                count = author.get('commits', 0)
                pct = (count / history.total_commits * 100) if history.total_commits > 0 else 0
                self.console.print(f"  {name:30} {count:3} commits ({pct:5.1f}%)")

        # Changes
        display_count = recent if recent else len(history.changes)
        display_count = min(display_count, len(history.changes))

        self.console.print(f"\n[bold cyan]Recent Changes - Last {display_count} of {history.total_commits}[/bold cyan]")
        self.console.print("-" * 80)

        for change in history.changes[:display_count]:
            self.console.print(f"\n[yellow]{change.commit_date[:19]}[/yellow] | [cyan]{change.commit_hash[:7]}[/cyan] | {change.author_name}")

            msg_lines = change.commit_message.strip().split('\n')
            self.console.print(f"  {msg_lines[0]}")

            self.console.print(f"  [green]+{change.lines_added}[/green] [red]-{change.lines_removed}[/red] lines")

        self.console.print("\n" + "=" * 80)

        json_path = Path(output) / "files" / f"{file_path}.json"
        history_path = Path(output) / "files" / f"{file_path}.history"
        self.console.print(f"\n[dim]Full history available at:[/dim]")
        self.console.print(f"  [dim]• {json_path}[/dim]")
        self.console.print(f"  [dim]• {history_path}[/dim]")
        self.console.print()
