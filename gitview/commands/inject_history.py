"""Inject history command implementation."""

import json
import sys
from pathlib import Path

from .base import BaseCommand
from ..file_tracker import FileHistoryTracker
from ..history_injector import HistoryInjector


class InjectHistoryCommand(BaseCommand):
    """Inject file change history as header comments into source files.

    This command injects change histories as formatted comments at the top
    of source files. Supports multiple programming languages with appropriate
    comment styles.
    """

    def validate(self) -> None:
        """Validate command options."""
        paths = self.get_option('paths', [])
        inject_all = self.get_option('inject_all', False)

        if not paths and not inject_all:
            self.print_error("Error: Either provide file paths or use --all")
            sys.exit(1)

    def execute(self):
        """Execute history injection."""
        self.console.print("\n[bold cyan]GitView History Header Injection[/bold cyan]")
        self.console.print("=" * 70)

        repo = self.get_option('repo', '.')
        output = self.get_option('output', 'output/file_histories')
        paths = self.get_option('paths', [])
        max_entries = self.get_option('max_entries', 10)
        dry_run = self.get_option('dry_run', False)
        inject_all = self.get_option('inject_all', False)

        # Initialize tracker and injector
        try:
            tracker = FileHistoryTracker(repo_path=repo, output_dir=output)
            injector = HistoryInjector(tracker)
        except Exception as e:
            self.print_error(f"Error initializing: {e}")
            sys.exit(1)

        # Determine files to inject
        file_list = self._get_file_list(paths, inject_all, output)

        if dry_run:
            self.print_warning("DRY RUN MODE - No files will be modified")

        self.console.print()

        # Inject headers
        successes = 0
        failures = 0
        skipped = 0

        for file_path in file_list:
            if not Path(file_path).exists():
                self.console.print(f"[red]✗[/red] {file_path} - File not found")
                skipped += 1
                continue

            success, message = injector.inject_history(
                file_path,
                max_entries=max_entries,
                dry_run=dry_run
            )

            if success:
                if dry_run:
                    self.console.print(f"[green]✓[/green] {file_path} - Would inject header")
                    if len(file_list) == 1:
                        self._show_preview(message)
                else:
                    self.console.print(f"[green]✓[/green] {message}")
                successes += 1
            else:
                self.console.print(f"[red]✗[/red] {message}")
                failures += 1

        # Summary
        self.console.print("\n" + "=" * 70)
        self.console.print("[bold cyan]Summary:[/bold cyan]")
        self.console.print(f"  Successful: {successes}")
        self.console.print(f"  Failed: {failures}")
        self.console.print(f"  Skipped: {skipped}")

        if dry_run:
            self.print_warning("\nDry run complete. Run without --dry-run to apply changes.")
        elif successes > 0:
            self.print_success("\n✓ Headers injected successfully!")
            self.console.print("To remove headers: [green]gitview remove-history <files>[/green]")

        self.console.print()

    def _get_file_list(self, paths, inject_all, output):
        """Get list of files to inject headers into."""
        if inject_all:
            index_path = Path(output) / "index.json"
            if not index_path.exists():
                self.print_error("Error: No tracked files found")
                self.console.print(f"Run: [green]gitview track-files[/green] first")
                sys.exit(1)

            with open(index_path, 'r') as f:
                index = json.load(f)

            file_list = [f['path'] for f in index.get('files', [])]
            self.console.print(f"\n[green]Injecting headers into {len(file_list)} tracked files[/green]")
        else:
            file_list = list(paths)
            self.console.print(f"\n[green]Injecting headers into {len(file_list)} file(s)[/green]")

        return file_list

    def _show_preview(self, message):
        """Show preview of header to be injected."""
        self.console.print("\n[bold cyan]Preview:[/bold cyan]")
        message_lines = message.split('\n')
        preview = message_lines[:30]
        for line in preview:
            self.console.print(f"  {line}")
        if len(message_lines) > 30:
            remaining = len(message_lines) - 30
            self.console.print(f"  ... ({remaining} more lines)")
