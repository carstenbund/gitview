"""Remove history command implementation."""

import os
import sys
from pathlib import Path

from .base import BaseCommand
from ..file_tracker import FileHistoryTracker
from ..history_injector import HistoryInjector


class RemoveHistoryCommand(BaseCommand):
    """Remove injected file change history headers from source files.

    This command removes previously injected history headers from files.
    Only removes headers that were injected by gitview.
    """

    def validate(self) -> None:
        """Validate command options."""
        paths = self.get_option('paths', [])
        remove_all = self.get_option('remove_all', False)

        if not paths and not remove_all:
            self.print_error("Error: Either provide file paths or use --all")
            sys.exit(1)

    def execute(self):
        """Execute history header removal."""
        self.console.print("\n[bold cyan]GitView History Header Removal[/bold cyan]")
        self.console.print("=" * 70)

        repo = self.get_option('repo', '.')
        paths = self.get_option('paths', [])
        dry_run = self.get_option('dry_run', False)
        remove_all = self.get_option('remove_all', False)

        # Initialize tracker and injector
        try:
            tracker = FileHistoryTracker(repo_path=repo, output_dir='output/file_histories')
            injector = HistoryInjector(tracker)
        except Exception as e:
            self.print_error(f"Error initializing: {e}")
            sys.exit(1)

        # Determine files to process
        if remove_all:
            file_list = self._find_files_with_headers(injector)
        else:
            file_list = list(paths)
            self.console.print(f"\n[green]Removing headers from {len(file_list)} file(s)[/green]")

        if dry_run:
            self.print_warning("DRY RUN MODE - No files will be modified")

        self.console.print()

        # Remove headers
        successes = 0
        failures = 0

        for file_path in file_list:
            success, message = injector.remove_history(file_path, dry_run=dry_run)

            if success:
                if dry_run:
                    self.console.print(f"[green]✓[/green] {file_path} - Would remove header")
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

        if dry_run:
            self.print_warning("\nDry run complete. Run without --dry-run to apply changes.")
        elif successes > 0:
            self.print_success("\n✓ Headers removed successfully!")

        self.console.print()

    def _find_files_with_headers(self, injector):
        """Find all files with injected headers."""
        self.print_warning("Scanning for files with injected headers...")
        file_list = []

        for root, dirs, files in os.walk('.'):
            # Skip hidden and common ignore directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', 'venv', '__pycache__']]

            for file in files:
                file_path = os.path.join(root, file)
                if injector.detect_language(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        if injector.has_injected_header(content):
                            file_list.append(file_path)
                    except:
                        pass

        self.print_success(f"Found {len(file_list)} file(s) with injected headers")
        return file_list
