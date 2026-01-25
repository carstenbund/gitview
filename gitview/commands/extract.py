"""Extract command implementation."""

import sys
from pathlib import Path

from .base import BaseCommand
from ..extractor import GitHistoryExtractor


class ExtractCommand(BaseCommand):
    """Extract git history to JSONL file.

    This command extracts detailed metadata from git commits without using an LLM.
    Useful for quick history extraction, pre-processing, or exploring repository metrics.
    """

    def validate(self) -> None:
        """Validate command options."""
        repo = self.get_option('repo', '.')
        repo_path = Path(repo).resolve()
        if not (repo_path / '.git').exists():
            self.print_error(f"Error: {repo_path} is not a git repository")
            sys.exit(1)
        self._repo_path = repo_path

    def execute(self):
        """Execute git history extraction."""
        self.print_header("Extracting Git History")

        output = self.get_option('output', 'output/repo_history.jsonl')
        max_commits = self.get_option('max_commits')
        branch = self.get_option('branch', 'HEAD')

        try:
            extractor = GitHistoryExtractor(str(self._repo_path))

            with self.create_progress() as progress:
                task = progress.add_task("Extracting commits...", total=None)
                records = extractor.extract_history(max_commits=max_commits, branch=branch)
                progress.update(task, completed=True)

            # Ensure output directory exists
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            extractor.save_to_jsonl(records, output)

            self.print_success(f"\nExtracted {len(records)} commits to {output}\n")
            return records

        except Exception as e:
            self.print_error(f"\nError: {e}")
            sys.exit(1)
