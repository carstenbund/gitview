"""Compare branches command implementation."""

import sys

from .base import BaseCommand
from ..branch_comparator import BranchComparator


class CompareBranchesCommand(BaseCommand):
    """Compare file histories between two git branches.

    This command tracks files on both branches and generates a detailed
    comparison report showing:
      - Files unique to each branch
      - Files with divergent changes
      - Commit differences
      - Line change statistics
      - Top most divergent files
    """

    def validate(self) -> None:
        """Validate command options."""
        branch_a = self.get_option('branch_a')
        branch_b = self.get_option('branch_b')

        if not branch_a or not branch_b:
            self.print_error("Error: Both branch_a and branch_b are required")
            sys.exit(1)

    def execute(self):
        """Execute branch comparison."""
        self.console.print("\n[bold cyan]GitView Branch Comparison[/bold cyan]")
        self.console.print("=" * 70)

        branch_a = self.get_option('branch_a')
        branch_b = self.get_option('branch_b')
        repo = self.get_option('repo', '.')
        output = self.get_option('output', 'output/branch_comparisons')
        patterns = self.get_option('patterns')
        exclude = self.get_option('exclude')
        top_n = self.get_option('top_n', 10)

        # Parse patterns
        file_patterns = None
        if patterns:
            file_patterns = [p.strip() for p in patterns.split(',')]
            self.console.print(f"[green]Include patterns:[/green] {', '.join(file_patterns)}")

        exclude_patterns = None
        if exclude:
            exclude_patterns = [p.strip() for p in exclude.split(',')]
            self.console.print(f"[green]Exclude patterns:[/green] {', '.join(exclude_patterns)}")

        # Initialize comparator
        try:
            comparator = BranchComparator(repo_path=repo, output_dir=output)
        except Exception as e:
            self.print_error(f"Error initializing comparator: {e}")
            sys.exit(1)

        # Compare branches
        try:
            summary, divergences = comparator.compare_branches(
                branch_a=branch_a,
                branch_b=branch_b,
                file_patterns=file_patterns,
                exclude_patterns=exclude_patterns
            )

            # Generate and display report
            report = comparator.generate_comparison_report(branch_a, branch_b, top_n=top_n)

            self.print_success("\n✓ Comparison Complete")
            self.console.print("=" * 70)
            self.console.print()
            self.console.print(report)

            # Summary table
            table = self.create_table()
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right", style="green")

            table.add_row("Total files compared", str(summary.total_files_compared))
            table.add_row(f"Files only in {branch_a}", str(summary.files_only_in_a))
            table.add_row(f"Files only in {branch_b}", str(summary.files_only_in_b))
            table.add_row("Files in both", str(summary.files_in_both))
            table.add_row("Files diverged", str(summary.files_diverged))
            table.add_row("", "")
            table.add_row(f"Commits only in {branch_a}", str(summary.commits_only_in_a))
            table.add_row(f"Commits only in {branch_b}", str(summary.commits_only_in_b))
            table.add_row("Commits in both", str(summary.commits_in_both))

            self.console.print("\n[bold cyan]Quick Summary:[/bold cyan]")
            self.console.print(table)

            # Top divergent files
            if summary.top_divergent_files:
                self.console.print(f"\n[bold cyan]Top {min(top_n, len(summary.top_divergent_files))} Most Divergent Files:[/bold cyan]")
                for i, (file_path, score) in enumerate(summary.top_divergent_files[:top_n], 1):
                    color = "red" if score > 70 else "yellow" if score > 40 else "green"
                    self.console.print(f"  {i}. [{color}]{score:5.1f}[/{color}] - {file_path}")

            self.console.print(f"\n[green]Full report saved to:[/green]")
            sanitized_a = branch_a.replace('/', '_')
            sanitized_b = branch_b.replace('/', '_')
            self.console.print(f"  {output}/comparisons/{sanitized_a}_vs_{sanitized_b}/report.txt")
            self.console.print()

        except Exception as e:
            self.print_error(f"\nError during comparison: {e}")
            import traceback
            self.console.print(traceback.format_exc())
            sys.exit(1)
