"""Graph command — build/update the persistent repository graph (no LLM)."""

import json
import sys
from pathlib import Path

from .base import BaseCommand
from ..graph import GraphStore, GraphUpdater, compute_graph_stats
from ..graph.models import DEFAULT_MAX_PROJECTION_FILES


class GraphCommand(BaseCommand):
    """Build or update ``<repo>/.gitview/graph.sqlite`` and print statistics."""

    def validate(self) -> None:
        top = self.get_option("top", 10)
        if top is not None and top < 1:
            import click
            raise click.UsageError("--top must be at least 1")

    def execute(self):
        repo_spec = self.get_option("repo", ".")
        branch = self.get_option("branch", "HEAD")
        rebuild = self.get_option("rebuild", False)
        show_stats = self.get_option("stats", False)
        as_json = self.get_option("json_output", False)
        top = self.get_option("top", 10) or 10
        cap = self.get_option("max_projection_files") or DEFAULT_MAX_PROJECTION_FILES

        repo_path = Path(repo_spec).resolve()
        if not (repo_path / ".git").exists():
            self.print_error(f"Error: {repo_path} is not a git repository")
            sys.exit(1)

        updater = GraphUpdater(repo_path, branch=branch, max_projection_files=cap)
        try:
            result = updater.sync(rebuild=rebuild)
        except Exception as exc:
            self.print_error(f"Error building graph: {exc}")
            sys.exit(1)

        structural = self.get_option("structural")
        observation = None
        if structural:
            from .observe import run_observation
            observation = run_observation(
                self, repo_path, structural, branch=branch,
                source=self.get_option("source"), refresh=self.get_option("refresh", False),
                quiet=as_json)

        with GraphStore(updater.store_path) as store:
            stats = compute_graph_stats(store, top=top)

        if as_json:
            payload = {
                'action': result.action,
                'reason': result.reason,
                'new_commits': result.new_commits,
                'structural_kept': result.structural_kept,
                'structural_unplaced': result.structural_unplaced,
                'graph_path': str(updater.store_path),
                'metadata': result.metadata.to_dict(),
                'stats': stats.to_dict(),
            }
            if observation is not None:
                payload['structural'] = {
                    'inserted': observation.inserted,
                    'drift': observation.drift,
                    'observation': observation.observation.to_dict(),
                }
            print(json.dumps(payload, indent=2))
            return payload

        self._print_summary(result, updater.store_path, stats)
        if show_stats:
            self._print_top_lists(stats)
        return stats

    # ------------------------------------------------------------------

    def _print_summary(self, result, store_path, stats) -> None:
        verb = {'built': 'Built', 'updated': 'Updated', 'unchanged': 'Up to date'}[result.action]
        detail = f" ({result.reason})" if result.reason and result.action != 'unchanged' else ''
        self.print_success(f"{verb}: {store_path}{detail}")
        if result.structural_kept:
            self.console.print(
                f"Kept {result.structural_kept:,} structural observation(s) across the rebuild")
        if result.structural_unplaced:
            self.print_warning(
                f"{result.structural_unplaced:,} structural observation(s) point at commits no longer "
                f"in the history; series motifs cannot order them")
        self.console.print()
        self.console.print("[bold]Repository graph[/bold]")
        self.console.print("----------------")
        rows = [
            ("Commits", f"{stats.commits:,}",
             f"  (merges: {stats.merge_commits:,}, projection-suppressed: {stats.suppressed_commits:,})"),
            ("Files", f"{stats.files:,}", ""),
            ("Authors", f"{stats.authors:,}", ""),
            ("Pull requests", f"{stats.pull_requests:,}", ""),
            ("Commit/file edges", f"{stats.commit_file_edges:,}", ""),
            ("File coupling edges", f"{stats.file_edges:,}", ""),
            ("Structural snapshots", f"{stats.structural_snapshots:,}", ""),
        ]
        for label, value, extra in rows:
            self.console.print(f"{label + ':':<22}{value:>8}{extra}")
        self.console.print()

    def _print_top_lists(self, stats) -> None:
        table = self.create_table("Most changed")
        table.add_column("File")
        table.add_column("Commits", justify="right")
        table.add_column("+/-", justify="right")
        for f in stats.most_changed:
            table.add_row(f.path, str(f.touch_count), f"+{f.total_insertions}/-{f.total_deletions}")
        self.console.print(table)

        table = self.create_table("Most coupled (co-changes ≥ 2, merges excluded)")
        table.add_column("File A")
        table.add_column("File B")
        table.add_column("Co-changes", justify="right")
        table.add_column("Jaccard", justify="right")
        for e in stats.most_coupled:
            table.add_row(e.path_a, e.path_b, str(e.cochange_count), f"{e.jaccard:.2f}")
        self.console.print(table)

        table = self.create_table("Most connected")
        table.add_column("File")
        table.add_column("Degree", justify="right")
        table.add_column("Weighted", justify="right")
        for d in stats.most_connected:
            table.add_row(d.path, str(d.degree), str(d.weighted_degree))
        self.console.print(table)
