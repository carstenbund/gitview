"""Motifs command — list recurring historical / architectural motifs."""

import json
import sys
from pathlib import Path

from .base import BaseCommand
from ..graph import GraphStore, GraphUpdater
from ..motifs import Evidence, Thresholds, motif_catalog, run_motifs


class MotifsCommand(BaseCommand):
    """Detect motifs in ``.gitview/graph.sqlite`` and print them grouped by evidence."""

    def validate(self) -> None:
        import click
        for name, floor in (('top', 1), ('min_cochanges', 1), ('min_touches', 1), ('min_degree_growth', 1)):
            value = self.get_option(name)
            if value is not None and value < floor:
                raise click.UsageError(f"--{name.replace('_', '-')} must be at least {floor}")

    def execute(self):
        if self.get_option("list_motifs", False):
            return self._print_catalog()

        repo_path = Path(self.get_option("repo", ".")).resolve()
        if not (repo_path / ".git").exists():
            self.print_error(f"Error: {repo_path} is not a git repository")
            sys.exit(1)
        branch = self.get_option("branch", "HEAD")
        as_json = self.get_option("json_output", False)
        only = self.get_option("only")
        only = [x.strip() for x in only.split(',') if x.strip()] if only else None

        thresholds = Thresholds()
        for name in ('min_cochanges', 'min_jaccard', 'min_touches', 'min_degree_growth', 'top'):
            value = self.get_option(name)
            if value is not None:
                setattr(thresholds, name, value)

        updater = GraphUpdater(repo_path, branch=branch)
        try:
            updater.sync()
        except Exception as exc:
            self.print_error(f"Error building graph: {exc}")
            sys.exit(1)

        with GraphStore(updater.store_path) as store:
            store.initialize()
            report = run_motifs(store, only=only, thresholds=thresholds,
                                provider=self.get_option("provider"))

        if as_json:
            print(json.dumps(report.to_dict(), indent=2))
            return report

        self._print_report(report)
        return report

    # ------------------------------------------------------------------

    def _print_catalog(self):
        catalog = motif_catalog()
        for heading, structural in (("Historical motifs", False), ("Structural + historical motifs", True)):
            self.console.print(f"\n[bold]{heading}[/bold]")
            for m in catalog:
                if m.structural == structural:
                    needs = ', '.join(sorted(e.value for e in m.requires))
                    self.console.print(f"  {m.id:<24}{m.title:<24}[dim]{m.interpretation}[/dim]")
                    self.console.print(f"  {'':<24}[dim]needs: {needs}[/dim]")
        return catalog

    def _print_report(self, report) -> None:
        grouped = report.by_motif()
        catalog = motif_catalog()
        obs = report.observations
        self.console.print()
        if obs:
            latest = obs[-1]
            span = (f"{obs[0].observed_sha[:8]} … {latest.observed_sha[:8]} ({len(obs)} observations)"
                    if len(obs) > 1 else latest.observed_sha[:8])
            self.console.print(f"[dim]Structural evidence: {latest.provider} {latest.provider_version} at {span}[/dim]")
        else:
            self.console.print("[dim]Structural evidence: none (historical motifs only)[/dim]")

        for heading, structural in (("Historical motifs", False), ("Structural + historical motifs", True)):
            self.console.print(f"\n[bold]{heading}[/bold]")
            self.console.print("-" * len(heading))
            any_printed = False
            for m in catalog:
                if m.structural != structural:
                    continue
                if m.id in report.skipped:
                    self.console.print(f"[dim]{m.title}: skipped — {report.skipped[m.id]}[/dim]")
                    any_printed = True
                    continue
                findings = grouped.get(m.id, [])
                if not findings:
                    continue
                any_printed = True
                self.console.print(f"[bold]{m.title}[/bold]  [dim]({len(findings)})[/dim]")
                for f in findings:
                    for path in f.files:
                        self.console.print(f"  {path}")
                    self.console.print(f"  [dim]{f.summary}[/dim]")
                    self.console.print(f"  [dim]confidence: {f.level} ({f.confidence:.2f})[/dim]")
                    self.console.print()
            if not any_printed:
                self.console.print("[dim]nothing detected at current thresholds[/dim]")
        self.console.print()
