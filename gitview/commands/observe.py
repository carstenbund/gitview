"""Observe command — store a structural observation as optional graph evidence."""

import json
import sys
from pathlib import Path

from .base import BaseCommand
from ..graph import GraphUpdater
from ..structural import StructuralProviderError, get_provider, provider_names
from ..structural.observe import observe_structure


def run_observation(cmd: BaseCommand, repo_path: Path, provider_name: str, *,
                    branch: str, source, refresh: bool, quiet: bool = False):
    """Shared by ``observe`` and ``graph --structural``. Exits on failure."""
    try:
        provider = get_provider(provider_name)
        ok, reason = provider.available()
        if not ok and (refresh or not source):
            # A pre-existing output file may still be usable without the tool.
            try:
                return _observe(cmd, repo_path, provider_name, branch, source, refresh, quiet)
            except StructuralProviderError as exc:
                cmd.print_error(f"Error: {reason}\n       ({exc})")
                sys.exit(1)
        return _observe(cmd, repo_path, provider_name, branch, source, refresh, quiet)
    except StructuralProviderError as exc:
        cmd.print_error(f"Error: {exc}")
        sys.exit(1)


def _observe(cmd, repo_path, provider_name, branch, source, refresh, quiet):
    result = observe_structure(repo_path, provider_name, branch=branch, source=source, refresh=refresh)
    if quiet:
        return result
    obs, snap = result.observation, result.snapshot
    verb = 'Stored' if result.inserted else 'Already stored'
    cmd.print_success(
        f"{verb}: {provider_name} {obs.provider_version} observation at {obs.observed_sha[:8]} "
        f"({obs.node_count:,} files, {obs.edge_count:,} file edges)")
    if result.drift:
        cmd.print_warning(
            f"Observation is for {snap.observed_sha[:8]} but {branch} is at {result.head_sha[:8]}; "
            f"re-run with --refresh to observe the current tree")
    if not result.in_history:
        cmd.print_warning(
            f"{obs.observed_sha[:8]} is not in the history graph; series motifs cannot order it")
    return result


class ObserveCommand(BaseCommand):
    """Run a structural provider and store its observation in ``.gitview/graph.sqlite``."""

    def validate(self) -> None:
        pass

    def execute(self):
        repo_spec = self.get_option("repo", ".")
        branch = self.get_option("branch", "HEAD")
        provider_name = self.get_option("structural") or "graphify"
        source = self.get_option("source")
        refresh = self.get_option("refresh", False)
        as_json = self.get_option("json_output", False)

        if self.get_option("list_providers", False):
            for name in provider_names():
                ok, reason = get_provider(name).available()
                self.console.print(f"{name:<12}{'available' if ok else 'unavailable: ' + reason}")
            return provider_names()

        repo_path = Path(repo_spec).resolve()
        if not (repo_path / ".git").exists():
            self.print_error(f"Error: {repo_path} is not a git repository")
            sys.exit(1)

        # Make sure the history graph exists so the observation lands on the timeline.
        try:
            GraphUpdater(repo_path, branch=branch).sync()
        except Exception as exc:
            self.print_error(f"Error building graph: {exc}")
            sys.exit(1)

        result = run_observation(self, repo_path, provider_name, branch=branch,
                                 source=source, refresh=refresh, quiet=as_json)
        if as_json:
            payload = {
                'inserted': result.inserted,
                'drift': result.drift,
                'head_sha': result.head_sha,
                'observation': result.observation.to_dict(),
            }
            print(json.dumps(payload, indent=2))
            return payload
        return result
