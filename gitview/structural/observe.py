"""Take a structural observation of a repository and store it in the graph."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from git import Repo

from ..graph.store import GraphStore
from ..graph.updater import default_graph_path
from .models import StructuralObservation, StructuralSnapshot
from .provider import get_provider


@dataclass
class ObserveResult:
    observation: StructuralObservation
    snapshot: StructuralSnapshot
    inserted: bool
    head_sha: str

    @property
    def drift(self) -> bool:
        """The analyser observed a different commit than the current branch tip."""
        return self.snapshot.observed_sha != self.head_sha

    @property
    def in_history(self) -> bool:
        return self.observation.sequence is not None


def observe_structure(
    repo_path: Union[str, Path],
    provider_name: str,
    *,
    branch: str = 'HEAD',
    source: Optional[Union[str, Path]] = None,
    refresh: bool = False,
    store_path: Optional[Union[str, Path]] = None,
) -> ObserveResult:
    """Ask ``provider_name`` for a snapshot at the branch tip and persist it.

    The history graph must already exist (``GraphUpdater.sync``) for the
    observation to be placed on the commit timeline; without it the snapshot
    is still stored, just with an unknown position.
    """
    repo_path = Path(repo_path).resolve()
    repo = Repo(repo_path)
    head_sha = repo.commit(branch).hexsha

    provider = get_provider(provider_name)
    snapshot = provider.snapshot(repo_path, head_sha, source=source, refresh=refresh)

    with GraphStore(store_path or default_graph_path(repo_path)) as store:
        store.initialize()
        sid, inserted = store.insert_structural_snapshot(snapshot)
        observation = next(o for o in store.structural_observations() if o.id == sid)
    return ObserveResult(observation, snapshot, inserted, head_sha)
