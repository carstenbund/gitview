"""Keep ``<repo>/.gitview/graph.sqlite`` in sync with the repository.

Decision table (evaluated in order):

* no store / no metadata / schema or projection version mismatch / branch
  changed / stored head no longer an ancestor of the branch tip → **rebuild**
* stored head == branch tip → **unchanged**
* otherwise → **update** with the commits after the stored head

History rewrites (rebase, reset, force push, branch replacement) fall into
the first case, so incremental updates are never applied across them.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Union

from git import Repo

from ..extractor import CommitRecord, GitHistoryExtractor
from ..history_cache import ensure_gitview_dir, gitview_dir, head_descends_from, load_or_extract_history
from .builder import GraphBuilder
from .models import (
    DEFAULT_MAX_PROJECTION_FILES,
    GRAPH_SCHEMA_VERSION,
    PROJECTION_VERSION,
    GraphMetadata,
)
from .store import GraphStore

GRAPH_DB_FILE = "graph.sqlite"

RecordsLoader = Callable[[], List[CommitRecord]]


@dataclass
class SyncResult:
    action: str            # 'built' | 'updated' | 'unchanged'
    new_commits: int
    metadata: GraphMetadata
    reason: str = ''


def default_graph_path(repo_path: Union[str, Path]) -> Path:
    return gitview_dir(Path(repo_path)) / GRAPH_DB_FILE


class GraphUpdater:
    def __init__(
        self,
        repo_path: Union[str, Path],
        *,
        branch: str = "HEAD",
        max_projection_files: int = DEFAULT_MAX_PROJECTION_FILES,
        store_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.branch = branch
        self.max_projection_files = max_projection_files
        self.store_path = Path(store_path) if store_path else default_graph_path(self.repo_path)

    # ------------------------------------------------------------------

    def _default_loader(self, repo: Repo, tip_sha: str) -> RecordsLoader:
        def load() -> List[CommitRecord]:
            extractor = GitHistoryExtractor(str(self.repo_path))
            return load_or_extract_history(
                extractor, repo, self.repo_path, branch=self.branch, head_sha=tip_sha,
            )
        return load

    def sync(self, *, rebuild: bool = False, records_loader: Optional[RecordsLoader] = None) -> SyncResult:
        """Bring the store up to date with the branch tip and return what happened."""
        repo = Repo(str(self.repo_path))
        tip_sha = repo.commit(self.branch).hexsha
        loader = records_loader or self._default_loader(repo, tip_sha)

        if self.store_path.parent == gitview_dir(self.repo_path):
            ensure_gitview_dir(self.repo_path)
        else:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)

        with GraphStore(self.store_path) as store:
            store.initialize()
            builder = GraphBuilder(
                store,
                repository_path=str(self.repo_path),
                branch=self.branch,
                max_projection_files=self.max_projection_files,
            )
            meta = store.get_metadata()

            reason = self._rebuild_reason(repo, meta, tip_sha, rebuild)
            if reason:
                records = loader()
                metadata = builder.build(records)
                return SyncResult('built', len(records), metadata, reason)

            if meta.last_commit_hash == tip_sha:
                return SyncResult('unchanged', 0, meta, 'graph head matches branch tip')

            records = loader()
            new_records = _records_after(records, meta.last_commit_hash)
            if new_records is None:
                # Cached history does not contain the stored head: be safe.
                metadata = builder.build(records)
                return SyncResult('built', len(records), metadata,
                                  'stored head not found in extracted history')

            metadata = builder.update(new_records)
            return SyncResult('updated', len(new_records), metadata,
                              f'{len(new_records)} new commit(s) since {meta.last_commit_hash[:8]}')

    def _rebuild_reason(self, repo: Repo, meta: Optional[GraphMetadata],
                        tip_sha: str, rebuild: bool) -> str:
        if rebuild:
            return 'rebuild requested'
        if meta is None:
            return 'no existing graph'
        if meta.schema_version != GRAPH_SCHEMA_VERSION:
            return f'schema version changed ({meta.schema_version} -> {GRAPH_SCHEMA_VERSION})'
        if meta.projection_version != PROJECTION_VERSION:
            return f'projection version changed ({meta.projection_version} -> {PROJECTION_VERSION})'
        if meta.max_projection_files != self.max_projection_files:
            return 'projection cap changed'
        if meta.branch != self.branch:
            return f'branch changed ({meta.branch} -> {self.branch})'
        if not meta.last_commit_hash:
            return 'existing graph is empty'
        if not head_descends_from(repo, meta.last_commit_hash, tip_sha):
            return 'history rewritten (stored head is not an ancestor of the branch tip)'
        return ''


def _records_after(records: List[CommitRecord], last_hash: str) -> Optional[List[CommitRecord]]:
    """Records strictly after ``last_hash`` in chronological order, or None if absent."""
    for index in range(len(records) - 1, -1, -1):
        if records[index].commit_hash == last_hash:
            return records[index + 1:]
    return None
