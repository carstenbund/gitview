"""Graph construction from CommitRecords (no LLM)."""

from datetime import datetime, timezone
from typing import List

from ..extractor import CommitRecord
from .models import (
    DEFAULT_MAX_PROJECTION_FILES,
    GRAPH_SCHEMA_VERSION,
    PROJECTION_VERSION,
    GraphMetadata,
)
from .store import GraphStore


class GraphBuilder:
    """Insert primary evidence into a ``GraphStore`` and run the projections.

    ``build`` starts from an empty store; ``update`` appends commits that are
    not yet stored and folds only those into the co-change projection.
    """

    def __init__(
        self,
        store: GraphStore,
        *,
        repository_path: str,
        branch: str = "HEAD",
        max_projection_files: int = DEFAULT_MAX_PROJECTION_FILES,
    ) -> None:
        self.store = store
        self.repository_path = repository_path
        self.branch = branch
        self.max_projection_files = max_projection_files

    def build(self, records: List[CommitRecord]) -> GraphMetadata:
        """Reset the store and build the whole graph from ``records``."""
        self.store.reset()
        self.store.insert_commits(records, self.max_projection_files)
        self.store.apply_cochange_projection(since_sequence=0)
        return self._write_metadata(records)

    def update(self, records: List[CommitRecord]) -> GraphMetadata:
        """Append new commits and extend the projection incrementally."""
        self.store.initialize()
        previous_max = self.store.max_sequence()
        self.store.insert_commits(records, self.max_projection_files)
        self.store.apply_cochange_projection(since_sequence=previous_max)
        return self._write_metadata(records)

    def _write_metadata(self, records: List[CommitRecord]) -> GraphMetadata:
        previous = self.store.get_metadata()
        if records:
            last_hash = records[-1].commit_hash
            last_ts = records[-1].timestamp
        elif previous is not None:
            last_hash, last_ts = previous.last_commit_hash, previous.last_commit_timestamp
        else:
            last_hash, last_ts = '', ''

        metadata = GraphMetadata(
            schema_version=GRAPH_SCHEMA_VERSION,
            repository_path=self.repository_path,
            branch=self.branch,
            built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            last_commit_hash=last_hash,
            last_commit_timestamp=last_ts,
            projection_version=PROJECTION_VERSION,
            max_projection_files=self.max_projection_files,
        )
        self.store.set_metadata(metadata)
        return metadata
