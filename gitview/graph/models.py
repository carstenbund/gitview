"""Data models for the repository graph."""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List

#: Bump when the SQLite schema changes incompatibly (forces a rebuild).
GRAPH_SCHEMA_VERSION = 1

#: Bump when the co-change projection semantics change (forces re-projection).
PROJECTION_VERSION = 1

#: Commits touching more files than this contribute commit→file edges but no
#: file↔file pairs (a 6,000-file mechanical commit must not create ~18M edges).
DEFAULT_MAX_PROJECTION_FILES = 100


@dataclass(frozen=True)
class GraphMetadata:
    """Provenance of a built graph, stored in ``graph_metadata``."""

    schema_version: int
    repository_path: str
    branch: str
    built_at: str
    last_commit_hash: str
    last_commit_timestamp: str
    projection_version: int
    max_projection_files: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphMetadata':
        return cls(
            schema_version=int(data.get('schema_version', 0)),
            repository_path=str(data.get('repository_path', '')),
            branch=str(data.get('branch', 'HEAD')),
            built_at=str(data.get('built_at', '')),
            last_commit_hash=str(data.get('last_commit_hash', '')),
            last_commit_timestamp=str(data.get('last_commit_timestamp', '')),
            projection_version=int(data.get('projection_version', 0)),
            max_projection_files=int(data.get('max_projection_files', DEFAULT_MAX_PROJECTION_FILES)),
        )


@dataclass(frozen=True)
class FileTouchStats:
    """Per-file change activity ("most changed")."""

    path: str
    touch_count: int
    total_insertions: int
    total_deletions: int


@dataclass(frozen=True)
class FileCouplingEdge:
    """A file↔file co-change relationship ("most coupled")."""

    path_a: str
    path_b: str
    cochange_count: int
    jaccard: float
    first_seen: str
    last_seen: str


@dataclass(frozen=True)
class FileDegree:
    """Structural connectivity of a file in the co-change graph ("most connected")."""

    path: str
    degree: int
    weighted_degree: int


@dataclass
class GraphStats:
    """Summary statistics of a built graph; JSON-serializable via ``to_dict``."""

    commits: int
    merge_commits: int
    suppressed_commits: int
    files: int
    authors: int
    pull_requests: int
    commit_file_edges: int
    file_edges: int
    structural_snapshots: int = 0
    most_changed: List[FileTouchStats] = field(default_factory=list)
    most_coupled: List[FileCouplingEdge] = field(default_factory=list)
    most_connected: List[FileDegree] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'commits': self.commits,
            'merge_commits': self.merge_commits,
            'suppressed_commits': self.suppressed_commits,
            'files': self.files,
            'authors': self.authors,
            'pull_requests': self.pull_requests,
            'commit_file_edges': self.commit_file_edges,
            'file_edges': self.file_edges,
            'structural_snapshots': self.structural_snapshots,
            'most_changed': [asdict(x) for x in self.most_changed],
            'most_coupled': [asdict(x) for x in self.most_coupled],
            'most_connected': [asdict(x) for x in self.most_connected],
        }
