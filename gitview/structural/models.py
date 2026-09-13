"""Neutral structural evidence model.

Structural observations describe the *present* shape of the code base
(which file depends on which, how files cluster) as seen by an external
analyser at one commit. GitView owns this model; providers translate their
own output into it, so nothing above this layer knows which tool produced
the observation.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

#: Bump when the structural tables change incompatibly.
STRUCTURAL_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StructuralNode:
    """One file as seen by the structural analyser."""

    path: str
    kind: str = 'code'                    # code | document | other (provider-normalized)
    community: Optional[str] = None       # opaque cluster id, if the provider clusters
    community_name: Optional[str] = None
    symbols: int = 0                      # number of symbols the provider found in the file

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StructuralEdge:
    """A directed file → file dependency (``source`` depends on ``target``)."""

    source: str
    target: str
    relation: str                         # imports | calls | inherits | references | ...
    weight: float = 1.0
    count: int = 1                        # symbol-level edges folded into this file edge

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StructuralSnapshot:
    """A complete structural observation of the repository at one commit."""

    provider: str
    provider_version: str
    observed_sha: str
    observed_at: str                      # ISO timestamp of when the observation was taken
    source: str                           # where the raw observation came from (path/URL)
    content_hash: str                     # hash of the raw observation, for provenance
    nodes: List[StructuralNode] = field(default_factory=list)
    edges: List[StructuralEdge] = field(default_factory=list)

    # ------------------------------------------------------------- helpers

    def paths(self) -> Set[str]:
        return {n.path for n in self.nodes}

    def edge_index(self) -> Dict[Tuple[str, str], List[StructuralEdge]]:
        index: Dict[Tuple[str, str], List[StructuralEdge]] = {}
        for e in self.edges:
            index.setdefault((e.source, e.target), []).append(e)
        return index

    def degree(self) -> Dict[str, int]:
        """Undirected number of distinct files each file is connected to."""
        neighbours: Dict[str, Set[str]] = {}
        for e in self.edges:
            neighbours.setdefault(e.source, set()).add(e.target)
            neighbours.setdefault(e.target, set()).add(e.source)
        return {p: len(n) for p, n in neighbours.items()}

    def to_dict(self) -> Dict[str, Any]:
        return {
            'provider': self.provider,
            'provider_version': self.provider_version,
            'observed_sha': self.observed_sha,
            'observed_at': self.observed_at,
            'source': self.source,
            'content_hash': self.content_hash,
            'nodes': [n.to_dict() for n in self.nodes],
            'edges': [e.to_dict() for e in self.edges],
        }


@dataclass(frozen=True)
class StructuralObservation:
    """Provenance of a stored snapshot (one row of ``structural_snapshots``)."""

    id: int
    provider: str
    provider_version: str
    observed_sha: str
    observed_at: str
    source: str
    content_hash: str
    node_count: int
    edge_count: int
    sequence: Optional[int] = None        # position of observed_sha in the history graph, if known

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def fold_symbol_edges(
    pairs: Iterable[Tuple[str, str, str, float]],
) -> List[StructuralEdge]:
    """Fold ``(source_file, target_file, relation, weight)`` symbol edges into file edges.

    Self-edges (intra-file) are dropped; the result is sorted for determinism.
    """
    acc: Dict[Tuple[str, str, str], Tuple[float, int]] = {}
    for src, dst, relation, weight in pairs:
        if not src or not dst or src == dst:
            continue
        w, c = acc.get((src, dst, relation), (0.0, 0))
        acc[(src, dst, relation)] = (w + float(weight), c + 1)
    return [StructuralEdge(s, d, r, round(w, 4), c)
            for (s, d, r), (w, c) in sorted(acc.items())]
