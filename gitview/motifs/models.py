"""Motif models: findings, evidence requirements and the shared detection context."""

from dataclasses import dataclass, field, asdict
from enum import Enum
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple

from ..graph.models import FileCouplingEdge
from ..graph.store import GraphStore
from ..structural.models import StructuralObservation, StructuralSnapshot


class Evidence(str, Enum):
    """What a motif needs before it can run."""

    HISTORICAL = 'historical'                  # the git-derived graph (always present)
    STRUCTURAL = 'structural'                  # at least one structural observation
    STRUCTURAL_SERIES = 'structural_series'    # two or more observations at different commits


@dataclass
class Thresholds:
    min_cochanges: int = 3       # co-changes before a pair counts as "repeatedly" coupled
    min_jaccard: float = 0.2     # co-change share of the pair's combined activity
    min_touches: int = 3         # commits a file needs before it is judged at all
    min_degree_growth: int = 3   # structural neighbours gained before centrality "grows"
    top: int = 20                # findings per motif

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MotifFinding:
    motif: str
    title: str
    files: List[str]
    summary: str
    confidence: float                      # 0.0 – 1.0
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.motif}:{'|'.join(self.files)}"

    @property
    def level(self) -> str:
        return 'high' if self.confidence >= 0.75 else 'medium' if self.confidence >= 0.5 else 'low'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'key': self.key, 'motif': self.motif, 'title': self.title, 'files': list(self.files),
            'summary': self.summary, 'confidence': round(self.confidence, 3), 'level': self.level,
            'evidence': self.evidence,
        }


@dataclass
class MotifReport:
    findings: List[MotifFinding]
    skipped: Dict[str, str]                # motif id → reason it did not run
    available: List[Evidence]
    observations: List[StructuralObservation]
    thresholds: Thresholds

    def by_motif(self) -> Dict[str, List[MotifFinding]]:
        grouped: Dict[str, List[MotifFinding]] = {}
        for f in self.findings:
            grouped.setdefault(f.motif, []).append(f)
        return grouped

    def to_dict(self) -> Dict[str, Any]:
        return {
            'available_evidence': [e.value for e in self.available],
            'observations': [o.to_dict() for o in self.observations],
            'thresholds': self.thresholds.to_dict(),
            'skipped': dict(self.skipped),
            'findings': [f.to_dict() for f in self.findings],
        }


class MotifContext:
    """Evidence shared by all motifs during one run. Loads lazily and caches."""

    def __init__(self, store: GraphStore, thresholds: Thresholds,
                 observations: List[StructuralObservation]) -> None:
        self.store = store
        self.t = thresholds
        self.observations = observations

    # ------------------------------------------------------------ historical

    @cached_property
    def coupling(self) -> List[FileCouplingEdge]:
        """Every co-change pair with at least one shared commit, strongest first."""
        return self.store.most_coupled(limit=10 ** 9, min_cochanges=1)

    @cached_property
    def coupling_index(self) -> Dict[Tuple[str, str], FileCouplingEdge]:
        return {(e.path_a, e.path_b): e for e in self.coupling}

    def cochange(self, a: str, b: str) -> Optional[FileCouplingEdge]:
        return self.coupling_index.get((min(a, b), max(a, b)))

    @cached_property
    def touches(self) -> Dict[str, int]:
        return self.store.touch_counts()

    # ------------------------------------------------------------ structural

    @cached_property
    def latest_observation(self) -> Optional[StructuralObservation]:
        return self.observations[-1] if self.observations else None

    @cached_property
    def earliest_observation(self) -> Optional[StructuralObservation]:
        return self.observations[0] if self.observations else None

    @cached_property
    def latest(self) -> Optional[StructuralSnapshot]:
        obs = self.latest_observation
        return self.store.load_structural_snapshot(obs.id) if obs else None

    @cached_property
    def earliest(self) -> Optional[StructuralSnapshot]:
        obs = self.earliest_observation
        if obs is None:
            return None
        if self.latest_observation is not None and obs.id == self.latest_observation.id:
            return self.latest
        return self.store.load_structural_snapshot(obs.id)

    @staticmethod
    def edges_between(snapshot: StructuralSnapshot, a: str, b: str):
        """Structural edges in either direction between two files."""
        index = _edge_index(snapshot)
        return index.get((a, b), []) + index.get((b, a), [])

    @staticmethod
    def undirected_pairs(snapshot: StructuralSnapshot):
        return {tuple(sorted((e.source, e.target))) for e in snapshot.edges}


_INDEX_CACHE: Dict[int, Dict[Tuple[str, str], list]] = {}


def _edge_index(snapshot: StructuralSnapshot):
    key = id(snapshot)
    if key not in _INDEX_CACHE:
        _INDEX_CACHE.clear()
        _INDEX_CACHE[key] = snapshot.edge_index()
    return _INDEX_CACHE[key]
