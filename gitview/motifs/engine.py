"""Run motifs against a graph store, honouring each motif's evidence requirements."""

from typing import Dict, Iterable, List, Optional, Type

from ..graph.store import GraphStore
from .base import Motif
from .historical import HISTORICAL_MOTIFS
from .models import Evidence, MotifContext, MotifReport, Thresholds
from .structural import STRUCTURAL_MOTIFS

ALL_MOTIFS: List[Type[Motif]] = [*HISTORICAL_MOTIFS, *STRUCTURAL_MOTIFS]


def motif_catalog() -> List[Motif]:
    return [cls() for cls in ALL_MOTIFS]


def available_evidence(store: GraphStore, provider: Optional[str] = None):
    observations = store.structural_observations(provider)
    available = [Evidence.HISTORICAL]
    if observations:
        available.append(Evidence.STRUCTURAL)
        if len({o.observed_sha for o in observations}) >= 2:
            available.append(Evidence.STRUCTURAL_SERIES)
    return available, observations


def run_motifs(
    store: GraphStore,
    *,
    only: Optional[Iterable[str]] = None,
    thresholds: Optional[Thresholds] = None,
    provider: Optional[str] = None,
) -> MotifReport:
    thresholds = thresholds or Thresholds()
    available, observations = available_evidence(store, provider)
    if provider is None and observations:
        # Series comparisons must not mix analysers; use the most recent one.
        latest_provider = observations[-1].provider
        observations = [o for o in observations if o.provider == latest_provider]
        available, _ = available_evidence(store, latest_provider)

    wanted = set(only) if only else None
    ctx = MotifContext(store, thresholds, observations)
    findings, skipped = [], {}
    _reason = {
        Evidence.STRUCTURAL: 'no structural observation stored (run `gitview observe --structural <provider>`)',
        Evidence.STRUCTURAL_SERIES: 'needs structural observations at two or more commits',
    }
    for motif in motif_catalog():
        if wanted is not None and motif.id not in wanted:
            continue
        missing = [e for e in motif.requires if e not in available]
        if missing:
            skipped[motif.id] = _reason.get(missing[0], f'missing evidence: {missing[0].value}')
            continue
        findings.extend(motif.detect(ctx))
    return MotifReport(findings, skipped, available, observations, thresholds)
