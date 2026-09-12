"""Deterministic repository graph: persistent evidence substrate for GitView.

The graph layer calculates relationships (commit → file, file ↔ file
co-change, commit → PR, author → commit) itself, so later stages can ask an
LLM to *interpret* structure instead of rediscovering it. Nothing in this
package depends on an LLM backend.
"""

from .models import (
    GRAPH_SCHEMA_VERSION,
    PROJECTION_VERSION,
    FileCouplingEdge,
    FileDegree,
    FileTouchStats,
    GraphMetadata,
    GraphStats,
)
from .store import GraphStore
from .builder import GraphBuilder
from .updater import GraphUpdater, SyncResult
from .analysis.stats import compute_graph_stats

__all__ = [
    'GRAPH_SCHEMA_VERSION',
    'PROJECTION_VERSION',
    'FileCouplingEdge',
    'FileDegree',
    'FileTouchStats',
    'GraphMetadata',
    'GraphStats',
    'GraphStore',
    'GraphBuilder',
    'GraphUpdater',
    'SyncResult',
    'compute_graph_stats',
]
