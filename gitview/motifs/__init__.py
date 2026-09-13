"""Recurring historical and architectural motifs detected from GitView evidence.

Historical motifs run everywhere. Structural motifs combine git history with
structural observations (see ``gitview.structural``) and are skipped, with a
reason, until such an observation exists.
"""

from .base import Motif
from .engine import ALL_MOTIFS, available_evidence, motif_catalog, run_motifs
from .models import Evidence, MotifContext, MotifFinding, MotifReport, Thresholds

__all__ = [
    'ALL_MOTIFS', 'Evidence', 'Motif', 'MotifContext', 'MotifFinding', 'MotifReport',
    'Thresholds', 'available_evidence', 'motif_catalog', 'run_motifs',
]
