"""Base class for motifs.

A motif is a recurring historical or architectural pattern. Each motif
declares the evidence it needs; the engine only runs it when that evidence
exists, so historical motifs work with nothing but git and structural motifs
appear once an observation has been stored.
"""

from abc import ABC, abstractmethod
from typing import FrozenSet, List

from .models import Evidence, MotifContext, MotifFinding


class Motif(ABC):
    id: str = ''
    title: str = ''
    #: One-line interpretation shown in listings.
    interpretation: str = ''
    requires: FrozenSet[Evidence] = frozenset({Evidence.HISTORICAL})

    @abstractmethod
    def detect(self, ctx: MotifContext) -> List[MotifFinding]:
        """Return findings, strongest first."""

    @property
    def structural(self) -> bool:
        return Evidence.STRUCTURAL in self.requires or Evidence.STRUCTURAL_SERIES in self.requires

    def _finding(self, files, summary, confidence, **evidence) -> MotifFinding:
        return MotifFinding(self.id, self.title, list(files), summary,
                            max(0.0, min(1.0, confidence)), evidence)
