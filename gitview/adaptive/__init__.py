"""Adaptive Review Agent - Discovery-driven git history analysis.

This module transforms GitView from a procedural pipeline into an adaptive
analysis system that reacts to discoveries during the review process.
"""

from .models import (
    Discovery,
    DiscoveryType,
    AnalysisDepth,
    AnalysisState,
    Decision,
    DecisionType,
    AnalysisContext,
    AnalysisResult,
)
from .discovery_extractor import DiscoveryExtractor
from .decision_engine import DecisionEngine
from .agent import AdaptiveReviewAgent

__all__ = [
    # Models
    "Discovery",
    "DiscoveryType",
    "AnalysisDepth",
    "AnalysisState",
    "Decision",
    "DecisionType",
    "AnalysisContext",
    "AnalysisResult",
    # Components
    "DiscoveryExtractor",
    "DecisionEngine",
    "AdaptiveReviewAgent",
]
