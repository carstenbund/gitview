"""Data models for the Adaptive Review Agent.

Defines discoveries, decisions, analysis context, and results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class DiscoveryType(Enum):
    """Types of discoveries the agent can make."""

    PATTERN = "pattern"  # Recurring behavior or structure
    ANOMALY = "anomaly"  # Unexpected deviation from norm
    RISK = "risk"  # Potential security/stability concern
    INSIGHT = "insight"  # Meaningful observation about codebase
    QUESTION = "question"  # Something needing clarification
    MILESTONE = "milestone"  # Significant achievement or checkpoint
    TREND = "trend"  # Direction of change over time


class AnalysisDepth(Enum):
    """How deeply to analyze a phase or component."""

    LIGHT = "light"  # Quick summary, minimal detail
    STANDARD = "standard"  # Normal analysis depth
    DEEP = "deep"  # Detailed hierarchical analysis
    EXHAUSTIVE = "exhaustive"  # Maximum detail, file-level analysis


class AnalysisState(Enum):
    """Current state of the analysis workflow."""

    INITIALIZING = "initializing"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    SUMMARIZING = "summarizing"
    INVESTIGATING = "investigating"  # Deep-diving into a discovery
    SYNTHESIZING = "synthesizing"
    NARRATING = "narrating"
    COMPLETED = "completed"
    PAUSED = "paused"  # Waiting for user input (future)


class DecisionType(Enum):
    """Types of decisions the agent can make."""

    CONTINUE = "continue"  # Proceed with current plan
    DEEPEN = "deepen"  # Investigate more deeply
    BROADEN = "broaden"  # Expand scope of investigation
    PIVOT = "pivot"  # Shift focus to new discovery
    SKIP = "skip"  # De-prioritize and move on
    CONCLUDE = "conclude"  # Mark investigation complete
    RECHUNK = "rechunk"  # Re-partition phases based on findings
    CLARIFY = "clarify"  # Flag for user attention (future)


@dataclass
class Discovery:
    """A finding that may influence subsequent analysis.

    Discoveries are first-class objects that capture what the agent
    learns during analysis. They drive adaptive decision-making.
    """

    discovery_type: DiscoveryType
    title: str
    description: str
    source: str  # Which component found this (e.g., "phase_summarizer")
    confidence: float  # 0.0 - 1.0 how certain we are
    significance: float  # 0.0 - 1.0 how important this is

    # Context
    phase_number: Optional[int] = None
    commit_hashes: List[str] = field(default_factory=list)
    file_paths: List[str] = field(default_factory=list)

    # Evidence and implications
    evidence: Dict[str, Any] = field(default_factory=dict)
    implications: List[str] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)

    # Metadata
    id: str = field(default_factory=lambda: f"disc_{uuid.uuid4().hex[:8]}")
    timestamp: datetime = field(default_factory=datetime.now)
    goal_relevance: float = 0.0  # Set when goals are provided

    # Tracking
    investigated: bool = False
    investigation_depth: Optional[AnalysisDepth] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for persistence."""
        return {
            "id": self.id,
            "discovery_type": self.discovery_type.value,
            "title": self.title,
            "description": self.description,
            "source": self.source,
            "confidence": self.confidence,
            "significance": self.significance,
            "phase_number": self.phase_number,
            "commit_hashes": self.commit_hashes,
            "file_paths": self.file_paths,
            "evidence": self.evidence,
            "implications": self.implications,
            "suggested_actions": self.suggested_actions,
            "timestamp": self.timestamp.isoformat(),
            "goal_relevance": self.goal_relevance,
            "investigated": self.investigated,
            "investigation_depth": (
                self.investigation_depth.value if self.investigation_depth else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Discovery":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            discovery_type=DiscoveryType(data["discovery_type"]),
            title=data["title"],
            description=data["description"],
            source=data["source"],
            confidence=data["confidence"],
            significance=data["significance"],
            phase_number=data.get("phase_number"),
            commit_hashes=data.get("commit_hashes", []),
            file_paths=data.get("file_paths", []),
            evidence=data.get("evidence", {}),
            implications=data.get("implications", []),
            suggested_actions=data.get("suggested_actions", []),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            goal_relevance=data.get("goal_relevance", 0.0),
            investigated=data.get("investigated", False),
            investigation_depth=(
                AnalysisDepth(data["investigation_depth"])
                if data.get("investigation_depth")
                else None
            ),
        )

    @property
    def priority_score(self) -> float:
        """Compute priority for decision-making."""
        base = self.significance * self.confidence
        if self.goal_relevance > 0:
            base *= 1 + self.goal_relevance
        if self.discovery_type == DiscoveryType.RISK:
            base *= 1.5  # Risks always get priority
        return min(base, 1.0)


@dataclass
class Decision:
    """A decision about what action to take next.

    Decisions are made by the DecisionEngine based on accumulated
    discoveries and the current analysis context.
    """

    decision_type: DecisionType
    reason: str
    target: Optional[Any] = None  # What to act on (phase, discovery, etc.)
    priority: float = 0.5  # 0.0 - 1.0
    parameters: Dict[str, Any] = field(default_factory=dict)

    # For tracking
    id: str = field(default_factory=lambda: f"dec_{uuid.uuid4().hex[:8]}")
    timestamp: datetime = field(default_factory=datetime.now)
    triggered_by: Optional[str] = None  # Discovery ID that triggered this

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "decision_type": self.decision_type.value,
            "reason": self.reason,
            "priority": self.priority,
            "parameters": self.parameters,
            "timestamp": self.timestamp.isoformat(),
            "triggered_by": self.triggered_by,
        }


@dataclass
class AnalysisContext:
    """Context maintained throughout adaptive analysis.

    Tracks the current state, accumulated discoveries, user goals,
    and configuration that guides decision-making.
    """

    repo_path: str
    state: AnalysisState = AnalysisState.INITIALIZING

    # User inputs
    user_goals: List[str] = field(default_factory=list)
    directives: Optional[str] = None
    critical_mode: bool = False

    # Accumulated knowledge
    discoveries: List[Discovery] = field(default_factory=list)
    decisions_made: List[Decision] = field(default_factory=list)

    # Focus and priorities
    current_focus: Optional[str] = None  # What we're currently investigating
    high_priority_items: List[str] = field(default_factory=list)

    # Adaptive settings
    max_iterations: int = 10
    max_deepening_per_phase: int = 2
    significance_threshold_high: float = 0.7
    significance_threshold_medium: float = 0.4
    discovery_confidence_threshold: float = 0.5

    # Tracking
    iteration_count: int = 0
    phases_deepened: Dict[int, int] = field(default_factory=dict)  # phase -> depth count

    def add_discovery(self, discovery: Discovery) -> None:
        """Add a discovery and update priorities if needed."""
        self.discoveries.append(discovery)

        if discovery.priority_score > self.significance_threshold_high:
            self.high_priority_items.append(discovery.id)

    def add_decision(self, decision: Decision) -> None:
        """Record a decision made."""
        self.decisions_made.append(decision)

    def can_deepen_phase(self, phase_number: int) -> bool:
        """Check if we can deepen analysis of a phase."""
        current_depth = self.phases_deepened.get(phase_number, 0)
        return current_depth < self.max_deepening_per_phase

    def record_phase_deepening(self, phase_number: int) -> None:
        """Record that we deepened a phase's analysis."""
        self.phases_deepened[phase_number] = (
            self.phases_deepened.get(phase_number, 0) + 1
        )

    def get_uninvestigated_discoveries(self) -> List[Discovery]:
        """Get discoveries that haven't been investigated yet."""
        return [d for d in self.discoveries if not d.investigated]

    def get_high_significance_discoveries(self) -> List[Discovery]:
        """Get discoveries above high significance threshold."""
        return [
            d
            for d in self.discoveries
            if d.significance > self.significance_threshold_high
        ]


@dataclass
class AnalysisResult:
    """Final result of adaptive analysis.

    Contains phases, discoveries, synthesis, and narrative.
    """

    phases: List[Any]  # List[Phase] - avoiding circular import
    discoveries: List[Discovery]
    narrative: Dict[str, str]
    context: AnalysisContext

    # Synthesis of findings
    key_findings: List[str] = field(default_factory=list)
    risk_summary: List[str] = field(default_factory=list)
    goal_alignment: Dict[str, float] = field(default_factory=dict)

    # Metadata
    total_iterations: int = 0
    total_decisions: int = 0
    analysis_depth_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "discoveries": [d.to_dict() for d in self.discoveries],
            "key_findings": self.key_findings,
            "risk_summary": self.risk_summary,
            "goal_alignment": self.goal_alignment,
            "total_iterations": self.total_iterations,
            "total_decisions": self.total_decisions,
            "analysis_depth_distribution": self.analysis_depth_distribution,
            "decisions_made": [d.to_dict() for d in self.context.decisions_made],
        }


# Signal patterns for discovery extraction
SECURITY_KEYWORDS = [
    "security",
    "vulnerability",
    "cve",
    "exploit",
    "injection",
    "xss",
    "csrf",
    "auth",
    "authentication",
    "authorization",
    "password",
    "credential",
    "secret",
    "token",
    "encrypt",
    "decrypt",
    "sanitize",
    "escape",
    "ssl",
    "tls",
    "https",
    "certificate",
]

BREAKING_CHANGE_KEYWORDS = [
    "breaking",
    "deprecated",
    "removal",
    "removed",
    "migration",
    "upgrade",
    "downgrade",
    "incompatible",
    "major version",
    "api change",
]

REFACTOR_KEYWORDS = [
    "refactor",
    "restructure",
    "reorganize",
    "cleanup",
    "clean up",
    "simplify",
    "extract",
    "rename",
    "move",
    "split",
    "merge",
    "consolidate",
    "modularize",
]

ARCHITECTURE_KEYWORDS = [
    "architecture",
    "microservice",
    "monolith",
    "service",
    "module",
    "layer",
    "component",
    "pattern",
    "design",
    "infrastructure",
    "framework",
    "migration",
]
