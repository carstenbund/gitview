"""Decision Engine - Makes adaptive decisions based on discoveries.

Evaluates accumulated discoveries and context to decide what
analysis actions to take next.
"""

from typing import Any, List, Optional, Tuple
import re

from .models import (
    Decision,
    DecisionType,
    Discovery,
    DiscoveryType,
    AnalysisContext,
    AnalysisDepth,
)


class DecisionEngine:
    """Makes decisions about analysis workflow based on discoveries.

    Implements the OODA loop's Decide phase by evaluating discoveries,
    considering context and goals, and choosing appropriate next actions.
    """

    def __init__(
        self,
        risk_priority_boost: float = 1.5,
        goal_relevance_boost: float = 1.3,
        deepening_threshold: float = 0.7,
    ):
        """Initialize the decision engine.

        Args:
            risk_priority_boost: Multiplier for risk-type discoveries
            goal_relevance_boost: Multiplier for goal-relevant discoveries
            deepening_threshold: Minimum score to trigger deepening
        """
        self.risk_priority_boost = risk_priority_boost
        self.goal_relevance_boost = goal_relevance_boost
        self.deepening_threshold = deepening_threshold

    def evaluate(
        self,
        discoveries: List[Discovery],
        context: AnalysisContext,
    ) -> List[Decision]:
        """Evaluate discoveries and decide on next actions.

        Args:
            discoveries: Recent discoveries to evaluate
            context: Current analysis context

        Returns:
            List of decisions, ordered by priority
        """
        decisions = []

        # Check for iteration limits
        if context.iteration_count >= context.max_iterations:
            decisions.append(
                Decision(
                    decision_type=DecisionType.CONCLUDE,
                    reason="Maximum iterations reached",
                    priority=1.0,
                )
            )
            return decisions

        # Process each discovery
        for discovery in discoveries:
            decision = self._evaluate_discovery(discovery, context)
            if decision:
                decisions.append(decision)

        # Apply goal-based prioritization
        if context.user_goals:
            self._apply_goal_priorities(decisions, discoveries, context)

        # Sort by priority
        decisions.sort(key=lambda d: d.priority, reverse=True)

        # Filter to actionable decisions
        return self._filter_actionable(decisions, context)

    def determine_phase_depth(
        self,
        phase: Any,
        discoveries: List[Discovery],
        context: AnalysisContext,
    ) -> AnalysisDepth:
        """Determine how deeply to analyze a phase.

        Args:
            phase: Phase to analyze
            discoveries: Discoveries so far (may include from this phase)
            context: Analysis context

        Returns:
            Recommended analysis depth
        """
        signals = []

        # Large LOC changes suggest significant work
        if hasattr(phase, "loc_delta_percent"):
            if abs(phase.loc_delta_percent) > 50:
                signals.append(("major_loc_change", 0.8))
            elif abs(phase.loc_delta_percent) > 30:
                signals.append(("significant_loc_change", 0.6))

        # Phase-specific flags
        if hasattr(phase, "has_large_deletion") and phase.has_large_deletion:
            signals.append(("large_deletion", 0.7))
        if hasattr(phase, "has_large_addition") and phase.has_large_addition:
            signals.append(("large_addition", 0.6))
        if hasattr(phase, "has_refactor") and phase.has_refactor:
            signals.append(("refactoring", 0.5))
        if hasattr(phase, "readme_changed") and phase.readme_changed:
            signals.append(("readme_change", 0.4))

        # High-significance discoveries for this phase
        phase_discoveries = [
            d for d in discoveries if d.phase_number == phase.phase_number
        ]
        high_sig_count = sum(1 for d in phase_discoveries if d.significance > 0.7)
        if high_sig_count >= 2:
            signals.append(("multiple_discoveries", 0.8))
        elif high_sig_count == 1:
            signals.append(("significant_discovery", 0.6))

        # Security discoveries always warrant deep analysis
        security_discoveries = [
            d for d in phase_discoveries if d.discovery_type == DiscoveryType.RISK
        ]
        if security_discoveries:
            signals.append(("security_concern", 0.9))

        # Goal-relevant discoveries
        goal_relevant = [d for d in phase_discoveries if d.goal_relevance > 0.5]
        if goal_relevant:
            signals.append(("goal_relevant", 0.7))

        # Aggregate score
        score = self._aggregate_signals(signals)

        # Map score to depth
        if score > 0.75:
            return AnalysisDepth.DEEP
        elif score > 0.5:
            return AnalysisDepth.STANDARD
        elif score > 0.25:
            return AnalysisDepth.LIGHT
        else:
            return AnalysisDepth.LIGHT

    def should_rechunk(
        self,
        phases: List[Any],
        discoveries: List[Discovery],
        context: AnalysisContext,
    ) -> Tuple[bool, Optional[str]]:
        """Determine if phases should be re-chunked based on discoveries.

        Args:
            phases: Current phase list
            discoveries: Accumulated discoveries
            context: Analysis context

        Returns:
            (should_rechunk, reason) tuple
        """
        # Look for boundary anomalies
        boundary_anomalies = [
            d
            for d in discoveries
            if d.discovery_type == DiscoveryType.ANOMALY
            and "boundary" in d.title.lower()
        ]
        if boundary_anomalies:
            return True, f"Boundary anomaly detected: {boundary_anomalies[0].title}"

        # Check for phases that are too small or too large
        tiny_phases = [p for p in phases if p.commit_count < 3]
        huge_phases = [p for p in phases if p.commit_count > 200]

        if len(tiny_phases) > len(phases) * 0.3:
            return True, f"Too many tiny phases ({len(tiny_phases)})"

        if huge_phases:
            return True, f"Oversized phases detected ({len(huge_phases)})"

        return False, None

    def prioritize_discoveries(
        self,
        discoveries: List[Discovery],
        context: AnalysisContext,
    ) -> List[Discovery]:
        """Prioritize discoveries based on context and goals.

        Args:
            discoveries: Discoveries to prioritize
            context: Analysis context with goals

        Returns:
            Discoveries sorted by priority
        """
        if context.user_goals:
            self._compute_goal_relevance(discoveries, context.user_goals)

        # Sort by computed priority
        return sorted(discoveries, key=lambda d: d.priority_score, reverse=True)

    def _evaluate_discovery(
        self,
        discovery: Discovery,
        context: AnalysisContext,
    ) -> Optional[Decision]:
        """Evaluate a single discovery and return a decision if warranted.

        Args:
            discovery: Discovery to evaluate
            context: Analysis context

        Returns:
            Decision if action warranted, None otherwise
        """
        # Already investigated
        if discovery.investigated:
            return None

        priority = discovery.priority_score

        # Risk discoveries get boosted priority
        if discovery.discovery_type == DiscoveryType.RISK:
            priority *= self.risk_priority_boost

        # High significance discoveries should be deepened
        if discovery.significance > self.deepening_threshold:
            if discovery.phase_number is not None:
                if context.can_deepen_phase(discovery.phase_number):
                    return Decision(
                        decision_type=DecisionType.DEEPEN,
                        reason=f"High significance discovery: {discovery.title}",
                        target=discovery,
                        priority=priority,
                        triggered_by=discovery.id,
                        parameters={
                            "phase_number": discovery.phase_number,
                            "discovery_type": discovery.discovery_type.value,
                        },
                    )

        # Anomalies warrant investigation
        if discovery.discovery_type == DiscoveryType.ANOMALY:
            if priority > 0.5:
                return Decision(
                    decision_type=DecisionType.DEEPEN,
                    reason=f"Anomaly requires investigation: {discovery.title}",
                    target=discovery,
                    priority=priority * 0.9,
                    triggered_by=discovery.id,
                )

        # Questions may need clarification (future: interactive mode)
        if discovery.discovery_type == DiscoveryType.QUESTION:
            return Decision(
                decision_type=DecisionType.CLARIFY,
                reason=f"Clarification needed: {discovery.title}",
                target=discovery,
                priority=priority * 0.7,
                triggered_by=discovery.id,
            )

        return None

    def _apply_goal_priorities(
        self,
        decisions: List[Decision],
        discoveries: List[Discovery],
        context: AnalysisContext,
    ) -> None:
        """Boost priorities for goal-relevant decisions.

        Args:
            decisions: Decisions to modify
            discoveries: Source discoveries
            context: Context with user goals
        """
        # Build discovery ID to relevance map
        relevance_map = {d.id: d.goal_relevance for d in discoveries}

        for decision in decisions:
            if decision.triggered_by and decision.triggered_by in relevance_map:
                relevance = relevance_map[decision.triggered_by]
                if relevance > 0.5:
                    decision.priority *= self.goal_relevance_boost
                    decision.reason += f" (goal-relevant: {relevance:.0%})"

    def _compute_goal_relevance(
        self,
        discoveries: List[Discovery],
        goals: List[str],
    ) -> None:
        """Compute goal relevance for each discovery.

        Args:
            discoveries: Discoveries to annotate
            goals: User-provided goals
        """
        # Extract keywords from goals
        goal_text = " ".join(goals).lower()
        goal_keywords = set(re.findall(r"\b\w{4,}\b", goal_text))

        for discovery in discoveries:
            # Simple keyword overlap scoring
            disc_text = f"{discovery.title} {discovery.description}".lower()
            disc_keywords = set(re.findall(r"\b\w{4,}\b", disc_text))

            if goal_keywords and disc_keywords:
                overlap = len(goal_keywords & disc_keywords)
                max_possible = min(len(goal_keywords), len(disc_keywords))
                discovery.goal_relevance = min(overlap / max_possible, 1.0) if max_possible > 0 else 0.0

    def _filter_actionable(
        self,
        decisions: List[Decision],
        context: AnalysisContext,
    ) -> List[Decision]:
        """Filter decisions to those that are actionable.

        Args:
            decisions: All decisions
            context: Analysis context

        Returns:
            Filtered list of actionable decisions
        """
        actionable = []

        for decision in decisions:
            # Skip clarify decisions for now (no interactive mode)
            if decision.decision_type == DecisionType.CLARIFY:
                continue

            # Check phase deepening limits
            if decision.decision_type == DecisionType.DEEPEN:
                phase_num = decision.parameters.get("phase_number")
                if phase_num is not None and not context.can_deepen_phase(phase_num):
                    continue

            actionable.append(decision)

        return actionable

    def _aggregate_signals(self, signals: List[Tuple[str, float]]) -> float:
        """Aggregate signal weights into a single score.

        Uses a soft-max approach to prevent saturation.

        Args:
            signals: List of (signal_name, weight) tuples

        Returns:
            Aggregated score between 0 and 1
        """
        if not signals:
            return 0.0

        weights = [w for _, w in signals]

        # Use complement product for soft-max effect
        complement_product = 1.0
        for w in weights:
            complement_product *= 1 - w

        return 1 - complement_product


class GoalMatcher:
    """Matches discoveries and phases against user-provided goals."""

    def __init__(self, goals: List[str]):
        """Initialize with user goals.

        Args:
            goals: List of goal statements
        """
        self.goals = goals
        self.goal_keywords = self._extract_keywords(goals)

    def match_discovery(self, discovery: Discovery) -> float:
        """Compute relevance score for a discovery.

        Args:
            discovery: Discovery to match

        Returns:
            Relevance score 0.0 to 1.0
        """
        disc_keywords = self._extract_keywords(
            [discovery.title, discovery.description]
        )

        return self._keyword_overlap(disc_keywords)

    def match_phase(self, phase: Any) -> float:
        """Compute relevance score for a phase.

        Args:
            phase: Phase to match

        Returns:
            Relevance score 0.0 to 1.0
        """
        # Build text from phase content
        texts = []
        if hasattr(phase, "summary") and phase.summary:
            texts.append(phase.summary)

        for commit in phase.commits:
            texts.append(commit.commit_message)

        if not texts:
            return 0.0

        phase_keywords = self._extract_keywords(texts)
        return self._keyword_overlap(phase_keywords)

    def _extract_keywords(self, texts: List[str]) -> set:
        """Extract keywords from text list.

        Args:
            texts: List of text strings

        Returns:
            Set of lowercase keywords
        """
        combined = " ".join(texts).lower()
        # Extract words with 4+ characters
        words = set(re.findall(r"\b\w{4,}\b", combined))
        # Filter common stop words
        stop_words = {
            "that",
            "this",
            "with",
            "from",
            "have",
            "been",
            "were",
            "being",
            "would",
            "could",
            "should",
            "their",
            "there",
            "these",
            "those",
            "which",
            "about",
            "into",
            "more",
            "some",
            "such",
            "than",
            "then",
            "them",
            "when",
            "what",
            "where",
            "will",
            "also",
        }
        return words - stop_words

    def _keyword_overlap(self, text_keywords: set) -> float:
        """Compute keyword overlap score.

        Args:
            text_keywords: Keywords from text to match

        Returns:
            Overlap score 0.0 to 1.0
        """
        if not self.goal_keywords or not text_keywords:
            return 0.0

        overlap = len(self.goal_keywords & text_keywords)
        max_possible = min(len(self.goal_keywords), len(text_keywords))

        return min(overlap / max_possible, 1.0) if max_possible > 0 else 0.0
