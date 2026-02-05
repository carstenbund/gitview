"""Tests for the Adaptive Review Agent components."""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock

from gitview.adaptive.models import (
    Discovery,
    DiscoveryType,
    Decision,
    DecisionType,
    AnalysisDepth,
    AnalysisState,
    AnalysisContext,
    AnalysisResult,
)
from gitview.adaptive.discovery_extractor import DiscoveryExtractor
from gitview.adaptive.decision_engine import DecisionEngine, GoalMatcher


class TestDiscoveryModel:
    """Tests for Discovery data model."""

    def test_discovery_creation(self):
        """Test basic discovery creation."""
        disc = Discovery(
            discovery_type=DiscoveryType.RISK,
            title="Security vulnerability detected",
            description="SQL injection possible in user input",
            source="commit_analyzer",
            confidence=0.9,
            significance=0.95,
            phase_number=3,
        )

        assert disc.discovery_type == DiscoveryType.RISK
        assert disc.title == "Security vulnerability detected"
        assert disc.confidence == 0.9
        assert disc.significance == 0.95
        assert disc.phase_number == 3
        assert disc.id.startswith("disc_")

    def test_discovery_priority_score(self):
        """Test priority score calculation."""
        # High significance, high confidence
        disc = Discovery(
            discovery_type=DiscoveryType.INSIGHT,
            title="Test",
            description="Test",
            source="test",
            confidence=0.9,
            significance=0.8,
        )
        assert disc.priority_score == pytest.approx(0.72, rel=0.1)

        # Risk type gets boost
        risk_disc = Discovery(
            discovery_type=DiscoveryType.RISK,
            title="Test",
            description="Test",
            source="test",
            confidence=0.9,
            significance=0.8,
        )
        assert risk_disc.priority_score > disc.priority_score

    def test_discovery_serialization(self):
        """Test discovery to_dict and from_dict."""
        disc = Discovery(
            discovery_type=DiscoveryType.ANOMALY,
            title="Large code deletion",
            description="50% of codebase removed",
            source="metrics_analyzer",
            confidence=0.95,
            significance=0.9,
            phase_number=5,
            evidence={"loc_delta": -10000},
            implications=["Major refactoring"],
        )

        data = disc.to_dict()
        restored = Discovery.from_dict(data)

        assert restored.discovery_type == disc.discovery_type
        assert restored.title == disc.title
        assert restored.confidence == disc.confidence
        assert restored.evidence == disc.evidence


class TestDecisionModel:
    """Tests for Decision data model."""

    def test_decision_creation(self):
        """Test basic decision creation."""
        decision = Decision(
            decision_type=DecisionType.DEEPEN,
            reason="High significance anomaly detected",
            priority=0.85,
        )

        assert decision.decision_type == DecisionType.DEEPEN
        assert decision.priority == 0.85
        assert decision.id.startswith("dec_")


class TestAnalysisContext:
    """Tests for AnalysisContext."""

    def test_context_creation(self):
        """Test context initialization."""
        ctx = AnalysisContext(
            repo_path="/path/to/repo",
            user_goals=["Implement OAuth", "Fix security bugs"],
            critical_mode=True,
        )

        assert ctx.repo_path == "/path/to/repo"
        assert len(ctx.user_goals) == 2
        assert ctx.critical_mode is True
        assert ctx.state == AnalysisState.INITIALIZING

    def test_context_add_discovery(self):
        """Test adding discoveries to context."""
        ctx = AnalysisContext(repo_path="/test")

        disc = Discovery(
            discovery_type=DiscoveryType.RISK,
            title="Security issue",
            description="Test",
            source="test",
            confidence=0.9,
            significance=0.85,
        )

        ctx.add_discovery(disc)

        assert len(ctx.discoveries) == 1
        assert disc.id in ctx.high_priority_items

    def test_context_phase_deepening_limits(self):
        """Test phase deepening tracking."""
        ctx = AnalysisContext(repo_path="/test", max_deepening_per_phase=2)

        assert ctx.can_deepen_phase(1) is True

        ctx.record_phase_deepening(1)
        assert ctx.can_deepen_phase(1) is True

        ctx.record_phase_deepening(1)
        assert ctx.can_deepen_phase(1) is False


class TestDiscoveryExtractor:
    """Tests for DiscoveryExtractor."""

    def create_mock_commit(self, message, insertions=10, deletions=5):
        """Helper to create mock commit objects."""
        commit = Mock()
        commit.commit_message = message
        commit.commit_hash = "abc123"
        commit.insertions = insertions
        commit.deletions = deletions
        commit.timestamp = "2025-01-15T10:00:00"
        commit.author = "test@example.com"
        return commit

    def create_mock_phase(self, commits, loc_delta_percent=10, phase_number=1):
        """Helper to create mock phase objects."""
        phase = Mock()
        phase.commits = commits
        phase.phase_number = phase_number
        phase.loc_delta = sum(c.insertions - c.deletions for c in commits)
        phase.loc_delta_percent = loc_delta_percent
        phase.total_insertions = sum(c.insertions for c in commits)
        phase.total_deletions = sum(c.deletions for c in commits)
        phase.has_large_deletion = loc_delta_percent < -30
        phase.has_large_addition = loc_delta_percent > 30
        phase.has_refactor = False
        phase.readme_changed = False
        return phase

    def test_extract_security_discovery(self):
        """Test extraction of security-related discoveries."""
        extractor = DiscoveryExtractor()

        commit = self.create_mock_commit("Fix SQL injection vulnerability in login")
        phase = self.create_mock_phase([commit])

        discoveries = extractor.extract_from_phase(phase)

        security_discoveries = [
            d for d in discoveries if d.discovery_type == DiscoveryType.RISK
        ]
        assert len(security_discoveries) >= 1
        assert any("security" in d.title.lower() or "sql" in d.evidence.get("keywords_found", [])
                  for d in security_discoveries)

    def test_extract_breaking_change_discovery(self):
        """Test extraction of breaking change discoveries."""
        extractor = DiscoveryExtractor()

        commit = self.create_mock_commit("BREAKING: Remove deprecated API endpoints")
        phase = self.create_mock_phase([commit])

        discoveries = extractor.extract_from_phase(phase)

        breaking_discoveries = [
            d for d in discoveries
            if "breaking" in d.title.lower()
        ]
        assert len(breaking_discoveries) >= 1

    def test_extract_large_loc_change(self):
        """Test extraction of large LOC change anomalies."""
        extractor = DiscoveryExtractor()

        commit = self.create_mock_commit("Major refactoring", insertions=5000, deletions=100)
        phase = self.create_mock_phase([commit], loc_delta_percent=60)

        discoveries = extractor.extract_from_phase(phase)

        loc_discoveries = [
            d for d in discoveries
            if d.discovery_type == DiscoveryType.ANOMALY and "code" in d.title.lower()
        ]
        assert len(loc_discoveries) >= 1

    def test_extract_from_summary(self):
        """Test extraction from LLM-generated summary."""
        extractor = DiscoveryExtractor()

        commit = self.create_mock_commit("Update readme")
        phase = self.create_mock_phase([commit])

        summary = """
        This phase achieved a major milestone with the OAuth implementation.
        [COMPLETED:feature] OAuth Implementation: Full OAuth2 support added.
        There are some security concerns that need attention.
        """

        discoveries = extractor.extract_from_phase(phase, summary)

        # Should find milestone from storyline marker
        milestones = [d for d in discoveries if d.discovery_type == DiscoveryType.MILESTONE]
        assert len(milestones) >= 1

    def test_deduplicate_discoveries(self):
        """Test that similar discoveries are deduplicated."""
        extractor = DiscoveryExtractor()

        # Create commits that would generate similar discoveries
        commit1 = self.create_mock_commit("Fix security vulnerability")
        commit2 = self.create_mock_commit("fix security vulnerability")  # Same, different case
        phase = self.create_mock_phase([commit1, commit2])

        discoveries = extractor.extract_from_phase(phase)

        # Should deduplicate similar discoveries
        security_discoveries = [d for d in discoveries if "security" in d.title.lower()]
        # After dedup, shouldn't have too many similar ones
        assert len(security_discoveries) <= 2


class TestDecisionEngine:
    """Tests for DecisionEngine."""

    def test_determine_phase_depth_deep(self):
        """Test that high-significance phases get deep analysis."""
        engine = DecisionEngine()

        # Create mock phase with significant changes
        phase = Mock()
        phase.phase_number = 1
        phase.loc_delta_percent = 60
        phase.has_large_deletion = True
        phase.has_large_addition = False
        phase.has_refactor = True
        phase.readme_changed = False

        ctx = AnalysisContext(repo_path="/test")

        # Add a high-significance discovery for this phase
        disc = Discovery(
            discovery_type=DiscoveryType.RISK,
            title="Security concern",
            description="Test",
            source="test",
            confidence=0.9,
            significance=0.85,
            phase_number=1,
        )
        ctx.add_discovery(disc)

        depth = engine.determine_phase_depth(phase, ctx.discoveries, ctx)

        assert depth == AnalysisDepth.DEEP

    def test_determine_phase_depth_light(self):
        """Test that low-significance phases get light analysis."""
        engine = DecisionEngine()

        # Create mock phase with minimal changes
        phase = Mock()
        phase.phase_number = 1
        phase.loc_delta_percent = 5
        phase.has_large_deletion = False
        phase.has_large_addition = False
        phase.has_refactor = False
        phase.readme_changed = False

        ctx = AnalysisContext(repo_path="/test")

        depth = engine.determine_phase_depth(phase, [], ctx)

        assert depth == AnalysisDepth.LIGHT

    def test_evaluate_generates_deepen_decision(self):
        """Test that high-significance discoveries trigger deepen decisions."""
        engine = DecisionEngine()

        ctx = AnalysisContext(repo_path="/test")

        disc = Discovery(
            discovery_type=DiscoveryType.ANOMALY,
            title="Major architectural shift",
            description="Complete restructuring detected",
            source="test",
            confidence=0.9,
            significance=0.9,
            phase_number=2,
        )

        decisions = engine.evaluate([disc], ctx)

        deepen_decisions = [d for d in decisions if d.decision_type == DecisionType.DEEPEN]
        assert len(deepen_decisions) >= 1

    def test_evaluate_respects_iteration_limits(self):
        """Test that evaluation stops at max iterations."""
        engine = DecisionEngine()

        ctx = AnalysisContext(repo_path="/test", max_iterations=5)
        ctx.iteration_count = 5  # At limit

        disc = Discovery(
            discovery_type=DiscoveryType.RISK,
            title="Test",
            description="Test",
            source="test",
            confidence=0.9,
            significance=0.9,
            phase_number=1,
        )

        decisions = engine.evaluate([disc], ctx)

        # Should get conclude decision
        conclude_decisions = [d for d in decisions if d.decision_type == DecisionType.CONCLUDE]
        assert len(conclude_decisions) == 1

    def test_prioritize_discoveries_with_goals(self):
        """Test that goal-relevant discoveries get prioritized."""
        engine = DecisionEngine()

        ctx = AnalysisContext(
            repo_path="/test",
            user_goals=["Implement OAuth authentication", "Add API rate limiting"],
        )

        disc1 = Discovery(
            discovery_type=DiscoveryType.INSIGHT,
            title="OAuth implementation progress",
            description="OAuth authentication partially complete",
            source="test",
            confidence=0.8,
            significance=0.6,
        )

        disc2 = Discovery(
            discovery_type=DiscoveryType.PATTERN,
            title="Code formatting changes",
            description="Linter updates across codebase",
            source="test",
            confidence=0.9,
            significance=0.5,
        )

        prioritized = engine.prioritize_discoveries([disc1, disc2], ctx)

        # OAuth-related discovery should have higher relevance
        assert prioritized[0].goal_relevance >= prioritized[1].goal_relevance


class TestGoalMatcher:
    """Tests for GoalMatcher."""

    def test_match_discovery_with_relevant_keywords(self):
        """Test matching discoveries against goals."""
        matcher = GoalMatcher(["Implement OAuth authentication", "Fix security vulnerabilities"])

        disc = Discovery(
            discovery_type=DiscoveryType.INSIGHT,
            title="OAuth implementation started",
            description="Initial OAuth2 authentication flow added",
            source="test",
            confidence=0.8,
            significance=0.7,
        )

        relevance = matcher.match_discovery(disc)

        assert relevance > 0.3  # Should have some relevance due to OAuth keyword

    def test_match_discovery_unrelated(self):
        """Test that unrelated discoveries get low relevance."""
        matcher = GoalMatcher(["Implement OAuth authentication"])

        disc = Discovery(
            discovery_type=DiscoveryType.PATTERN,
            title="Logging improvements",
            description="Added structured logging to all modules",
            source="test",
            confidence=0.8,
            significance=0.5,
        )

        relevance = matcher.match_discovery(disc)

        assert relevance < 0.3  # Should have low relevance


class TestIntegration:
    """Integration tests for adaptive components working together."""

    def test_discovery_to_decision_flow(self):
        """Test complete flow from discovery to decision."""
        extractor = DiscoveryExtractor()
        engine = DecisionEngine()
        ctx = AnalysisContext(repo_path="/test")

        # Create mock data
        commit = Mock()
        commit.commit_message = "CRITICAL: Fix authentication bypass vulnerability"
        commit.commit_hash = "abc123"
        commit.insertions = 50
        commit.deletions = 10

        phase = Mock()
        phase.commits = [commit]
        phase.phase_number = 1
        phase.loc_delta = 40
        phase.loc_delta_percent = 15
        phase.total_insertions = 50
        phase.total_deletions = 10
        phase.has_large_deletion = False
        phase.has_large_addition = False
        phase.has_refactor = False
        phase.readme_changed = False

        # Extract discoveries
        discoveries = extractor.extract_from_phase(phase)

        # Should find security-related discovery
        assert len(discoveries) > 0

        # Add to context
        for disc in discoveries:
            ctx.add_discovery(disc)

        # Make decisions
        decisions = engine.evaluate(discoveries, ctx)

        # Should recommend deeper investigation for security issue
        assert any(d.decision_type == DecisionType.DEEPEN for d in decisions)
