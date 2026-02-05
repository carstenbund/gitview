"""Adaptive Review Agent - Discovery-driven git history analysis.

Orchestrates analysis using an OODA loop (Observe-Orient-Decide-Act)
to adapt workflow based on discoveries during the review process.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .models import (
    AnalysisContext,
    AnalysisDepth,
    AnalysisResult,
    AnalysisState,
    Decision,
    DecisionType,
    Discovery,
    DiscoveryType,
)
from .discovery_extractor import DiscoveryExtractor
from .decision_engine import DecisionEngine, GoalMatcher


class AdaptiveReviewAgent:
    """Orchestrates adaptive git history analysis.

    Instead of mechanically executing fixed steps, this agent:
    1. Observes - Extracts data and detects patterns
    2. Orients - Evaluates significance and relevance
    3. Decides - Chooses next action based on discoveries
    4. Acts - Executes analysis and records findings

    The workflow loops until analysis is complete or max iterations reached.
    """

    def __init__(
        self,
        extractor: Any = None,  # GitHistoryExtractor
        chunker: Any = None,  # HistoryChunker
        summarizer: Any = None,  # PhaseSummarizer
        storyteller: Any = None,  # StoryTeller
        storyline_tracker: Any = None,  # StorylineTracker
        logger: Optional[Callable] = None,
    ):
        """Initialize the adaptive agent.

        Args:
            extractor: Git history extractor instance
            chunker: History chunker instance
            summarizer: Phase summarizer instance
            storyteller: Story generator instance
            storyline_tracker: Storyline tracker instance
            logger: Optional logging function
        """
        self.extractor = extractor
        self.chunker = chunker
        self.summarizer = summarizer
        self.storyteller = storyteller
        self.storyline_tracker = storyline_tracker
        self.logger = logger or (lambda msg: None)

        # Adaptive components
        self.discovery_extractor = DiscoveryExtractor()
        self.decision_engine = DecisionEngine()

    def analyze(
        self,
        repo_path: str,
        output_path: str,
        branch: str = "HEAD",
        user_goals: Optional[List[str]] = None,
        directives: Optional[str] = None,
        critical_mode: bool = False,
        max_commits: Optional[int] = None,
        skip_llm: bool = False,
        github_token: Optional[str] = None,
        **kwargs,
    ) -> AnalysisResult:
        """Run adaptive analysis on a repository.

        Args:
            repo_path: Path to git repository
            output_path: Output directory
            branch: Branch to analyze
            user_goals: Optional list of goals to measure against
            directives: Optional analysis directives
            critical_mode: Enable critical examination mode
            max_commits: Maximum commits to analyze
            skip_llm: Skip LLM summarization
            github_token: Optional GitHub token for enrichment

        Returns:
            AnalysisResult with phases, discoveries, and narrative
        """
        # Initialize context
        context = AnalysisContext(
            repo_path=repo_path,
            user_goals=user_goals or [],
            directives=directives,
            critical_mode=critical_mode,
        )

        self._log(f"Starting adaptive analysis of {repo_path}")

        # Goal matcher for relevance scoring
        goal_matcher = GoalMatcher(user_goals) if user_goals else None

        # === PHASE 1: OBSERVE - Extract History ===
        context.state = AnalysisState.EXTRACTING
        self._log("Phase 1: Extracting git history...")

        records = self._extract_history(context, branch, max_commits)
        if not records:
            self._log("No commits found, aborting analysis")
            return self._create_empty_result(context)

        # Extract initial discoveries from raw records
        initial_discoveries = self.discovery_extractor.extract_from_records(records)
        for disc in initial_discoveries:
            if goal_matcher:
                disc.goal_relevance = goal_matcher.match_discovery(disc)
            context.add_discovery(disc)

        self._log(f"Extracted {len(records)} commits, {len(initial_discoveries)} initial discoveries")

        # === PHASE 2: OBSERVE - Chunk into Phases ===
        context.state = AnalysisState.CHUNKING
        self._log("Phase 2: Chunking into phases...")

        phases = self._chunk_history(records, context)
        self._log(f"Created {len(phases)} phases")

        # Check if re-chunking is needed based on discoveries
        should_rechunk, reason = self.decision_engine.should_rechunk(
            phases, context.discoveries, context
        )
        if should_rechunk:
            self._log(f"Re-chunking triggered: {reason}")
            phases = self._adaptive_rechunk(phases, context)
            self._log(f"After re-chunking: {len(phases)} phases")

        # Save phases
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        phases_dir = output_dir / "phases"
        phases_dir.mkdir(exist_ok=True)
        self.chunker.save_phases(phases, str(phases_dir))

        if skip_llm:
            self._log("Skipping LLM summarization (--skip-llm)")
            return self._create_result_without_llm(phases, context, output_dir)

        # === PHASE 3: ORIENT + DECIDE + ACT - Adaptive Summarization ===
        context.state = AnalysisState.SUMMARIZING
        self._log("Phase 3: Adaptive summarization...")

        summarized_phases = self._adaptive_summarize(
            phases, context, goal_matcher, str(phases_dir)
        )

        # === PHASE 4: Track Storylines ===
        self._log("Phase 4: Tracking storylines...")
        self._track_storylines(summarized_phases, context, str(phases_dir), github_token)

        # === PHASE 5: SYNTHESIZE - Consolidate Discoveries ===
        context.state = AnalysisState.SYNTHESIZING
        self._log("Phase 5: Synthesizing discoveries...")

        key_findings = self._synthesize_findings(context)

        # === PHASE 6: NARRATE - Generate Stories ===
        context.state = AnalysisState.NARRATING
        self._log("Phase 6: Generating narrative...")

        narrative = self._generate_narrative(
            summarized_phases, context, key_findings, output_path
        )

        # === FINALIZE ===
        context.state = AnalysisState.COMPLETED
        self._log("Analysis complete")

        # Persist discoveries
        self._save_discoveries(context.discoveries, output_dir)

        return AnalysisResult(
            phases=summarized_phases,
            discoveries=context.discoveries,
            narrative=narrative,
            context=context,
            key_findings=key_findings,
            risk_summary=self._extract_risk_summary(context),
            goal_alignment=self._compute_goal_alignment(context),
            total_iterations=context.iteration_count,
            total_decisions=len(context.decisions_made),
            analysis_depth_distribution=self._compute_depth_distribution(context),
        )

    def _extract_history(
        self,
        context: AnalysisContext,
        branch: str,
        max_commits: Optional[int],
    ) -> List[Any]:
        """Extract git history."""
        if self.extractor is None:
            from ..extractor import GitHistoryExtractor

            self.extractor = GitHistoryExtractor(context.repo_path)

        return self.extractor.extract_history(max_commits=max_commits, branch=branch)

    def _chunk_history(
        self,
        records: List[Any],
        context: AnalysisContext,
    ) -> List[Any]:
        """Chunk records into phases."""
        if self.chunker is None:
            from ..chunker import HistoryChunker

            self.chunker = HistoryChunker("adaptive")

        return self.chunker.chunk(records)

    def _adaptive_rechunk(
        self,
        phases: List[Any],
        context: AnalysisContext,
    ) -> List[Any]:
        """Re-chunk phases based on discoveries.

        Merges tiny phases and splits oversized ones.
        """
        new_phases = []

        i = 0
        while i < len(phases):
            phase = phases[i]

            # Merge tiny phases
            if phase.commit_count < 3 and i + 1 < len(phases):
                next_phase = phases[i + 1]
                if next_phase.commit_count < 10:
                    # Merge with next
                    merged = self._merge_phases(phase, next_phase)
                    new_phases.append(merged)
                    i += 2
                    continue

            # Split oversized phases
            if phase.commit_count > 200:
                split_phases = self._split_phase(phase, phase.commit_count // 2)
                new_phases.extend(split_phases)
                i += 1
                continue

            new_phases.append(phase)
            i += 1

        # Renumber phases
        for idx, phase in enumerate(new_phases, 1):
            phase.phase_number = idx

        return new_phases

    def _merge_phases(self, phase1: Any, phase2: Any) -> Any:
        """Merge two phases into one."""
        from ..chunker import Phase

        merged_commits = phase1.commits + phase2.commits

        return Phase(
            phase_number=phase1.phase_number,
            commits=merged_commits,
            start_date=phase1.start_date,
            end_date=phase2.end_date,
            commit_count=len(merged_commits),
            total_insertions=phase1.total_insertions + phase2.total_insertions,
            total_deletions=phase1.total_deletions + phase2.total_deletions,
            loc_start=phase1.loc_start,
            loc_end=phase2.loc_end,
            loc_delta=phase2.loc_end - phase1.loc_start,
            loc_delta_percent=((phase2.loc_end - phase1.loc_start) / max(phase1.loc_start, 1)) * 100,
            has_large_deletion=phase1.has_large_deletion or phase2.has_large_deletion,
            has_large_addition=phase1.has_large_addition or phase2.has_large_addition,
            has_refactor=phase1.has_refactor or phase2.has_refactor,
            readme_changed=phase1.readme_changed or phase2.readme_changed,
        )

    def _split_phase(self, phase: Any, split_point: int) -> List[Any]:
        """Split a phase at the given commit index."""
        from ..chunker import Phase

        commits1 = phase.commits[:split_point]
        commits2 = phase.commits[split_point:]

        if not commits1 or not commits2:
            return [phase]

        def make_phase(commits, number, start_loc):
            insertions = sum(c.insertions for c in commits)
            deletions = sum(c.deletions for c in commits)
            loc_end = commits[-1].loc_total if hasattr(commits[-1], "loc_total") else start_loc

            return Phase(
                phase_number=number,
                commits=commits,
                start_date=commits[0].timestamp,
                end_date=commits[-1].timestamp,
                commit_count=len(commits),
                total_insertions=insertions,
                total_deletions=deletions,
                loc_start=start_loc,
                loc_end=loc_end,
                loc_delta=loc_end - start_loc,
                loc_delta_percent=((loc_end - start_loc) / max(start_loc, 1)) * 100,
                has_large_deletion=any(c.deletions > 500 for c in commits),
                has_large_addition=any(c.insertions > 500 for c in commits),
                has_refactor=False,
                readme_changed=any("readme" in (c.commit_message or "").lower() for c in commits),
            )

        phase1 = make_phase(commits1, phase.phase_number, phase.loc_start)
        mid_loc = commits1[-1].loc_total if hasattr(commits1[-1], "loc_total") else phase.loc_start
        phase2 = make_phase(commits2, phase.phase_number + 1, mid_loc)

        return [phase1, phase2]

    def _adaptive_summarize(
        self,
        phases: List[Any],
        context: AnalysisContext,
        goal_matcher: Optional[GoalMatcher],
        phases_dir: str,
    ) -> List[Any]:
        """Summarize phases with adaptive depth.

        Spends more effort on significant phases, triggered by discoveries.
        """
        if self.summarizer is None:
            from ..summarizer import PhaseSummarizer

            self.summarizer = PhaseSummarizer(
                critical_mode=context.critical_mode,
                directives=context.directives,
            )

        previous_summaries = []

        for phase in phases:
            context.iteration_count += 1

            # Determine analysis depth based on discoveries
            depth = self.decision_engine.determine_phase_depth(
                phase, context.discoveries, context
            )
            self._log(f"Phase {phase.phase_number}: depth={depth.value}")

            # Summarize with appropriate depth
            if depth == AnalysisDepth.DEEP:
                summary = self._deep_summarize(phase, previous_summaries, context)
                context.record_phase_deepening(phase.phase_number)
            elif depth == AnalysisDepth.LIGHT:
                summary = self._light_summarize(phase, previous_summaries, context)
            else:
                summary = self._standard_summarize(phase, previous_summaries, context)

            phase.summary = summary

            # Extract discoveries from summary
            phase_discoveries = self.discovery_extractor.extract_from_phase(
                phase, summary
            )
            for disc in phase_discoveries:
                if goal_matcher:
                    disc.goal_relevance = goal_matcher.match_discovery(disc)
                context.add_discovery(disc)

            # Make decisions based on new discoveries
            decisions = self.decision_engine.evaluate(phase_discoveries, context)
            for decision in decisions:
                context.add_decision(decision)
                self._execute_decision(decision, phase, context)

            previous_summaries.append({
                "phase_number": phase.phase_number,
                "summary": summary,
                "loc_delta": phase.loc_delta,
            })

            # Save phase with summary
            self._save_phase_summary(phase, phases_dir)

        return phases

    def _standard_summarize(
        self,
        phase: Any,
        previous_summaries: List[Dict],
        context: AnalysisContext,
    ) -> str:
        """Standard summarization."""
        context_str = self.summarizer._build_context(previous_summaries)
        return self.summarizer.summarize_phase(phase, context_str)

    def _deep_summarize(
        self,
        phase: Any,
        previous_summaries: List[Dict],
        context: AnalysisContext,
    ) -> str:
        """Deep summarization with hierarchical detail.

        Uses hierarchical summarizer if available for more detail.
        """
        try:
            from ..hierarchical_summarizer import HierarchicalPhaseSummarizer

            deep_summarizer = HierarchicalPhaseSummarizer(
                backend=getattr(self.summarizer, "backend", None),
                model=getattr(self.summarizer, "model", None),
                api_key=getattr(self.summarizer, "api_key", None),
            )
            result = deep_summarizer.summarize_phase(phase)
            return result["full_summary"]
        except Exception as e:
            self._log(f"Deep summarization fallback: {e}")
            # Fall back to standard with enhanced prompt
            return self._standard_summarize(phase, previous_summaries, context)

    def _light_summarize(
        self,
        phase: Any,
        previous_summaries: List[Dict],
        context: AnalysisContext,
    ) -> str:
        """Light summarization for low-significance phases."""
        # Generate a brief summary
        commit_types = {}
        for commit in phase.commits[:10]:  # Sample first 10
            msg = commit.commit_message.lower()
            if "fix" in msg:
                commit_types["fixes"] = commit_types.get("fixes", 0) + 1
            elif "feat" in msg or "add" in msg:
                commit_types["features"] = commit_types.get("features", 0) + 1
            elif "refactor" in msg or "clean" in msg:
                commit_types["refactoring"] = commit_types.get("refactoring", 0) + 1
            else:
                commit_types["other"] = commit_types.get("other", 0) + 1

        type_summary = ", ".join(f"{k}: {v}" for k, v in commit_types.items())

        return (
            f"Phase {phase.phase_number} contains {phase.commit_count} commits "
            f"({type_summary}). "
            f"LOC changed: {phase.loc_delta:+,d} ({phase.loc_delta_percent:+.1f}%)."
        )

    def _execute_decision(
        self,
        decision: Decision,
        current_phase: Any,
        context: AnalysisContext,
    ) -> None:
        """Execute a decision made by the decision engine."""
        self._log(f"Executing decision: {decision.decision_type.value} - {decision.reason}")

        if decision.decision_type == DecisionType.DEEPEN:
            # Mark for deeper analysis on next pass (already handled in _adaptive_summarize)
            if isinstance(decision.target, Discovery):
                decision.target.investigated = True
                decision.target.investigation_depth = AnalysisDepth.DEEP

        elif decision.decision_type == DecisionType.PIVOT:
            # Shift focus to a new area
            context.current_focus = decision.reason

        elif decision.decision_type == DecisionType.SKIP:
            # De-prioritize (no action needed, depth already determined)
            pass

    def _track_storylines(
        self,
        phases: List[Any],
        context: AnalysisContext,
        phases_dir: str,
        github_token: Optional[str],
    ) -> None:
        """Track storylines using StorylineTracker."""
        if self.storyline_tracker is None:
            from ..storyline import StorylineTracker

            db_path = str(Path(phases_dir) / "storylines.json")
            self.storyline_tracker = StorylineTracker(persist_path=db_path)

        from ..summarizer import _parse_storylines

        for phase in phases:
            llm_storylines = None
            if phase.summary:
                llm_storylines = _parse_storylines(phase.summary)

            result = self.storyline_tracker.process_phase(phase, llm_storylines)

            # Create discoveries from storyline transitions
            for transition in result.get("transitions", []):
                context.add_discovery(
                    Discovery(
                        discovery_type=DiscoveryType.MILESTONE
                        if transition["to"] == "completed"
                        else DiscoveryType.INSIGHT,
                        title=f"Storyline {transition['from']} -> {transition['to']}: {transition['title']}",
                        description=f"Storyline transitioned from {transition['from']} to {transition['to']}",
                        source="storyline_tracker",
                        confidence=0.9,
                        significance=0.6,
                        phase_number=phase.phase_number,
                        evidence={"transition": transition},
                    )
                )

    def _synthesize_findings(self, context: AnalysisContext) -> List[str]:
        """Synthesize discoveries into key findings."""
        findings = []

        # High-significance discoveries
        high_sig = context.get_high_significance_discoveries()
        for disc in high_sig[:5]:  # Top 5
            findings.append(f"[{disc.discovery_type.value.upper()}] {disc.title}")

        # Risk summary
        risks = [d for d in context.discoveries if d.discovery_type == DiscoveryType.RISK]
        if risks:
            findings.append(f"Identified {len(risks)} potential risk(s) requiring attention")

        # Goal alignment
        if context.user_goals:
            goal_relevant = [d for d in context.discoveries if d.goal_relevance > 0.5]
            findings.append(
                f"{len(goal_relevant)} discoveries directly relevant to stated goals"
            )

        return findings

    def _generate_narrative(
        self,
        phases: List[Any],
        context: AnalysisContext,
        key_findings: List[str],
        cache_dir: str,
    ) -> Dict[str, str]:
        """Generate final narrative with discovery context."""
        if self.storyteller is None:
            from ..storyteller import StoryTeller

            self.storyteller = StoryTeller(
                critical_mode=context.critical_mode,
                directives=context.directives,
            )

        # Get storylines for narrative
        storylines = []
        if self.storyline_tracker:
            storylines = self.storyline_tracker.get_storylines_for_prompt(limit=10)

        # Include discoveries in storyteller context
        discovery_context = self._format_discoveries_for_narrative(context.discoveries)

        # Generate stories
        stories = self.storyteller.generate_global_story(
            phases,
            repo_name=Path(context.repo_path).name,
            cache_dir=cache_dir,
        )

        # Enhance with discoveries section
        if discovery_context:
            stories["discoveries"] = discovery_context

        return stories

    def _format_discoveries_for_narrative(
        self, discoveries: List[Discovery]
    ) -> str:
        """Format discoveries for inclusion in narrative."""
        if not discoveries:
            return ""

        sections = []

        # Group by type
        by_type: Dict[DiscoveryType, List[Discovery]] = {}
        for disc in discoveries:
            by_type.setdefault(disc.discovery_type, []).append(disc)

        # Risks first
        if DiscoveryType.RISK in by_type:
            risks = by_type[DiscoveryType.RISK]
            sections.append("### Risk Indicators\n")
            for risk in risks[:5]:
                sections.append(f"- **{risk.title}**: {risk.description}")
            sections.append("")

        # Insights
        if DiscoveryType.INSIGHT in by_type:
            insights = by_type[DiscoveryType.INSIGHT]
            sections.append("### Key Insights\n")
            for insight in insights[:5]:
                sections.append(f"- **{insight.title}**: {insight.description}")
            sections.append("")

        # Patterns
        if DiscoveryType.PATTERN in by_type:
            patterns = by_type[DiscoveryType.PATTERN]
            sections.append("### Detected Patterns\n")
            for pattern in patterns[:5]:
                sections.append(f"- **{pattern.title}**: {pattern.description}")
            sections.append("")

        return "\n".join(sections)

    def _save_discoveries(
        self, discoveries: List[Discovery], output_dir: Path
    ) -> None:
        """Save discoveries to JSON file."""
        discoveries_file = output_dir / "discoveries.json"
        data = {
            "generated_at": datetime.now().isoformat(),
            "total_discoveries": len(discoveries),
            "by_type": {},
            "discoveries": [d.to_dict() for d in discoveries],
        }

        # Count by type
        for disc in discoveries:
            type_name = disc.discovery_type.value
            data["by_type"][type_name] = data["by_type"].get(type_name, 0) + 1

        with open(discoveries_file, "w") as f:
            json.dump(data, f, indent=2)

    def _save_phase_summary(self, phase: Any, phases_dir: str) -> None:
        """Save phase with summary to disk."""
        phase_file = Path(phases_dir) / f"phase_{phase.phase_number:02d}.json"
        phase_data = phase.to_dict()
        with open(phase_file, "w") as f:
            json.dump(phase_data, f, indent=2)

    def _extract_risk_summary(self, context: AnalysisContext) -> List[str]:
        """Extract risk summary from discoveries."""
        risks = [
            d for d in context.discoveries if d.discovery_type == DiscoveryType.RISK
        ]
        return [f"{r.title}: {r.description}" for r in risks]

    def _compute_goal_alignment(
        self, context: AnalysisContext
    ) -> Dict[str, float]:
        """Compute goal alignment scores."""
        if not context.user_goals:
            return {}

        alignment = {}
        for goal in context.user_goals:
            relevant = [
                d
                for d in context.discoveries
                if d.goal_relevance > 0.5 and goal.lower() in d.title.lower()
            ]
            alignment[goal[:50]] = len(relevant) / max(len(context.discoveries), 1)

        return alignment

    def _compute_depth_distribution(
        self, context: AnalysisContext
    ) -> Dict[str, int]:
        """Compute distribution of analysis depths."""
        distribution = {"deep": 0, "standard": 0, "light": 0}

        for phase_num, depth_count in context.phases_deepened.items():
            if depth_count > 0:
                distribution["deep"] += 1
            else:
                distribution["standard"] += 1

        return distribution

    def _create_empty_result(self, context: AnalysisContext) -> AnalysisResult:
        """Create result for empty repository."""
        return AnalysisResult(
            phases=[],
            discoveries=[],
            narrative={"error": "No commits found"},
            context=context,
        )

    def _create_result_without_llm(
        self,
        phases: List[Any],
        context: AnalysisContext,
        output_dir: Path,
    ) -> AnalysisResult:
        """Create result when LLM is skipped."""
        # Write simple timeline
        from ..writer import OutputWriter

        timeline_file = output_dir / "timeline.md"
        OutputWriter.write_simple_timeline(phases, str(timeline_file))

        return AnalysisResult(
            phases=phases,
            discoveries=context.discoveries,
            narrative={"timeline": str(timeline_file)},
            context=context,
        )

    def _log(self, message: str) -> None:
        """Log a message."""
        self.logger(f"[AdaptiveAgent] {message}")
