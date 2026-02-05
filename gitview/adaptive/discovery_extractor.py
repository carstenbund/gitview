"""Discovery Extractor - Extracts discoveries from analysis artifacts.

Analyzes phases, summaries, and commit patterns to identify discoveries
that may warrant further investigation.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    Discovery,
    DiscoveryType,
    SECURITY_KEYWORDS,
    BREAKING_CHANGE_KEYWORDS,
    REFACTOR_KEYWORDS,
    ARCHITECTURE_KEYWORDS,
)


class DiscoveryExtractor:
    """Extracts discoveries from phases, summaries, and patterns.

    Uses multiple signals to identify significant findings:
    - Keyword analysis for security, breaking changes, refactors
    - Statistical anomalies in LOC changes
    - Pattern detection in file clusters and commit patterns
    - LLM summary analysis for implicit discoveries
    """

    def __init__(
        self,
        security_sensitivity: float = 1.0,
        anomaly_threshold: float = 2.0,  # Standard deviations
    ):
        self.security_sensitivity = security_sensitivity
        self.anomaly_threshold = anomaly_threshold
        self._baseline_stats: Optional[Dict[str, float]] = None

    def extract_from_phase(
        self, phase: Any, summary: Optional[str] = None
    ) -> List[Discovery]:
        """Extract all discoveries from a phase.

        Args:
            phase: Phase object with commits and metadata
            summary: Optional LLM-generated summary of the phase

        Returns:
            List of discovered findings
        """
        discoveries = []

        # Extract from phase metadata and commits
        discoveries.extend(self._extract_from_commits(phase))
        discoveries.extend(self._extract_from_metrics(phase))

        # Extract from summary if available
        if summary:
            discoveries.extend(self._extract_from_summary(summary, phase.phase_number))

        # Check for file cluster patterns
        discoveries.extend(self._extract_from_file_patterns(phase))

        # Deduplicate similar discoveries
        return self._deduplicate(discoveries)

    def extract_from_records(self, records: List[Any]) -> List[Discovery]:
        """Extract discoveries from raw commit records before chunking.

        Used in the initial observation phase to identify patterns
        across the entire history.
        """
        discoveries = []

        # Compute baseline statistics
        self._compute_baseline(records)

        # Look for overall patterns
        discoveries.extend(self._detect_author_patterns(records))
        discoveries.extend(self._detect_temporal_patterns(records))
        discoveries.extend(self._detect_message_patterns(records))

        return discoveries

    def _extract_from_commits(self, phase: Any) -> List[Discovery]:
        """Extract discoveries from commit messages and metadata."""
        discoveries = []

        for commit in phase.commits:
            msg_lower = commit.commit_message.lower()

            # Security-related discoveries
            security_matches = [kw for kw in SECURITY_KEYWORDS if kw in msg_lower]
            if security_matches:
                significance = min(
                    0.5 + len(security_matches) * 0.1 * self.security_sensitivity, 1.0
                )
                discoveries.append(
                    Discovery(
                        discovery_type=DiscoveryType.RISK,
                        title=f"Security-related change: {commit.commit_message[:50]}",
                        description=f"Commit contains security keywords: {', '.join(security_matches)}",
                        source="commit_analyzer",
                        confidence=0.8,
                        significance=significance,
                        phase_number=phase.phase_number,
                        commit_hashes=[commit.commit_hash],
                        file_paths=commit.files_changed if hasattr(commit, "files_changed") else [],
                        evidence={
                            "keywords_found": security_matches,
                            "commit_message": commit.commit_message,
                        },
                        implications=[
                            "May involve security fixes or enhancements",
                            "Should be reviewed for completeness",
                        ],
                        suggested_actions=[
                            "Verify security change is complete",
                            "Check for related security commits",
                        ],
                    )
                )

            # Breaking change discoveries
            breaking_matches = [
                kw for kw in BREAKING_CHANGE_KEYWORDS if kw in msg_lower
            ]
            if breaking_matches:
                discoveries.append(
                    Discovery(
                        discovery_type=DiscoveryType.RISK,
                        title=f"Breaking change detected: {commit.commit_message[:50]}",
                        description=f"Commit indicates breaking changes: {', '.join(breaking_matches)}",
                        source="commit_analyzer",
                        confidence=0.75,
                        significance=0.8,
                        phase_number=phase.phase_number,
                        commit_hashes=[commit.commit_hash],
                        evidence={
                            "keywords_found": breaking_matches,
                            "commit_message": commit.commit_message,
                        },
                        implications=[
                            "API or behavior changes may affect consumers",
                            "May require migration documentation",
                        ],
                        suggested_actions=[
                            "Document breaking changes",
                            "Verify migration path exists",
                        ],
                    )
                )

            # Architecture changes
            arch_matches = [kw for kw in ARCHITECTURE_KEYWORDS if kw in msg_lower]
            if len(arch_matches) >= 2:  # Multiple keywords suggest real arch change
                discoveries.append(
                    Discovery(
                        discovery_type=DiscoveryType.INSIGHT,
                        title=f"Architecture change: {commit.commit_message[:50]}",
                        description=f"Commit suggests architectural work: {', '.join(arch_matches)}",
                        source="commit_analyzer",
                        confidence=0.7,
                        significance=0.7,
                        phase_number=phase.phase_number,
                        commit_hashes=[commit.commit_hash],
                        evidence={"keywords_found": arch_matches},
                        implications=["May be part of larger restructuring effort"],
                        suggested_actions=["Track related architectural commits"],
                    )
                )

        return discoveries

    def _extract_from_metrics(self, phase: Any) -> List[Discovery]:
        """Extract discoveries from phase metrics (LOC changes, etc.)."""
        discoveries = []

        # Large LOC changes
        if abs(phase.loc_delta_percent) > 50:
            direction = "addition" if phase.loc_delta > 0 else "deletion"
            discoveries.append(
                Discovery(
                    discovery_type=DiscoveryType.ANOMALY,
                    title=f"Major code {direction}: {phase.loc_delta_percent:.0f}% change",
                    description=f"Phase {phase.phase_number} shows {abs(phase.loc_delta):,} LOC {direction}",
                    source="metrics_analyzer",
                    confidence=0.95,
                    significance=min(abs(phase.loc_delta_percent) / 100, 1.0),
                    phase_number=phase.phase_number,
                    evidence={
                        "loc_delta": phase.loc_delta,
                        "loc_delta_percent": phase.loc_delta_percent,
                        "insertions": phase.total_insertions,
                        "deletions": phase.total_deletions,
                    },
                    implications=[
                        f"Significant {'growth' if phase.loc_delta > 0 else 'cleanup'} occurred",
                        "May indicate major feature or refactoring",
                    ],
                    suggested_actions=[
                        "Analyze file-level changes",
                        "Identify primary contributors",
                    ],
                )
            )

        # High churn (lots of both additions and deletions)
        if phase.total_insertions > 1000 and phase.total_deletions > 1000:
            churn_ratio = min(phase.total_insertions, phase.total_deletions) / max(
                phase.total_insertions, phase.total_deletions
            )
            if churn_ratio > 0.5:  # Balanced adds/deletes suggests refactoring
                discoveries.append(
                    Discovery(
                        discovery_type=DiscoveryType.PATTERN,
                        title=f"High code churn: {phase.total_insertions:,}+ / {phase.total_deletions:,}-",
                        description="Balanced additions and deletions suggest significant refactoring",
                        source="metrics_analyzer",
                        confidence=0.85,
                        significance=0.65,
                        phase_number=phase.phase_number,
                        evidence={
                            "insertions": phase.total_insertions,
                            "deletions": phase.total_deletions,
                            "churn_ratio": churn_ratio,
                        },
                        implications=[
                            "Code is being actively restructured",
                            "May indicate tech debt reduction",
                        ],
                        suggested_actions=["Investigate refactoring patterns"],
                    )
                )

        # Flag phases with specific events
        if phase.has_large_deletion:
            discoveries.append(
                Discovery(
                    discovery_type=DiscoveryType.INSIGHT,
                    title="Large deletion event detected",
                    description=f"Phase {phase.phase_number} contains significant code removal",
                    source="metrics_analyzer",
                    confidence=0.9,
                    significance=0.6,
                    phase_number=phase.phase_number,
                    evidence={"has_large_deletion": True},
                    implications=["Code cleanup or feature removal occurred"],
                    suggested_actions=["Identify what was removed and why"],
                )
            )

        if phase.has_refactor:
            discoveries.append(
                Discovery(
                    discovery_type=DiscoveryType.PATTERN,
                    title="Refactoring activity detected",
                    description=f"Phase {phase.phase_number} shows refactoring patterns",
                    source="metrics_analyzer",
                    confidence=0.85,
                    significance=0.55,
                    phase_number=phase.phase_number,
                    evidence={"has_refactor": True},
                    implications=["Code quality improvements ongoing"],
                    suggested_actions=["Track refactoring progress"],
                )
            )

        return discoveries

    def _extract_from_summary(
        self, summary: str, phase_number: int
    ) -> List[Discovery]:
        """Extract discoveries from LLM-generated summary text."""
        discoveries = []
        summary_lower = summary.lower()

        # Look for explicit storyline markers
        storyline_pattern = r"\[(\w+):(\w+)\]\s+([^:\n]+)"
        matches = re.findall(storyline_pattern, summary)
        for status, category, title in matches:
            if status.lower() == "completed":
                discoveries.append(
                    Discovery(
                        discovery_type=DiscoveryType.MILESTONE,
                        title=f"Completed: {title.strip()}",
                        description=f"Storyline '{title.strip()}' marked as completed",
                        source="summary_analyzer",
                        confidence=0.9,
                        significance=0.75,
                        phase_number=phase_number,
                        evidence={
                            "status": status,
                            "category": category,
                            "title": title.strip(),
                        },
                        implications=["Development initiative reached completion"],
                        suggested_actions=["Verify completion is genuine"],
                    )
                )

        # Look for risk/concern language in summary
        risk_phrases = [
            "security concern",
            "vulnerability",
            "risk",
            "breaking change",
            "deprecated",
            "technical debt",
            "needs attention",
            "incomplete",
            "workaround",
            "hack",
            "todo",
            "fixme",
        ]
        found_risks = [phrase for phrase in risk_phrases if phrase in summary_lower]
        if found_risks:
            discoveries.append(
                Discovery(
                    discovery_type=DiscoveryType.RISK,
                    title="Potential concerns identified in summary",
                    description=f"Summary mentions: {', '.join(found_risks)}",
                    source="summary_analyzer",
                    confidence=0.7,
                    significance=0.6,
                    phase_number=phase_number,
                    evidence={"risk_phrases": found_risks},
                    implications=["Issues may need follow-up"],
                    suggested_actions=["Investigate mentioned concerns"],
                )
            )

        # Look for milestone/achievement language
        achievement_phrases = [
            "major milestone",
            "significant achievement",
            "breakthrough",
            "completed",
            "launched",
            "released",
            "deployed",
            "production ready",
        ]
        found_achievements = [
            phrase for phrase in achievement_phrases if phrase in summary_lower
        ]
        if found_achievements:
            discoveries.append(
                Discovery(
                    discovery_type=DiscoveryType.MILESTONE,
                    title="Achievement detected in phase",
                    description=f"Summary indicates: {', '.join(found_achievements)}",
                    source="summary_analyzer",
                    confidence=0.75,
                    significance=0.7,
                    phase_number=phase_number,
                    evidence={"achievement_phrases": found_achievements},
                    implications=["Significant progress milestone reached"],
                    suggested_actions=["Highlight in narrative"],
                )
            )

        return discoveries

    def _extract_from_file_patterns(self, phase: Any) -> List[Discovery]:
        """Extract discoveries from file change patterns."""
        discoveries = []

        # Collect all files changed in this phase
        files_changed: Dict[str, int] = {}
        for commit in phase.commits:
            if hasattr(commit, "files_changed"):
                for file in commit.files_changed:
                    files_changed[file] = files_changed.get(file, 0) + 1

        if not files_changed:
            return discoveries

        # Detect hot spots (files changed many times)
        hot_spots = [(f, c) for f, c in files_changed.items() if c >= 5]
        if hot_spots:
            discoveries.append(
                Discovery(
                    discovery_type=DiscoveryType.PATTERN,
                    title=f"Hot spot files: {len(hot_spots)} files changed 5+ times",
                    description="Multiple files were modified frequently in this phase",
                    source="file_pattern_analyzer",
                    confidence=0.85,
                    significance=0.5,
                    phase_number=phase.phase_number,
                    file_paths=[f for f, _ in hot_spots],
                    evidence={
                        "hot_spots": dict(hot_spots[:10]),
                        "total_hot_spots": len(hot_spots),
                    },
                    implications=["Active development areas identified"],
                    suggested_actions=["Analyze hot spot file histories"],
                )
            )

        # Detect test file patterns
        test_files = [f for f in files_changed.keys() if "test" in f.lower()]
        non_test_files = [f for f in files_changed.keys() if "test" not in f.lower()]
        if test_files and non_test_files:
            test_ratio = len(test_files) / (len(test_files) + len(non_test_files))
            if test_ratio > 0.4:
                discoveries.append(
                    Discovery(
                        discovery_type=DiscoveryType.PATTERN,
                        title=f"High test activity: {test_ratio:.0%} of changes are tests",
                        description="Significant testing effort in this phase",
                        source="file_pattern_analyzer",
                        confidence=0.9,
                        significance=0.4,
                        phase_number=phase.phase_number,
                        evidence={
                            "test_file_count": len(test_files),
                            "non_test_file_count": len(non_test_files),
                            "test_ratio": test_ratio,
                        },
                        implications=["Quality-focused development period"],
                        suggested_actions=["Check coverage improvements"],
                    )
                )

        # Detect config/infrastructure changes
        config_files = [
            f
            for f in files_changed.keys()
            if any(
                pattern in f.lower()
                for pattern in [
                    "config",
                    ".yml",
                    ".yaml",
                    ".json",
                    "dockerfile",
                    "docker-compose",
                    "makefile",
                    ".env",
                    "requirements",
                    "package.json",
                    "setup.py",
                    "pyproject",
                ]
            )
        ]
        if len(config_files) >= 3:
            discoveries.append(
                Discovery(
                    discovery_type=DiscoveryType.PATTERN,
                    title=f"Infrastructure changes: {len(config_files)} config files modified",
                    description="Multiple configuration/infrastructure files changed",
                    source="file_pattern_analyzer",
                    confidence=0.85,
                    significance=0.55,
                    phase_number=phase.phase_number,
                    file_paths=config_files,
                    evidence={"config_files": config_files},
                    implications=["Build or deployment changes likely"],
                    suggested_actions=["Review infrastructure changes"],
                )
            )

        return discoveries

    def _detect_author_patterns(self, records: List[Any]) -> List[Discovery]:
        """Detect patterns in author activity across all records."""
        discoveries = []

        # Count commits per author
        author_commits: Dict[str, int] = {}
        for record in records:
            author = record.author
            author_commits[author] = author_commits.get(author, 0) + 1

        if not author_commits:
            return discoveries

        total_commits = len(records)
        sorted_authors = sorted(author_commits.items(), key=lambda x: x[1], reverse=True)

        # Check for dominant author (>50% of commits)
        top_author, top_count = sorted_authors[0]
        if top_count > total_commits * 0.5:
            discoveries.append(
                Discovery(
                    discovery_type=DiscoveryType.PATTERN,
                    title=f"Dominant contributor: {top_author}",
                    description=f"Single author responsible for {top_count}/{total_commits} ({top_count/total_commits:.0%}) commits",
                    source="author_analyzer",
                    confidence=0.95,
                    significance=0.4,
                    evidence={
                        "top_author": top_author,
                        "commit_count": top_count,
                        "total_commits": total_commits,
                        "percentage": top_count / total_commits,
                    },
                    implications=[
                        "Project may have bus factor risk",
                        "Knowledge concentrated in one person",
                    ],
                    suggested_actions=["Consider knowledge sharing initiatives"],
                )
            )

        # Check for many contributors (healthy diversity)
        if len(author_commits) > 10:
            discoveries.append(
                Discovery(
                    discovery_type=DiscoveryType.INSIGHT,
                    title=f"Diverse contributor base: {len(author_commits)} authors",
                    description="Repository has contributions from many authors",
                    source="author_analyzer",
                    confidence=0.95,
                    significance=0.3,
                    evidence={
                        "author_count": len(author_commits),
                        "top_authors": dict(sorted_authors[:5]),
                    },
                    implications=["Healthy project with distributed knowledge"],
                    suggested_actions=[],
                )
            )

        return discoveries

    def _detect_temporal_patterns(self, records: List[Any]) -> List[Discovery]:
        """Detect patterns in commit timing."""
        discoveries = []

        if len(records) < 10:
            return discoveries

        # Group commits by month for trend analysis
        monthly_commits: Dict[str, int] = {}
        for record in records:
            month_key = record.timestamp[:7]  # YYYY-MM
            monthly_commits[month_key] = monthly_commits.get(month_key, 0) + 1

        if len(monthly_commits) < 3:
            return discoveries

        # Detect activity spikes
        sorted_months = sorted(monthly_commits.items())
        values = [v for _, v in sorted_months]
        avg = sum(values) / len(values)

        for month, count in sorted_months:
            if count > avg * 2.5:  # More than 2.5x average
                discoveries.append(
                    Discovery(
                        discovery_type=DiscoveryType.ANOMALY,
                        title=f"Activity spike in {month}",
                        description=f"{count} commits vs {avg:.0f} monthly average (2.5x+)",
                        source="temporal_analyzer",
                        confidence=0.85,
                        significance=0.5,
                        evidence={
                            "month": month,
                            "commit_count": count,
                            "average": avg,
                            "ratio": count / avg,
                        },
                        implications=["Unusual development intensity"],
                        suggested_actions=["Investigate cause of spike"],
                    )
                )

        # Detect long gaps (>60 days between commits)
        dates = sorted(set(r.timestamp[:10] for r in records))
        if len(dates) > 1:
            from datetime import datetime

            for i in range(1, len(dates)):
                d1 = datetime.strptime(dates[i - 1], "%Y-%m-%d")
                d2 = datetime.strptime(dates[i], "%Y-%m-%d")
                gap = (d2 - d1).days
                if gap > 60:
                    discoveries.append(
                        Discovery(
                            discovery_type=DiscoveryType.ANOMALY,
                            title=f"Development gap: {gap} days",
                            description=f"No commits between {dates[i-1]} and {dates[i]}",
                            source="temporal_analyzer",
                            confidence=0.95,
                            significance=0.4,
                            evidence={
                                "start_date": dates[i - 1],
                                "end_date": dates[i],
                                "gap_days": gap,
                            },
                            implications=["Project may have been paused"],
                            suggested_actions=["Note gap in timeline narrative"],
                        )
                    )

        return discoveries

    def _detect_message_patterns(self, records: List[Any]) -> List[Discovery]:
        """Detect patterns in commit messages."""
        discoveries = []

        # Count empty or very short messages
        short_messages = [r for r in records if len(r.commit_message.strip()) < 10]
        if len(short_messages) > len(records) * 0.1:  # >10% short messages
            discoveries.append(
                Discovery(
                    discovery_type=DiscoveryType.PATTERN,
                    title=f"Commit message quality: {len(short_messages)} short messages",
                    description=f"{len(short_messages)}/{len(records)} commits have messages <10 chars",
                    source="message_analyzer",
                    confidence=0.9,
                    significance=0.35,
                    evidence={
                        "short_message_count": len(short_messages),
                        "total_commits": len(records),
                        "ratio": len(short_messages) / len(records),
                    },
                    implications=["Commit history may lack context"],
                    suggested_actions=["Consider commit message guidelines"],
                )
            )

        # Detect conventional commit usage
        conventional_pattern = r"^(feat|fix|docs|style|refactor|test|chore|perf|ci|build|revert)(\(.+\))?:"
        conventional_commits = [
            r
            for r in records
            if re.match(conventional_pattern, r.commit_message.strip(), re.IGNORECASE)
        ]
        if len(conventional_commits) > len(records) * 0.5:
            discoveries.append(
                Discovery(
                    discovery_type=DiscoveryType.INSIGHT,
                    title="Conventional commits pattern detected",
                    description=f"{len(conventional_commits)}/{len(records)} commits follow conventional format",
                    source="message_analyzer",
                    confidence=0.95,
                    significance=0.3,
                    evidence={
                        "conventional_count": len(conventional_commits),
                        "total_commits": len(records),
                        "ratio": len(conventional_commits) / len(records),
                    },
                    implications=["Well-structured commit history"],
                    suggested_actions=["Leverage commit types in analysis"],
                )
            )

        return discoveries

    def _compute_baseline(self, records: List[Any]) -> None:
        """Compute baseline statistics for anomaly detection."""
        if not records:
            return

        loc_deltas = [r.insertions - r.deletions for r in records if hasattr(r, "insertions")]
        if loc_deltas:
            import statistics

            self._baseline_stats = {
                "loc_mean": statistics.mean(loc_deltas),
                "loc_stdev": statistics.stdev(loc_deltas) if len(loc_deltas) > 1 else 0,
                "total_records": len(records),
            }

    def _deduplicate(self, discoveries: List[Discovery]) -> List[Discovery]:
        """Remove duplicate or very similar discoveries."""
        if not discoveries:
            return discoveries

        unique = []
        seen_titles = set()

        for disc in sorted(discoveries, key=lambda d: d.significance, reverse=True):
            # Simple dedup by normalized title
            normalized = disc.title.lower().strip()
            if normalized not in seen_titles:
                seen_titles.add(normalized)
                unique.append(disc)

        return unique
