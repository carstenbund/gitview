"""
Branch comparison and analysis for file histories.

This module provides functionality to compare file histories across different
git branches, identify diverging changes, and analyze differences.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Set
import json
from datetime import datetime

from .file_tracker import FileHistory, FileChange, FileHistoryTracker


@dataclass
class BranchFileHistory:
    """File history annotated with branch information"""
    branch_name: str
    file_path: str
    history: FileHistory

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            'branch_name': self.branch_name,
            'file_path': self.file_path,
            'history': self.history.to_dict()
        }


@dataclass
class FileDivergence:
    """Represents divergence of a file between two branches"""
    file_path: str
    branch_a_name: str
    branch_b_name: str

    # Existence
    exists_in_a: bool
    exists_in_b: bool

    # Commits
    commits_only_in_a: List[str] = field(default_factory=list)
    commits_only_in_b: List[str] = field(default_factory=list)
    commits_in_both: List[str] = field(default_factory=list)

    # Changes
    changes_only_in_a: List[FileChange] = field(default_factory=list)
    changes_only_in_b: List[FileChange] = field(default_factory=list)

    # Stats
    lines_added_in_a: int = 0
    lines_removed_in_a: int = 0
    lines_added_in_b: int = 0
    lines_removed_in_b: int = 0

    # Authors
    unique_authors_a: Set[str] = field(default_factory=set)
    unique_authors_b: Set[str] = field(default_factory=set)

    def get_divergence_score(self) -> float:
        """
        Calculate divergence score (0-100)
        Higher score = more divergent
        """
        score = 0.0

        # File existence difference
        if self.exists_in_a != self.exists_in_b:
            score += 50.0

        # Unique commits
        total_unique = len(self.commits_only_in_a) + len(self.commits_only_in_b)
        if total_unique > 0:
            score += min(30.0, total_unique * 2)

        # Line changes
        total_lines = (self.lines_added_in_a + self.lines_removed_in_a +
                      self.lines_added_in_b + self.lines_removed_in_b)
        if total_lines > 0:
            score += min(20.0, total_lines / 10)

        return min(100.0, score)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        data = asdict(self)
        # Convert sets to lists for JSON serialization
        data['unique_authors_a'] = list(self.unique_authors_a)
        data['unique_authors_b'] = list(self.unique_authors_b)
        # Convert FileChange objects to dicts
        data['changes_only_in_a'] = [c.to_dict() for c in self.changes_only_in_a]
        data['changes_only_in_b'] = [c.to_dict() for c in self.changes_only_in_b]
        return data


@dataclass
class BranchComparisonSummary:
    """Summary of comparison between two branches"""
    branch_a_name: str
    branch_b_name: str
    comparison_date: str

    # File counts
    total_files_compared: int
    files_only_in_a: int
    files_only_in_b: int
    files_in_both: int
    files_diverged: int

    # Commit counts
    commits_only_in_a: int
    commits_only_in_b: int
    commits_in_both: int

    # Top divergent files
    top_divergent_files: List[Tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return asdict(self)


class BranchComparator:
    """
    Compare file histories across git branches
    """

    def __init__(self, repo_path: str, output_dir: str = "output/branch_comparisons"):
        self.repo_path = Path(repo_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def track_branch(
        self,
        branch_name: str,
        file_patterns: List[str] = None,
        exclude_patterns: List[str] = None,
        max_entries: int = 100
    ) -> Dict[str, BranchFileHistory]:
        """
        Track all files on a specific branch

        Args:
            branch_name: Branch to track
            file_patterns: File patterns to include
            exclude_patterns: File patterns to exclude
            max_entries: Max history entries per file

        Returns:
            Dictionary mapping file paths to BranchFileHistory objects
        """
        # Create branch-specific output directory
        branch_output_dir = self.output_dir / "branches" / self._sanitize_branch_name(branch_name)

        # Track files
        tracker = FileHistoryTracker(
            repo_path=str(self.repo_path),
            output_dir=str(branch_output_dir)
        )

        print(f"\n[Branch: {branch_name}]")
        tracker.track_all_files(
            file_patterns=file_patterns,
            exclude_patterns=exclude_patterns,
            incremental=False,  # Always full scan for branch comparison
            max_history_entries=max_entries
        )

        # Load all tracked histories and annotate with branch
        branch_histories = {}

        histories_dir = branch_output_dir / "files"
        if histories_dir.exists():
            for json_file in histories_dir.rglob("*.json"):
                # Skip non-history files
                if json_file.name in ['index.json', 'checkpoint.json']:
                    continue

                try:
                    with open(json_file, 'r') as f:
                        history_data = json.load(f)

                    history = FileHistory.from_dict(history_data)
                    file_path = history.file_path

                    branch_history = BranchFileHistory(
                        branch_name=branch_name,
                        file_path=file_path,
                        history=history
                    )

                    branch_histories[file_path] = branch_history

                except Exception as e:
                    print(f"Warning: Failed to load history for {json_file}: {e}")

        # Save branch metadata
        metadata = {
            'branch_name': branch_name,
            'tracked_at': datetime.now().isoformat(),
            'total_files': len(branch_histories),
            'file_list': list(branch_histories.keys())
        }

        metadata_path = branch_output_dir / "branch_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return branch_histories

    def compare_branches(
        self,
        branch_a: str,
        branch_b: str,
        file_patterns: List[str] = None,
        exclude_patterns: List[str] = None
    ) -> Tuple[BranchComparisonSummary, Dict[str, FileDivergence]]:
        """
        Compare two branches and identify divergences

        Args:
            branch_a: First branch name
            branch_b: Second branch name
            file_patterns: File patterns to include
            exclude_patterns: File patterns to exclude

        Returns:
            Tuple of (summary, divergences_dict)
        """
        print(f"\n{'='*70}")
        print(f"Comparing branches: {branch_a} vs {branch_b}")
        print(f"{'='*70}")

        # Track both branches
        histories_a = self.track_branch(branch_a, file_patterns, exclude_patterns)
        histories_b = self.track_branch(branch_b, file_patterns, exclude_patterns)

        print(f"\nAnalyzing differences...")

        # Get all files from both branches
        all_files = set(histories_a.keys()) | set(histories_b.keys())

        # Analyze divergences
        divergences = {}
        files_only_in_a = 0
        files_only_in_b = 0
        files_diverged = 0

        for file_path in all_files:
            divergence = self._analyze_file_divergence(
                file_path,
                branch_a,
                branch_b,
                histories_a.get(file_path),
                histories_b.get(file_path)
            )

            divergences[file_path] = divergence

            if not divergence.exists_in_b:
                files_only_in_a += 1
            elif not divergence.exists_in_a:
                files_only_in_b += 1
            elif divergence.get_divergence_score() > 0:
                files_diverged += 1

        # Calculate summary stats
        all_commits_a = set()
        all_commits_b = set()

        for hist in histories_a.values():
            all_commits_a.update(c.commit_hash for c in hist.history.changes)

        for hist in histories_b.values():
            all_commits_b.update(c.commit_hash for c in hist.history.changes)

        commits_only_in_a = len(all_commits_a - all_commits_b)
        commits_only_in_b = len(all_commits_b - all_commits_a)
        commits_in_both = len(all_commits_a & all_commits_b)

        # Get top divergent files
        divergent_files = [
            (fp, div.get_divergence_score())
            for fp, div in divergences.items()
            if div.get_divergence_score() > 0
        ]
        divergent_files.sort(key=lambda x: x[1], reverse=True)
        top_divergent = divergent_files[:20]  # Top 20

        # Create summary
        summary = BranchComparisonSummary(
            branch_a_name=branch_a,
            branch_b_name=branch_b,
            comparison_date=datetime.now().isoformat(),
            total_files_compared=len(all_files),
            files_only_in_a=files_only_in_a,
            files_only_in_b=files_only_in_b,
            files_in_both=len(all_files) - files_only_in_a - files_only_in_b,
            files_diverged=files_diverged,
            commits_only_in_a=commits_only_in_a,
            commits_only_in_b=commits_only_in_b,
            commits_in_both=commits_in_both,
            top_divergent_files=top_divergent
        )

        # Save comparison results
        self._save_comparison(branch_a, branch_b, summary, divergences)

        return summary, divergences

    def _analyze_file_divergence(
        self,
        file_path: str,
        branch_a_name: str,
        branch_b_name: str,
        history_a: Optional[BranchFileHistory],
        history_b: Optional[BranchFileHistory]
    ) -> FileDivergence:
        """Analyze divergence for a single file"""

        divergence = FileDivergence(
            file_path=file_path,
            branch_a_name=branch_a_name,
            branch_b_name=branch_b_name,
            exists_in_a=history_a is not None,
            exists_in_b=history_b is not None
        )

        if not history_a or not history_b:
            # File only exists in one branch
            return divergence

        # Get commits from both branches
        commits_a = {c.commit_hash for c in history_a.history.changes}
        commits_b = {c.commit_hash for c in history_b.history.changes}

        divergence.commits_only_in_a = list(commits_a - commits_b)
        divergence.commits_only_in_b = list(commits_b - commits_a)
        divergence.commits_in_both = list(commits_a & commits_b)

        # Get changes unique to each branch
        for change in history_a.history.changes:
            if change.commit_hash in divergence.commits_only_in_a:
                divergence.changes_only_in_a.append(change)
                divergence.lines_added_in_a += change.lines_added
                divergence.lines_removed_in_a += change.lines_removed
                divergence.unique_authors_a.add(change.author_name)

        for change in history_b.history.changes:
            if change.commit_hash in divergence.commits_only_in_b:
                divergence.changes_only_in_b.append(change)
                divergence.lines_added_in_b += change.lines_added
                divergence.lines_removed_in_b += change.lines_removed
                divergence.unique_authors_b.add(change.author_name)

        return divergence

    def _save_comparison(
        self,
        branch_a: str,
        branch_b: str,
        summary: BranchComparisonSummary,
        divergences: Dict[str, FileDivergence]
    ):
        """Save comparison results to disk"""

        comparison_name = f"{self._sanitize_branch_name(branch_a)}_vs_{self._sanitize_branch_name(branch_b)}"
        comparison_dir = self.output_dir / "comparisons" / comparison_name
        comparison_dir.mkdir(parents=True, exist_ok=True)

        # Save summary
        summary_path = comparison_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2)

        # Save divergences
        divergences_path = comparison_dir / "divergences.json"
        divergences_data = {fp: div.to_dict() for fp, div in divergences.items()}
        with open(divergences_path, 'w') as f:
            json.dump(divergences_data, f, indent=2)

        print(f"\nComparison saved to: {comparison_dir}")

    def _sanitize_branch_name(self, branch_name: str) -> str:
        """Convert branch name to filesystem-safe name"""
        return branch_name.replace('/', '_').replace('\\', '_').replace(' ', '_')

    def generate_comparison_report(
        self,
        branch_a: str,
        branch_b: str,
        top_n: int = 10
    ) -> str:
        """
        Generate human-readable comparison report

        Args:
            branch_a: First branch
            branch_b: Second branch
            top_n: Number of top divergent files to include

        Returns:
            Formatted report string
        """
        comparison_name = f"{self._sanitize_branch_name(branch_a)}_vs_{self._sanitize_branch_name(branch_b)}"
        comparison_dir = self.output_dir / "comparisons" / comparison_name

        # Load summary and divergences
        with open(comparison_dir / "summary.json", 'r') as f:
            summary_data = json.load(f)
        summary = BranchComparisonSummary(**summary_data)

        with open(comparison_dir / "divergences.json", 'r') as f:
            divergences_data = json.load(f)

        # Generate report
        lines = []
        lines.append("=" * 80)
        lines.append(f"BRANCH COMPARISON REPORT")
        lines.append("=" * 80)
        lines.append(f"Branch A: {summary.branch_a_name}")
        lines.append(f"Branch B: {summary.branch_b_name}")
        lines.append(f"Generated: {summary.comparison_date[:19]}")
        lines.append("")

        lines.append("FILE STATISTICS")
        lines.append("-" * 80)
        lines.append(f"Total files compared:    {summary.total_files_compared}")
        lines.append(f"Files only in {branch_a:20}: {summary.files_only_in_a}")
        lines.append(f"Files only in {branch_b:20}: {summary.files_only_in_b}")
        lines.append(f"Files in both branches:  {summary.files_in_both}")
        lines.append(f"Files with divergence:   {summary.files_diverged}")
        lines.append("")

        lines.append("COMMIT STATISTICS")
        lines.append("-" * 80)
        lines.append(f"Commits only in {branch_a:20}: {summary.commits_only_in_a}")
        lines.append(f"Commits only in {branch_b:20}: {summary.commits_only_in_b}")
        lines.append(f"Commits in both branches: {summary.commits_in_both}")
        lines.append("")

        lines.append(f"TOP {top_n} MOST DIVERGENT FILES")
        lines.append("-" * 80)

        for i, (file_path, score) in enumerate(summary.top_divergent_files[:top_n], 1):
            div_data = divergences_data.get(file_path, {})
            commits_a = len(div_data.get('commits_only_in_a', []))
            commits_b = len(div_data.get('commits_only_in_b', []))

            lines.append(f"\n{i}. {file_path}")
            lines.append(f"   Divergence score: {score:.1f}/100")
            lines.append(f"   Unique commits in {branch_a}: {commits_a}")
            lines.append(f"   Unique commits in {branch_b}: {commits_b}")

        lines.append("")
        lines.append("=" * 80)

        # Save report
        report_path = comparison_dir / "report.txt"
        report = '\n'.join(lines)
        with open(report_path, 'w') as f:
            f.write(report)

        return report
