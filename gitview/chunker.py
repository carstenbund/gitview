"""Chunk git history into meaningful epochs/phases."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from .extractor import CommitRecord
from .significance_analyzer import SignificanceAnalyzer, CommitCluster


@dataclass
class Phase:
    """Represents a phase/epoch in repository history."""

    phase_number: int
    start_date: str
    end_date: str
    commit_count: int
    commits: List[CommitRecord]

    # Phase characteristics
    loc_start: int
    loc_end: int
    loc_delta: int
    loc_delta_percent: float

    total_insertions: int
    total_deletions: int

    # Language evolution
    languages_start: Dict[str, int]
    languages_end: Dict[str, int]

    # Major events
    has_large_deletion: bool
    has_large_addition: bool
    has_refactor: bool
    readme_changed: bool

    # Authors
    authors: List[str]
    primary_author: str

    # Summary (will be filled by LLM later)
    summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert CommitRecord objects to dicts
        data['commits'] = [c.to_dict() for c in self.commits]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Phase':
        """Create Phase from dictionary."""
        # Copy to avoid mutating callers and handle legacy caches gracefully
        data = dict(data)

        commits_data = data.pop('commits', [])
        commits = [CommitRecord(**c) for c in commits_data]

        return cls(commits=commits, **data)


class HistoryChunker:
    """Chunk repository history into meaningful phases."""

    def __init__(self, strategy: str = "adaptive"):
        """
        Initialize chunker with strategy.

        Args:
            strategy: 'fixed', 'time', 'adaptive', or 'semantic'
        """
        self.strategy = strategy

    def chunk(self, records: List[CommitRecord], **kwargs) -> List[Phase]:
        """
        Chunk commit records into phases.

        Args:
            records: List of CommitRecord objects (chronologically sorted)
            **kwargs: Strategy-specific parameters

        Returns:
            List of Phase objects
        """
        if self.strategy == "fixed":
            return self._chunk_fixed(records, kwargs.get('chunk_size', 50))
        elif self.strategy == "time":
            return self._chunk_time(records, kwargs.get('period', 'quarter'))
        elif self.strategy == "adaptive":
            return self._chunk_adaptive(records, kwargs)
        elif self.strategy == "semantic":
            return self._chunk_semantic(records, kwargs)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _chunk_fixed(self, records: List[CommitRecord], chunk_size: int) -> List[Phase]:
        """Split into fixed-size chunks."""
        phases = []

        for i in range(0, len(records), chunk_size):
            chunk = records[i:i + chunk_size]
            phase = self._create_phase(len(phases) + 1, chunk)
            phases.append(phase)

        return phases

    def _chunk_time(self, records: List[CommitRecord], period: str) -> List[Phase]:
        """
        Split by time period.

        Args:
            period: 'week', 'month', 'quarter', or 'year'
        """
        from dateutil.relativedelta import relativedelta

        if not records:
            return []

        phases = []
        current_chunk = []

        # Parse period
        delta_map = {
            'week': relativedelta(weeks=1),
            'month': relativedelta(months=1),
            'quarter': relativedelta(months=3),
            'year': relativedelta(years=1),
        }

        if period not in delta_map:
            raise ValueError(f"Unknown period: {period}")

        delta = delta_map[period]

        # Start from first commit
        current_start = datetime.fromisoformat(records[0].timestamp)
        current_end = current_start + delta

        for record in records:
            record_time = datetime.fromisoformat(record.timestamp)

            if record_time >= current_end:
                # Start new phase
                if current_chunk:
                    phase = self._create_phase(len(phases) + 1, current_chunk)
                    phases.append(phase)

                current_chunk = [record]
                current_start = current_end
                current_end = current_start + delta
            else:
                current_chunk.append(record)

        # Add final chunk
        if current_chunk:
            phase = self._create_phase(len(phases) + 1, current_chunk)
            phases.append(phase)

        return phases

    def _chunk_adaptive(self, records: List[CommitRecord], config: Dict[str, Any]) -> List[Phase]:
        """
        Adaptive chunking based on significant changes.

        Splits when:
        - LOC changes by more than threshold (default 30%)
        - Large deletions/additions detected
        - Language mix changes significantly
        - README is rewritten
        - Comment density shifts significantly
        - Refactoring detected

        Args:
            config: Configuration dict with thresholds
        """
        # Default thresholds (tuned for smaller, more detailed phases)
        loc_threshold = config.get('loc_threshold', 0.25)
        min_chunk_size = config.get('min_chunk_size', 5)
        max_chunk_size = config.get('max_chunk_size', 60)
        readme_change_split = config.get('readme_change_split', True)
        refactor_split = config.get('refactor_split', True)

        if not records:
            return []

        phases = []
        current_chunk = [records[0]]
        chunk_start_loc = records[0].loc_total

        for i, record in enumerate(records[1:], 1):
            should_split = False

            # Check LOC change
            if chunk_start_loc > 0:
                loc_change = abs(record.loc_total - chunk_start_loc) / chunk_start_loc
                if loc_change > loc_threshold:
                    should_split = True

            # Check large deletions/additions
            if record.is_large_deletion or record.is_large_addition:
                should_split = True

            # Check README rewrite
            if readme_change_split and record.readme_exists:
                if current_chunk:
                    last_readme_size = current_chunk[-1].readme_size
                    if last_readme_size > 0:
                        readme_change = abs(record.readme_size - last_readme_size) / last_readme_size
                        if readme_change > 0.5:  # README changed by >50%
                            should_split = True

            # Check refactoring
            if refactor_split and record.is_refactor:
                should_split = True

            # Enforce min/max chunk sizes
            if len(current_chunk) < min_chunk_size:
                should_split = False
            elif len(current_chunk) >= max_chunk_size:
                should_split = True

            if should_split:
                # Create phase from current chunk
                phase = self._create_phase(len(phases) + 1, current_chunk)
                phases.append(phase)

                # Start new chunk
                current_chunk = [record]
                chunk_start_loc = record.loc_total
            else:
                current_chunk.append(record)

        # Add final chunk
        if current_chunk:
            phase = self._create_phase(len(phases) + 1, current_chunk)
            phases.append(phase)

        return phases

    def _chunk_semantic(self, records: List[CommitRecord], config: Dict[str, Any]) -> List[Phase]:
        """Chunk commits into phases using semantic clustering.

        This uses the significance analyzer's semantic clustering to group
        consecutive clusters that share topic, file overlap, or PR lineage
        into phases. The intent is to align phase boundaries with meaningful
        workstreams rather than raw commit counts or LOC swings.

        Configurable knobs:
            semantic_threshold: Minimum affinity score to keep clusters in the
                same phase (default: 1)
            max_clusters_per_phase: Hard limit on clusters per phase to avoid
                overly large groupings (default: 6)
            max_gap_days: Split when clusters are separated by long time gaps
                (default: 30)
        """
        if not records:
            return []

        analyzer = SignificanceAnalyzer()
        clusters = analyzer.cluster_commits(records)

        if not clusters:
            return []

        phases: List[Phase] = []
        current_clusters: List[CommitCluster] = [clusters[0]]

        semantic_threshold = config.get('semantic_threshold', 1)
        max_clusters_per_phase = config.get('max_clusters_per_phase', 6)
        max_gap_days = config.get('max_gap_days', 30)

        for cluster in clusters[1:]:
            last_cluster = current_clusters[-1]

            if self._should_split_cluster(
                last_cluster,
                cluster,
                current_clusters,
                semantic_threshold,
                max_clusters_per_phase,
                max_gap_days,
            ):
                phase_commits = [c for cl in current_clusters for c in cl.commits]
                phase = self._create_phase(len(phases) + 1, phase_commits)
                phases.append(phase)
                current_clusters = [cluster]
            else:
                current_clusters.append(cluster)

        if current_clusters:
            phase_commits = [c for cl in current_clusters for c in cl.commits]
            phase = self._create_phase(len(phases) + 1, phase_commits)
            phases.append(phase)

        return phases

    def _should_split_cluster(
        self,
        previous: CommitCluster,
        current: CommitCluster,
        clusters_in_phase: List[CommitCluster],
        threshold: int,
        max_clusters_per_phase: int,
        max_gap_days: int,
    ) -> bool:
        """Determine whether to start a new phase before `current`.

        A split occurs when semantic affinity is low, when too many clusters
        have accumulated, or when a large temporal gap appears between
        clusters.
        """
        if len(clusters_in_phase) >= max_clusters_per_phase:
            return True

        gap_days = self._cluster_gap_in_days(previous, current)
        if gap_days is not None and gap_days > max_gap_days:
            return True

        affinity = 0

        if previous.cluster_type == current.cluster_type:
            affinity += 1

        if self._clusters_share_pr(previous, current):
            affinity += 2

        overlap = self._file_overlap(previous, current)
        if overlap >= 0.25:
            affinity += 1

        return affinity < threshold

    @staticmethod
    def _cluster_gap_in_days(previous: CommitCluster, current: CommitCluster) -> Optional[int]:
        """Calculate days between the last commit of previous and first of current."""
        try:
            prev_time = datetime.fromisoformat(previous.commits[-1].timestamp)
            current_time = datetime.fromisoformat(current.commits[0].timestamp)
        except Exception:
            return None

        return (current_time - prev_time).days

    @staticmethod
    def _clusters_share_pr(first: CommitCluster, second: CommitCluster) -> bool:
        """Check if clusters share PR lineage via PR numbers."""
        first_prs = {
            c.get_pr_number()
            for c in first.commits
            if c.has_github_context() and c.get_pr_number() is not None
        }
        second_prs = {
            c.get_pr_number()
            for c in second.commits
            if c.has_github_context() and c.get_pr_number() is not None
        }

        return bool(first_prs and second_prs and first_prs.intersection(second_prs))

    @staticmethod
    def _file_overlap(first: CommitCluster, second: CommitCluster) -> float:
        """Calculate normalized file path overlap between two clusters."""
        first_files = set(first.file_changes.keys())
        second_files = set(second.file_changes.keys())

        if not first_files or not second_files:
            return 0.0

        shared = len(first_files.intersection(second_files))
        return shared / min(len(first_files), len(second_files))

    def _create_phase(self, phase_number: int, commits: List[CommitRecord]) -> Phase:
        """Create a Phase object from a list of commits."""
        if not commits:
            raise ValueError("Cannot create phase from empty commit list")

        # Basic info
        start_date = commits[0].timestamp
        end_date = commits[-1].timestamp
        commit_count = len(commits)

        # LOC metrics
        loc_start = commits[0].loc_total
        loc_end = commits[-1].loc_total
        loc_delta = loc_end - loc_start
        loc_delta_percent = (loc_delta / loc_start * 100) if loc_start > 0 else 0

        # Insertions/deletions
        total_insertions = sum(c.insertions for c in commits)
        total_deletions = sum(c.deletions for c in commits)

        # Language breakdown
        languages_start = commits[0].language_breakdown
        languages_end = commits[-1].language_breakdown

        # Major events
        has_large_deletion = any(c.is_large_deletion for c in commits)
        has_large_addition = any(c.is_large_addition for c in commits)
        has_refactor = any(c.is_refactor for c in commits)

        # README changes
        readme_sizes = [c.readme_size for c in commits if c.readme_exists]
        readme_changed = len(readme_sizes) > 1 and max(readme_sizes) - min(readme_sizes) > 100

        # Authors
        authors = list(set(c.author for c in commits))
        author_counts = {}
        for c in commits:
            author_counts[c.author] = author_counts.get(c.author, 0) + 1
        primary_author = max(author_counts.items(), key=lambda x: x[1])[0]

        return Phase(
            phase_number=phase_number,
            start_date=start_date,
            end_date=end_date,
            commit_count=commit_count,
            commits=commits,
            loc_start=loc_start,
            loc_end=loc_end,
            loc_delta=loc_delta,
            loc_delta_percent=loc_delta_percent,
            total_insertions=total_insertions,
            total_deletions=total_deletions,
            languages_start=languages_start,
            languages_end=languages_end,
            has_large_deletion=has_large_deletion,
            has_large_addition=has_large_addition,
            has_refactor=has_refactor,
            readme_changed=readme_changed,
            authors=authors,
            primary_author=primary_author,
        )

    def save_phases(self, phases: List[Phase], output_dir: str):
        """Save phases to JSON files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save each phase separately
        for phase in phases:
            phase_file = output_path / f"phase_{phase.phase_number:02d}.json"
            with open(phase_file, 'w') as f:
                json.dump(phase.to_dict(), f, indent=2)

        # Save phase index
        index = {
            'total_phases': len(phases),
            'phases': [
                {
                    'phase_number': p.phase_number,
                    'start_date': p.start_date,
                    'end_date': p.end_date,
                    'commit_count': p.commit_count,
                    'loc_delta': p.loc_delta,
                }
                for p in phases
            ]
        }

        index_file = output_path / "phase_index.json"
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)

    @staticmethod
    def load_phases(input_dir: str) -> List[Phase]:
        """Load phases from JSON files."""
        input_path = Path(input_dir)
        phases = []

        # Find all phase files
        phase_files = sorted(input_path.glob("phase_*.json"))

        for phase_file in phase_files:
            with open(phase_file, 'r') as f:
                data = json.load(f)
                phase = Phase.from_dict(data)
                phases.append(phase)

        return phases


def chunk_history(records: List[CommitRecord],
                  strategy: str = "adaptive",
                  output_dir: str = "output/phases",
                  **kwargs) -> List[Phase]:
    """
    Chunk commit history into phases.

    Args:
        records: List of CommitRecord objects
        strategy: 'fixed', 'time', or 'adaptive'
        output_dir: Directory to save phase files
        **kwargs: Strategy-specific parameters

    Returns:
        List of Phase objects
    """
    chunker = HistoryChunker(strategy)
    phases = chunker.chunk(records, **kwargs)
    chunker.save_phases(phases, output_dir)
    return phases
