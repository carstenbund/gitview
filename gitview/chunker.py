"""Chunk git history into meaningful epochs/phases."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from .extractor import CommitRecord


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
        commits = [CommitRecord(**c) for c in data.pop('commits')]
        return cls(commits=commits, **data)


class HistoryChunker:
    """Chunk repository history into meaningful phases."""

    def __init__(self, strategy: str = "adaptive"):
        """
        Initialize chunker with strategy.

        Args:
            strategy: 'fixed', 'time', or 'adaptive'
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
        # Default thresholds
        loc_threshold = config.get('loc_threshold', 0.3)
        min_chunk_size = config.get('min_chunk_size', 5)
        max_chunk_size = config.get('max_chunk_size', 100)
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
