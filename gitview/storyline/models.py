"""Core data models for storyline tracking."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum
from datetime import datetime
import hashlib
import json


class StorylineStatus(Enum):
    """State machine states for storyline lifecycle."""
    EMERGING = "emerging"        # First signal detected, not yet confirmed
    ACTIVE = "active"            # Confirmed ongoing work
    PROGRESSING = "progressing"  # Active with recent updates
    STALLED = "stalled"          # No updates in N phases
    COMPLETED = "completed"      # Explicitly resolved
    ABANDONED = "abandoned"      # No activity, never completed


class StorylineCategory(Enum):
    """Categories of storylines."""
    FEATURE = "feature"
    REFACTOR = "refactor"
    BUGFIX = "bugfix"
    TECH_DEBT = "tech_debt"
    INFRASTRUCTURE = "infrastructure"
    DOCUMENTATION = "documentation"
    MIGRATION = "migration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str) -> 'StorylineCategory':
        """Convert string to category, with fuzzy matching."""
        value = value.lower().strip()

        # Direct mappings
        mappings = {
            'feature': cls.FEATURE,
            'feat': cls.FEATURE,
            'enhancement': cls.FEATURE,
            'refactor': cls.REFACTOR,
            'refactoring': cls.REFACTOR,
            'cleanup': cls.REFACTOR,
            'bugfix': cls.BUGFIX,
            'bug': cls.BUGFIX,
            'fix': cls.BUGFIX,
            'hotfix': cls.BUGFIX,
            'tech_debt': cls.TECH_DEBT,
            'debt': cls.TECH_DEBT,
            'technical-debt': cls.TECH_DEBT,
            'infrastructure': cls.INFRASTRUCTURE,
            'infra': cls.INFRASTRUCTURE,
            'ci': cls.INFRASTRUCTURE,
            'ci/cd': cls.INFRASTRUCTURE,
            'build': cls.INFRASTRUCTURE,
            'documentation': cls.DOCUMENTATION,
            'docs': cls.DOCUMENTATION,
            'doc': cls.DOCUMENTATION,
            'migration': cls.MIGRATION,
            'migrate': cls.MIGRATION,
            'performance': cls.PERFORMANCE,
            'perf': cls.PERFORMANCE,
            'optimization': cls.PERFORMANCE,
            'security': cls.SECURITY,
            'sec': cls.SECURITY,
        }

        return mappings.get(value, cls.UNKNOWN)


@dataclass
class StorylineSignal:
    """A single signal that contributes to storyline detection."""
    source: str                  # 'pr_label', 'pr_title', 'commit_pattern', 'file_cluster', 'llm_extraction'
    confidence: float            # 0.0 to 1.0
    phase_number: int
    commit_hashes: List[str]
    title: str                   # Suggested storyline title
    category: StorylineCategory
    description: str
    data: Dict[str, Any] = field(default_factory=dict)  # Source-specific data
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    files: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            'source': self.source,
            'confidence': self.confidence,
            'phase_number': self.phase_number,
            'commit_hashes': self.commit_hashes,
            'title': self.title,
            'category': self.category.value,
            'description': self.description,
            'data': self.data,
            'timestamp': self.timestamp,
            'files': self.files,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StorylineSignal':
        """Deserialize from dict."""
        return cls(
            source=data['source'],
            confidence=data['confidence'],
            phase_number=data['phase_number'],
            commit_hashes=data['commit_hashes'],
            title=data['title'],
            category=StorylineCategory(data['category']),
            description=data['description'],
            data=data.get('data', {}),
            timestamp=data.get('timestamp', datetime.now().isoformat()),
            files=data.get('files', []),
        )


@dataclass
class StorylineUpdate:
    """A single update/progress entry for a storyline."""
    phase_number: int
    timestamp: str
    description: str
    signals: List[StorylineSignal]
    commit_count: int
    insertions: int
    deletions: int
    key_files: List[str]
    key_commits: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            'phase_number': self.phase_number,
            'timestamp': self.timestamp,
            'description': self.description,
            'signals': [s.to_dict() for s in self.signals],
            'commit_count': self.commit_count,
            'insertions': self.insertions,
            'deletions': self.deletions,
            'key_files': self.key_files,
            'key_commits': self.key_commits,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StorylineUpdate':
        """Deserialize from dict."""
        return cls(
            phase_number=data['phase_number'],
            timestamp=data['timestamp'],
            description=data['description'],
            signals=[StorylineSignal.from_dict(s) for s in data.get('signals', [])],
            commit_count=data['commit_count'],
            insertions=data['insertions'],
            deletions=data['deletions'],
            key_files=data['key_files'],
            key_commits=data['key_commits'],
        )


@dataclass
class Storyline:
    """A tracked narrative thread across phases."""
    id: str                                  # Unique identifier
    title: str                               # Canonical name
    category: StorylineCategory
    status: StorylineStatus
    confidence: float                        # Aggregate confidence (0.0-1.0)

    # Temporal tracking
    first_phase: int
    last_phase: int
    phases_involved: List[int] = field(default_factory=list)

    # Content
    description: str = ""                    # Initial description
    current_summary: str = ""                # Latest status summary
    updates: List[StorylineUpdate] = field(default_factory=list)

    # Signals that contributed to detection
    signals: List[StorylineSignal] = field(default_factory=list)

    # Related entities
    related_storylines: List[str] = field(default_factory=list)  # IDs of connected storylines
    key_files: Set[str] = field(default_factory=set)             # Files most associated
    key_authors: Set[str] = field(default_factory=set)           # Authors most involved
    pr_numbers: List[int] = field(default_factory=list)          # Associated PRs

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = field(default_factory=list)                # Additional tags

    @staticmethod
    def generate_id(title: str, first_phase: int) -> str:
        """Generate a unique ID for a storyline."""
        content = f"{title.lower().strip()}:{first_phase}"
        return f"sl_{hashlib.sha256(content.encode()).hexdigest()[:12]}"

    @staticmethod
    def normalize_title(title: str) -> str:
        """Normalize a title for comparison."""
        import re
        # Lowercase, remove special chars, collapse whitespace
        normalized = title.lower().strip()
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized.strip()

    def add_signal(self, signal: StorylineSignal) -> None:
        """Add a signal and update aggregate confidence."""
        self.signals.append(signal)
        self._recalculate_confidence()
        self.updated_at = datetime.now().isoformat()

    def add_update(self, update: StorylineUpdate) -> None:
        """Add an update to the storyline."""
        self.updates.append(update)
        if update.phase_number not in self.phases_involved:
            self.phases_involved.append(update.phase_number)
            self.phases_involved.sort()
        self.last_phase = max(self.last_phase, update.phase_number)
        self.current_summary = update.description
        self.key_files.update(update.key_files)
        self.updated_at = datetime.now().isoformat()

    def _recalculate_confidence(self) -> None:
        """Recalculate aggregate confidence from signals."""
        if not self.signals:
            self.confidence = 0.0
            return

        # Weighted average with diminishing returns for multiple signals
        # First signal counts full, subsequent signals add with decay
        sorted_signals = sorted(self.signals, key=lambda s: s.confidence, reverse=True)
        total_weight = 0.0
        weighted_sum = 0.0
        decay = 1.0

        for signal in sorted_signals:
            weighted_sum += signal.confidence * decay
            total_weight += decay
            decay *= 0.7  # Each additional signal contributes less

        self.confidence = min(1.0, weighted_sum / total_weight) if total_weight > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON persistence."""
        return {
            'id': self.id,
            'title': self.title,
            'category': self.category.value,
            'status': self.status.value,
            'confidence': self.confidence,
            'first_phase': self.first_phase,
            'last_phase': self.last_phase,
            'phases_involved': self.phases_involved,
            'description': self.description,
            'current_summary': self.current_summary,
            'updates': [u.to_dict() for u in self.updates],
            'signals': [s.to_dict() for s in self.signals],
            'related_storylines': self.related_storylines,
            'key_files': list(self.key_files),
            'key_authors': list(self.key_authors),
            'pr_numbers': self.pr_numbers,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'tags': self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Storyline':
        """Deserialize from dict."""
        storyline = cls(
            id=data['id'],
            title=data['title'],
            category=StorylineCategory(data['category']),
            status=StorylineStatus(data['status']),
            confidence=data['confidence'],
            first_phase=data['first_phase'],
            last_phase=data['last_phase'],
            phases_involved=data.get('phases_involved', []),
            description=data.get('description', ''),
            current_summary=data.get('current_summary', ''),
            related_storylines=data.get('related_storylines', []),
            key_files=set(data.get('key_files', [])),
            key_authors=set(data.get('key_authors', [])),
            pr_numbers=data.get('pr_numbers', []),
            created_at=data.get('created_at', datetime.now().isoformat()),
            updated_at=data.get('updated_at', datetime.now().isoformat()),
            tags=data.get('tags', []),
        )

        # Deserialize nested objects
        storyline.updates = [StorylineUpdate.from_dict(u) for u in data.get('updates', [])]
        storyline.signals = [StorylineSignal.from_dict(s) for s in data.get('signals', [])]

        return storyline

    @classmethod
    def from_signal(cls, signal: StorylineSignal) -> 'Storyline':
        """Create a new storyline from an initial signal."""
        storyline_id = cls.generate_id(signal.title, signal.phase_number)

        storyline = cls(
            id=storyline_id,
            title=signal.title,
            category=signal.category,
            status=StorylineStatus.EMERGING,
            confidence=signal.confidence,
            first_phase=signal.phase_number,
            last_phase=signal.phase_number,
            phases_involved=[signal.phase_number],
            description=signal.description,
            current_summary=signal.description,
        )

        storyline.signals.append(signal)
        storyline.key_files.update(signal.files)

        return storyline


@dataclass
class StorylineDatabase:
    """Container for all storylines with indexing."""
    storylines: Dict[str, Storyline] = field(default_factory=dict)     # id -> Storyline
    title_index: Dict[str, str] = field(default_factory=dict)          # normalized_title -> id
    phase_index: Dict[int, List[str]] = field(default_factory=dict)    # phase_number -> [ids]
    file_index: Dict[str, List[str]] = field(default_factory=dict)     # file_path -> [ids]
    pr_index: Dict[int, List[str]] = field(default_factory=dict)       # pr_number -> [ids]

    # Metadata
    last_phase_analyzed: int = 0
    last_commit_hash: str = ""
    version: str = "1.0.0"

    def add_storyline(self, storyline: Storyline) -> None:
        """Add a storyline and update indexes."""
        self.storylines[storyline.id] = storyline

        # Update title index
        normalized = Storyline.normalize_title(storyline.title)
        self.title_index[normalized] = storyline.id

        # Update phase index
        for phase in storyline.phases_involved:
            if phase not in self.phase_index:
                self.phase_index[phase] = []
            if storyline.id not in self.phase_index[phase]:
                self.phase_index[phase].append(storyline.id)

        # Update file index
        for file_path in storyline.key_files:
            if file_path not in self.file_index:
                self.file_index[file_path] = []
            if storyline.id not in self.file_index[file_path]:
                self.file_index[file_path].append(storyline.id)

        # Update PR index
        for pr_num in storyline.pr_numbers:
            if pr_num not in self.pr_index:
                self.pr_index[pr_num] = []
            if storyline.id not in self.pr_index[pr_num]:
                self.pr_index[pr_num].append(storyline.id)

    def get_by_title(self, title: str) -> Optional[Storyline]:
        """Find storyline by title (normalized)."""
        normalized = Storyline.normalize_title(title)
        storyline_id = self.title_index.get(normalized)
        if storyline_id:
            return self.storylines.get(storyline_id)
        return None

    def get_by_phase(self, phase_number: int) -> List[Storyline]:
        """Get all storylines involved in a phase."""
        ids = self.phase_index.get(phase_number, [])
        return [self.storylines[id] for id in ids if id in self.storylines]

    def get_by_file(self, file_path: str) -> List[Storyline]:
        """Get all storylines associated with a file."""
        ids = self.file_index.get(file_path, [])
        return [self.storylines[id] for id in ids if id in self.storylines]

    def get_by_pr(self, pr_number: int) -> List[Storyline]:
        """Get all storylines associated with a PR."""
        ids = self.pr_index.get(pr_number, [])
        return [self.storylines[id] for id in ids if id in self.storylines]

    def get_active(self) -> List[Storyline]:
        """Get all non-completed storylines sorted by recency."""
        active = [
            s for s in self.storylines.values()
            if s.status not in (StorylineStatus.COMPLETED, StorylineStatus.ABANDONED)
        ]
        return sorted(active, key=lambda s: (s.last_phase, s.confidence), reverse=True)

    def get_by_status(self, status: StorylineStatus) -> List[Storyline]:
        """Get all storylines with a specific status."""
        return [s for s in self.storylines.values() if s.status == status]

    def update_indexes(self, storyline: Storyline) -> None:
        """Update indexes after a storyline is modified."""
        # Re-add to rebuild indexes for this storyline
        self.add_storyline(storyline)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON persistence."""
        return {
            'version': self.version,
            'metadata': {
                'last_phase_analyzed': self.last_phase_analyzed,
                'last_commit_hash': self.last_commit_hash,
                'last_updated': datetime.now().isoformat(),
                'total_storylines': len(self.storylines),
                'status_counts': {
                    status.value: len(self.get_by_status(status))
                    for status in StorylineStatus
                },
            },
            'storylines': [s.to_dict() for s in self.storylines.values()],
            'indexes': {
                'title_index': self.title_index,
                'phase_index': {str(k): v for k, v in self.phase_index.items()},
                'pr_index': {str(k): v for k, v in self.pr_index.items()},
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StorylineDatabase':
        """Deserialize from dict."""
        db = cls(
            version=data.get('version', '1.0.0'),
            last_phase_analyzed=data.get('metadata', {}).get('last_phase_analyzed', 0),
            last_commit_hash=data.get('metadata', {}).get('last_commit_hash', ''),
        )

        # Load storylines and rebuild indexes
        for storyline_data in data.get('storylines', []):
            storyline = Storyline.from_dict(storyline_data)
            db.add_storyline(storyline)

        return db

    def save(self, path: str) -> None:
        """Save database to JSON file."""
        from pathlib import Path
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'StorylineDatabase':
        """Load database from JSON file."""
        from pathlib import Path
        file_path = Path(path)

        if not file_path.exists():
            return cls()

        with open(file_path, 'r') as f:
            data = json.load(f)

        return cls.from_dict(data)
