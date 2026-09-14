"""Neutral model for repository versions (no git, no I/O).

A repository describes its own versioning in a *descriptor*; every source it
names is translated into :class:`VersionEvent` objects, and phases are derived
from those events only.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

DESCRIPTOR_SCHEMA = 1

# Ordered from most to least significant. ``label`` components are displayed
# but never seal anything.
ROLES = ('generation', 'boundary', 'step', 'label')
SEALING_ROLES = ('generation', 'boundary')
UNRESOLVED = '?'

SOURCE_KINDS = ('tag', 'file', 'sequence')
FILE_PARSERS = ('python-assign', 'python-dunder', 'json', 'toml', 'regex', 'plain')

UNVERSIONED_PHASE_ID = 'unversioned'


class DescriptorError(ValueError):
    """The descriptor itself is malformed (as opposed to history not matching it)."""


@dataclass(frozen=True)
class FieldSpec:
    name: str
    role: str
    meaning: str = ''
    suggested: str = ''


@dataclass
class SourceSpec:
    kind: str
    fields: List[FieldSpec]
    options: Dict[str, Any] = field(default_factory=dict)

    def field(self, name: str) -> Optional[FieldSpec]:
        return next((f for f in self.fields if f.name == name), None)

    def names_for(self, role: str) -> List[str]:
        return [f.name for f in self.fields if f.role == role]

    def describe(self) -> str:
        if self.kind == 'tag':
            return f"tag:{self.options.get('pattern', '')}"
        if self.kind == 'file':
            return f"file:{self.options.get('path', '')}"
        return f"sequence:{self.options.get('glob', '')}"


@dataclass
class Descriptor:
    """How one repository (or one project inside a monorepo) is versioned."""
    origin: str                  # file the descriptor was read from, e.g. "pyproject.toml"
    scope: str                   # directory it applies to, relative to the git root ("" = root)
    sources: List[SourceSpec]
    schema: int = DESCRIPTOR_SCHEMA
    description: str = ''

    def unresolved_fields(self) -> List[Tuple[SourceSpec, FieldSpec]]:
        return [(s, f) for s in self.sources for f in s.fields if f.role == UNRESOLVED]


@dataclass(frozen=True)
class VersionEvent:
    """A version taking effect at one commit of the analysed branch."""
    commit: str
    timestamp: str
    source: str
    raw: str
    label: str
    fields: Dict[str, int]
    level: str                   # highest role whose value changed
    key: Tuple[Optional[int], Optional[int], Optional[int]]   # (generation, boundary, step)
    message: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return {
            'commit': self.commit, 'timestamp': self.timestamp, 'source': self.source,
            'raw': self.raw, 'label': self.label, 'fields': dict(self.fields),
            'level': self.level, 'message': self.message,
        }


@dataclass
class VersionPhase:
    id: str
    level: str                   # 'generation' | 'boundary' | 'unversioned'
    start: str                   # first commit on the branch's first-parent chain
    end: str                     # last commit on that chain
    start_timestamp: str
    end_timestamp: str
    commits: int                 # all commits in the git range, merged branches included
    sealed: bool
    event: Optional[VersionEvent] = None
    steps: List[VersionEvent] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id, 'level': self.level, 'sealed': self.sealed,
            'start': self.start, 'end': self.end,
            'from': self.start_timestamp, 'to': self.end_timestamp,
            'commits': self.commits,
            'version': self.event.to_dict() if self.event else None,
            'steps': [s.to_dict() for s in self.steps],
        }


@dataclass(frozen=True)
class Problem:
    severity: str                # 'error' | 'warning'
    message: str
    commit: str = ''
    source: str = ''

    def to_dict(self) -> Dict[str, str]:
        return {'severity': self.severity, 'message': self.message,
                'commit': self.commit, 'source': self.source}


@dataclass
class VersionTimeline:
    descriptor: Descriptor
    branch: str
    tip: str
    events: List[VersionEvent]
    phases: List[VersionPhase]
    problems: List[Problem]

    @property
    def errors(self) -> List[Problem]:
        return [p for p in self.problems if p.severity == 'error']

    @property
    def warnings(self) -> List[Problem]:
        return [p for p in self.problems if p.severity == 'warning']

    def to_dict(self) -> Dict[str, Any]:
        return {
            'descriptor': {'origin': self.descriptor.origin, 'scope': self.descriptor.scope,
                           'description': self.descriptor.description,
                           'sources': [s.describe() for s in self.descriptor.sources]},
            'branch': self.branch, 'tip': self.tip,
            'phases': [p.to_dict() for p in self.phases],
            'problems': [p.to_dict() for p in self.problems],
        }
