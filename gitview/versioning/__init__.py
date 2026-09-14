"""Repository versioning: a descriptor the owner confirms, and the phases it yields.

GitView does not guess what a version number means. A repository describes
where it is versioned (tags, version files, numbered migrations) and which
component seals a phase in a *version descriptor*; ``detect`` drafts one,
``load_timeline`` replays it over a branch. See docs/PHASED_HISTORY_DESIGN.md.
"""

from .descriptor import find_descriptor, parse_descriptor
from .detect import Detection, detect, render_draft, write_draft
from .models import (
    Descriptor,
    DescriptorError,
    FieldSpec,
    Problem,
    SourceSpec,
    VersionEvent,
    VersionPhase,
    VersionTimeline,
)
from .sources import BranchHistory, GitError
from .timeline import build_timeline, load_timeline

__all__ = [
    'BranchHistory',
    'Descriptor',
    'DescriptorError',
    'Detection',
    'FieldSpec',
    'GitError',
    'Problem',
    'SourceSpec',
    'VersionEvent',
    'VersionPhase',
    'VersionTimeline',
    'build_timeline',
    'detect',
    'find_descriptor',
    'load_timeline',
    'parse_descriptor',
    'render_draft',
    'write_draft',
]
