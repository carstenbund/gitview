"""Storyline tracking for narrative continuity across phases."""

from .models import (
    StorylineStatus,
    StorylineCategory,
    StorylineSignal,
    StorylineUpdate,
    Storyline,
    StorylineDatabase,
)
from .state_machine import StorylineStateMachine
from .detector import (
    StorylineDetector,
    PRLabelDetector,
    PRTitlePatternDetector,
    CommitMessagePatternDetector,
    FileClusterDetector,
)

__all__ = [
    # Models
    'StorylineStatus',
    'StorylineCategory',
    'StorylineSignal',
    'StorylineUpdate',
    'Storyline',
    'StorylineDatabase',
    # State machine
    'StorylineStateMachine',
    # Detectors
    'StorylineDetector',
    'PRLabelDetector',
    'PRTitlePatternDetector',
    'CommitMessagePatternDetector',
    'FileClusterDetector',
]
