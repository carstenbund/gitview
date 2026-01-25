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
from .tracker import StorylineTracker, create_tracker_from_legacy
from .parser import StorylineResponseParser, parse_storylines, build_storyline_prompt_section
from .reporter import StorylineReporter, generate_storyline_section_for_markdown
from .extractor import StorylineExtractor, ExtractedStoryline, parse_storylines as extract_storylines

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
    # Tracker
    'StorylineTracker',
    'create_tracker_from_legacy',
    # Parser
    'StorylineResponseParser',
    'parse_storylines',
    'build_storyline_prompt_section',
    # Reporter
    'StorylineReporter',
    'generate_storyline_section_for_markdown',
    # Extractor (unified JSON extraction)
    'StorylineExtractor',
    'ExtractedStoryline',
    'extract_storylines',
]
