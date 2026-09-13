"""GitView CLI commands module.

This module contains the command implementations extracted from cli.py
for better modularity, testability, and maintainability.
"""

from .base import BaseCommand
from .analyze import AnalyzeCommand
from .extract import ExtractCommand
from .chunk import ChunkCommand
from .track_files import TrackFilesCommand
from .file_history import FileHistoryCommand
from .inject_history import InjectHistoryCommand
from .remove_history import RemoveHistoryCommand
from .compare_branches import CompareBranchesCommand
from .worklog import WorklogCommand
from .brief import BriefCommand
from .graph import GraphCommand
from .observe import ObserveCommand
from .motifs import MotifsCommand

__all__ = [
    'BaseCommand',
    'AnalyzeCommand',
    'ExtractCommand',
    'ChunkCommand',
    'TrackFilesCommand',
    'FileHistoryCommand',
    'InjectHistoryCommand',
    'RemoveHistoryCommand',
    'CompareBranchesCommand',
    'WorklogCommand',
    'BriefCommand',
    'GraphCommand',
    'ObserveCommand',
    'MotifsCommand',
]
