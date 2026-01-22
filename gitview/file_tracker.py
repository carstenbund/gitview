"""
File history tracking for detailed per-file change analysis.

This module provides functionality to track the complete history of each file
in a repository, extract change details, and generate companion .history files
that can be injected as header comments for debugging and accountability.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Set
import json
import hashlib
from git import Repo, GitCommandError


@dataclass
class FileChange:
    """Represents a single commit's change to one file"""
    file_path: str
    commit_hash: str
    commit_date: str
    author_name: str
    author_email: str
    commit_message: str

    # Change details
    lines_added: int
    lines_removed: int

    # Diff information
    diff_snippet: str = ""  # Truncated or key sections

    # Future: Will be populated in Phase 4
    functions_added: List[str] = field(default_factory=list)
    functions_removed: List[str] = field(default_factory=list)
    functions_modified: List[str] = field(default_factory=list)
    classes_added: List[str] = field(default_factory=list)
    classes_modified: List[str] = field(default_factory=list)

    # AI summary (Phase 2)
    ai_summary: Optional[str] = None
    ai_summary_model: Optional[str] = None
    ai_summary_generated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileChange':
        """Deserialize from dictionary"""
        return cls(**data)

    def get_summary_line(self) -> str:
        """Get one-line summary for compact display"""
        msg = self.commit_message.split('\n')[0][:60]
        return f"{self.commit_date[:10]} | {self.commit_hash[:7]} | {self.author_name}"


@dataclass
class FileHistory:
    """Complete change history for a single file"""
    file_path: str
    current_path: str  # May differ if renamed

    # Timeline
    first_commit: str
    first_commit_date: str
    last_commit: str
    last_commit_date: str

    # Statistics
    total_commits: int
    total_lines_added: int
    total_lines_removed: int

    # Contributors (sorted by commit count)
    authors: List[Dict[str, Any]] = field(default_factory=list)

    # Rename history
    previous_paths: List[str] = field(default_factory=list)

    # Chronological changes (newest first)
    changes: List[FileChange] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        data = asdict(self)
        # Convert FileChange objects to dicts
        data['changes'] = [change.to_dict() if hasattr(change, 'to_dict') else change
                          for change in self.changes]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileHistory':
        """Deserialize from dictionary"""
        # Convert change dicts to FileChange objects
        changes_data = data.pop('changes', [])
        changes = [FileChange.from_dict(c) if isinstance(c, dict) else c
                  for c in changes_data]
        history = cls(**data)
        history.changes = changes
        return history

    def format_as_text(self, max_entries: int = 10, include_diffs: bool = False) -> str:
        """
        Format history as human-readable text

        Args:
            max_entries: Maximum number of changes to include
            include_diffs: Include diff snippets

        Returns:
            Formatted text suitable for .history file
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"FILE CHANGE HISTORY: {self.file_path}")
        lines.append("=" * 80)
        lines.append(f"First seen: {self.first_commit_date[:10]} (commit {self.first_commit[:7]})")
        lines.append(f"Last modified: {self.last_commit_date[:10]} (commit {self.last_commit[:7]})")
        lines.append(f"Total commits: {self.total_commits}")
        lines.append(f"Lines added: +{self.total_lines_added}")
        lines.append(f"Lines removed: -{self.total_lines_removed}")
        lines.append(f"Net change: {self.total_lines_added - self.total_lines_removed:+d} lines")
        lines.append("")

        # Authors
        if self.authors:
            lines.append("Contributors:")
            for author in self.authors[:5]:  # Top 5 contributors
                name = author.get('name', 'Unknown')
                count = author.get('commits', 0)
                pct = (count / self.total_commits * 100) if self.total_commits > 0 else 0
                lines.append(f"  {name:30} {count:3} commits ({pct:5.1f}%)")
            lines.append("")

        # Rename history
        if self.previous_paths:
            lines.append("Previous paths:")
            for path in self.previous_paths:
                lines.append(f"  {path}")
            lines.append("")

        # Changes
        display_count = min(max_entries, len(self.changes))
        lines.append("-" * 80)
        lines.append(f"[Recent Changes - Last {display_count} of {self.total_commits}]")
        lines.append("-" * 80)
        lines.append("")

        for change in self.changes[:display_count]:
            lines.append(f"{change.commit_date} | {change.commit_hash[:7]} | {change.author_name}")

            # Commit message (first line)
            msg_lines = change.commit_message.strip().split('\n')
            lines.append(f"  {msg_lines[0]}")

            # Change stats
            lines.append(f"  Changes: +{change.lines_added} lines, -{change.lines_removed} lines")

            # Function/class changes (if available)
            if change.functions_modified or change.functions_added or change.classes_modified:
                changes_parts = []
                if change.functions_modified:
                    changes_parts.append(f"Modified: {', '.join(change.functions_modified[:3])}")
                if change.functions_added:
                    changes_parts.append(f"Added: {', '.join(change.functions_added[:3])}")
                if change.classes_modified:
                    changes_parts.append(f"Classes: {', '.join(change.classes_modified[:2])}")
                if changes_parts:
                    lines.append(f"  {' | '.join(changes_parts)}")

            # AI Summary (if available)
            if change.ai_summary:
                lines.append("")
                lines.append("  AI Summary:")
                # Wrap summary text at reasonable length
                summary_words = change.ai_summary.split()
                current_line = "    "
                for word in summary_words:
                    if len(current_line) + len(word) + 1 > 78:
                        lines.append(current_line)
                        current_line = "    " + word
                    else:
                        current_line += " " + word if current_line != "    " else word
                if current_line.strip():
                    lines.append(current_line)

            # Diff snippet (if requested)
            if include_diffs and change.diff_snippet:
                lines.append("")
                lines.append("  Diff:")
                for diff_line in change.diff_snippet.split('\n')[:10]:  # Max 10 lines
                    lines.append(f"    {diff_line}")

            lines.append("")

        if len(self.changes) > display_count:
            lines.append(f"... and {len(self.changes) - display_count} more changes")
            lines.append("")

        lines.append("=" * 80)
        lines.append(f"Full history available in JSON format")
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append("=" * 80)

        return '\n'.join(lines)


@dataclass
class FileTrackerCheckpoint:
    """Tracks last processed state for incremental runs"""
    last_processed_commit: str
    last_processed_timestamp: str
    total_commits_processed: int
    total_files_tracked: int

    # Next run should start after this commit
    resume_from_commit: str

    # Metadata
    cache_version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileTrackerCheckpoint':
        """Deserialize from dictionary"""
        return cls(**data)

    def save(self, path: Path):
        """Save checkpoint to JSON file"""
        self.updated_at = datetime.now().isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> Optional['FileTrackerCheckpoint']:
        """Load checkpoint from JSON file"""
        if not path.exists():
            return None
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except (json.JSONDecodeError, FileNotFoundError):
            return None


class FileChangeExtractor:
    """
    Extracts detailed change information for files across git history
    """

    def __init__(self, repo_path: str):
        self.repo = Repo(repo_path)

    def extract_file_history(
        self,
        file_path: str,
        since_commit: Optional[str] = None,
        max_count: Optional[int] = None
    ) -> FileHistory:
        """
        Extract complete history for a single file

        Args:
            file_path: Relative path to file in repository
            since_commit: Only get commits after this commit
            max_count: Maximum number of commits to process

        Returns:
            FileHistory object with all changes
        """
        changes = []
        authors_map = {}  # Track commit counts per author
        total_added = 0
        total_removed = 0

        # Get all commits that touched this file
        try:
            if since_commit:
                # Get commits after since_commit
                commits = list(self.repo.iter_commits(f'{since_commit}..HEAD', paths=file_path, max_count=max_count))
            elif max_count:
                # Limit number of commits
                commits = list(self.repo.iter_commits(paths=file_path, max_count=max_count))
            else:
                # Get all commits for this file
                commits = list(self.repo.iter_commits(paths=file_path))

        except GitCommandError:
            # File might not exist or path is invalid
            commits = []

        if not commits:
            # Return empty history
            now = datetime.now().isoformat()
            return FileHistory(
                file_path=file_path,
                current_path=file_path,
                first_commit="",
                first_commit_date=now,
                last_commit="",
                last_commit_date=now,
                total_commits=0,
                total_lines_added=0,
                total_lines_removed=0,
                authors=[],
                previous_paths=[],
                changes=[]
            )

        # Process each commit (reverse chronological order)
        for commit in commits:
            change = self._extract_single_change(commit, file_path)
            if change:
                changes.append(change)
                total_added += change.lines_added
                total_removed += change.lines_removed

                # Track author
                author = change.author_name
                if author not in authors_map:
                    authors_map[author] = {
                        'name': author,
                        'email': change.author_email,
                        'commits': 0
                    }
                authors_map[author]['commits'] += 1

        # Sort authors by commit count
        authors = sorted(
            authors_map.values(),
            key=lambda x: x['commits'],
            reverse=True
        )

        # Build FileHistory object
        first_commit = commits[-1] if commits else None
        last_commit = commits[0] if commits else None

        return FileHistory(
            file_path=file_path,
            current_path=file_path,
            first_commit=first_commit.hexsha if first_commit else "",
            first_commit_date=first_commit.committed_datetime.isoformat() if first_commit else "",
            last_commit=last_commit.hexsha if last_commit else "",
            last_commit_date=last_commit.committed_datetime.isoformat() if last_commit else "",
            total_commits=len(changes),
            total_lines_added=total_added,
            total_lines_removed=total_removed,
            authors=authors,
            previous_paths=[],  # TODO: Track renames in Phase 4
            changes=changes
        )

    def _extract_single_change(self, commit, file_path: str) -> Optional[FileChange]:
        """
        Extract change details for a single commit

        Args:
            commit: GitPython commit object
            file_path: File path to analyze

        Returns:
            FileChange object or None if file wasn't changed
        """
        try:
            # Get diff stats for this file
            lines_added = 0
            lines_removed = 0
            diff_text = ""

            # Get parent commit for diff
            if commit.parents:
                parent = commit.parents[0]
                diffs = parent.diff(commit, paths=file_path, create_patch=True)

                if diffs:
                    diff = diffs[0]

                    # Parse diff stats
                    if diff.diff:
                        diff_text = diff.diff.decode('utf-8', errors='ignore')
                        for line in diff_text.split('\n'):
                            if line.startswith('+') and not line.startswith('+++'):
                                lines_added += 1
                            elif line.startswith('-') and not line.startswith('---'):
                                lines_removed += 1
            else:
                # First commit - all lines are additions
                try:
                    blob = commit.tree / file_path
                    if blob:
                        content = blob.data_stream.read().decode('utf-8', errors='ignore')
                        lines_added = len(content.split('\n'))
                except (KeyError, AttributeError):
                    pass

            # Generate diff snippet (first 20 lines)
            diff_snippet = self._generate_diff_snippet(diff_text, max_lines=20)

            return FileChange(
                file_path=file_path,
                commit_hash=commit.hexsha,
                commit_date=commit.committed_datetime.isoformat(),
                author_name=commit.author.name,
                author_email=commit.author.email,
                commit_message=commit.message.strip(),
                lines_added=lines_added,
                lines_removed=lines_removed,
                diff_snippet=diff_snippet
            )

        except Exception as e:
            # Log error but continue processing
            print(f"Warning: Could not extract change for {file_path} in commit {commit.hexsha[:7]}: {e}")
            return None

    def _generate_diff_snippet(self, diff_text: str, max_lines: int = 20) -> str:
        """
        Extract key sections from diff

        Args:
            diff_text: Full diff text
            max_lines: Maximum lines to include

        Returns:
            Truncated diff snippet
        """
        if not diff_text:
            return ""

        lines = diff_text.split('\n')

        # Skip diff headers (---, +++, @@)
        content_lines = []
        for line in lines:
            if line.startswith('---') or line.startswith('+++'):
                continue
            if line.startswith('@@'):
                continue
            content_lines.append(line)

        # Truncate to max_lines
        if len(content_lines) > max_lines:
            snippet = '\n'.join(content_lines[:max_lines])
            snippet += f"\n... ({len(content_lines) - max_lines} more lines)"
        else:
            snippet = '\n'.join(content_lines)

        return snippet

    def get_tracked_files(
        self,
        patterns: List[str] = None,
        exclude_patterns: List[str] = None
    ) -> List[str]:
        """
        Get list of files to track based on patterns

        Args:
            patterns: Glob patterns to include (e.g., ['*.py', '*.js'])
            exclude_patterns: Glob patterns to exclude

        Returns:
            List of file paths
        """
        from fnmatch import fnmatch

        # Get all files in current tree
        all_files = []
        for item in self.repo.tree().traverse():
            if item.type == 'blob':  # Is a file
                all_files.append(item.path)

        # Apply filters
        filtered_files = []
        for file_path in all_files:
            # Check include patterns
            if patterns:
                if not any(fnmatch(file_path, pattern) for pattern in patterns):
                    continue

            # Check exclude patterns
            if exclude_patterns:
                if any(fnmatch(file_path, pattern) for pattern in exclude_patterns):
                    continue

            filtered_files.append(file_path)

        return sorted(filtered_files)


class FileHistoryTracker:
    """
    Main orchestrator for file history tracking
    """

    def __init__(
        self,
        repo_path: str,
        output_dir: str = "output/file_histories",
        llm_router = None
    ):
        self.repo_path = Path(repo_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.extractor = FileChangeExtractor(str(repo_path))
        self.checkpoint_path = self.output_dir / "checkpoint.json"
        self.histories_dir = self.output_dir / "files"
        self.histories_dir.mkdir(parents=True, exist_ok=True)

        # AI summarizer (optional)
        self.llm_router = llm_router
        self.ai_summarizer = None
        if llm_router:
            from .file_ai_summarizer import ChangeAISummarizer
            cache_path = self.output_dir / "summaries_cache.json"
            self.ai_summarizer = ChangeAISummarizer(llm_router, cache_path)

    def track_all_files(
        self,
        file_patterns: List[str] = None,
        exclude_patterns: List[str] = None,
        since_commit: Optional[str] = None,
        incremental: bool = True,
        max_history_entries: int = 100,
        with_ai_summaries: bool = False
    ) -> Dict[str, Any]:
        """
        Track all files matching patterns

        Args:
            file_patterns: Patterns to include (default: all files)
            exclude_patterns: Patterns to exclude
            since_commit: Start from specific commit (overrides checkpoint)
            incremental: Use checkpoint for incremental processing
            max_history_entries: Max changes to keep per file
            with_ai_summaries: Generate AI summaries for changes (requires llm_router)

        Returns:
            Summary statistics
        """
        # Load checkpoint if incremental
        checkpoint = None
        if incremental and not since_commit:
            checkpoint = FileTrackerCheckpoint.load(self.checkpoint_path)
            if checkpoint:
                # Only use checkpoint if it's different from current HEAD
                repo = Repo(str(self.repo_path))
                current_head = repo.head.commit.hexsha
                if checkpoint.resume_from_commit != current_head:
                    since_commit = checkpoint.resume_from_commit
                    print(f"Resuming from checkpoint: {since_commit[:7]}")
                else:
                    print(f"Checkpoint is current (at HEAD), doing full scan")

        # Get files to track
        tracked_files = self.extractor.get_tracked_files(
            patterns=file_patterns,
            exclude_patterns=exclude_patterns
        )

        print(f"Tracking {len(tracked_files)} files...")

        # Track each file
        histories = {}
        total_changes = 0

        for i, file_path in enumerate(tracked_files, 1):
            if i % 10 == 0 or i == len(tracked_files):
                print(f"  Processing {i}/{len(tracked_files)}: {file_path}")

            try:
                history = self.extractor.extract_file_history(
                    file_path,
                    since_commit=since_commit,
                    max_count=max_history_entries
                )

                if history.total_commits > 0:
                    histories[file_path] = history
                    total_changes += history.total_commits

            except Exception as e:
                print(f"  Warning: Failed to process {file_path}: {e}")
                continue

        # Generate AI summaries if requested
        ai_summaries_generated = 0
        if with_ai_summaries and self.ai_summarizer:
            print(f"\nGenerating AI summaries for {total_changes} changes...")

            for file_path, history in histories.items():
                for change in history.changes:
                    # Skip if already has summary (from cache)
                    if change.ai_summary:
                        continue

                    try:
                        summary = self.ai_summarizer.summarize_change(
                            commit_hash=change.commit_hash,
                            file_path=file_path,
                            commit_message=change.commit_message,
                            author=change.author_name,
                            date=change.commit_date,
                            lines_added=change.lines_added,
                            lines_removed=change.lines_removed,
                            diff_snippet=change.diff_snippet,
                            functions_modified=change.functions_modified,
                            functions_added=change.functions_added
                        )

                        change.ai_summary = summary
                        change.ai_summary_model = self.llm_router.model
                        change.ai_summary_generated_at = datetime.now().isoformat()
                        ai_summaries_generated += 1

                        # Progress indicator
                        if ai_summaries_generated % 5 == 0:
                            print(f"  Generated {ai_summaries_generated}/{total_changes} summaries...")

                    except Exception as e:
                        print(f"  Warning: Failed to generate summary for {file_path} in {change.commit_hash[:7]}: {e}")
                        continue

            # Print cache stats
            cache_stats = self.ai_summarizer.get_cache_stats()
            print(f"\nAI Summary Stats:")
            print(f"  Generated: {ai_summaries_generated}")
            print(f"  From cache: {cache_stats['cache_hits']}")
            print(f"  Cache hit rate: {cache_stats['hit_rate_percent']:.1f}%")

        # Save all histories (with AI summaries if generated)
        for file_path, history in histories.items():
            try:
                self._save_file_history(history)
            except Exception as e:
                print(f"  Warning: Failed to save history for {file_path}: {e}")

        # Create/update checkpoint
        repo = self.extractor.repo
        latest_commit = repo.head.commit

        new_checkpoint = FileTrackerCheckpoint(
            last_processed_commit=latest_commit.hexsha,
            last_processed_timestamp=latest_commit.committed_datetime.isoformat(),
            total_commits_processed=total_changes,
            total_files_tracked=len(histories),
            resume_from_commit=latest_commit.hexsha
        )
        new_checkpoint.save(self.checkpoint_path)

        # Generate summary
        summary = {
            'files_tracked': len(histories),
            'total_changes': total_changes,
            'checkpoint_commit': latest_commit.hexsha[:7],
            'checkpoint_date': latest_commit.committed_datetime.isoformat(),
            'output_dir': str(self.output_dir)
        }

        # Save index
        self._save_index(histories)

        return summary

    def _save_file_history(self, history: FileHistory):
        """Save individual file history to both JSON and text formats"""
        # Create directory structure
        file_dir = self.histories_dir / Path(history.file_path).parent
        file_dir.mkdir(parents=True, exist_ok=True)

        file_name = Path(history.file_path).name

        # Save JSON (machine-readable)
        json_path = file_dir / f"{file_name}.json"
        with open(json_path, 'w') as f:
            json.dump(history.to_dict(), f, indent=2)

        # Save text (human-readable)
        text_path = file_dir / f"{file_name}.history"
        with open(text_path, 'w') as f:
            f.write(history.format_as_text(max_entries=10))

    def _save_index(self, histories: Dict[str, FileHistory]):
        """Save master index of all tracked files"""
        index = {
            'generated_at': datetime.now().isoformat(),
            'total_files': len(histories),
            'files': []
        }

        for file_path, history in sorted(histories.items()):
            index['files'].append({
                'path': file_path,
                'commits': history.total_commits,
                'first_commit': history.first_commit[:7],
                'last_commit': history.last_commit[:7],
                'total_changes': f"+{history.total_lines_added}/-{history.total_lines_removed}",
                'top_author': history.authors[0]['name'] if history.authors else 'Unknown'
            })

        index_path = self.output_dir / "index.json"
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)

    def get_file_history(self, file_path: str) -> Optional[FileHistory]:
        """Retrieve history for a specific file"""
        json_path = self.histories_dir / f"{file_path}.json"

        if not json_path.exists():
            return None

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return FileHistory.from_dict(data)
        except (json.JSONDecodeError, FileNotFoundError):
            return None
