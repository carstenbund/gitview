"""
History header injection for source files.

This module provides functionality to inject file change histories as header
comments into source files, with support for multiple programming languages
and comment styles.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

from .file_tracker import FileHistory, FileHistoryTracker


# Language detection patterns
LANGUAGE_PATTERNS = {
    'python': ['.py', '.pyw'],
    'javascript': ['.js', '.mjs', '.cjs'],
    'typescript': ['.ts', '.tsx'],
    'java': ['.java'],
    'go': ['.go'],
    'rust': ['.rs'],
    'c': ['.c', '.h'],
    'cpp': ['.cpp', '.hpp', '.cc', '.hh', '.cxx'],
    'csharp': ['.cs'],
    'ruby': ['.rb'],
    'php': ['.php'],
    'swift': ['.swift'],
    'kotlin': ['.kt', '.kts'],
    'scala': ['.scala'],
    'shell': ['.sh', '.bash', '.zsh'],
    'sql': ['.sql'],
    'r': ['.r', '.R'],
    'perl': ['.pl', '.pm'],
    'lua': ['.lua'],
    'yaml': ['.yml', '.yaml'],
}


@dataclass
class CommentStyle:
    """Defines comment style for a language"""
    single_line: Optional[str] = None  # e.g., "//" or "#"
    block_start: Optional[str] = None  # e.g., "/*"
    block_end: Optional[str] = None    # e.g., "*/"
    use_block: bool = False  # Prefer block comments over single-line


# Comment styles by language
COMMENT_STYLES = {
    'python': CommentStyle(single_line='#'),
    'javascript': CommentStyle(single_line='//', block_start='/*', block_end='*/', use_block=True),
    'typescript': CommentStyle(single_line='//', block_start='/*', block_end='*/', use_block=True),
    'java': CommentStyle(single_line='//', block_start='/*', block_end='*/', use_block=True),
    'go': CommentStyle(single_line='//', block_start='/*', block_end='*/', use_block=True),
    'rust': CommentStyle(single_line='//', block_start='/*', block_end='*/', use_block=True),
    'c': CommentStyle(single_line='//', block_start='/*', block_end='*/', use_block=True),
    'cpp': CommentStyle(single_line='//', block_start='/*', block_end='*/', use_block=True),
    'csharp': CommentStyle(single_line='//', block_start='/*', block_end='*/', use_block=True),
    'ruby': CommentStyle(single_line='#', block_start='=begin', block_end='=end'),
    'php': CommentStyle(single_line='//', block_start='/*', block_end='*/'),
    'swift': CommentStyle(single_line='//', block_start='/*', block_end='*/'),
    'kotlin': CommentStyle(single_line='//', block_start='/*', block_end='*/'),
    'scala': CommentStyle(single_line='//', block_start='/*', block_end='*/'),
    'shell': CommentStyle(single_line='#'),
    'sql': CommentStyle(single_line='--', block_start='/*', block_end='*/'),
    'r': CommentStyle(single_line='#'),
    'perl': CommentStyle(single_line='#'),
    'lua': CommentStyle(single_line='--', block_start='--[[', block_end=']]'),
    'yaml': CommentStyle(single_line='#'),
}


class HistoryInjector:
    """
    Injects file history as header comments into source files
    """

    # Marker to identify injected headers
    HEADER_START_MARKER = "FILE CHANGE HISTORY"
    HEADER_END_MARKER = "END FILE CHANGE HISTORY"

    def __init__(self, tracker: FileHistoryTracker):
        """
        Initialize history injector

        Args:
            tracker: FileHistoryTracker instance with file histories
        """
        self.tracker = tracker

    def detect_language(self, file_path: str) -> Optional[str]:
        """
        Detect programming language from file extension

        Args:
            file_path: Path to file

        Returns:
            Language name or None if unknown
        """
        path = Path(file_path)
        extension = path.suffix.lower()

        for language, extensions in LANGUAGE_PATTERNS.items():
            if extension in extensions:
                return language

        return None

    def get_comment_style(self, language: str) -> Optional[CommentStyle]:
        """
        Get comment style for a language

        Args:
            language: Language name

        Returns:
            CommentStyle or None if unknown
        """
        return COMMENT_STYLES.get(language)

    def format_header(
        self,
        history: FileHistory,
        language: str,
        max_entries: int = 10,
        include_diffs: bool = False
    ) -> str:
        """
        Format history as header comment

        Args:
            history: FileHistory object
            language: Programming language
            max_entries: Maximum number of recent changes to include
            include_diffs: Include diff snippets

        Returns:
            Formatted header string
        """
        comment_style = self.get_comment_style(language)
        if not comment_style:
            return ""

        # Generate header content (without comment markers)
        lines = []
        lines.append("=" * 78)
        lines.append(f"{self.HEADER_START_MARKER}")
        lines.append("=" * 78)
        lines.append(f"File: {history.file_path}")
        lines.append(f"Total changes: {history.total_commits} commits")

        # Contributors
        if history.authors:
            author_parts = []
            for author in history.authors[:3]:
                name = author.get('name', 'Unknown')
                count = author.get('commits', 0)
                author_parts.append(f"{name} ({count})")
            lines.append(f"Authors: {', '.join(author_parts)}")

        lines.append("")
        lines.append(f"[Recent Changes - Last {min(max_entries, len(history.changes))} of {history.total_commits}]")
        lines.append("")

        # Recent changes
        for change in history.changes[:max_entries]:
            date = change.commit_date[:10]
            commit = change.commit_hash[:7]
            author = change.author_name

            lines.append(f"{date} | {commit} | {author}")

            # Commit message (first line)
            msg = change.commit_message.split('\n')[0]
            if len(msg) > 70:
                msg = msg[:67] + "..."
            lines.append(f"  {msg}")

            # Stats
            lines.append(f"  Changes: +{change.lines_added} -{change.lines_removed} lines")

            # AI Summary (if available)
            if change.ai_summary:
                lines.append("")
                lines.append(f"  Summary: {change.ai_summary}")

            lines.append("")

        lines.append("=" * 78)
        lines.append(f"{self.HEADER_END_MARKER}")
        lines.append("=" * 78)

        # Apply comment formatting
        if comment_style.use_block and comment_style.block_start and comment_style.block_end:
            # Use block comments
            formatted = [comment_style.block_start]
            formatted.extend(lines)
            formatted.append(comment_style.block_end)
            header = '\n'.join(formatted)
        elif comment_style.single_line:
            # Use single-line comments
            formatted = [f"{comment_style.single_line} {line}" for line in lines]
            header = '\n'.join(formatted)
        else:
            # Fallback
            header = '\n'.join(lines)

        return header

    def has_injected_header(self, content: str) -> bool:
        """
        Check if file already has an injected header

        Args:
            content: File content

        Returns:
            True if header exists
        """
        return self.HEADER_START_MARKER in content

    def remove_existing_header(self, content: str, language: str) -> str:
        """
        Remove existing injected header from file content

        Args:
            content: File content
            language: Programming language

        Returns:
            Content with header removed
        """
        comment_style = self.get_comment_style(language)
        if not comment_style:
            return content

        lines = content.split('\n')
        filtered_lines = []
        in_header = False
        skip_next_separator = False
        skip_next_blank = False

        for line in lines:
            # Check if exiting header (must check before entering, since END contains START)
            if self.HEADER_END_MARKER in line:
                in_header = False
                skip_next_separator = True
                continue

            # Check if entering header
            if self.HEADER_START_MARKER in line:
                in_header = True
                # Also skip the preceding separator line if it exists
                if filtered_lines and '=' * 20 in filtered_lines[-1]:
                    filtered_lines.pop()
                continue

            # Skip lines inside header
            if in_header:
                continue

            # Skip separator line after END marker
            if skip_next_separator and '=' * 20 in line:
                skip_next_separator = False
                skip_next_blank = True
                continue

            # Skip one blank line after header
            if skip_next_blank and line.strip() == '':
                skip_next_blank = False
                continue

            filtered_lines.append(line)

        return '\n'.join(filtered_lines)

    def inject_history(
        self,
        file_path: str,
        max_entries: int = 10,
        include_diffs: bool = False,
        dry_run: bool = False
    ) -> Tuple[bool, str]:
        """
        Inject history header into a file

        Args:
            file_path: Path to file to inject
            max_entries: Maximum number of recent changes
            include_diffs: Include diff snippets
            dry_run: If True, don't write to file (just return result)

        Returns:
            Tuple of (success, message or modified content if dry_run)
        """
        # Detect language
        language = self.detect_language(file_path)
        if not language:
            return False, f"Unsupported file type: {file_path}"

        comment_style = self.get_comment_style(language)
        if not comment_style:
            return False, f"No comment style for language: {language}"

        # Load file history
        history = self.tracker.get_file_history(file_path)
        if not history or history.total_commits == 0:
            return False, f"No history found for: {file_path}"

        # Read current file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
        except FileNotFoundError:
            return False, f"File not found: {file_path}"
        except Exception as e:
            return False, f"Error reading file: {e}"

        # Remove existing header if present
        clean_content = self.remove_existing_header(original_content, language)

        # Format new header
        header = self.format_header(history, language, max_entries, include_diffs)

        # Handle shebang for scripts
        if clean_content.startswith('#!'):
            # Keep shebang at top
            lines = clean_content.split('\n', 1)
            shebang = lines[0]
            rest = lines[1] if len(lines) > 1 else ""
            new_content = f"{shebang}\n\n{header}\n\n{rest}"
        else:
            # Insert at beginning
            new_content = f"{header}\n\n{clean_content}"

        # Dry run - return content
        if dry_run:
            return True, new_content

        # Write to file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True, f"Injected header into: {file_path}"
        except Exception as e:
            return False, f"Error writing file: {e}"

    def remove_history(self, file_path: str, dry_run: bool = False) -> Tuple[bool, str]:
        """
        Remove injected history header from a file

        Args:
            file_path: Path to file
            dry_run: If True, don't write to file (just return result)

        Returns:
            Tuple of (success, message or modified content if dry_run)
        """
        # Detect language
        language = self.detect_language(file_path)
        if not language:
            return False, f"Unsupported file type: {file_path}"

        # Read current file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            return False, f"File not found: {file_path}"
        except Exception as e:
            return False, f"Error reading file: {e}"

        # Check if header exists
        if not self.has_injected_header(content):
            return False, f"No injected header found in: {file_path}"

        # Remove header
        clean_content = self.remove_existing_header(content, language)

        # Dry run - return content
        if dry_run:
            return True, clean_content

        # Write to file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(clean_content)
            return True, f"Removed header from: {file_path}"
        except Exception as e:
            return False, f"Error writing file: {e}"

    def inject_multiple(
        self,
        file_paths: List[str],
        max_entries: int = 10,
        dry_run: bool = False
    ) -> Dict[str, Tuple[bool, str]]:
        """
        Inject headers into multiple files

        Args:
            file_paths: List of file paths
            max_entries: Maximum number of recent changes
            dry_run: If True, don't write to files

        Returns:
            Dictionary mapping file paths to (success, message) tuples
        """
        results = {}
        for file_path in file_paths:
            results[file_path] = self.inject_history(file_path, max_entries, dry_run=dry_run)
        return results

    def remove_multiple(
        self,
        file_paths: List[str],
        dry_run: bool = False
    ) -> Dict[str, Tuple[bool, str]]:
        """
        Remove headers from multiple files

        Args:
            file_paths: List of file paths
            dry_run: If True, don't write to files

        Returns:
            Dictionary mapping file paths to (success, message) tuples
        """
        results = {}
        for file_path in file_paths:
            results[file_path] = self.remove_history(file_path, dry_run=dry_run)
        return results
