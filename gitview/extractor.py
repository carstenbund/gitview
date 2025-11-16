"""Extract git history with detailed metadata."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict

import git
from git import Repo


@dataclass
class CommitRecord:
    """Represents a single commit with extracted metadata."""

    commit_hash: str
    short_hash: str
    timestamp: str
    author: str
    author_email: str
    commit_message: str
    commit_subject: str
    commit_body: str
    parent_hashes: List[str]

    # Code metrics
    loc_added: int
    loc_deleted: int
    loc_total: int
    files_changed: int

    # Language breakdown
    language_breakdown: Dict[str, int]

    # README state
    readme_exists: bool
    readme_size: int
    readme_excerpt: Optional[str]

    # Comment analysis
    comment_samples: List[str]
    comment_density: float

    # Diff stats
    insertions: int
    deletions: int
    files_stats: Dict[str, Dict[str, int]]

    # Large changes detection
    is_large_deletion: bool
    is_large_addition: bool
    is_refactor: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class GitHistoryExtractor:
    """Extract detailed git history from a repository."""

    def __init__(self, repo_path: str = "."):
        """Initialize with repository path."""
        self.repo_path = Path(repo_path)
        self.repo = Repo(repo_path)

        # Language extensions mapping
        self.language_extensions = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.jsx': 'JSX',
            '.tsx': 'TSX',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.h': 'C/C++ Header',
            '.go': 'Go',
            '.rs': 'Rust',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.scala': 'Scala',
            '.sh': 'Shell',
            '.bash': 'Bash',
            '.md': 'Markdown',
            '.html': 'HTML',
            '.css': 'CSS',
            '.sql': 'SQL',
            '.yaml': 'YAML',
            '.yml': 'YAML',
            '.json': 'JSON',
            '.xml': 'XML',
        }

        # Comment patterns for different languages
        self.comment_patterns = {
            'Python': [r'#.*', r'"""[\s\S]*?"""', r"'''[\s\S]*?'''"],
            'JavaScript': [r'//.*', r'/\*[\s\S]*?\*/'],
            'TypeScript': [r'//.*', r'/\*[\s\S]*?\*/'],
            'Java': [r'//.*', r'/\*[\s\S]*?\*/'],
            'C++': [r'//.*', r'/\*[\s\S]*?\*/'],
            'C': [r'/\*[\s\S]*?\*/'],
            'Go': [r'//.*', r'/\*[\s\S]*?\*/'],
            'Rust': [r'//.*', r'/\*[\s\S]*?\*/'],
            'Ruby': [r'#.*'],
            'Shell': [r'#.*'],
        }

    def extract_history(self, max_commits: Optional[int] = None,
                       branch: str = "HEAD") -> List[CommitRecord]:
        """Extract full git history with metadata.

        Args:
            max_commits: Maximum number of commits to extract (None for all)
            branch: Branch to extract from (default: HEAD)

        Returns:
            List of CommitRecord objects, sorted chronologically (oldest first)
        """
        commits = []
        commit_iterator = self.repo.iter_commits(branch, max_count=max_commits)

        for commit in commit_iterator:
            try:
                record = self._extract_commit_record(commit)
                commits.append(record)
            except Exception as e:
                print(f"Warning: Failed to extract commit {commit.hexsha[:8]}: {e}")
                continue

        # Sort chronologically (oldest first)
        commits.reverse()

        # Calculate cumulative LOC
        total_loc = 0
        for record in commits:
            total_loc += (record.loc_added - record.loc_deleted)
            record.loc_total = max(0, total_loc)

        return commits

    def _extract_commit_record(self, commit: git.Commit) -> CommitRecord:
        """Extract detailed metadata from a single commit."""

        # Basic commit info
        commit_hash = commit.hexsha
        short_hash = commit.hexsha[:8]
        timestamp = datetime.fromtimestamp(commit.committed_date).isoformat()
        author = commit.author.name
        author_email = commit.author.email

        # Parse commit message
        message_lines = commit.message.strip().split('\n')
        subject = message_lines[0] if message_lines else ""
        body = '\n'.join(message_lines[1:]).strip() if len(message_lines) > 1 else ""

        # Parent commits
        parent_hashes = [p.hexsha for p in commit.parents]

        # Get diff stats
        stats = self._get_diff_stats(commit)

        # Language breakdown
        language_breakdown = self._get_language_breakdown(commit)

        # README analysis
        readme_info = self._get_readme_info(commit)

        # Comment analysis
        comment_info = self._get_comment_info(commit)

        # Detect large changes
        is_large_deletion = stats['deletions'] > 1000
        is_large_addition = stats['insertions'] > 1000
        is_refactor = self._detect_refactor(commit, stats)

        return CommitRecord(
            commit_hash=commit_hash,
            short_hash=short_hash,
            timestamp=timestamp,
            author=author,
            author_email=author_email,
            commit_message=commit.message.strip(),
            commit_subject=subject,
            commit_body=body,
            parent_hashes=parent_hashes,
            loc_added=stats['insertions'],
            loc_deleted=stats['deletions'],
            loc_total=0,  # Will be calculated later
            files_changed=stats['files_changed'],
            language_breakdown=language_breakdown,
            readme_exists=readme_info['exists'],
            readme_size=readme_info['size'],
            readme_excerpt=readme_info['excerpt'],
            comment_samples=comment_info['samples'],
            comment_density=comment_info['density'],
            insertions=stats['insertions'],
            deletions=stats['deletions'],
            files_stats=stats['files'],
            is_large_deletion=is_large_deletion,
            is_large_addition=is_large_addition,
            is_refactor=is_refactor,
        )

    def _get_diff_stats(self, commit: git.Commit) -> Dict[str, Any]:
        """Get diff statistics for a commit."""
        stats = {
            'insertions': 0,
            'deletions': 0,
            'files_changed': 0,
            'files': {}
        }

        if not commit.parents:
            # Initial commit
            try:
                diff_index = commit.diff(git.NULL_TREE)
            except:
                return stats
        else:
            diff_index = commit.parents[0].diff(commit)

        for diff in diff_index:
            try:
                # Skip binary files
                if diff.a_blob and diff.a_blob.mime_type.startswith('image/'):
                    continue
                if diff.b_blob and diff.b_blob.mime_type.startswith('image/'):
                    continue

                file_path = diff.b_path or diff.a_path
                if not file_path:
                    continue

                # Get line changes
                if diff.diff:
                    diff_text = diff.diff.decode('utf-8', errors='ignore')
                    insertions = len([l for l in diff_text.split('\n') if l.startswith('+')])
                    deletions = len([l for l in diff_text.split('\n') if l.startswith('-')])

                    stats['insertions'] += insertions
                    stats['deletions'] += deletions
                    stats['files'][file_path] = {
                        'insertions': insertions,
                        'deletions': deletions
                    }

                stats['files_changed'] += 1

            except Exception as e:
                continue

        return stats

    def _get_language_breakdown(self, commit: git.Commit) -> Dict[str, int]:
        """Get language breakdown for files in commit."""
        breakdown = {}

        try:
            for item in commit.tree.traverse():
                if item.type == 'blob':
                    ext = Path(item.path).suffix.lower()
                    language = self.language_extensions.get(ext, 'Other')
                    breakdown[language] = breakdown.get(language, 0) + 1
        except:
            pass

        return breakdown

    def _get_readme_info(self, commit: git.Commit) -> Dict[str, Any]:
        """Extract README information at this commit."""
        info = {
            'exists': False,
            'size': 0,
            'excerpt': None
        }

        try:
            readme_names = ['README.md', 'README', 'README.txt', 'README.rst', 'readme.md']

            for item in commit.tree.traverse():
                if item.type == 'blob' and item.name in readme_names:
                    info['exists'] = True
                    info['size'] = item.size

                    # Get excerpt (first 200 chars)
                    try:
                        content = item.data_stream.read().decode('utf-8', errors='ignore')
                        info['excerpt'] = content[:200].strip()
                    except:
                        pass

                    break
        except:
            pass

        return info

    def _get_comment_info(self, commit: git.Commit) -> Dict[str, Any]:
        """Extract comment samples and density from code files."""
        info = {
            'samples': [],
            'density': 0.0
        }

        total_lines = 0
        comment_lines = 0

        try:
            if not commit.parents:
                diff_index = commit.diff(git.NULL_TREE)
            else:
                diff_index = commit.parents[0].diff(commit)

            for diff in diff_index:
                if not diff.b_blob:
                    continue

                file_path = diff.b_path
                if not file_path:
                    continue

                ext = Path(file_path).suffix.lower()
                language = self.language_extensions.get(ext)

                if language not in self.comment_patterns:
                    continue

                try:
                    content = diff.b_blob.data_stream.read().decode('utf-8', errors='ignore')
                    lines = content.split('\n')
                    total_lines += len(lines)

                    # Find comments
                    for pattern in self.comment_patterns[language]:
                        for match in re.finditer(pattern, content, re.MULTILINE):
                            comment_text = match.group(0).strip()
                            if len(comment_text) > 10:  # Skip very short comments
                                comment_lines += len(comment_text.split('\n'))
                                if len(info['samples']) < 5:
                                    info['samples'].append(comment_text[:100])

                except:
                    continue

        except:
            pass

        # Calculate density
        if total_lines > 0:
            info['density'] = comment_lines / total_lines

        return info

    def _detect_refactor(self, commit: git.Commit, stats: Dict[str, Any]) -> bool:
        """Detect if this is likely a refactoring commit."""
        # Heuristics for refactoring:
        # 1. Similar insertions and deletions
        # 2. Message contains refactor keywords
        # 3. Multiple files changed

        insertions = stats['insertions']
        deletions = stats['deletions']

        if insertions == 0 or deletions == 0:
            return False

        ratio = min(insertions, deletions) / max(insertions, deletions)

        refactor_keywords = ['refactor', 'rename', 'reorganize', 'restructure', 'cleanup', 'rewrite']
        message_lower = commit.message.lower()
        has_keyword = any(keyword in message_lower for keyword in refactor_keywords)

        return ratio > 0.7 and (has_keyword or stats['files_changed'] > 3)

    def save_to_jsonl(self, records: List[CommitRecord], output_path: str):
        """Save commit records to JSONL file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            for record in records:
                json.dump(record.to_dict(), f)
                f.write('\n')

    @staticmethod
    def load_from_jsonl(input_path: str) -> List[CommitRecord]:
        """Load commit records from JSONL file."""
        records = []

        with open(input_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                # Convert dict back to CommitRecord
                records.append(CommitRecord(**data))

        return records


def extract_git_history(repo_path: str = ".",
                       output_path: str = "output/repo_history.jsonl",
                       max_commits: Optional[int] = None,
                       branch: str = "HEAD") -> List[CommitRecord]:
    """
    Extract git history and save to JSONL.

    Args:
        repo_path: Path to git repository
        output_path: Path to output JSONL file
        max_commits: Maximum number of commits to extract (None for all)
        branch: Branch to extract from

    Returns:
        List of CommitRecord objects
    """
    extractor = GitHistoryExtractor(repo_path)
    records = extractor.extract_history(max_commits=max_commits, branch=branch)
    extractor.save_to_jsonl(records, output_path)
    return records
