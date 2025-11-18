"""Branch discovery and management for multi-branch analysis."""

import fnmatch
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime

import git
from git import Repo


@dataclass
class BranchInfo:
    """Information about a git branch."""
    name: str                    # Full branch name (e.g., "origin/main")
    short_name: str              # Short name (e.g., "main")
    is_remote: bool              # True if remote tracking branch
    remote_name: Optional[str]   # Remote name if remote branch (e.g., "origin")
    commit_hash: str             # Latest commit hash
    commit_count: int            # Total commits in branch
    last_commit_date: str        # ISO format date of last commit
    sanitized_name: str          # Filesystem-safe name for directory

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class BranchManager:
    """Manage branch discovery and filtering for git repositories."""

    def __init__(self, repo_path: str):
        """Initialize with repository path.

        Args:
            repo_path: Path to git repository
        """
        self.repo_path = Path(repo_path)
        self.repo = Repo(repo_path)

    def list_all_branches(self, include_remote: bool = True) -> List[BranchInfo]:
        """List all branches in repository.

        Args:
            include_remote: Include remote tracking branches

        Returns:
            List of BranchInfo objects
        """
        branches = []

        # Local branches
        for ref in self.repo.heads:
            branch_info = self._create_branch_info(ref, is_remote=False)
            if branch_info:
                branches.append(branch_info)

        # Remote tracking branches
        if include_remote:
            for ref in self.repo.remotes:
                for remote_ref in ref.refs:
                    # Skip HEAD references
                    if remote_ref.name.endswith('/HEAD'):
                        continue

                    branch_info = self._create_branch_info(remote_ref, is_remote=True)
                    if branch_info:
                        branches.append(branch_info)

        return branches

    def _create_branch_info(self, ref, is_remote: bool) -> Optional[BranchInfo]:
        """Create BranchInfo from a git reference.

        Args:
            ref: Git reference object
            is_remote: Whether this is a remote tracking branch

        Returns:
            BranchInfo or None if creation fails
        """
        try:
            commit = ref.commit
            name = ref.name

            # Extract remote name and short name for remote branches
            if is_remote:
                # name is like "origin/main"
                parts = name.split('/', 1)
                if len(parts) == 2:
                    remote_name = parts[0]
                    short_name = parts[1]
                else:
                    remote_name = None
                    short_name = name
            else:
                remote_name = None
                short_name = name

            # Count commits in branch
            commit_count = sum(1 for _ in self.repo.iter_commits(ref))

            # Get last commit date
            last_commit_date = datetime.fromtimestamp(commit.committed_date).isoformat()

            # Create sanitized name for filesystem
            sanitized_name = self._sanitize_branch_name(name)

            return BranchInfo(
                name=name,
                short_name=short_name,
                is_remote=is_remote,
                remote_name=remote_name,
                commit_hash=commit.hexsha,
                commit_count=commit_count,
                last_commit_date=last_commit_date,
                sanitized_name=sanitized_name
            )

        except Exception as e:
            print(f"Warning: Failed to get info for branch {ref.name}: {e}")
            return None

    def _sanitize_branch_name(self, branch_name: str) -> str:
        """Convert branch name to filesystem-safe directory name.

        Args:
            branch_name: Original branch name (e.g., "origin/feature/new-ui")

        Returns:
            Sanitized name (e.g., "origin-feature-new-ui")
        """
        # Replace problematic characters
        sanitized = branch_name.replace('/', '-')
        sanitized = sanitized.replace('\\', '-')
        sanitized = sanitized.replace(':', '-')
        sanitized = sanitized.replace('*', '-star-')
        sanitized = sanitized.replace('?', '-q-')
        sanitized = sanitized.replace('"', '-quote-')
        sanitized = sanitized.replace('<', '-lt-')
        sanitized = sanitized.replace('>', '-gt-')
        sanitized = sanitized.replace('|', '-pipe-')

        return sanitized

    def filter_branches(
        self,
        branches: List[BranchInfo],
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> List[BranchInfo]:
        """Filter branches by include/exclude patterns.

        Args:
            branches: List of BranchInfo objects
            include_patterns: Glob patterns to include (e.g., ["main", "feature/*"])
            exclude_patterns: Glob patterns to exclude (e.g., ["hotfix/*"])

        Returns:
            Filtered list of BranchInfo objects
        """
        filtered = branches

        # Apply include patterns
        if include_patterns:
            included = []
            for branch in filtered:
                for pattern in include_patterns:
                    # Try matching against full name and short name
                    if (fnmatch.fnmatch(branch.name, pattern) or
                        fnmatch.fnmatch(branch.short_name, pattern)):
                        included.append(branch)
                        break
            filtered = included

        # Apply exclude patterns
        if exclude_patterns:
            excluded = []
            for branch in filtered:
                should_exclude = False
                for pattern in exclude_patterns:
                    if (fnmatch.fnmatch(branch.name, pattern) or
                        fnmatch.fnmatch(branch.short_name, pattern)):
                        should_exclude = True
                        break
                if not should_exclude:
                    excluded.append(branch)
            filtered = excluded

        return filtered

    def get_branch_by_name(self, branch_name: str) -> Optional[BranchInfo]:
        """Get branch info for a specific branch name.

        Args:
            branch_name: Branch name to look up

        Returns:
            BranchInfo or None if not found
        """
        all_branches = self.list_all_branches(include_remote=True)

        for branch in all_branches:
            if branch.name == branch_name or branch.short_name == branch_name:
                return branch

        return None

    def estimate_total_commits(self, branches: List[BranchInfo]) -> int:
        """Estimate total unique commits across branches.

        This is an upper bound - actual unique commits may be less due to merges.

        Args:
            branches: List of branches to analyze

        Returns:
            Estimated total commits
        """
        # For now, return sum of all commits
        # TODO: Could be improved by finding unique commits across branches
        return sum(b.commit_count for b in branches)

    def get_branch_statistics(self, branches: List[BranchInfo]) -> Dict:
        """Get statistics about a set of branches.

        Args:
            branches: List of branches

        Returns:
            Dictionary with statistics
        """
        local_branches = [b for b in branches if not b.is_remote]
        remote_branches = [b for b in branches if b.is_remote]

        total_commits = sum(b.commit_count for b in branches)
        avg_commits = total_commits / len(branches) if branches else 0

        # Group by remote
        remotes: Dict[str, int] = {}
        for branch in remote_branches:
            if branch.remote_name:
                remotes[branch.remote_name] = remotes.get(branch.remote_name, 0) + 1

        return {
            'total_branches': len(branches),
            'local_branches': len(local_branches),
            'remote_branches': len(remote_branches),
            'total_commits': total_commits,
            'avg_commits_per_branch': avg_commits,
            'remotes': remotes,
        }


def parse_branch_spec(branch_spec: str) -> List[str]:
    """Parse branch specification string into list of patterns.

    Args:
        branch_spec: Comma-separated branch names or patterns

    Returns:
        List of branch patterns

    Examples:
        "main,develop" -> ["main", "develop"]
        "feature/*" -> ["feature/*"]
        "main,feature/*,hotfix/*" -> ["main", "feature/*", "hotfix/*"]
    """
    return [p.strip() for p in branch_spec.split(',') if p.strip()]
