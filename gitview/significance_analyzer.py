"""Analyze and preserve commit significance for large repositories."""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

from .extractor import CommitRecord


@dataclass
class CommitCluster:
    """A group of semantically related commits."""

    cluster_type: str  # 'feature', 'bugfix', 'refactor', 'docs', 'infrastructure'
    commits: List[CommitRecord]
    key_commit: CommitRecord  # Most significant commit in cluster
    summary: Optional[str] = None

    @property
    def total_insertions(self) -> int:
        return sum(c.insertions for c in self.commits)

    @property
    def total_deletions(self) -> int:
        return sum(c.deletions for c in self.commits)

    @property
    def file_changes(self) -> Dict[str, int]:
        """Aggregate file changes across all commits."""
        files = defaultdict(int)
        for commit in self.commits:
            for file_path in commit.files_stats.keys():
                files[file_path] += 1
        return dict(files)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            'cluster_type': self.cluster_type,
            'commit_count': len(self.commits),
            'total_insertions': self.total_insertions,
            'total_deletions': self.total_deletions,
            'key_commit': {
                'hash': self.key_commit.short_hash,
                'message': self.key_commit.commit_subject,
                'pr_title': self.key_commit.get_pr_title(),
                'pr_body': self.key_commit.get_pr_body(),
                'pr_labels': self.key_commit.get_pr_labels(),
            },
            'commits': [
                {
                    'hash': c.short_hash,
                    'message': c.commit_subject,
                    'insertions': c.insertions,
                    'deletions': c.deletions,
                    'files_changed': c.files_changed,
                }
                for c in self.commits[:10]  # First 10 for context
            ],
            'top_files': sorted(
                self.file_changes.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],  # Top 10 most-changed files
            'summary': self.summary,
        }


class SignificanceAnalyzer:
    """Analyze commit significance to preserve important details."""

    # Keywords for classifying commit types
    FEATURE_KEYWORDS = ['add', 'implement', 'create', 'new', 'feature', 'support']
    BUGFIX_KEYWORDS = ['fix', 'bug', 'issue', 'resolve', 'patch', 'correct']
    REFACTOR_KEYWORDS = ['refactor', 'restructure', 'reorganize', 'clean', 'simplify']
    DOCS_KEYWORDS = ['doc', 'readme', 'comment', 'documentation', 'guide']
    INFRA_KEYWORDS = ['ci', 'build', 'deploy', 'config', 'setup', 'dependency']

    def __init__(self):
        pass

    def cluster_commits(self, commits: List[CommitRecord]) -> List[CommitCluster]:
        """
        Group commits into semantic clusters.

        Args:
            commits: List of commits to cluster

        Returns:
            List of CommitCluster objects
        """
        # Strategy: Group consecutive commits by PR or semantic similarity
        clusters: List[CommitCluster] = []
        current_cluster: List[CommitRecord] = []
        current_type: Optional[str] = None

        for commit in commits:
            commit_type = self._classify_commit(commit)

            # Check if we should start a new cluster
            should_start_new = False

            if not current_cluster:
                # First commit
                should_start_new = False
            elif commit.has_github_context() and current_cluster[-1].has_github_context():
                # Different PRs = different clusters
                if commit.get_pr_number() != current_cluster[-1].get_pr_number():
                    should_start_new = True
            elif commit_type != current_type:
                # Different types = different clusters
                should_start_new = True
            elif len(current_cluster) >= 10:
                # Max cluster size to prevent oversized clusters
                should_start_new = True

            if should_start_new and current_cluster:
                # Finalize current cluster
                key_commit = self._find_key_commit(current_cluster)
                cluster = CommitCluster(
                    cluster_type=current_type or 'general',
                    commits=current_cluster,
                    key_commit=key_commit,
                )
                clusters.append(cluster)
                current_cluster = []
                current_type = None

            # Add to current cluster
            current_cluster.append(commit)
            if current_type is None:
                current_type = commit_type

        # Add final cluster
        if current_cluster:
            key_commit = self._find_key_commit(current_cluster)
            cluster = CommitCluster(
                cluster_type=current_type or 'general',
                commits=current_cluster,
                key_commit=key_commit,
            )
            clusters.append(cluster)

        return clusters

    def _classify_commit(self, commit: CommitRecord) -> str:
        """Classify commit by type based on message and PR labels."""
        # Check PR labels first (most reliable)
        if commit.has_github_context():
            labels = commit.get_pr_labels()
            for label in labels:
                label_lower = label.lower()
                if any(kw in label_lower for kw in ['feature', 'enhancement']):
                    return 'feature'
                elif any(kw in label_lower for kw in ['bug', 'fix']):
                    return 'bugfix'
                elif 'refactor' in label_lower:
                    return 'refactor'
                elif any(kw in label_lower for kw in ['doc', 'documentation']):
                    return 'docs'
                elif any(kw in label_lower for kw in ['ci', 'infrastructure', 'build']):
                    return 'infrastructure'

        # Fall back to message analysis
        message = (commit.commit_subject + ' ' + commit.commit_body).lower()

        # Check for refactor flags
        if commit.is_refactor:
            return 'refactor'

        # Check keywords
        if any(kw in message for kw in self.BUGFIX_KEYWORDS):
            return 'bugfix'
        elif any(kw in message for kw in self.FEATURE_KEYWORDS):
            return 'feature'
        elif any(kw in message for kw in self.REFACTOR_KEYWORDS):
            return 'refactor'
        elif any(kw in message for kw in self.DOCS_KEYWORDS):
            return 'docs'
        elif any(kw in message for kw in self.INFRA_KEYWORDS):
            return 'infrastructure'

        # Check file changes
        files = list(commit.files_stats.keys())
        if any(f.endswith('.md') or 'readme' in f.lower() for f in files):
            return 'docs'

        return 'general'

    def _find_key_commit(self, commits: List[CommitRecord]) -> CommitRecord:
        """Find the most significant commit in a cluster."""
        # Scoring: PR info > large changes > merge commits
        def score_commit(c: CommitRecord) -> int:
            score = 0

            # Has PR context
            if c.has_github_context() and c.get_pr_title():
                score += 100

            # Large changes
            if c.is_large_addition or c.is_large_deletion:
                score += 50

            # Refactor
            if c.is_refactor:
                score += 30

            # Merge commits often have good descriptions
            if c.github_context and c.github_context.get('is_merge_commit'):
                score += 20

            # More files changed = more significant
            score += min(c.files_changed, 20)

            return score

        return max(commits, key=score_commit)

    def extract_significant_details(self, cluster: CommitCluster) -> Dict[str, Any]:
        """Extract the most significant details from a cluster for LLM context."""
        key = cluster.key_commit

        details = {
            'type': cluster.cluster_type,
            'summary': key.commit_subject,
            'commit_count': len(cluster.commits),
            'insertions': cluster.total_insertions,
            'deletions': cluster.total_deletions,
        }

        # Add PR context if available
        if key.has_github_context():
            pr_title = key.get_pr_title()
            pr_body = key.get_pr_body()
            pr_labels = key.get_pr_labels()

            if pr_title:
                details['pr_title'] = pr_title
            if pr_body:
                # Keep first 500 chars of PR body
                details['pr_description'] = pr_body[:500]
            if pr_labels:
                details['labels'] = pr_labels

        # Add commit message if no PR
        if 'pr_title' not in details:
            if key.commit_body:
                details['description'] = key.commit_body[:500]

        # Add top changed files
        details['top_files'] = sorted(
            cluster.file_changes.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        # Add all commit messages if cluster is small
        if len(cluster.commits) <= 5:
            details['all_commits'] = [
                {
                    'hash': c.short_hash,
                    'message': c.commit_subject,
                }
                for c in cluster.commits
            ]

        return details


def analyze_commit_significance(commits: List[CommitRecord]) -> List[CommitCluster]:
    """
    Analyze commits and group by significance.

    Args:
        commits: List of commits to analyze

    Returns:
        List of CommitCluster objects
    """
    analyzer = SignificanceAnalyzer()
    return analyzer.cluster_commits(commits)
