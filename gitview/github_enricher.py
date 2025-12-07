"""Enrich git commit data with GitHub collaboration context."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .github_graphql import (
    GitHubGraphQLClient,
    GitHubGraphQLError,
    GitHubRateLimitError,
    CommitContext,
    PullRequest,
    Review,
    parse_github_url,
)
from .extractor import CommitRecord


@dataclass
class GitHubContext:
    """GitHub context data for a commit."""

    # PR Information
    pr_number: Optional[int] = None
    pr_title: Optional[str] = None
    pr_body: Optional[str] = None
    pr_labels: List[str] = field(default_factory=list)
    pr_state: Optional[str] = None
    pr_merged: bool = False

    # Author/Reviewer Information
    pr_author: Optional[str] = None
    pr_reviewers: List[str] = field(default_factory=list)
    pr_review_states: Dict[str, str] = field(default_factory=dict)  # reviewer -> state

    # Review Comments (meaningful feedback)
    review_comments: List[str] = field(default_factory=list)  # Extracted review feedback
    review_summary: Optional[str] = None  # AI-generated or extracted summary

    # Branch Information
    source_branch: Optional[str] = None
    target_branch: Optional[str] = None

    # Merge Information
    is_merge_commit: bool = False
    merge_commit_sha: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'pr_number': self.pr_number,
            'pr_title': self.pr_title,
            'pr_body': self.pr_body,
            'pr_labels': self.pr_labels,
            'pr_state': self.pr_state,
            'pr_merged': self.pr_merged,
            'pr_author': self.pr_author,
            'pr_reviewers': self.pr_reviewers,
            'pr_review_states': self.pr_review_states,
            'review_comments': self.review_comments,
            'review_summary': self.review_summary,
            'source_branch': self.source_branch,
            'target_branch': self.target_branch,
            'is_merge_commit': self.is_merge_commit,
            'merge_commit_sha': self.merge_commit_sha,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GitHubContext':
        """Create from dictionary."""
        return cls(
            pr_number=data.get('pr_number'),
            pr_title=data.get('pr_title'),
            pr_body=data.get('pr_body'),
            pr_labels=data.get('pr_labels', []),
            pr_state=data.get('pr_state'),
            pr_merged=data.get('pr_merged', False),
            pr_author=data.get('pr_author'),
            pr_reviewers=data.get('pr_reviewers', []),
            pr_review_states=data.get('pr_review_states', {}),
            review_comments=data.get('review_comments', []),
            review_summary=data.get('review_summary'),
            source_branch=data.get('source_branch'),
            target_branch=data.get('target_branch'),
            is_merge_commit=data.get('is_merge_commit', False),
            merge_commit_sha=data.get('merge_commit_sha'),
        )

    def get_context_summary(self) -> str:
        """Get a human-readable summary of the GitHub context."""
        parts = []

        if self.pr_number and self.pr_title:
            parts.append(f"PR #{self.pr_number}: {self.pr_title}")

        if self.pr_labels:
            parts.append(f"Labels: {', '.join(self.pr_labels)}")

        if self.pr_reviewers:
            parts.append(f"Reviewed by: {', '.join(self.pr_reviewers)}")

        if self.pr_review_states:
            states = [f"{r}: {s}" for r, s in self.pr_review_states.items()]
            parts.append(f"Review states: {', '.join(states)}")

        if self.review_comments:
            # Include first 2-3 meaningful comments
            for i, comment in enumerate(self.review_comments[:3]):
                parts.append(f"Review feedback: {comment[:200]}...")

        return "\n".join(parts) if parts else ""


class GitHubEnricher:
    """Enriches commit records with GitHub collaboration data."""

    def __init__(
        self,
        token: str,
        owner: str,
        repo: str,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the enricher.

        Args:
            token: GitHub Personal Access Token
            owner: Repository owner
            repo: Repository name
            cache_dir: Optional cache directory
        """
        self.token = token
        self.owner = owner
        self.repo = repo
        self.client = GitHubGraphQLClient(token, cache_dir=cache_dir)

        # Cache for PR data (commit SHA -> GitHubContext)
        self._commit_context_cache: Dict[str, GitHubContext] = {}
        self._pr_cache: Dict[int, PullRequest] = {}

    @classmethod
    def from_repo_url(
        cls,
        token: str,
        repo_url: str,
        cache_dir: Optional[str] = None,
    ) -> 'GitHubEnricher':
        """
        Create enricher from repository URL.

        Args:
            token: GitHub Personal Access Token
            repo_url: GitHub repository URL or owner/repo
            cache_dir: Optional cache directory

        Returns:
            GitHubEnricher instance
        """
        owner, repo = parse_github_url(repo_url)
        return cls(token, owner, repo, cache_dir)

    def get_rate_limit(self) -> Dict[str, Any]:
        """Get current rate limit status."""
        rate_limit = self.client.get_rate_limit_status()
        return {
            'limit': rate_limit.limit,
            'remaining': rate_limit.remaining,
            'reset_at': rate_limit.reset_at.isoformat(),
        }

    def enrich_commits(
        self,
        commits: List[CommitRecord],
        branch: str = "HEAD",
        progress_callback: Optional[callable] = None,
    ) -> List[Tuple[CommitRecord, GitHubContext]]:
        """
        Enrich a list of commits with GitHub context.

        Args:
            commits: List of CommitRecord objects
            branch: Branch name for context
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of tuples (CommitRecord, GitHubContext)
        """
        if not commits:
            return []

        # Build a map of commit SHA to CommitRecord
        commit_map = {c.commit_hash: c for c in commits}

        # Get date range from commits
        since = commits[0].timestamp if commits else None
        until = commits[-1].timestamp if commits else None

        # Fetch GitHub commit data with associated PRs
        try:
            github_commits = self.client.get_all_commits_with_prs(
                owner=self.owner,
                repo=self.repo,
                branch=branch,
                since=since,
                until=until,
                max_commits=len(commits) * 2,  # Some buffer for merge commits
                progress_callback=lambda n: progress_callback(n, len(commits)) if progress_callback else None,
            )
        except GitHubGraphQLError as e:
            print(f"Warning: GitHub API error: {e}")
            # Return commits without enrichment
            return [(c, GitHubContext()) for c in commits]

        # Build GitHub context map
        github_context_map: Dict[str, CommitContext] = {
            gc.sha: gc for gc in github_commits
        }

        # Enrich each commit
        results = []
        for commit in commits:
            github_commit = github_context_map.get(commit.commit_hash)
            context = self._build_context(commit, github_commit)
            results.append((commit, context))

        return results

    def enrich_single_commit(
        self,
        commit: CommitRecord,
    ) -> GitHubContext:
        """
        Enrich a single commit with GitHub context.

        Args:
            commit: CommitRecord to enrich

        Returns:
            GitHubContext for the commit
        """
        # Check cache first
        if commit.commit_hash in self._commit_context_cache:
            return self._commit_context_cache[commit.commit_hash]

        # Fetch from GitHub
        try:
            github_commits, _, _ = self.client.get_commits_with_prs(
                owner=self.owner,
                repo=self.repo,
                branch="HEAD",
                first=1,
            )

            # Find matching commit
            github_commit = None
            for gc in github_commits:
                if gc.sha == commit.commit_hash:
                    github_commit = gc
                    break

            context = self._build_context(commit, github_commit)
            self._commit_context_cache[commit.commit_hash] = context
            return context

        except GitHubGraphQLError as e:
            print(f"Warning: GitHub API error: {e}")
            return GitHubContext()

    def get_merged_prs_for_range(
        self,
        since: str,
        until: str,
    ) -> List[PullRequest]:
        """
        Get all merged PRs in a date range.

        Args:
            since: Start date (ISO format)
            until: End date (ISO format)

        Returns:
            List of merged PullRequest objects
        """
        all_prs = []
        cursor = None
        has_next = True

        while has_next:
            prs, cursor, has_next = self.client.get_merged_prs(
                owner=self.owner,
                repo=self.repo,
                since=since,
                until=until,
                after=cursor,
            )
            all_prs.extend(prs)

        return all_prs

    def _build_context(
        self,
        commit: CommitRecord,
        github_commit: Optional[CommitContext],
    ) -> GitHubContext:
        """Build GitHubContext from commit and GitHub data."""
        context = GitHubContext()

        # Check if this is a merge commit
        if len(commit.parent_hashes) > 1:
            context.is_merge_commit = True

        if not github_commit or not github_commit.associated_prs:
            return context

        # Find the most relevant PR (usually there's only one)
        # Prefer the one where this commit is the merge commit
        pr = None
        for associated_pr in github_commit.associated_prs:
            if associated_pr.merge_commit_sha == commit.commit_hash:
                pr = associated_pr
                break

        # Fall back to first associated PR
        if not pr and github_commit.associated_prs:
            pr = github_commit.associated_prs[0]

        if pr:
            context.pr_number = pr.number
            context.pr_title = pr.title
            context.pr_body = pr.body
            context.pr_labels = pr.labels
            context.pr_state = pr.state
            context.pr_merged = pr.merged
            context.pr_author = pr.author.login if pr.author else None
            context.source_branch = pr.head_ref_name
            context.target_branch = pr.base_ref_name
            context.merge_commit_sha = pr.merge_commit_sha

            # Extract reviewers
            context.pr_reviewers = [r.login for r in pr.reviewers if r.login]

            # Add reviewers from actual reviews
            for review in pr.reviews:
                if review.author and review.author.login:
                    if review.author.login not in context.pr_reviewers:
                        context.pr_reviewers.append(review.author.login)
                    context.pr_review_states[review.author.login] = review.state

            # Extract meaningful review comments
            context.review_comments = self._extract_review_comments(pr.reviews)

            # Generate review summary
            context.review_summary = self._generate_review_summary(pr)

        return context

    def _extract_review_comments(self, reviews: List[Review]) -> List[str]:
        """Extract meaningful comments from reviews."""
        comments = []

        for review in reviews:
            # Include review body if substantial
            if review.body and len(review.body.strip()) > 20:
                comments.append(f"[{review.state}] {review.body.strip()}")

            # Include inline review comments
            for comment in review.comments:
                if comment.body and len(comment.body.strip()) > 20:
                    path_prefix = f"({comment.path}) " if comment.path else ""
                    comments.append(f"{path_prefix}{comment.body.strip()}")

        # Limit and deduplicate
        seen = set()
        unique_comments = []
        for c in comments:
            # Normalize for dedup
            key = c.lower()[:100]
            if key not in seen:
                seen.add(key)
                unique_comments.append(c)

        return unique_comments[:10]  # Limit to 10 most relevant

    def _generate_review_summary(self, pr: PullRequest) -> Optional[str]:
        """Generate a summary of the review activity."""
        if not pr.reviews:
            return None

        parts = []

        # Count review states
        state_counts: Dict[str, int] = {}
        for review in pr.reviews:
            state_counts[review.state] = state_counts.get(review.state, 0) + 1

        if state_counts:
            state_summary = ", ".join(
                f"{count} {state.lower()}" for state, count in state_counts.items()
            )
            parts.append(f"Reviews: {state_summary}")

        # Reviewer summary
        reviewers = set()
        for review in pr.reviews:
            if review.author:
                reviewers.add(review.author.login)

        if reviewers:
            parts.append(f"Reviewers: {', '.join(sorted(reviewers))}")

        return "; ".join(parts) if parts else None


def enrich_commits_with_github(
    commits: List[CommitRecord],
    github_token: str,
    repo_url: str,
    branch: str = "HEAD",
    progress_callback: Optional[callable] = None,
) -> Dict[str, GitHubContext]:
    """
    Convenience function to enrich commits with GitHub data.

    Args:
        commits: List of CommitRecord objects
        github_token: GitHub Personal Access Token
        repo_url: Repository URL or owner/repo
        branch: Branch name
        progress_callback: Optional progress callback

    Returns:
        Dict mapping commit SHA to GitHubContext
    """
    enricher = GitHubEnricher.from_repo_url(github_token, repo_url)
    results = enricher.enrich_commits(commits, branch, progress_callback)

    return {commit.commit_hash: context for commit, context in results}


def extract_pr_narratives(github_contexts: Dict[str, GitHubContext]) -> Dict[str, str]:
    """
    Extract narrative-friendly descriptions from PR data.

    Args:
        github_contexts: Dict mapping commit SHA to GitHubContext

    Returns:
        Dict mapping commit SHA to narrative description
    """
    narratives = {}

    for sha, context in github_contexts.items():
        if not context.pr_number:
            continue

        parts = []

        # PR title and description
        if context.pr_title:
            parts.append(f"This change was introduced via PR #{context.pr_number}: \"{context.pr_title}\"")

        if context.pr_body:
            # Extract first paragraph or summary
            body = context.pr_body.strip()
            if body:
                # Look for summary section or use first paragraph
                lines = body.split('\n')
                summary_lines = []
                for line in lines:
                    if line.strip().startswith('#'):
                        break
                    if len(summary_lines) < 5:
                        summary_lines.append(line)
                if summary_lines:
                    parts.append("PR description: " + ' '.join(summary_lines).strip()[:500])

        # Labels context
        if context.pr_labels:
            label_str = ', '.join(context.pr_labels)
            parts.append(f"Categorized as: {label_str}")

        # Review context
        if context.pr_reviewers:
            reviewer_str = ', '.join(context.pr_reviewers)
            parts.append(f"Reviewed by: {reviewer_str}")

        if context.pr_review_states:
            approved_by = [r for r, s in context.pr_review_states.items() if s == 'APPROVED']
            if approved_by:
                parts.append(f"Approved by: {', '.join(approved_by)}")

        # Key review feedback
        if context.review_comments:
            parts.append("Key review feedback:")
            for comment in context.review_comments[:3]:
                parts.append(f"  - {comment[:200]}")

        if parts:
            narratives[sha] = "\n".join(parts)

    return narratives
