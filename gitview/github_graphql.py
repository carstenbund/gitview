"""GitHub GraphQL API client for fetching repository collaboration data."""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


@dataclass
class GitHubAuthor:
    """Represents a GitHub user."""
    login: str
    name: Optional[str] = None
    email: Optional[str] = None
    avatar_url: Optional[str] = None


@dataclass
class ReviewComment:
    """Represents a review comment on a PR."""
    author: Optional[GitHubAuthor]
    body: str
    path: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class Review:
    """Represents a PR review."""
    author: Optional[GitHubAuthor]
    state: str  # APPROVED, CHANGES_REQUESTED, COMMENTED, DISMISSED, PENDING
    body: Optional[str] = None
    submitted_at: Optional[str] = None
    comments: List[ReviewComment] = field(default_factory=list)


@dataclass
class PullRequest:
    """Represents a GitHub Pull Request."""
    number: int
    title: str
    body: Optional[str]
    state: str  # OPEN, CLOSED, MERGED
    merged: bool
    author: Optional[GitHubAuthor]
    created_at: str
    merged_at: Optional[str] = None
    closed_at: Optional[str] = None
    merge_commit_sha: Optional[str] = None
    head_ref_name: Optional[str] = None
    base_ref_name: Optional[str] = None
    labels: List[str] = field(default_factory=list)
    reviewers: List[GitHubAuthor] = field(default_factory=list)
    reviews: List[Review] = field(default_factory=list)
    comments_count: int = 0
    review_comments_count: int = 0
    additions: int = 0
    deletions: int = 0
    changed_files: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        return result


@dataclass
class CommitContext:
    """GitHub context for a commit."""
    sha: str
    message: str
    author: Optional[GitHubAuthor] = None
    committer: Optional[GitHubAuthor] = None
    committed_date: Optional[str] = None
    associated_prs: List[PullRequest] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class RateLimitInfo:
    """GitHub API rate limit information."""
    limit: int
    remaining: int
    reset_at: datetime
    cost: int = 0


class GitHubGraphQLError(Exception):
    """Exception for GitHub GraphQL API errors."""
    def __init__(self, message: str, errors: Optional[List[Dict]] = None):
        super().__init__(message)
        self.errors = errors or []


class GitHubRateLimitError(GitHubGraphQLError):
    """Exception for rate limit errors."""
    def __init__(self, reset_at: datetime, remaining: int = 0):
        super().__init__(f"Rate limit exceeded. Resets at {reset_at}")
        self.reset_at = reset_at
        self.remaining = remaining


class GitHubGraphQLClient:
    """Client for GitHub's GraphQL API."""

    GRAPHQL_ENDPOINT = "https://api.github.com/graphql"

    def __init__(
        self,
        token: str,
        cache_dir: Optional[str] = None,
        cache_ttl_hours: int = 24,
    ):
        """
        Initialize the GitHub GraphQL client.

        Args:
            token: GitHub Personal Access Token (classic with 'repo' scope)
            cache_dir: Directory for caching responses (default: ~/.gitview/cache/github)
            cache_ttl_hours: Cache time-to-live in hours (default: 24)
        """
        self.token = token
        self.cache_ttl = timedelta(hours=cache_ttl_hours)

        # Set up cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".gitview" / "cache" / "github"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Rate limit tracking
        self.rate_limit: Optional[RateLimitInfo] = None

        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

    def _get_cache_key(self, query: str, variables: Dict[str, Any]) -> str:
        """Generate a cache key for a query."""
        content = json.dumps({"query": query, "variables": variables}, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_cached(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if valid."""
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r') as f:
                cached = json.load(f)

            cached_at = datetime.fromisoformat(cached['cached_at'])
            if datetime.now() - cached_at > self.cache_ttl:
                # Cache expired
                cache_file.unlink()
                return None

            return cached['data']
        except (json.JSONDecodeError, KeyError, ValueError):
            # Invalid cache file
            cache_file.unlink(missing_ok=True)
            return None

    def _set_cached(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Cache a response."""
        cache_file = self.cache_dir / f"{cache_key}.json"

        with open(cache_file, 'w') as f:
            json.dump({
                'cached_at': datetime.now().isoformat(),
                'data': data,
            }, f)

    def _update_rate_limit(self, response_headers: Dict[str, str], cost: int = 1) -> None:
        """Update rate limit info from response headers."""
        try:
            self.rate_limit = RateLimitInfo(
                limit=int(response_headers.get('x-ratelimit-limit', 5000)),
                remaining=int(response_headers.get('x-ratelimit-remaining', 5000)),
                reset_at=datetime.fromtimestamp(
                    int(response_headers.get('x-ratelimit-reset', time.time() + 3600))
                ),
                cost=cost,
            )
        except (ValueError, TypeError):
            pass

    def execute(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute a GraphQL query.

        Args:
            query: GraphQL query string
            variables: Query variables
            use_cache: Whether to use caching (default: True)

        Returns:
            Query response data

        Raises:
            GitHubGraphQLError: On API errors
            GitHubRateLimitError: On rate limit errors
        """
        variables = variables or {}

        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(query, variables)
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached

        # Check rate limit
        if self.rate_limit and self.rate_limit.remaining <= 10:
            wait_time = (self.rate_limit.reset_at - datetime.now()).total_seconds()
            if wait_time > 0:
                raise GitHubRateLimitError(
                    self.rate_limit.reset_at,
                    self.rate_limit.remaining
                )

        # Execute query
        payload = {"query": query, "variables": variables}

        response = self.session.post(self.GRAPHQL_ENDPOINT, json=payload)

        # Update rate limit info
        self._update_rate_limit(response.headers)

        if response.status_code == 403:
            # Rate limit or permission error
            if 'rate limit' in response.text.lower():
                raise GitHubRateLimitError(
                    self.rate_limit.reset_at if self.rate_limit else datetime.now() + timedelta(hours=1),
                    0
                )
            raise GitHubGraphQLError(f"Permission denied: {response.text}")

        if response.status_code != 200:
            raise GitHubGraphQLError(
                f"HTTP {response.status_code}: {response.text}"
            )

        result = response.json()

        if 'errors' in result:
            errors = result['errors']
            error_messages = [e.get('message', str(e)) for e in errors]
            raise GitHubGraphQLError(
                f"GraphQL errors: {'; '.join(error_messages)}",
                errors=errors
            )

        data = result.get('data', {})

        # Cache successful response
        if use_cache:
            self._set_cached(cache_key, data)

        return data

    def get_rate_limit_status(self) -> RateLimitInfo:
        """Get current rate limit status."""
        query = """
        query {
            rateLimit {
                limit
                remaining
                resetAt
                cost
            }
        }
        """

        data = self.execute(query, use_cache=False)
        rate_limit = data.get('rateLimit', {})

        return RateLimitInfo(
            limit=rate_limit.get('limit', 5000),
            remaining=rate_limit.get('remaining', 5000),
            reset_at=datetime.fromisoformat(
                rate_limit.get('resetAt', datetime.now().isoformat()).replace('Z', '+00:00')
            ),
            cost=rate_limit.get('cost', 1),
        )

    def get_repository_info(self, owner: str, repo: str) -> Dict[str, Any]:
        """Get basic repository information."""
        query = """
        query($owner: String!, $repo: String!) {
            repository(owner: $owner, name: $repo) {
                name
                description
                url
                createdAt
                updatedAt
                defaultBranchRef {
                    name
                }
                primaryLanguage {
                    name
                }
                languages(first: 10) {
                    nodes {
                        name
                    }
                }
                stargazerCount
                forkCount
                isPrivate
                isArchived
            }
        }
        """

        data = self.execute(query, {"owner": owner, "repo": repo})
        return data.get('repository', {})

    def get_commits_with_prs(
        self,
        owner: str,
        repo: str,
        branch: str = "HEAD",
        since: Optional[str] = None,
        until: Optional[str] = None,
        first: int = 100,
        after: Optional[str] = None,
    ) -> Tuple[List[CommitContext], Optional[str], bool]:
        """
        Get commits with associated PR information.

        Args:
            owner: Repository owner
            repo: Repository name
            branch: Branch name (default: HEAD)
            since: ISO date string to start from
            until: ISO date string to end at
            first: Number of commits per page (max 100)
            after: Cursor for pagination

        Returns:
            Tuple of (commits, next_cursor, has_next_page)
        """
        query = """
        query($owner: String!, $repo: String!, $branch: String!, $first: Int!, $after: String, $since: GitTimestamp, $until: GitTimestamp) {
            repository(owner: $owner, name: $repo) {
                ref(qualifiedName: $branch) {
                    target {
                        ... on Commit {
                            history(first: $first, after: $after, since: $since, until: $until) {
                                pageInfo {
                                    hasNextPage
                                    endCursor
                                }
                                nodes {
                                    oid
                                    message
                                    committedDate
                                    author {
                                        user {
                                            login
                                            name
                                            email
                                            avatarUrl
                                        }
                                        name
                                        email
                                    }
                                    committer {
                                        user {
                                            login
                                            name
                                            email
                                        }
                                        name
                                        email
                                    }
                                    associatedPullRequests(first: 5) {
                                        nodes {
                                            number
                                            title
                                            body
                                            state
                                            merged
                                            createdAt
                                            mergedAt
                                            closedAt
                                            mergeCommit {
                                                oid
                                            }
                                            headRefName
                                            baseRefName
                                            author {
                                                login
                                                ... on User {
                                                    name
                                                    email
                                                }
                                            }
                                            labels(first: 10) {
                                                nodes {
                                                    name
                                                }
                                            }
                                            reviews(first: 10) {
                                                nodes {
                                                    author {
                                                        login
                                                        ... on User {
                                                            name
                                                        }
                                                    }
                                                    state
                                                    body
                                                    submittedAt
                                                }
                                            }
                                            reviewRequests(first: 10) {
                                                nodes {
                                                    requestedReviewer {
                                                        ... on User {
                                                            login
                                                            name
                                                        }
                                                    }
                                                }
                                            }
                                            additions
                                            deletions
                                            changedFiles
                                            comments {
                                                totalCount
                                            }
                                            reviewThreads {
                                                totalCount
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        """

        # Handle branch name - if HEAD, use default branch
        if branch == "HEAD":
            repo_info = self.get_repository_info(owner, repo)
            default_branch = repo_info.get('defaultBranchRef', {})
            branch = default_branch.get('name', 'main') if default_branch else 'main'

        # Qualify branch name for ref lookup
        qualified_branch = f"refs/heads/{branch}" if not branch.startswith("refs/") else branch

        variables = {
            "owner": owner,
            "repo": repo,
            "branch": qualified_branch,
            "first": min(first, 100),
            "after": after,
        }

        if since:
            variables["since"] = since
        if until:
            variables["until"] = until

        data = self.execute(query, variables)

        repository = data.get('repository', {})
        ref = repository.get('ref', {})
        target = ref.get('target', {}) if ref else {}
        history = target.get('history', {})

        page_info = history.get('pageInfo', {})
        nodes = history.get('nodes', [])

        commits = []
        for node in nodes:
            commit = self._parse_commit_node(node)
            commits.append(commit)

        return (
            commits,
            page_info.get('endCursor'),
            page_info.get('hasNextPage', False)
        )

    def get_all_commits_with_prs(
        self,
        owner: str,
        repo: str,
        branch: str = "HEAD",
        since: Optional[str] = None,
        until: Optional[str] = None,
        max_commits: Optional[int] = None,
        progress_callback: Optional[callable] = None,
    ) -> List[CommitContext]:
        """
        Get all commits with associated PR information (handles pagination).

        Args:
            owner: Repository owner
            repo: Repository name
            branch: Branch name (default: HEAD)
            since: ISO date string to start from
            until: ISO date string to end at
            max_commits: Maximum number of commits to fetch
            progress_callback: Optional callback for progress updates

        Returns:
            List of CommitContext objects
        """
        all_commits = []
        cursor = None
        has_next = True

        while has_next:
            remaining = None
            if max_commits:
                remaining = max_commits - len(all_commits)
                if remaining <= 0:
                    break

            page_size = min(100, remaining) if remaining else 100

            commits, cursor, has_next = self.get_commits_with_prs(
                owner=owner,
                repo=repo,
                branch=branch,
                since=since,
                until=until,
                first=page_size,
                after=cursor,
            )

            all_commits.extend(commits)

            if progress_callback:
                progress_callback(len(all_commits))

        return all_commits

    def get_pull_request(
        self,
        owner: str,
        repo: str,
        number: int,
    ) -> Optional[PullRequest]:
        """
        Get detailed information about a specific PR.

        Args:
            owner: Repository owner
            repo: Repository name
            number: PR number

        Returns:
            PullRequest object or None if not found
        """
        query = """
        query($owner: String!, $repo: String!, $number: Int!) {
            repository(owner: $owner, name: $repo) {
                pullRequest(number: $number) {
                    number
                    title
                    body
                    state
                    merged
                    createdAt
                    mergedAt
                    closedAt
                    mergeCommit {
                        oid
                    }
                    headRefName
                    baseRefName
                    author {
                        login
                        ... on User {
                            name
                            email
                        }
                    }
                    labels(first: 20) {
                        nodes {
                            name
                        }
                    }
                    reviews(first: 50) {
                        nodes {
                            author {
                                login
                                ... on User {
                                    name
                                }
                            }
                            state
                            body
                            submittedAt
                            comments(first: 20) {
                                nodes {
                                    author {
                                        login
                                    }
                                    body
                                    path
                                    createdAt
                                }
                            }
                        }
                    }
                    reviewRequests(first: 20) {
                        nodes {
                            requestedReviewer {
                                ... on User {
                                    login
                                    name
                                }
                            }
                        }
                    }
                    additions
                    deletions
                    changedFiles
                    comments {
                        totalCount
                    }
                    reviewThreads {
                        totalCount
                    }
                }
            }
        }
        """

        data = self.execute(
            query,
            {"owner": owner, "repo": repo, "number": number}
        )

        pr_data = data.get('repository', {}).get('pullRequest')
        if not pr_data:
            return None

        return self._parse_pr_node(pr_data)

    def get_merged_prs(
        self,
        owner: str,
        repo: str,
        since: Optional[str] = None,
        until: Optional[str] = None,
        first: int = 100,
        after: Optional[str] = None,
    ) -> Tuple[List[PullRequest], Optional[str], bool]:
        """
        Get merged pull requests in a time range.

        Args:
            owner: Repository owner
            repo: Repository name
            since: ISO date string to start from
            until: ISO date string to end at
            first: Number of PRs per page
            after: Cursor for pagination

        Returns:
            Tuple of (prs, next_cursor, has_next_page)
        """
        # Build search query
        search_query = f"repo:{owner}/{repo} is:pr is:merged"
        if since:
            search_query += f" merged:>={since[:10]}"
        if until:
            search_query += f" merged:<={until[:10]}"

        query = """
        query($searchQuery: String!, $first: Int!, $after: String) {
            search(query: $searchQuery, type: ISSUE, first: $first, after: $after) {
                pageInfo {
                    hasNextPage
                    endCursor
                }
                nodes {
                    ... on PullRequest {
                        number
                        title
                        body
                        state
                        merged
                        createdAt
                        mergedAt
                        closedAt
                        mergeCommit {
                            oid
                        }
                        headRefName
                        baseRefName
                        author {
                            login
                            ... on User {
                                name
                                email
                            }
                        }
                        labels(first: 10) {
                            nodes {
                                name
                            }
                        }
                        reviews(first: 10) {
                            nodes {
                                author {
                                    login
                                }
                                state
                                body
                                submittedAt
                            }
                        }
                        additions
                        deletions
                        changedFiles
                        comments {
                            totalCount
                        }
                    }
                }
            }
        }
        """

        data = self.execute(
            query,
            {
                "searchQuery": search_query,
                "first": min(first, 100),
                "after": after,
            }
        )

        search = data.get('search', {})
        page_info = search.get('pageInfo', {})
        nodes = search.get('nodes', [])

        prs = [self._parse_pr_node(node) for node in nodes if node]

        return (
            prs,
            page_info.get('endCursor'),
            page_info.get('hasNextPage', False)
        )

    def _parse_commit_node(self, node: Dict[str, Any]) -> CommitContext:
        """Parse a commit node from GraphQL response."""
        # Parse author
        author_data = node.get('author', {})
        author = None
        if author_data:
            user = author_data.get('user', {})
            if user:
                author = GitHubAuthor(
                    login=user.get('login', ''),
                    name=user.get('name'),
                    email=user.get('email'),
                    avatar_url=user.get('avatarUrl'),
                )
            elif author_data.get('name'):
                author = GitHubAuthor(
                    login=author_data.get('email', '').split('@')[0],
                    name=author_data.get('name'),
                    email=author_data.get('email'),
                )

        # Parse committer
        committer_data = node.get('committer', {})
        committer = None
        if committer_data:
            user = committer_data.get('user', {})
            if user:
                committer = GitHubAuthor(
                    login=user.get('login', ''),
                    name=user.get('name'),
                    email=user.get('email'),
                )

        # Parse associated PRs
        associated_prs = []
        prs_data = node.get('associatedPullRequests', {}).get('nodes', [])
        for pr_data in prs_data:
            if pr_data:
                pr = self._parse_pr_node(pr_data)
                associated_prs.append(pr)

        return CommitContext(
            sha=node.get('oid', ''),
            message=node.get('message', ''),
            author=author,
            committer=committer,
            committed_date=node.get('committedDate'),
            associated_prs=associated_prs,
        )

    def _parse_pr_node(self, node: Dict[str, Any]) -> PullRequest:
        """Parse a PR node from GraphQL response."""
        # Parse author
        author_data = node.get('author', {})
        author = None
        if author_data:
            author = GitHubAuthor(
                login=author_data.get('login', ''),
                name=author_data.get('name'),
                email=author_data.get('email'),
            )

        # Parse labels
        labels = [
            label.get('name', '')
            for label in node.get('labels', {}).get('nodes', [])
            if label
        ]

        # Parse reviewers from review requests
        reviewers = []
        for req in node.get('reviewRequests', {}).get('nodes', []):
            if req:
                reviewer_data = req.get('requestedReviewer', {})
                if reviewer_data:
                    reviewers.append(GitHubAuthor(
                        login=reviewer_data.get('login', ''),
                        name=reviewer_data.get('name'),
                    ))

        # Parse reviews
        reviews = []
        for review_data in node.get('reviews', {}).get('nodes', []):
            if review_data:
                review_author = None
                if review_data.get('author'):
                    review_author = GitHubAuthor(
                        login=review_data['author'].get('login', ''),
                        name=review_data['author'].get('name'),
                    )

                # Parse review comments
                review_comments = []
                for comment_data in review_data.get('comments', {}).get('nodes', []):
                    if comment_data:
                        comment_author = None
                        if comment_data.get('author'):
                            comment_author = GitHubAuthor(
                                login=comment_data['author'].get('login', '')
                            )
                        review_comments.append(ReviewComment(
                            author=comment_author,
                            body=comment_data.get('body', ''),
                            path=comment_data.get('path'),
                            created_at=comment_data.get('createdAt'),
                        ))

                reviews.append(Review(
                    author=review_author,
                    state=review_data.get('state', ''),
                    body=review_data.get('body'),
                    submitted_at=review_data.get('submittedAt'),
                    comments=review_comments,
                ))

        # Get merge commit SHA
        merge_commit = node.get('mergeCommit', {})
        merge_commit_sha = merge_commit.get('oid') if merge_commit else None

        return PullRequest(
            number=node.get('number', 0),
            title=node.get('title', ''),
            body=node.get('body'),
            state=node.get('state', ''),
            merged=node.get('merged', False),
            author=author,
            created_at=node.get('createdAt', ''),
            merged_at=node.get('mergedAt'),
            closed_at=node.get('closedAt'),
            merge_commit_sha=merge_commit_sha,
            head_ref_name=node.get('headRefName'),
            base_ref_name=node.get('baseRefName'),
            labels=labels,
            reviewers=reviewers,
            reviews=reviews,
            comments_count=node.get('comments', {}).get('totalCount', 0),
            review_comments_count=node.get('reviewThreads', {}).get('totalCount', 0),
            additions=node.get('additions', 0),
            deletions=node.get('deletions', 0),
            changed_files=node.get('changedFiles', 0),
        )

    def clear_cache(self) -> int:
        """
        Clear all cached responses.

        Returns:
            Number of cache files removed
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1
        return count

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            'cache_dir': str(self.cache_dir),
            'file_count': len(cache_files),
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
        }


def parse_github_url(url: str) -> Tuple[str, str]:
    """
    Parse a GitHub URL to extract owner and repo.

    Args:
        url: GitHub URL or owner/repo string

    Returns:
        Tuple of (owner, repo)

    Raises:
        ValueError: If URL format is invalid
    """
    # Handle owner/repo format
    if '/' in url and not url.startswith('http') and not url.startswith('git@'):
        parts = url.strip('/').split('/')
        if len(parts) >= 2:
            return parts[-2], parts[-1].replace('.git', '')

    # Handle HTTPS URL
    if url.startswith('https://github.com/'):
        path = url.replace('https://github.com/', '')
        parts = path.strip('/').split('/')
        if len(parts) >= 2:
            return parts[0], parts[1].replace('.git', '')

    # Handle SSH URL
    if url.startswith('git@github.com:'):
        path = url.replace('git@github.com:', '')
        parts = path.strip('/').split('/')
        if len(parts) >= 2:
            return parts[0], parts[1].replace('.git', '')

    raise ValueError(f"Invalid GitHub URL format: {url}")
