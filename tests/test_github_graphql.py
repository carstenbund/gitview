"""Tests for GitHub GraphQL module."""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from gitview.github_graphql import (
    GitHubGraphQLClient,
    GitHubGraphQLError,
    GitHubRateLimitError,
    GitHubAuthor,
    PullRequest,
    Review,
    ReviewComment,
    CommitContext,
    RateLimitInfo,
    parse_github_url,
)


class TestParseGitHubUrl:
    """Tests for parse_github_url function."""

    def test_parse_owner_repo_format(self):
        owner, repo = parse_github_url("owner/repo")
        assert owner == "owner"
        assert repo == "repo"

    def test_parse_owner_repo_with_trailing_slash(self):
        owner, repo = parse_github_url("owner/repo/")
        assert owner == "owner"
        assert repo == "repo"

    def test_parse_https_url(self):
        owner, repo = parse_github_url("https://github.com/owner/repo")
        assert owner == "owner"
        assert repo == "repo"

    def test_parse_https_url_with_git_suffix(self):
        owner, repo = parse_github_url("https://github.com/owner/repo.git")
        assert owner == "owner"
        assert repo == "repo"

    def test_parse_ssh_url(self):
        owner, repo = parse_github_url("git@github.com:owner/repo.git")
        assert owner == "owner"
        assert repo == "repo"

    def test_parse_invalid_url_raises_error(self):
        with pytest.raises(ValueError):
            parse_github_url("invalid-url")


class TestGitHubAuthor:
    """Tests for GitHubAuthor dataclass."""

    def test_create_author(self):
        author = GitHubAuthor(
            login="testuser",
            name="Test User",
            email="test@example.com",
            avatar_url="https://avatars.githubusercontent.com/testuser"
        )
        assert author.login == "testuser"
        assert author.name == "Test User"
        assert author.email == "test@example.com"

    def test_create_author_minimal(self):
        author = GitHubAuthor(login="testuser")
        assert author.login == "testuser"
        assert author.name is None
        assert author.email is None


class TestPullRequest:
    """Tests for PullRequest dataclass."""

    def test_create_pull_request(self):
        author = GitHubAuthor(login="author")
        pr = PullRequest(
            number=123,
            title="Test PR",
            body="Test body",
            state="MERGED",
            merged=True,
            author=author,
            created_at="2025-01-01T00:00:00Z",
            merged_at="2025-01-02T00:00:00Z",
        )
        assert pr.number == 123
        assert pr.title == "Test PR"
        assert pr.merged is True

    def test_pull_request_to_dict(self):
        pr = PullRequest(
            number=1,
            title="Test",
            body=None,
            state="MERGED",
            merged=True,
            author=None,
            created_at="2025-01-01T00:00:00Z",
            labels=["bug", "fix"],
        )
        d = pr.to_dict()
        assert d['number'] == 1
        assert d['labels'] == ["bug", "fix"]


class TestReview:
    """Tests for Review dataclass."""

    def test_create_review(self):
        author = GitHubAuthor(login="reviewer")
        review = Review(
            author=author,
            state="APPROVED",
            body="LGTM",
            submitted_at="2025-01-01T00:00:00Z",
        )
        assert review.state == "APPROVED"
        assert review.body == "LGTM"

    def test_review_with_comments(self):
        comment = ReviewComment(
            author=GitHubAuthor(login="reviewer"),
            body="Consider refactoring this",
            path="src/main.py",
        )
        review = Review(
            author=GitHubAuthor(login="reviewer"),
            state="CHANGES_REQUESTED",
            comments=[comment],
        )
        assert len(review.comments) == 1
        assert review.comments[0].body == "Consider refactoring this"


class TestCommitContext:
    """Tests for CommitContext dataclass."""

    def test_create_commit_context(self):
        ctx = CommitContext(
            sha="abc123def456",
            message="Test commit message",
            author=GitHubAuthor(login="author"),
            committed_date="2025-01-01T00:00:00Z",
        )
        assert ctx.sha == "abc123def456"
        assert ctx.message == "Test commit message"

    def test_commit_context_with_prs(self):
        pr = PullRequest(
            number=1,
            title="Test PR",
            body=None,
            state="MERGED",
            merged=True,
            author=None,
            created_at="2025-01-01T00:00:00Z",
        )
        ctx = CommitContext(
            sha="abc123",
            message="Test",
            associated_prs=[pr],
        )
        assert len(ctx.associated_prs) == 1
        assert ctx.associated_prs[0].number == 1


class TestGitHubGraphQLClient:
    """Tests for GitHubGraphQLClient."""

    @pytest.fixture
    def client(self, tmp_path):
        """Create a client with a temp cache directory."""
        return GitHubGraphQLClient(
            token="test_token",
            cache_dir=str(tmp_path / "cache"),
        )

    def test_client_initialization(self, tmp_path):
        client = GitHubGraphQLClient(
            token="test_token",
            cache_dir=str(tmp_path / "cache"),
            cache_ttl_hours=48,
        )
        assert client.token == "test_token"
        assert client.cache_dir.exists()

    def test_cache_key_generation(self, client):
        key1 = client._get_cache_key("query { test }", {"var": "value"})
        key2 = client._get_cache_key("query { test }", {"var": "value"})
        key3 = client._get_cache_key("query { test }", {"var": "different"})

        assert key1 == key2  # Same query and vars should produce same key
        assert key1 != key3  # Different vars should produce different key

    def test_cache_set_and_get(self, client):
        cache_key = "test_key"
        test_data = {"result": "data"}

        client._set_cached(cache_key, test_data)
        retrieved = client._get_cached(cache_key)

        assert retrieved == test_data

    def test_cache_expiry(self, client):
        cache_key = "test_key"
        test_data = {"result": "data"}

        # Set cache with expired TTL
        client.cache_ttl = timedelta(seconds=-1)
        client._set_cached(cache_key, test_data)

        # Should return None for expired cache
        retrieved = client._get_cached(cache_key)
        assert retrieved is None

    def test_execute_query_success(self, client):
        """Test successful query execution."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {"repository": {"name": "test"}}
        }
        mock_response.headers = {
            'x-ratelimit-limit': '5000',
            'x-ratelimit-remaining': '4999',
            'x-ratelimit-reset': str(int(datetime.now().timestamp()) + 3600),
        }

        with patch.object(client.session, 'post', return_value=mock_response):
            result = client.execute("query { test }", use_cache=False)
            assert result == {"repository": {"name": "test"}}

    def test_execute_query_with_errors(self, client):
        """Test query execution with GraphQL errors."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "errors": [{"message": "Test error"}]
        }
        mock_response.headers = {}

        with patch.object(client.session, 'post', return_value=mock_response):
            with pytest.raises(GitHubGraphQLError) as exc_info:
                client.execute("query { test }", use_cache=False)
            assert "Test error" in str(exc_info.value)

    def test_execute_query_rate_limited(self, client):
        """Test query execution when rate limited."""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.text = "rate limit exceeded"
        mock_response.headers = {}

        with patch.object(client.session, 'post', return_value=mock_response):
            with pytest.raises(GitHubRateLimitError):
                client.execute("query { test }", use_cache=False)

    def test_clear_cache(self, client):
        """Test cache clearing."""
        # Add some cache entries
        client._set_cached("key1", {"data": 1})
        client._set_cached("key2", {"data": 2})

        count = client.clear_cache()
        assert count == 2

        # Verify cache is empty
        assert client._get_cached("key1") is None
        assert client._get_cached("key2") is None

    def test_get_cache_stats(self, client):
        """Test cache statistics."""
        client._set_cached("key1", {"data": "test data"})

        stats = client.get_cache_stats()
        assert stats['file_count'] == 1
        assert stats['total_size_bytes'] > 0

    def test_parse_commit_node(self, client):
        """Test parsing a commit node from GraphQL response."""
        node = {
            'oid': 'abc123def456789',
            'message': 'Test commit\n\nDetailed description',
            'committedDate': '2025-01-01T12:00:00Z',
            'author': {
                'user': {
                    'login': 'testuser',
                    'name': 'Test User',
                    'email': 'test@example.com',
                },
                'name': 'Test User',
                'email': 'test@example.com',
            },
            'committer': {
                'user': {
                    'login': 'testuser',
                },
            },
            'associatedPullRequests': {
                'nodes': [
                    {
                        'number': 123,
                        'title': 'Test PR',
                        'body': 'PR description',
                        'state': 'MERGED',
                        'merged': True,
                        'createdAt': '2025-01-01T00:00:00Z',
                        'mergedAt': '2025-01-02T00:00:00Z',
                        'author': {'login': 'prauthor'},
                        'labels': {'nodes': [{'name': 'enhancement'}]},
                        'reviews': {'nodes': []},
                        'reviewRequests': {'nodes': []},
                        'additions': 100,
                        'deletions': 50,
                        'changedFiles': 5,
                        'comments': {'totalCount': 3},
                        'reviewThreads': {'totalCount': 2},
                    }
                ]
            }
        }

        commit = client._parse_commit_node(node)

        assert commit.sha == 'abc123def456789'
        assert commit.message == 'Test commit\n\nDetailed description'
        assert commit.author.login == 'testuser'
        assert len(commit.associated_prs) == 1
        assert commit.associated_prs[0].number == 123
        assert commit.associated_prs[0].title == 'Test PR'

    def test_parse_pr_node(self, client):
        """Test parsing a PR node from GraphQL response."""
        node = {
            'number': 456,
            'title': 'Feature: Add new functionality',
            'body': 'This PR adds...',
            'state': 'MERGED',
            'merged': True,
            'createdAt': '2025-01-01T00:00:00Z',
            'mergedAt': '2025-01-02T00:00:00Z',
            'closedAt': '2025-01-02T00:00:00Z',
            'mergeCommit': {'oid': 'mergesha123'},
            'headRefName': 'feature/new-feature',
            'baseRefName': 'main',
            'author': {'login': 'developer', 'name': 'Developer'},
            'labels': {'nodes': [{'name': 'feature'}, {'name': 'reviewed'}]},
            'reviews': {
                'nodes': [
                    {
                        'author': {'login': 'reviewer1', 'name': 'Reviewer'},
                        'state': 'APPROVED',
                        'body': 'LGTM',
                        'submittedAt': '2025-01-01T12:00:00Z',
                        'comments': {
                            'nodes': [
                                {
                                    'author': {'login': 'reviewer1'},
                                    'body': 'Nice work!',
                                    'path': 'src/main.py',
                                    'createdAt': '2025-01-01T11:00:00Z',
                                }
                            ]
                        }
                    }
                ]
            },
            'reviewRequests': {'nodes': []},
            'additions': 200,
            'deletions': 50,
            'changedFiles': 10,
            'comments': {'totalCount': 5},
            'reviewThreads': {'totalCount': 3},
        }

        pr = client._parse_pr_node(node)

        assert pr.number == 456
        assert pr.title == 'Feature: Add new functionality'
        assert pr.merged is True
        assert pr.merge_commit_sha == 'mergesha123'
        assert pr.head_ref_name == 'feature/new-feature'
        assert pr.labels == ['feature', 'reviewed']
        assert len(pr.reviews) == 1
        assert pr.reviews[0].state == 'APPROVED'
        assert len(pr.reviews[0].comments) == 1


class TestRateLimitInfo:
    """Tests for RateLimitInfo dataclass."""

    def test_create_rate_limit_info(self):
        reset_at = datetime.now() + timedelta(hours=1)
        info = RateLimitInfo(
            limit=5000,
            remaining=4500,
            reset_at=reset_at,
            cost=5,
        )
        assert info.limit == 5000
        assert info.remaining == 4500
        assert info.cost == 5


# Integration-style tests (would need real token to run)
class TestGitHubGraphQLIntegration:
    """Integration tests that require a real GitHub token.

    These tests are skipped by default. To run them:
    GITHUB_TOKEN=your_token pytest -m integration
    """

    @pytest.fixture
    def real_client(self):
        """Create a client with real token if available."""
        import os
        token = os.environ.get('GITHUB_TOKEN')
        if not token:
            pytest.skip("GITHUB_TOKEN not set")
        return GitHubGraphQLClient(token)

    @pytest.mark.integration
    def test_get_repository_info(self, real_client):
        """Test fetching real repository info."""
        info = real_client.get_repository_info("octocat", "Hello-World")
        assert info['name'] == 'Hello-World'

    @pytest.mark.integration
    def test_get_rate_limit_status(self, real_client):
        """Test fetching rate limit status."""
        status = real_client.get_rate_limit_status()
        assert status.limit > 0
        assert status.remaining >= 0
