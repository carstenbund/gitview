"""Tests for GitHub enricher module."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from gitview.github_enricher import (
    GitHubContext,
    GitHubEnricher,
    enrich_commits_with_github,
    extract_pr_narratives,
)
from gitview.github_graphql import (
    GitHubAuthor,
    PullRequest,
    Review,
    ReviewComment,
    CommitContext,
)
from gitview.extractor import CommitRecord


class TestGitHubContext:
    """Tests for GitHubContext dataclass."""

    def test_create_empty_context(self):
        ctx = GitHubContext()
        assert ctx.pr_number is None
        assert ctx.pr_title is None
        assert ctx.review_comments == []

    def test_create_full_context(self):
        ctx = GitHubContext(
            pr_number=123,
            pr_title="Test PR",
            pr_body="Description",
            pr_labels=["bug", "fix"],
            pr_state="MERGED",
            pr_merged=True,
            pr_author="testuser",
            pr_reviewers=["reviewer1", "reviewer2"],
            pr_review_states={"reviewer1": "APPROVED"},
            review_comments=["Great work!"],
            source_branch="feature/test",
            target_branch="main",
            is_merge_commit=True,
        )
        assert ctx.pr_number == 123
        assert ctx.pr_merged is True
        assert "bug" in ctx.pr_labels

    def test_to_dict(self):
        ctx = GitHubContext(
            pr_number=456,
            pr_title="Feature",
            pr_labels=["enhancement"],
        )
        d = ctx.to_dict()
        assert d['pr_number'] == 456
        assert d['pr_title'] == "Feature"
        assert d['pr_labels'] == ["enhancement"]

    def test_from_dict(self):
        data = {
            'pr_number': 789,
            'pr_title': 'Bug fix',
            'pr_labels': ['bug'],
            'pr_merged': True,
        }
        ctx = GitHubContext.from_dict(data)
        assert ctx.pr_number == 789
        assert ctx.pr_title == 'Bug fix'
        assert ctx.pr_merged is True

    def test_get_context_summary_with_pr(self):
        ctx = GitHubContext(
            pr_number=100,
            pr_title="Add feature X",
            pr_labels=["feature", "priority"],
            pr_reviewers=["alice", "bob"],
            pr_review_states={"alice": "APPROVED", "bob": "COMMENTED"},
            review_comments=["Consider adding tests", "Nice implementation"],
        )
        summary = ctx.get_context_summary()
        assert "PR #100" in summary
        assert "Add feature X" in summary
        assert "feature" in summary
        assert "alice" in summary

    def test_get_context_summary_empty(self):
        ctx = GitHubContext()
        summary = ctx.get_context_summary()
        assert summary == ""


class TestGitHubEnricher:
    """Tests for GitHubEnricher class."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock GraphQL client."""
        with patch('gitview.github_enricher.GitHubGraphQLClient') as mock:
            yield mock

    @pytest.fixture
    def sample_commit_record(self):
        """Create a sample CommitRecord."""
        return CommitRecord(
            commit_hash="abc123def456789",
            short_hash="abc123d",
            timestamp="2025-01-01T12:00:00",
            author="Test User",
            author_email="test@example.com",
            commit_message="Test commit",
            commit_subject="Test commit",
            commit_body="",
            parent_hashes=["parent123"],
            loc_added=100,
            loc_deleted=50,
            loc_total=1000,
            files_changed=5,
            language_breakdown={"Python": 3},
            readme_exists=True,
            readme_size=500,
            readme_excerpt="# Test",
            comment_samples=[],
            comment_density=0.1,
            insertions=100,
            deletions=50,
            files_stats={},
            is_large_deletion=False,
            is_large_addition=False,
            is_refactor=False,
        )

    def test_from_repo_url(self, mock_client):
        """Test creating enricher from repo URL."""
        enricher = GitHubEnricher.from_repo_url(
            token="test_token",
            repo_url="owner/repo",
        )
        assert enricher.owner == "owner"
        assert enricher.repo == "repo"

    def test_from_repo_url_https(self, mock_client):
        """Test creating enricher from HTTPS URL."""
        enricher = GitHubEnricher.from_repo_url(
            token="test_token",
            repo_url="https://github.com/owner/repo.git",
        )
        assert enricher.owner == "owner"
        assert enricher.repo == "repo"

    def test_build_context_no_github_data(self, mock_client):
        """Test building context without GitHub data."""
        enricher = GitHubEnricher("token", "owner", "repo")
        commit = Mock()
        commit.parent_hashes = ["parent1"]

        ctx = enricher._build_context(commit, None)
        assert ctx.pr_number is None
        assert ctx.is_merge_commit is False

    def test_build_context_merge_commit(self, mock_client):
        """Test building context for merge commit."""
        enricher = GitHubEnricher("token", "owner", "repo")
        commit = Mock()
        commit.parent_hashes = ["parent1", "parent2"]  # Multiple parents = merge

        ctx = enricher._build_context(commit, None)
        assert ctx.is_merge_commit is True

    def test_build_context_with_pr(self, mock_client):
        """Test building context with PR data."""
        enricher = GitHubEnricher("token", "owner", "repo")

        commit = Mock()
        commit.commit_hash = "abc123"
        commit.parent_hashes = ["parent1"]

        pr = PullRequest(
            number=42,
            title="Test PR",
            body="PR body",
            state="MERGED",
            merged=True,
            author=GitHubAuthor(login="author"),
            created_at="2025-01-01T00:00:00Z",
            merged_at="2025-01-02T00:00:00Z",
            merge_commit_sha="abc123",
            head_ref_name="feature/test",
            base_ref_name="main",
            labels=["enhancement"],
            reviewers=[GitHubAuthor(login="reviewer")],
            reviews=[
                Review(
                    author=GitHubAuthor(login="reviewer"),
                    state="APPROVED",
                    body="LGTM",
                )
            ],
        )

        github_commit = CommitContext(
            sha="abc123",
            message="Test",
            associated_prs=[pr],
        )

        ctx = enricher._build_context(commit, github_commit)

        assert ctx.pr_number == 42
        assert ctx.pr_title == "Test PR"
        assert ctx.pr_merged is True
        assert ctx.source_branch == "feature/test"
        assert "reviewer" in ctx.pr_reviewers
        assert ctx.pr_review_states.get("reviewer") == "APPROVED"

    def test_extract_review_comments(self, mock_client):
        """Test extracting review comments."""
        enricher = GitHubEnricher("token", "owner", "repo")

        reviews = [
            Review(
                author=GitHubAuthor(login="reviewer1"),
                state="APPROVED",
                body="Great implementation!",
                comments=[
                    ReviewComment(
                        author=GitHubAuthor(login="reviewer1"),
                        body="Consider using a constant here",
                        path="src/main.py",
                    )
                ],
            ),
            Review(
                author=GitHubAuthor(login="reviewer2"),
                state="CHANGES_REQUESTED",
                body="Needs more tests",
            ),
        ]

        comments = enricher._extract_review_comments(reviews)

        assert len(comments) > 0
        assert any("APPROVED" in c for c in comments)
        assert any("constant" in c for c in comments)

    def test_generate_review_summary(self, mock_client):
        """Test generating review summary."""
        enricher = GitHubEnricher("token", "owner", "repo")

        pr = PullRequest(
            number=1,
            title="Test",
            body=None,
            state="MERGED",
            merged=True,
            author=None,
            created_at="2025-01-01T00:00:00Z",
            reviews=[
                Review(
                    author=GitHubAuthor(login="alice"),
                    state="APPROVED",
                ),
                Review(
                    author=GitHubAuthor(login="bob"),
                    state="APPROVED",
                ),
                Review(
                    author=GitHubAuthor(login="charlie"),
                    state="COMMENTED",
                ),
            ],
        )

        summary = enricher._generate_review_summary(pr)

        assert summary is not None
        assert "2 approved" in summary.lower()
        assert "alice" in summary


class TestExtractPrNarratives:
    """Tests for extract_pr_narratives function."""

    def test_extract_narratives_empty(self):
        narratives = extract_pr_narratives({})
        assert narratives == {}

    def test_extract_narratives_no_pr(self):
        contexts = {
            "sha1": GitHubContext(),  # No PR data
        }
        narratives = extract_pr_narratives(contexts)
        assert "sha1" not in narratives

    def test_extract_narratives_with_pr(self):
        contexts = {
            "sha1": GitHubContext(
                pr_number=123,
                pr_title="Add awesome feature",
                pr_body="This PR implements the awesome feature by...",
                pr_labels=["feature", "enhancement"],
                pr_reviewers=["reviewer1", "reviewer2"],
                pr_review_states={"reviewer1": "APPROVED"},
                review_comments=["Great implementation!"],
            ),
        }
        narratives = extract_pr_narratives(contexts)

        assert "sha1" in narratives
        narrative = narratives["sha1"]
        assert "PR #123" in narrative
        assert "awesome feature" in narrative.lower()
        assert "reviewer1" in narrative


class TestEnrichCommitsWithGitHub:
    """Tests for enrich_commits_with_github function."""

    @patch('gitview.github_enricher.GitHubEnricher')
    def test_enrich_commits(self, mock_enricher_class):
        """Test enriching commits with GitHub data."""
        # Setup mock
        mock_enricher = Mock()
        mock_enricher_class.from_repo_url.return_value = mock_enricher

        ctx = GitHubContext(pr_number=1)
        mock_enricher.enrich_commits.return_value = [
            (Mock(commit_hash="sha1"), ctx)
        ]

        # Create sample commit
        commit = Mock()
        commit.commit_hash = "sha1"

        result = enrich_commits_with_github(
            commits=[commit],
            github_token="token",
            repo_url="owner/repo",
        )

        assert "sha1" in result
        assert result["sha1"].pr_number == 1
