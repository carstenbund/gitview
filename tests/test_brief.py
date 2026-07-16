"""Tests for the 'gitview brief' command and its supporting helpers."""

import subprocess
from unittest.mock import MagicMock

import pytest

from gitview.extractor import CommitRecord
from gitview.chunker import Phase
from gitview.storyline.detector import CommitTrailerDetector, StorylineDetector
from gitview.storyline.models import StorylineCategory
from gitview.commands.brief import (
    _detect_repo_identity,
    _first_paragraph,
    _top_churn_files,
    _all_contributors,
    _language_mix,
    _increase_heading_level,
    _read_stamp,
    _looks_like_public_remote,
    render_brief,
    BriefCommand,
)


# ---------------------------------------------------------------------------
# Fixtures / factories
# ---------------------------------------------------------------------------

def _make_commit(
    commit_hash="abc123def456",
    author="Test Author",
    timestamp="2026-01-01T00:00:00",
    commit_message="Add a feature",
    files=None,
    insertions=10,
    deletions=2,
    language_breakdown=None,
):
    files = files if files is not None else {"gitview/foo.py": {"insertions": insertions, "deletions": deletions}}
    lines = commit_message.split("\n")
    return CommitRecord(
        commit_hash=commit_hash,
        short_hash=commit_hash[:8],
        timestamp=timestamp,
        author=author,
        author_email=f"{author.replace(' ', '.').lower()}@example.com",
        commit_message=commit_message,
        commit_subject=lines[0],
        commit_body="\n".join(lines[1:]).strip(),
        parent_hashes=[],
        loc_added=insertions,
        loc_deleted=deletions,
        loc_total=insertions - deletions,
        files_changed=len(files),
        language_breakdown=language_breakdown or {"Python": 100},
        readme_exists=True,
        readme_size=100,
        readme_excerpt=None,
        comment_samples=[],
        comment_density=0.0,
        insertions=insertions,
        deletions=deletions,
        files_stats=files,
        is_large_deletion=False,
        is_large_addition=False,
        is_refactor=False,
    )


def _make_phase(phase_number, commits):
    return Phase(
        phase_number=phase_number,
        start_date=commits[0].timestamp,
        end_date=commits[-1].timestamp,
        commit_count=len(commits),
        commits=commits,
        loc_start=0,
        loc_end=sum(c.loc_total for c in commits),
        loc_delta=sum(c.insertions - c.deletions for c in commits),
        loc_delta_percent=0.0,
        total_insertions=sum(c.insertions for c in commits),
        total_deletions=sum(c.deletions for c in commits),
        languages_start={},
        languages_end={},
        has_large_deletion=False,
        has_large_addition=False,
        has_refactor=False,
        readme_changed=False,
        authors=sorted({c.author for c in commits}),
        primary_author=commits[0].author,
    )


# ---------------------------------------------------------------------------
# CommitTrailerDetector
# ---------------------------------------------------------------------------

class TestCommitTrailerDetector:
    """Test the 'Storyline: [status:category] Title' commit trailer detector."""

    def _mock_phase(self, messages):
        commits = []
        for i, message in enumerate(messages):
            c = MagicMock()
            c.commit_message = message
            c.short_hash = f"sha{i}"
            commits.append(c)
        phase = MagicMock()
        phase.phase_number = 1
        phase.commits = commits
        return phase

    def test_detects_basic_trailer(self):
        detector = CommitTrailerDetector()
        phase = self._mock_phase([
            "feat(worklog): add date range\n\nStoryline: [new:feature] GitHub worklog analyze mode"
        ])

        signals = detector.detect(phase)

        assert len(signals) == 1
        assert signals[0].title == "GitHub worklog analyze mode"
        assert signals[0].category == StorylineCategory.FEATURE
        assert signals[0].data["status"] == "new"
        assert signals[0].data["is_new"] is True
        assert signals[0].data["source"] == "commit_trailer"

    def test_case_insensitive_status_and_category(self):
        detector = CommitTrailerDetector()
        phase = self._mock_phase([
            "feat: x\n\nStoryline: [NEW:FEATURE] Some Initiative"
        ])

        signals = detector.detect(phase)

        assert len(signals) == 1
        assert signals[0].category == StorylineCategory.FEATURE
        assert signals[0].data["status"] == "new"

    def test_completed_status_sets_completion_flag(self):
        detector = CommitTrailerDetector()
        phase = self._mock_phase([
            "docs: finish up\n\nStoryline: [completed:docs] GitView Documentation"
        ])

        signals = detector.detect(phase)

        assert signals[0].data["is_completion"] is True
        assert signals[0].category == StorylineCategory.DOCUMENTATION

    def test_no_trailer_produces_no_signal(self):
        detector = CommitTrailerDetector()
        phase = self._mock_phase(["Just a normal commit message"])

        assert detector.detect(phase) == []

    def test_multiple_commits_same_title_grouped_latest_status_wins(self):
        detector = CommitTrailerDetector()
        phase = self._mock_phase([
            "Storyline: [new:feature] Big Initiative",
            "Storyline: [continued:feature] Big Initiative",
        ])

        signals = detector.detect(phase)

        assert len(signals) == 1
        assert signals[0].data["status"] == "continued"
        assert len(signals[0].commit_hashes) == 2

    def test_unknown_category_still_produces_signal(self):
        """Unlike inferred detectors, explicit trailers are trusted even when
        the category doesn't map to a known StorylineCategory."""
        detector = CommitTrailerDetector()
        phase = self._mock_phase([
            "Storyline: [new:test] Add coverage for worklog"
        ])

        signals = detector.detect(phase)

        assert len(signals) == 1
        assert signals[0].category == StorylineCategory.UNKNOWN
        assert signals[0].title == "Add coverage for worklog"

    def test_highest_confidence_weight_of_all_detectors(self):
        detector = CommitTrailerDetector()
        assert detector.weight == 0.95

    def test_wired_into_storyline_detector(self):
        combined = StorylineDetector()
        assert any(isinstance(d, CommitTrailerDetector) for d in combined.detectors)


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------

class TestBriefHelpers:

    def test_top_churn_files_counts_and_orders(self):
        records = [
            _make_commit(commit_hash="a", files={"a.py": {}, "b.py": {}}),
            _make_commit(commit_hash="b", files={"a.py": {}}),
            _make_commit(commit_hash="c", files={"a.py": {}, "c.py": {}}),
        ]

        result = _top_churn_files(records, limit=2)

        assert result[0] == ("a.py", 3)
        assert len(result) == 2

    def test_all_contributors_sorted_desc(self):
        records = [
            _make_commit(commit_hash="a", author="Alice"),
            _make_commit(commit_hash="b", author="Bob"),
            _make_commit(commit_hash="c", author="Alice"),
        ]

        result = _all_contributors(records)

        assert result[0] == ("Alice", 2)
        assert result[1] == ("Bob", 1)

    def test_language_mix_uses_last_commit_snapshot(self):
        records = [
            _make_commit(commit_hash="a", language_breakdown={"Python": 1}),
            _make_commit(commit_hash="b", language_breakdown={"Python": 3, "Markdown": 1}),
        ]

        result = _language_mix(records)

        assert result[0][0] == "Python"
        assert result[0][1] == pytest.approx(0.75)

    def test_language_mix_empty_when_no_records(self):
        assert _language_mix([]) == []

    def test_increase_heading_level(self):
        markdown = "# Title\n\nSome text\n\n## Subsection\n"
        shifted = _increase_heading_level(markdown, 2)
        assert "### Title" in shifted
        assert "#### Subsection" in shifted
        assert "Some text" in shifted

    def test_first_paragraph_skips_headings_and_badges(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text(
            "# My Project\n\n[![CI](badge.svg)](link)\n\n"
            "A tool that does the thing, quite well actually.\n"
        )

        result = _first_paragraph(readme)

        assert result.startswith("A tool that does the thing")

    def test_detect_repo_identity_from_pyproject(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "widget"\ndescription = "Makes widgets"\n'
        )

        identity = _detect_repo_identity(tmp_path)

        assert identity["name"] == "widget"
        assert identity["description"] == "Makes widgets"

    def test_detect_repo_identity_falls_back_to_dirname(self, tmp_path):
        project_dir = tmp_path / "my-project"
        project_dir.mkdir()

        identity = _detect_repo_identity(project_dir)

        assert identity["name"] == "my-project"
        assert identity["description"] == ""

    def test_detect_repo_identity_prefers_remote_slug(self, tmp_path):
        # HTTPS and SSH remotes both resolve to the repo slug.
        identity = _detect_repo_identity(
            tmp_path, remote_url="https://github.com/org/ado-api.git"
        )
        assert identity["name"] == "ado-api"

        identity = _detect_repo_identity(
            tmp_path, remote_url="git@github.com:org/ado-api.git"
        )
        assert identity["name"] == "ado-api"

    def test_detect_repo_identity_remote_overrides_subcomponent_pyproject(self, tmp_path):
        # A monorepo root pyproject that names a packaged subcomponent must not
        # override the repo's own identity (from the remote).
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "ocs-pipeline"\n'
            'description = "OCS XML order processing pipeline"\n'
        )

        identity = _detect_repo_identity(
            tmp_path, remote_url="https://github.com/org/ado-api.git"
        )

        assert identity["name"] == "ado-api"
        # The subcomponent's description is not trusted for the whole repo.
        assert identity["description"] != "OCS XML order processing pipeline"

    def test_detect_repo_identity_trusts_pyproject_when_name_matches_repo(self, tmp_path):
        # When the pyproject name matches the repo (single-package repo), keep
        # using its description.
        repo_dir = tmp_path / "widget"
        repo_dir.mkdir()
        (repo_dir / "pyproject.toml").write_text(
            '[project]\nname = "widget"\ndescription = "Makes widgets"\n'
        )

        identity = _detect_repo_identity(
            repo_dir, remote_url="https://github.com/org/widget.git"
        )

        assert identity["name"] == "widget"
        assert identity["description"] == "Makes widgets"

    def test_read_stamp_missing_file_returns_none(self, tmp_path):
        assert _read_stamp(tmp_path / "nope.md") is None

    def test_read_stamp_round_trip(self, tmp_path):
        target = tmp_path / "AGENT_BRIEF.md"
        target.write_text(
            "<!-- gitview:brief head=deadbeef generated=2026-01-01T00:00:00+00:00 "
            "commits=5 gitview_version=0.6.2 -->\n# Agent Brief\n"
        )

        stamp = _read_stamp(target)

        assert stamp["head"] == "deadbeef"
        assert stamp["commits"] == "5"

    def test_looks_like_public_remote_filters_loopback(self):
        assert not _looks_like_public_remote("http://local_proxy@127.0.0.1:1234/git/x")
        assert _looks_like_public_remote("https://github.com/carstenbund/gitview")


# ---------------------------------------------------------------------------
# render_brief
# ---------------------------------------------------------------------------

class TestRenderBrief:

    def test_render_brief_includes_expected_sections(self, tmp_path):
        commits = [
            _make_commit(
                commit_hash="a1", author="Alice",
                timestamp="2026-01-01T00:00:00",
                commit_message="feat: add widgets\n\nStoryline: [new:feature] Widget Support",
            ),
            _make_commit(
                commit_hash="a2", author="Alice",
                timestamp="2026-01-02T00:00:00",
                commit_message="feat: more widgets\n\nStoryline: [completed:feature] Widget Support",
            ),
        ]
        phase = _make_phase(1, commits)

        content = render_brief(
            repo_path=tmp_path,
            records=commits,
            phases=[phase],
            head_sha="a" * 40,
            is_shallow=False,
            remote_url=None,
            top_files_limit=5,
        )

        assert content.startswith("<!-- gitview:brief head=")
        assert "## Project" in content
        assert "## At a Glance" in content
        assert "## Timeline" in content
        assert "## Storylines" in content
        assert "## Most-Changed Files" in content
        assert "Widget Support" in content
        assert "Alice" in content

    def test_render_brief_omits_remote_when_none(self, tmp_path):
        commits = [_make_commit()]
        phase = _make_phase(1, commits)

        content = render_brief(
            repo_path=tmp_path, records=commits, phases=[phase],
            head_sha="b" * 40, is_shallow=False, remote_url=None,
        )

        assert "**Remote:**" not in content

    def test_render_brief_notes_shallow_clone(self, tmp_path):
        commits = [_make_commit()]
        phase = _make_phase(1, commits)

        content = render_brief(
            repo_path=tmp_path, records=commits, phases=[phase],
            head_sha="c" * 40, is_shallow=True, remote_url=None,
        )

        assert "shallow clone" in content


# ---------------------------------------------------------------------------
# BriefCommand integration (real temp git repo)
# ---------------------------------------------------------------------------

def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def tiny_repo(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    _git("init", "-q", cwd=repo_dir)
    _git("config", "user.email", "test@example.com", cwd=repo_dir)
    _git("config", "user.name", "Test User", cwd=repo_dir)

    (repo_dir / "README.md").write_text("# Tiny Repo\n\nA tiny test repo.\n")
    _git("add", "README.md", cwd=repo_dir)
    _git("commit", "-q", "-m", "Initial commit", cwd=repo_dir)

    (repo_dir / "feature.py").write_text("def feature():\n    return 1\n")
    _git("add", "feature.py", cwd=repo_dir)
    _git(
        "commit", "-q", "-m",
        "feat: add feature\n\nStoryline: [new:feature] Feature Work",
        cwd=repo_dir,
    )

    return repo_dir


class TestBriefCommandIntegration:

    def test_generates_brief_with_stamp(self, tiny_repo):
        output = tiny_repo / "AGENT_BRIEF.md"
        BriefCommand(repo=str(tiny_repo), output=str(output)).run()

        assert output.exists()
        content = output.read_text()
        assert "<!-- gitview:brief head=" in content
        assert "Feature Work" in content

    def test_second_run_is_noop_without_force(self, tiny_repo):
        output = tiny_repo / "AGENT_BRIEF.md"
        BriefCommand(repo=str(tiny_repo), output=str(output)).run()
        first_mtime = output.stat().st_mtime_ns

        BriefCommand(repo=str(tiny_repo), output=str(output)).run()
        second_mtime = output.stat().st_mtime_ns

        assert first_mtime == second_mtime

    def test_force_regenerates(self, tiny_repo):
        output = tiny_repo / "AGENT_BRIEF.md"
        BriefCommand(repo=str(tiny_repo), output=str(output)).run()
        first_mtime = output.stat().st_mtime_ns

        BriefCommand(repo=str(tiny_repo), output=str(output), force=True).run()
        second_mtime = output.stat().st_mtime_ns

        assert second_mtime > first_mtime

    def test_check_exits_nonzero_when_no_brief_yet(self, tiny_repo):
        output = tiny_repo / "AGENT_BRIEF.md"

        with pytest.raises(SystemExit) as exc_info:
            BriefCommand(repo=str(tiny_repo), output=str(output), check=True).run()

        assert exc_info.value.code == 1
        assert not output.exists()

    def test_new_commit_makes_it_stale(self, tiny_repo):
        output = tiny_repo / "AGENT_BRIEF.md"
        BriefCommand(repo=str(tiny_repo), output=str(output)).run()

        (tiny_repo / "more.py").write_text("x = 1\n")
        _git("add", "more.py", cwd=tiny_repo)
        _git("commit", "-q", "-m", "chore: more", cwd=tiny_repo)

        with pytest.raises(SystemExit) as exc_info:
            BriefCommand(repo=str(tiny_repo), output=str(output), check=True).run()
        assert exc_info.value.code == 1

        # Regenerating (no --check) should succeed and pick up the new commit
        BriefCommand(repo=str(tiny_repo), output=str(output)).run()
        content = output.read_text()
        assert "commits=3" in content.split("\n", 1)[0]
