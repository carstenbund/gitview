"""Tests for the canonical CommitRecord accessors and the detectors that rely on them."""

from gitview.extractor import CommitRecord
from gitview.significance_analyzer import SignificanceAnalyzer
from gitview.storyline.detector import (
    CommitMessagePatternDetector,
    FileClusterDetector,
    PRLabelDetector,
    PRTitlePatternDetector,
)

from tests.test_brief import _make_commit, _make_phase


def _with_pr(commit, number, title="Add login flow", labels=None):
    commit.github_context = {
        'pr_number': number,
        'pr_title': title,
        'pr_body': "Implements login.",
        'pr_labels': labels if labels is not None else [],
    }
    return commit


class TestCommitRecordAccessors:
    def test_get_changed_files_returns_files_stats_keys_in_order(self):
        commit = _make_commit(files={
            "b.py": {"insertions": 1, "deletions": 0},
            "a.py": {"insertions": 1, "deletions": 0},
        })
        assert commit.get_changed_files() == ["b.py", "a.py"]

    def test_get_changed_files_empty(self):
        assert _make_commit(files={}).get_changed_files() == []

    def test_get_pr_number_without_context(self):
        assert _make_commit().get_pr_number() is None

    def test_get_pr_number_with_context(self):
        assert _with_pr(_make_commit(), 42).get_pr_number() == 42

    def test_get_pr_number_coerces_strings_and_rejects_garbage(self):
        assert _with_pr(_make_commit(), "7").get_pr_number() == 7
        assert _with_pr(_make_commit(), "n/a").get_pr_number() is None

    def test_is_merge(self):
        assert _make_commit().is_merge is False
        single = _make_commit()
        single.parent_hashes = ["p1"]
        assert single.is_merge is False
        merge = _make_commit()
        merge.parent_hashes = ["p1", "p2"]
        assert merge.is_merge is True

    def test_from_dict_roundtrip_keeps_accessors(self):
        original = _with_pr(_make_commit(), 3)
        restored = CommitRecord.from_dict(original.to_dict())
        assert restored.get_pr_number() == 3
        assert restored.get_changed_files() == original.get_changed_files()


class TestFileClusterDetectorUsesRealRecords:
    """Regression: the detector used to look for attributes CommitRecord never had."""

    def test_emits_cluster_signal_with_files(self):
        parser = {"src/parser.py": {"insertions": 5, "deletions": 1},
                  "tests/test_parser.py": {"insertions": 3, "deletions": 0}}
        commits = [
            _make_commit(commit_hash=f"c{i}" * 6, commit_message=f"Work {i}", files=parser)
            for i in range(3)
        ]
        signals = FileClusterDetector().detect(_make_phase(1, commits))

        assert len(signals) == 1
        assert set(signals[0].files) == {"src/parser.py", "tests/test_parser.py"}
        assert len(signals[0].commit_hashes) == 3

    def test_no_signal_when_files_never_cochange(self):
        commits = [
            _make_commit(commit_hash="a" * 12, files={"a.py": {"insertions": 1, "deletions": 0}}),
            _make_commit(commit_hash="b" * 12, files={"b.py": {"insertions": 1, "deletions": 0}}),
        ]
        assert FileClusterDetector().detect(_make_phase(1, commits)) == []


class TestPRDetectorsCarryFiles:
    def test_pr_label_signal_has_files(self):
        commits = [
            _with_pr(_make_commit(commit_hash="1" * 12, files={"auth.py": {"insertions": 1, "deletions": 0}}),
                     10, labels=["feature"]),
            _with_pr(_make_commit(commit_hash="2" * 12, files={"auth_test.py": {"insertions": 1, "deletions": 0}}),
                     10, labels=["feature"]),
        ]
        signals = PRLabelDetector().detect(_make_phase(1, commits))
        assert len(signals) == 1
        assert set(signals[0].files) == {"auth.py", "auth_test.py"}
        assert signals[0].data['pr_number'] == 10

    def test_pr_title_signal_has_files(self):
        commit = _with_pr(_make_commit(files={"cli.py": {"insertions": 1, "deletions": 0}}),
                          11, title="feat: add cli flag")
        signals = PRTitlePatternDetector().detect(_make_phase(1, [commit]))
        assert len(signals) == 1
        assert signals[0].files == ["cli.py"]

    def test_commit_message_signal_has_files(self):
        commits = [
            _make_commit(commit_hash="3" * 12, commit_message="Add parser support",
                         files={"parser.py": {"insertions": 1, "deletions": 0}}),
            _make_commit(commit_hash="4" * 12, commit_message="Add parser support tests",
                         files={"test_parser.py": {"insertions": 1, "deletions": 0}}),
        ]
        signals = CommitMessagePatternDetector().detect(_make_phase(1, commits))
        assert len(signals) == 1
        assert set(signals[0].files) == {"parser.py", "test_parser.py"}


class TestSignificanceAnalyzerWithGitHubContext:
    """Regression: cluster_commits called a get_pr_number() that did not exist."""

    def test_splits_clusters_on_pr_number(self):
        commits = [
            _with_pr(_make_commit(commit_hash="a" * 12, commit_message="Add x"), 1),
            _with_pr(_make_commit(commit_hash="b" * 12, commit_message="Add y"), 1),
            _with_pr(_make_commit(commit_hash="c" * 12, commit_message="Add z"), 2),
        ]
        clusters = SignificanceAnalyzer().cluster_commits(commits)
        assert [len(c.commits) for c in clusters] == [2, 1]
