"""Tests for the persistent repository graph (gitview.graph)."""

import json
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from gitview.cli import cli
from gitview.graph import (
    GRAPH_SCHEMA_VERSION,
    GraphBuilder,
    GraphStore,
    GraphUpdater,
    compute_graph_stats,
)
from gitview.graph.updater import _records_after
from gitview.history_cache import head_descends_from

from tests.test_brief import _make_commit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _files(*paths, ins=1, dels=0):
    return {p: {"insertions": ins, "deletions": dels} for p in paths}


def _commit(n, *paths, parents=None, ts=None, author="Dev", pr=None, **kw):
    record = _make_commit(
        commit_hash=f"{n:040x}",
        timestamp=ts or f"2026-01-{n:02d}T00:00:00",
        author=author,
        files=_files(*paths),
        **kw,
    )
    record.parent_hashes = parents if parents is not None else ([f"{n-1:040x}"] if n > 1 else [])
    if pr is not None:
        record.github_context = {"pr_number": pr, "pr_title": f"PR {pr}", "pr_labels": ["feature"],
                                 "pr_state": "merged", "pr_merged": True}
    return record


def _build(records, **kw):
    store = GraphStore(":memory:")
    store.initialize()
    GraphBuilder(store, repository_path="/tmp/repo", **kw).build(records)
    return store


# The proposal's worked example.
PROPOSAL_RECORDS = [
    _commit(1, "parser.py", "models.py", "tests/test_parser.py"),
    _commit(2, "parser.py", "tests/test_parser.py"),
    _commit(3, "models.py", "serializer.py"),
]


# ---------------------------------------------------------------------------
# Store + builder
# ---------------------------------------------------------------------------

class TestStoreAndBuilder:
    def test_schema_and_metadata_roundtrip(self):
        store = _build(PROPOSAL_RECORDS, branch="main", max_projection_files=50)
        meta = store.get_metadata()
        assert meta.schema_version == GRAPH_SCHEMA_VERSION
        assert meta.branch == "main"
        assert meta.max_projection_files == 50
        assert meta.last_commit_hash == PROPOSAL_RECORDS[-1].commit_hash
        assert meta.last_commit_timestamp == PROPOSAL_RECORDS[-1].timestamp

    def test_empty_store_has_no_metadata(self):
        store = GraphStore(":memory:")
        store.initialize()
        assert store.get_metadata() is None

    def test_counts_after_build(self):
        store = _build(PROPOSAL_RECORDS)
        counts = store.counts()
        assert counts["commits"] == 3
        assert counts["files"] == 4
        assert counts["commit_file_edges"] == 7
        assert counts["authors"] == 1
        assert counts["merge_commits"] == 0

    def test_parent_edges_preserved(self):
        merge = _commit(4, "x.py", parents=[PROPOSAL_RECORDS[2].commit_hash, "f" * 40])
        store = _build(PROPOSAL_RECORDS + [merge])
        assert store.parents_of(merge.commit_hash) == [PROPOSAL_RECORDS[2].commit_hash, "f" * 40]
        assert store.parents_of(PROPOSAL_RECORDS[0].commit_hash) == []

    def test_files_of_commit(self):
        store = _build(PROPOSAL_RECORDS)
        assert store.files_of(PROPOSAL_RECORDS[0].commit_hash) == ["models.py", "parser.py", "tests/test_parser.py"]

    def test_file_touch_stats_accumulate(self):
        store = _build(PROPOSAL_RECORDS)
        top = store.most_changed(limit=2)
        assert [(f.path, f.touch_count) for f in top] == [("models.py", 2), ("parser.py", 2)]
        assert top[0].total_insertions == 2

    def test_authors_deduplicated_and_prs_linked(self):
        records = [
            _commit(1, "a.py", author="Alice", pr=7),
            _commit(2, "b.py", author="Alice", pr=7),
            _commit(3, "c.py", author="Bob"),
        ]
        store = _build(records)
        counts = store.counts()
        assert counts["authors"] == 2
        assert counts["pull_requests"] == 1
        row = store.conn.execute("SELECT number, title, merged FROM pull_requests").fetchone()
        assert (row["number"], row["title"], row["merged"]) == (7, "PR 7", 1)
        assert store.conn.execute("SELECT COUNT(*) AS n FROM commit_prs").fetchone()["n"] == 2

    def test_insert_skips_known_hashes(self):
        store = _build(PROPOSAL_RECORDS)
        assert store.insert_commits(PROPOSAL_RECORDS) == 0
        assert store.counts()["commits"] == 3
        assert store.max_sequence() == 3


# ---------------------------------------------------------------------------
# Co-change projection
# ---------------------------------------------------------------------------

class TestCochangeProjection:
    def test_single_commit_yields_all_pairs(self):
        store = _build([_commit(1, "a", "b", "c")])
        edges = store.all_edges()
        assert [(a, b, n) for a, b, n, *_ in edges] == [("a", "b", 1), ("a", "c", 1), ("b", "c", 1)]

    def test_proposal_jaccard_example(self):
        store = _build(PROPOSAL_RECORDS)
        assert store.edge("parser.py", "tests/test_parser.py").jaccard == 1.0
        assert store.edge("parser.py", "models.py").jaccard == pytest.approx(1 / 3, abs=1e-4)
        assert store.edge("models.py", "serializer.py").jaccard == 0.5
        assert store.edge("parser.py", "serializer.py") is None

    def test_edge_lookup_is_order_independent(self):
        store = _build(PROPOSAL_RECORDS)
        assert store.edge("tests/test_parser.py", "parser.py").cochange_count == 2

    def test_first_and_last_seen(self):
        store = _build(PROPOSAL_RECORDS)
        edge = store.edge("parser.py", "tests/test_parser.py")
        assert edge.first_seen == "2026-01-01T00:00:00"
        assert edge.last_seen == "2026-01-02T00:00:00"

    def test_merge_commits_do_not_create_pairs(self):
        merge = _commit(2, "a", "b", "c", parents=["1" * 40, "2" * 40])
        store = _build([_commit(1, "a"), merge])
        assert store.counts()["merge_commits"] == 1
        assert store.counts()["commit_file_edges"] == 4   # still evidence
        assert store.counts()["file_edges"] == 0
        # projected touch count excludes the merge; plain touch count includes it
        row = store.conn.execute("SELECT touch_count, projected_touch_count FROM files WHERE path='a'").fetchone()
        assert (row["touch_count"], row["projected_touch_count"]) == (2, 1)

    def test_giant_commit_is_suppressed(self):
        giant = _commit(1, *[f"gen/{i}.py" for i in range(5)])
        store = _build([giant, _commit(2, "gen/0.py", "gen/1.py")], max_projection_files=3)
        assert store.counts()["suppressed_commits"] == 1
        assert store.counts()["files"] == 5
        assert store.counts()["file_edges"] == 1
        assert store.edge("gen/0.py", "gen/1.py").cochange_count == 1

    def test_incremental_update_matches_full_build(self):
        records = PROPOSAL_RECORDS + [
            _commit(4, "parser.py", "serializer.py"),
            _commit(5, "models.py", "parser.py", "tests/test_parser.py"),
            _commit(6, "cli.py"),
        ]
        full = _build(records)

        partial = GraphStore(":memory:")
        partial.initialize()
        builder = GraphBuilder(partial, repository_path="/tmp/repo")
        builder.build(records[:2])
        builder.update(records[2:4])
        builder.update(records[4:])

        assert partial.all_edges() == full.all_edges()
        assert partial.counts() == full.counts()
        touches = "SELECT path, touch_count, projected_touch_count FROM files ORDER BY path"
        assert [tuple(r) for r in partial.conn.execute(touches)] == \
               [tuple(r) for r in full.conn.execute(touches)]
        assert partial.most_coupled(10) == full.most_coupled(10)
        assert partial.most_connected(10) == full.most_connected(10)

    def test_top_lists_are_deterministic_with_ties(self):
        # Every pair co-changes exactly once: ordering must fall back to path.
        store = _build([_commit(1, "b", "a"), _commit(2, "d", "c")])
        coupled = store.most_coupled(10, min_cochanges=1)
        assert [(e.path_a, e.path_b) for e in coupled] == [("a", "b"), ("c", "d")]
        assert [d.path for d in store.most_connected(10)] == ["a", "b", "c", "d"]

    def test_rebuild_after_clear_projection_is_idempotent(self):
        store = _build(PROPOSAL_RECORDS)
        before = store.all_edges()
        store.clear_projection()
        assert store.counts()["file_edges"] == 0
        store.apply_cochange_projection(since_sequence=0)
        assert store.all_edges() == before


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_to_dict_is_json_serializable(self):
        store = _build(PROPOSAL_RECORDS)
        stats = compute_graph_stats(store, top=5)
        payload = json.loads(json.dumps(stats.to_dict()))
        assert payload["commits"] == 3
        assert payload["file_edges"] == 4
        assert payload["most_coupled"][0]["path_a"] == "parser.py"
        assert payload["most_connected"][0]["path"] in {"models.py", "parser.py"}

    def test_records_after(self):
        assert _records_after(PROPOSAL_RECORDS, PROPOSAL_RECORDS[0].commit_hash) == PROPOSAL_RECORDS[1:]
        assert _records_after(PROPOSAL_RECORDS, PROPOSAL_RECORDS[-1].commit_hash) == []
        assert _records_after(PROPOSAL_RECORDS, "missing") is None


# ---------------------------------------------------------------------------
# Synthetic repository: updater + CLI
# ---------------------------------------------------------------------------

def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _write_and_commit(repo, message, **files):
    for name, content in files.items():
        path = Path(repo) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        _git(repo, "add", name)
    _git(repo, "-c", "user.name=T", "-c", "user.email=t@example.com", "commit", "-q", "-m", message)


@pytest.fixture
def synthetic_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _write_and_commit(repo, "init", **{"src/a.py": "a\n", "src/b.py": "b\n"})
    _write_and_commit(repo, "touch a and b again", **{"src/a.py": "aa\n", "src/b.py": "bb\n"})
    _write_and_commit(repo, "docs", **{"README.md": "hi\n"})
    return repo


class TestGraphUpdater:
    def test_build_then_unchanged_then_update(self, synthetic_repo):
        updater = GraphUpdater(synthetic_repo)

        first = updater.sync()
        assert first.action == "built"
        assert first.new_commits == 3

        second = updater.sync()
        assert second.action == "unchanged"

        _write_and_commit(synthetic_repo, "more", **{"src/a.py": "aaa\n", "src/c.py": "c\n"})
        third = updater.sync()
        assert third.action == "updated"
        assert third.new_commits == 1

        with GraphStore(updater.store_path) as store:
            assert store.counts()["commits"] == 4
            assert store.edge("src/a.py", "src/b.py").cochange_count == 2
            assert store.edge("src/a.py", "src/c.py").cochange_count == 1

    def test_history_rewrite_triggers_rebuild(self, synthetic_repo):
        updater = GraphUpdater(synthetic_repo)
        updater.sync()

        _git(synthetic_repo, "reset", "-q", "--hard", "HEAD~1")
        _write_and_commit(synthetic_repo, "replacement", **{"src/z.py": "z\n"})

        result = updater.sync()
        assert result.action == "built"
        assert "rewritten" in result.reason
        with GraphStore(updater.store_path) as store:
            assert store.counts()["commits"] == 3
            assert store.files_of(result.metadata.last_commit_hash) == ["src/z.py"]

    def test_explicit_rebuild(self, synthetic_repo):
        updater = GraphUpdater(synthetic_repo)
        updater.sync()
        assert updater.sync(rebuild=True).action == "built"

    def test_store_lives_in_gitview_dir_and_is_ignored(self, synthetic_repo):
        GraphUpdater(synthetic_repo).sync()
        assert (synthetic_repo / ".gitview" / "graph.sqlite").exists()
        assert (synthetic_repo / ".gitview" / ".gitignore").read_text() == "*\n"
        status = subprocess.run(["git", "status", "--porcelain"], cwd=synthetic_repo,
                                capture_output=True, text=True, check=True).stdout
        assert ".gitview" not in status

    def test_head_descends_from(self, synthetic_repo):
        from git import Repo
        repo = Repo(str(synthetic_repo))
        head = repo.head.commit
        assert head_descends_from(repo, head.hexsha)
        assert head_descends_from(repo, head.parents[0].hexsha)
        assert not head_descends_from(repo, "0" * 40)
        assert not head_descends_from(repo, "")


class TestGraphCli:
    def test_json_output(self, synthetic_repo):
        result = CliRunner().invoke(cli, ["graph", "--repo", str(synthetic_repo), "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["action"] == "built"
        assert payload["stats"]["commits"] == 3
        assert payload["stats"]["files"] == 3
        assert payload["stats"]["file_edges"] == 1

    def test_stats_tables(self, synthetic_repo):
        result = CliRunner().invoke(cli, ["graph", "--repo", str(synthetic_repo), "--stats"])
        assert result.exit_code == 0, result.output
        assert "Repository graph" in result.output
        assert "Most coupled" in result.output
        assert "src/a.py" in result.output

    def test_rejects_non_repo(self, tmp_path):
        result = CliRunner().invoke(cli, ["graph", "--repo", str(tmp_path)])
        assert result.exit_code == 1
        assert "not a git repository" in " ".join(result.output.split())
