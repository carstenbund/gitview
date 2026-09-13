"""Line counting must match git's own: no churn on merges, renames detected, versioned caches."""

import json
import subprocess
from pathlib import Path

from gitview.extractor import EXTRACTION_VERSION, GitHistoryExtractor
from gitview.history_cache import load_or_extract_history
from git import Repo


def _git(repo, *args):
    subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True, text=True)


def _commit(repo, message, **files):
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    _git(repo, 'add', '-A')
    _git(repo, '-c', 'user.name=T', '-c', 'user.email=t@example.com', 'commit', '-q', '-m', message)


def _repo(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(repo, 'init', '-q', '-b', 'main')
    _commit(repo, 'init', **{'a.py': 'line\n' * 100})
    return repo


def test_rename_is_not_delete_plus_insert(tmp_path):
    repo = _repo(tmp_path)
    _git(repo, 'mv', 'a.py', 'b.py')
    (repo / 'b.py').write_text('line\n' * 100 + 'extra\n')
    _git(repo, 'add', '-A')
    _git(repo, '-c', 'user.name=T', '-c', 'user.email=t@example.com', 'commit', '-q', '-m', 'move a to b')

    last = GitHistoryExtractor(str(repo)).extract_history()[-1]
    assert (last.insertions, last.deletions) == (1, 0)
    assert list(last.files_stats) == ['b.py']
    assert not last.is_large_deletion and not last.is_large_addition


def test_merge_commit_carries_no_churn(tmp_path):
    repo = _repo(tmp_path)
    _git(repo, 'checkout', '-q', '-b', 'feature')
    _commit(repo, 'feature work', **{'f.py': 'x = 1\n' * 500})
    _git(repo, 'checkout', '-q', 'main')
    _commit(repo, 'main work', **{'m.py': 'y = 1\n' * 5})
    _git(repo, '-c', 'user.name=T', '-c', 'user.email=t@example.com', 'merge', '-q', '--no-ff', '-m', 'merge feature', 'feature')

    records = GitHistoryExtractor(str(repo)).extract_history()
    merge = records[-1]
    assert merge.is_merge
    assert (merge.insertions, merge.deletions, merge.files_stats) == (0, 0, {})
    # The branch work is counted exactly once, on its own commit.
    assert sum(r.insertions for r in records) == 100 + 500 + 5
    assert records[-1].loc_total == 605


def test_jsonl_round_trip_carries_extraction_version(tmp_path):
    repo = _repo(tmp_path)
    extractor = GitHistoryExtractor(str(repo))
    records = extractor.extract_history()
    out = tmp_path / 'history.jsonl'
    extractor.save_to_jsonl(records, str(out))

    assert json.loads(out.read_text().splitlines()[0]) == {'gitview_extraction_version': EXTRACTION_VERSION}
    assert GitHistoryExtractor.jsonl_extraction_version(str(out)) == EXTRACTION_VERSION
    loaded = GitHistoryExtractor.load_from_jsonl(str(out))
    assert [r.commit_hash for r in loaded] == [r.commit_hash for r in records]

    legacy = tmp_path / 'legacy.jsonl'
    legacy.write_text('\n'.join(json.dumps(r.to_dict()) for r in records) + '\n')
    assert GitHistoryExtractor.jsonl_extraction_version(str(legacy)) == 1
    assert len(GitHistoryExtractor.load_from_jsonl(str(legacy))) == len(records)


def test_history_cache_ignores_legacy_files(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    extractor = GitHistoryExtractor(str(repo))
    cache_dir = repo / '.gitview'
    cache_dir.mkdir()
    stale = extractor.extract_history()
    stale[0].insertions = 999_999                      # numbers from the old counting
    (cache_dir / 'history.jsonl').write_text(json.dumps(stale[0].to_dict()) + '\n')   # no header: v1

    records = load_or_extract_history(extractor, Repo(str(repo)), repo)
    assert records[0].insertions == 100                # re-extracted, not the stale cache
    assert GitHistoryExtractor.jsonl_extraction_version(str(cache_dir / 'history.jsonl')) == EXTRACTION_VERSION
