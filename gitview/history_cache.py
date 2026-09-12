"""Shared, incremental extraction cache for deterministic commands.

`gitview brief` and `gitview graph` are both meant to be re-run after every
batch of commits, so re-extracting the entire history each time is wasteful
(and on very large repositories is exactly what exhausts memory). Extracted
records are persisted to ``<repo>/.gitview/history.jsonl``; on a subsequent
run whose HEAD descends from the cached HEAD only the new commits are
extracted and appended.

Every failure path falls back to a full extraction, so a missing, stale or
unreadable cache is never fatal.
"""

from pathlib import Path
from typing import List, Optional

from git import Repo

from .extractor import CommitRecord, GitHistoryExtractor

GITVIEW_DIR = ".gitview"
HISTORY_CACHE_FILE = "history.jsonl"


def gitview_dir(repo_path: Path) -> Path:
    """Per-repository working directory for GitView caches (git-ignored)."""
    return Path(repo_path) / GITVIEW_DIR


def ensure_gitview_dir(repo_path: Path) -> Path:
    """Create ``.gitview/`` with a self-ignoring ``.gitignore`` and return it."""
    directory = gitview_dir(repo_path)
    directory.mkdir(parents=True, exist_ok=True)
    gitignore = directory / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*\n", encoding="utf-8")
    return directory


def head_descends_from(repo: Repo, cached_head: str, head_sha: Optional[str] = None) -> bool:
    """True when ``cached_head`` is an ancestor of (or equal to) HEAD.

    A false result means history was rewritten (rebase, reset, force push,
    branch replacement) and any incremental "since cached_head" update is
    unsafe.
    """
    if not cached_head:
        return False
    try:
        head_sha = head_sha or repo.head.commit.hexsha
        if cached_head == head_sha:
            return True
        return bool(repo.is_ancestor(cached_head, head_sha))
    except Exception:
        return False


def load_or_extract_history(
    extractor: GitHistoryExtractor,
    repo: Repo,
    repo_path: Path,
    branch: str = "HEAD",
    max_commits: Optional[int] = None,
    head_sha: Optional[str] = None,
) -> List[CommitRecord]:
    """Extract commit records, reusing the per-repo cache when possible.

    A bounded run (``max_commits``) is a partial view and neither reads nor
    writes the cache.
    """
    if max_commits:
        return extractor.extract_history(max_commits=max_commits, branch=branch)

    repo_path = Path(repo_path)
    if head_sha is None:
        head_sha = repo.head.commit.hexsha

    cache_path = gitview_dir(repo_path) / HISTORY_CACHE_FILE

    cached: Optional[List[CommitRecord]] = None
    if cache_path.exists():
        try:
            cached = extractor.load_from_jsonl(str(cache_path))
        except Exception:
            cached = None

    records: Optional[List[CommitRecord]] = None
    if cached:
        cached_head = cached[-1].commit_hash
        if cached_head == head_sha:
            records = cached
        elif head_descends_from(repo, cached_head, head_sha):
            try:
                new_records = extractor.extract_incremental(
                    since_commit=cached_head, branch=branch
                )
                # extract_incremental leaves loc_total at 0 for the range;
                # continue the cumulative count from where the cache left off.
                extractor._calculate_cumulative_loc(
                    new_records, starting_loc=cached[-1].loc_total
                )
                records = cached + new_records
            except Exception:
                records = None

    if records is None:
        records = extractor.extract_history(branch=branch)

    # Refresh the cache (best-effort; never fail the caller over it).
    try:
        ensure_gitview_dir(repo_path)
        extractor.save_to_jsonl(records, str(cache_path))
    except Exception:
        pass

    return records
