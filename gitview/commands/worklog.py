"""Worklog command — GitHub-based work log for billing and reporting."""

import csv
import io
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

import click

from .base import BaseCommand
from ..cache import CacheManager
from ..extractor import CommitRecord
from ..github_graphql import (
    GitHubGraphQLClient,
    GitHubGraphQLError,
    GitHubRateLimitError,
    PullRequest,
    parse_github_url,
)
from ..remote import RemoteRepoHandler

# Tool-generated branch prefixes to ignore when resolving "work branch"
_TOOL_BRANCH_PREFIXES = ("claude/", "dependabot/", "renovate/", "snyk-")


# ---------------------------------------------------------------------------
# Minimal PR summary for cache-sourced commits (mirrors PullRequest fields)
# ---------------------------------------------------------------------------

@dataclass
class _PRSummary:
    number: int
    title: str
    state: str
    merged: bool
    merged_at: Optional[str]
    head_ref_name: Optional[str]
    base_ref_name: Optional[str]


# ---------------------------------------------------------------------------
# Token auto-discovery
# ---------------------------------------------------------------------------

def _resolve_token(explicit: Optional[str] = None) -> Optional[str]:
    """Discover a GitHub token without requiring the user to supply it.

    Search order:
      1. Explicitly passed value (CLI flag)
      2. GITHUB_TOKEN environment variable
      3. GH_TOKEN environment variable (GitHub CLI convention)
      4. `gh auth token` (GitHub CLI, if installed and authenticated)
      5. git credential helper for github.com
    """
    if explicit:
        return explicit

    for var in ("GITHUB_TOKEN", "GH_TOKEN"):
        val = os.environ.get(var)
        if val:
            return val

    # Try GitHub CLI
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True, text=True, timeout=5
        )
        token = result.stdout.strip()
        if token:
            return token
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Try git credential helper for api.github.com
    try:
        result = subprocess.run(
            ["git", "credential", "fill"],
            input="protocol=https\nhost=github.com\n\n",
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if line.startswith("password="):
                return line[9:]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


# ---------------------------------------------------------------------------
# Repo / owner resolution
# ---------------------------------------------------------------------------

def _resolve_owner_repo(repo_spec: str) -> tuple[str, str]:
    """Resolve owner and repo name from a repo spec.

    Accepts a local path (reads GitHub remote from git config),
    a GitHub shortcut org/repo, or a full URL.
    """
    handler = RemoteRepoHandler(repo_spec)

    if handler.is_local:
        try:
            import git
            local_repo = git.Repo(repo_spec, search_parent_directories=True)
            for remote in local_repo.remotes:
                try:
                    return parse_github_url(remote.url)
                except ValueError:
                    continue
        except Exception:
            pass
        raise click.UsageError(
            f"Could not determine GitHub owner/repo from local path '{repo_spec}'. "
            "Pass an explicit repo spec: --repo owner/repo"
        )

    return handler.repo_info.org, handler.repo_info.repo


def _local_repo_path(repo_spec: str) -> Optional[str]:
    """Return the local filesystem path for the repo spec, or None if remote."""
    handler = RemoteRepoHandler(repo_spec)
    if handler.is_local:
        try:
            import git
            r = git.Repo(repo_spec, search_parent_directories=True)
            return str(r.working_dir)
        except Exception:
            return repo_spec
    return None


def _to_iso(date_str: str, end_of_day: bool = False) -> str:
    """Normalise YYYY-MM-DD to a full ISO-8601 UTC timestamp."""
    if "T" in date_str:
        return date_str
    return f"{date_str}T23:59:59Z" if end_of_day else f"{date_str}T00:00:00Z"


# ---------------------------------------------------------------------------
# Cached-commit helpers
# ---------------------------------------------------------------------------

def _pr_from_context(ctx: dict) -> Optional[_PRSummary]:
    """Build a minimal PR summary from a CommitRecord's github_context dict."""
    if not ctx or not ctx.get("pr_number"):
        return None
    return _PRSummary(
        number=ctx["pr_number"],
        title=ctx.get("pr_title") or "",
        state=ctx.get("pr_state") or "",
        merged=ctx.get("pr_merged", False),
        merged_at=None,  # not stored in GitHubContext
        head_ref_name=ctx.get("source_branch"),
        base_ref_name=ctx.get("target_branch"),
    )


def _worklog_from_cache(
    records: list[CommitRecord],
    since: str,
    until: str,
    author_filter: Optional[str],
) -> list[dict]:
    """Build worklog entries from cached CommitRecords filtered by date range."""
    commits = []
    for r in records:
        ts = r.timestamp
        if ts < since or ts > until:
            continue
        if author_filter and author_filter.lower() not in (r.author or "").lower():
            continue

        ctx = r.github_context or {}
        commits.append({
            "sha": r.commit_hash,
            "date": ts,
            "author_name": r.author,
            "author_login": None,
            "message": r.commit_subject or r.commit_message.splitlines()[0],
            "branches": set(),          # not tracked per-commit in local cache
            "best_pr": _pr_from_context(ctx),
            "default_branch": "main",   # used only for fallback branch label
            "_source": "cache",
        })
    return sorted(commits, key=lambda c: c["date"])


# ---------------------------------------------------------------------------
# Live GitHub API helpers
# ---------------------------------------------------------------------------

def _list_all_branches(client: GitHubGraphQLClient, owner: str, repo: str) -> list[str]:
    query = """
    query($owner: String!, $repo: String!, $after: String) {
        repository(owner: $owner, name: $repo) {
            refs(refPrefix: "refs/heads/", first: 100, after: $after) {
                pageInfo { hasNextPage endCursor }
                nodes { name }
            }
        }
    }
    """
    branches, cursor, has_next = [], None, True
    while has_next:
        data = client.execute(query, {"owner": owner, "repo": repo, "after": cursor})
        refs = data.get("repository", {}).get("refs", {})
        branches.extend(n["name"] for n in refs.get("nodes", []) if n)
        has_next = refs.get("pageInfo", {}).get("hasNextPage", False)
        cursor = refs.get("pageInfo", {}).get("endCursor")
    return branches


def _choose_best_pr(prs: list[PullRequest], default_branch: str) -> Optional[PullRequest]:
    if not prs:
        return None
    return sorted(
        prs,
        key=lambda pr: (pr.merged, pr.base_ref_name == default_branch, pr.merged_at or ""),
        reverse=True,
    )[0]


def _worklog_from_api(
    token: str,
    owner: str,
    repo: str,
    since: str,
    until: str,
    author_login: Optional[str],
    progress_callback,
) -> list[dict]:
    """Fetch deduplicated commits across all branches via GitHub GraphQL."""

    def _progress(msg: str):
        if progress_callback:
            progress_callback(msg)

    client = GitHubGraphQLClient(token)
    repo_info = client.get_repository_info(owner, repo)
    default_branch = (repo_info.get("defaultBranchRef") or {}).get("name", "main")

    _progress("Fetching branch list…")
    all_branches = _list_all_branches(client, owner, repo)

    commits_by_sha: dict[str, dict] = {}
    for branch_name in all_branches:
        _progress(f"Scanning {branch_name}…")
        for cc in client.get_all_commits_with_prs(
            owner=owner, repo=repo, branch=branch_name,
            since=since, until=until,
            progress_callback=lambda n: _progress(f"  {branch_name}: {n} commits"),
        ):
            if author_login and (not cc.author or cc.author.login != author_login):
                continue
            if cc.sha not in commits_by_sha:
                commits_by_sha[cc.sha] = {
                    "sha": cc.sha,
                    "date": cc.committed_date or "",
                    "author_name": (cc.author.name or cc.author.login) if cc.author else "",
                    "author_login": cc.author.login if cc.author else None,
                    "message": cc.message.splitlines()[0] if cc.message else "",
                    "branches": set(),
                    "best_pr": _choose_best_pr(cc.associated_prs, default_branch),
                    "default_branch": default_branch,
                    "_source": "api",
                }
            commits_by_sha[cc.sha]["branches"].add(branch_name)

    return sorted(commits_by_sha.values(), key=lambda c: c["date"])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_worklog(
    token: Optional[str],
    owner: str,
    repo: str,
    since: str,
    until: str,
    author_login: Optional[str] = None,
    output_dir: Optional[str] = None,
    repo_path: Optional[str] = None,
    progress_callback=None,
) -> tuple[list[dict], str]:
    """Return (commits, source) where source is 'cache' or 'api'."""

    def _progress(msg: str):
        if progress_callback:
            progress_callback(msg)

    # 1. Try the local analysis cache (repo_history.jsonl from a prior run)
    cache_dir = output_dir or (os.path.join(repo_path, "output") if repo_path else "output")
    cache = CacheManager(cache_dir, repo_path)
    cached_records = cache.load_commit_history()
    if cached_records:
        _progress(f"Loading from cache ({len(cached_records)} commits)…")
        commits = _worklog_from_cache(cached_records, since, until, author_login)
        if commits:
            return commits, "cache"
        _progress("Cache hit but no commits in date range — querying GitHub API…")

    # 2. Fall back to live GitHub API
    if not token:
        raise click.UsageError(
            "No GitHub token found. Set GITHUB_TOKEN, run `gh auth login`, "
            "or pass --github-token."
        )
    commits = _worklog_from_api(token, owner, repo, since, until, author_login, progress_callback)
    return commits, "api"


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _work_branch(commit: dict) -> str:
    """Return the meaningful 'work branch' label for a commit."""
    pr = commit["best_pr"]
    if pr and pr.head_ref_name:
        return pr.head_ref_name

    default = commit["default_branch"]
    branches = commit["branches"]

    # Commit is on the default branch with no PR — it was committed directly there.
    if default in branches:
        return default

    # Feature branch commit with no PR — show the branch, skip tool branches.
    candidates = sorted(
        b for b in branches
        if not any(b.startswith(p) for p in _TOOL_BRANCH_PREFIXES)
    )
    return candidates[0] if candidates else (sorted(branches)[0] if branches else default)


def _merged_to(commit: dict) -> str:
    pr = commit["best_pr"]
    if pr and pr.merged and pr.base_ref_name:
        return pr.base_ref_name
    return ""


def render_markdown(
    commits: list[dict],
    repo_label: str = "",
    since: str = "",
    until: str = "",
    source: str = "",
) -> str:
    lines = []
    header = f"# Work Log — {repo_label}" if repo_label else "# Work Log"
    if since and until:
        header += f"  |  {since[:10]} → {until[:10]}"
    if source:
        header += f"  *(source: {source})*"
    lines.append(header)
    lines.append("")

    current_day = None
    day_rows: list[tuple] = []

    def _flush(day: str, rows: list[tuple]):
        lines.append(f"## {day}")
        lines.append("")
        lines.append("| SHA | Time | Author | Message | Branch | Merged to | PR |")
        lines.append("|-----|------|--------|---------|--------|-----------|----|")
        for row in rows:
            lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(*row))
        lines.append("")

    for c in commits:
        day = c["date"][:10]
        time = c["date"][11:16] if len(c["date"]) >= 16 else ""
        author = c["author_login"] or c["author_name"]
        pr = c["best_pr"]
        pr_cell = f"#{pr.number}" if pr else "—"

        if day != current_day:
            if current_day is not None:
                _flush(current_day, day_rows)
            current_day, day_rows = day, []

        day_rows.append((
            f"`{c['sha'][:7]}`",
            time,
            author,
            c["message"].replace("|", "\\|"),
            _work_branch(c),
            _merged_to(c) or "—",
            pr_cell,
        ))

    if current_day is not None:
        _flush(current_day, day_rows)

    return "\n".join(lines)


def render_csv(commits: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "date", "time", "sha", "author",
        "message", "work_branch", "merged_to",
        "pr_number", "pr_title", "pr_merged_at",
    ])
    for c in commits:
        pr = c["best_pr"]
        writer.writerow([
            c["date"][:10],
            c["date"][11:19] if len(c["date"]) >= 19 else "",
            c["sha"][:7],
            c["author_login"] or c["author_name"],
            c["message"],
            _work_branch(c),
            _merged_to(c),
            pr.number if pr else "",
            pr.title if pr else "",
            pr.merged_at if pr else "",
        ])
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------

class WorklogCommand(BaseCommand):
    """Generate a work log from GitHub commit history across all branches."""

    def validate(self) -> None:
        if not self.get_option("since"):
            raise click.UsageError("--since is required (start date, YYYY-MM-DD).")
        if not self.get_option("until"):
            raise click.UsageError("--until is required (end date, YYYY-MM-DD).")

    def execute(self):
        repo_spec = self.get_option("repo", ".")
        since = _to_iso(self.get_option("since"))
        until = _to_iso(self.get_option("until"), end_of_day=True)
        author = self.get_option("author")
        fmt = self.get_option("fmt", "markdown")
        output_dir = self.get_option("output", "output")   # same convention as analyze

        token = _resolve_token(self.get_option("github_token"))
        repo_path = _local_repo_path(repo_spec)

        try:
            owner, repo_name = _resolve_owner_repo(repo_spec)
        except (ValueError, click.UsageError) as exc:
            self.print_error(str(exc))
            sys.exit(1)

        repo_label = f"{owner}/{repo_name}"
        self.print_header(
            f"GitHub Work Log — {repo_label}  |  {since[:10]} → {until[:10]}"
        )
        if author:
            self.print_info(f"Filtering by author: {author}")

        with self.create_progress() as progress:
            task = progress.add_task("Loading commits…", total=None)

            def update(msg: str):
                progress.update(task, description=msg)

            try:
                commits, source = build_worklog(
                    token=token,
                    owner=owner,
                    repo=repo_name,
                    since=since,
                    until=until,
                    author_login=author,
                    output_dir=output_dir,
                    repo_path=repo_path,
                    progress_callback=update,
                )
            except GitHubRateLimitError as exc:
                self.print_error(f"GitHub rate limit exceeded. Resets at {exc.reset_at}")
                sys.exit(1)
            except GitHubGraphQLError as exc:
                self.print_error(f"GitHub API error: {exc}")
                sys.exit(1)
            except click.UsageError as exc:
                self.print_error(str(exc))
                sys.exit(1)

        self.print_success(
            f"Found {len(commits)} unique commit(s)  [source: {source}]"
        )

        ext = "csv" if fmt == "csv" else "md"
        filename = f"worklog_{owner}_{repo_name}_{since[:10]}_{until[:10]}.{ext}"

        if fmt == "csv":
            content = render_csv(commits)
        else:
            content = render_markdown(
                commits, repo_label=repo_label,
                since=since, until=until, source=source,
            )

        if not sys.stdout.isatty():
            print(content)
            return

        from pathlib import Path
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        dest = out_path / filename

        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(content)
        self.print_success(f"Work log written to: {dest}")
