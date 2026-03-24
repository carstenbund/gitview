"""GitHub REST API client for worklog generation.

Fetches commits across all branches with deduplication and PR association,
intended for billing/work-log reporting over a date range.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import requests

API_VERSION = "2022-11-28"


@dataclass
class PRInfo:
    number: int
    title: str
    state: str
    merged_at: Optional[str]
    html_url: str
    base_ref: str
    head_ref: str


@dataclass
class WorklogCommit:
    sha: str
    date: str
    author_name: str
    author_login: Optional[str]
    message: str
    html_url: str
    branches: set = field(default_factory=set)
    prs: list = field(default_factory=list)
    best_pr: Optional[PRInfo] = None


class GitHubWorklogClient:
    """REST API client for worklog data — commits, branches, and PRs."""

    def __init__(self, token: str, owner: str, repo: str):
        self.owner = owner
        self.repo = repo
        self.base = f"https://api.github.com/repos/{owner}/{repo}"
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": API_VERSION,
        })

    def _get_all_pages(self, url: str, params: Optional[dict] = None) -> list:
        results = []
        while url:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                results.extend(data)
            else:
                return data
            url = resp.links.get("next", {}).get("url")
            params = None  # pagination cursor is in the Link header URL
        return results

    def repo_info(self) -> dict:
        resp = self.session.get(self.base, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def list_branches(self) -> list[dict]:
        return self._get_all_pages(f"{self.base}/branches", params={"per_page": 100})

    def list_commits(self, branch: str, since: str, until: str) -> list[dict]:
        return self._get_all_pages(
            f"{self.base}/commits",
            params={
                "sha": branch,
                "since": since,
                "until": until,
                "per_page": 100,
            },
        )

    def list_prs_for_commit(self, sha: str) -> list[dict]:
        return self._get_all_pages(
            f"{self.base}/commits/{sha}/pulls",
            params={"per_page": 100},
        )


def _choose_best_pr(prs: list[PRInfo], default_branch: str) -> Optional[PRInfo]:
    """Prefer merged PRs targeting the default branch, then most-recently merged."""
    if not prs:
        return None

    def sort_key(pr: PRInfo):
        return (pr.merged_at is not None, pr.base_ref == default_branch, pr.merged_at or "")

    return sorted(prs, key=sort_key, reverse=True)[0]


def build_worklog(
    token: str,
    owner: str,
    repo: str,
    since: str,
    until: str,
    author_login: Optional[str] = None,
    progress_callback=None,
) -> list[WorklogCommit]:
    """Fetch and deduplicate commits across all branches for the given date range.

    Args:
        token: GitHub personal access token.
        owner: Repository owner/organisation.
        repo: Repository name.
        since: ISO-8601 start datetime (e.g. "2024-01-01T00:00:00Z").
        until: ISO-8601 end datetime (e.g. "2024-01-31T23:59:59Z").
        author_login: Filter to a specific GitHub login (optional).
        progress_callback: Optional callable(message: str) for status updates.

    Returns:
        Commits sorted by date, deduplicated by SHA, with branch and PR info.
    """
    gh = GitHubWorklogClient(token, owner, repo)

    info = gh.repo_info()
    default_branch = info["default_branch"]

    def _progress(msg: str):
        if progress_callback:
            progress_callback(msg)

    # --- Step 1: collect commits across all branches, deduplicate by SHA ---
    _progress("Fetching branch list…")
    branches = gh.list_branches()
    commits_by_sha: dict[str, WorklogCommit] = {}

    for branch in branches:
        branch_name = branch["name"]
        _progress(f"Scanning branch: {branch_name}")
        for raw in gh.list_commits(branch_name, since, until):
            sha = raw["sha"]
            github_author = raw.get("author")
            login = github_author["login"] if github_author else None

            if author_login and login != author_login:
                continue

            if sha not in commits_by_sha:
                commit_data = raw["commit"]
                commits_by_sha[sha] = WorklogCommit(
                    sha=sha,
                    date=commit_data["author"]["date"],
                    author_name=commit_data["author"]["name"],
                    author_login=login,
                    message=commit_data["message"].splitlines()[0],
                    html_url=raw["html_url"],
                )
            commits_by_sha[sha].branches.add(branch_name)

    # --- Step 2: enrich each unique commit with PR info ---
    total = len(commits_by_sha)
    _progress(f"Fetching PR associations for {total} unique commits…")

    for i, commit in enumerate(commits_by_sha.values(), 1):
        _progress(f"PR lookup {i}/{total}: {commit.sha[:7]}")
        try:
            raw_prs = gh.list_prs_for_commit(commit.sha)
            commit.prs = [
                PRInfo(
                    number=pr["number"],
                    title=pr["title"],
                    state=pr["state"],
                    merged_at=pr.get("merged_at"),
                    html_url=pr["html_url"],
                    base_ref=pr["base"]["ref"],
                    head_ref=pr["head"]["ref"],
                )
                for pr in raw_prs
            ]
            commit.best_pr = _choose_best_pr(commit.prs, default_branch)
        except requests.HTTPError:
            # Non-fatal: some commits may have no PR data accessible
            pass
        time.sleep(0.05)  # gentle rate-limiting

    return sorted(commits_by_sha.values(), key=lambda c: c.date)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_markdown(commits: list[WorklogCommit], repo_label: str = "") -> str:
    """Render a human-readable markdown worklog grouped by day."""
    lines = []
    if repo_label:
        lines.append(f"# Work Log — {repo_label}\n")

    current_day = None
    for c in commits:
        day = c.date[:10]
        if day != current_day:
            if lines:
                lines.append("")
            lines.append(f"## {day}")
            current_day = day

        author = c.author_login or c.author_name
        branches = ", ".join(sorted(c.branches))

        lines.append(f"- **{c.message}**")
        lines.append(f"  - author: `{author}`")
        lines.append(f"  - sha: `{c.sha[:7]}`")
        lines.append(f"  - branches: {branches}")
        lines.append(f"  - commit: {c.html_url}")

        if c.best_pr:
            pr = c.best_pr
            merged = pr.merged_at or "not merged"
            lines.append(f"  - pr: [#{pr.number} {pr.title}]({pr.html_url})")
            lines.append(f"  - pr state: {pr.state}")
            lines.append(f"  - base ← head: `{pr.base_ref}` ← `{pr.head_ref}`")
            lines.append(f"  - merged at: {merged}")

    return "\n".join(lines)


def render_csv(commits: list[WorklogCommit]) -> str:
    """Render a CSV worklog (one row per commit)."""
    import csv
    import io

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "date", "sha", "author", "message", "branches",
        "pr_number", "pr_title", "pr_state", "pr_merged_at",
        "pr_url", "commit_url",
    ])
    for c in commits:
        branches = "|".join(sorted(c.branches))
        pr = c.best_pr
        writer.writerow([
            c.date, c.sha[:7], c.author_login or c.author_name,
            c.message, branches,
            pr.number if pr else "",
            pr.title if pr else "",
            pr.state if pr else "",
            pr.merged_at if pr else "",
            pr.html_url if pr else "",
            c.html_url,
        ])
    return buf.getvalue()
