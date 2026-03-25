"""Worklog command — GitHub-based work log for billing and reporting."""

import csv
import io
import sys
from typing import Optional

import click

from .base import BaseCommand
from ..github_graphql import (
    GitHubGraphQLClient,
    GitHubGraphQLError,
    GitHubRateLimitError,
    PullRequest,
    parse_github_url,
)
from ..remote import RemoteRepoHandler

# Branch prefixes that are tool-generated and not useful as "work branch" labels
_TOOL_BRANCH_PREFIXES = ("claude/", "dependabot/", "renovate/", "snyk-")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_owner_repo(repo_spec: str) -> tuple[str, str]:
    """Resolve owner and repo name from a repo spec.

    Accepts:
      - Local path (reads GitHub remote from git config)
      - GitHub shortcut: org/repo
      - Full URL: https://github.com/org/repo
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


def _to_iso(date_str: str, end_of_day: bool = False) -> str:
    """Normalise YYYY-MM-DD to a full ISO-8601 UTC timestamp."""
    if "T" in date_str:
        return date_str
    return f"{date_str}T23:59:59Z" if end_of_day else f"{date_str}T00:00:00Z"


def _list_all_branches(client: GitHubGraphQLClient, owner: str, repo: str) -> list[str]:
    """Return all branch names for a GitHub repository via GraphQL."""
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
    branches = []
    cursor = None
    has_next = True

    while has_next:
        data = client.execute(query, {"owner": owner, "repo": repo, "after": cursor})
        refs = data.get("repository", {}).get("refs", {})
        page_info = refs.get("pageInfo", {})
        branches.extend(n["name"] for n in refs.get("nodes", []) if n)
        has_next = page_info.get("hasNextPage", False)
        cursor = page_info.get("endCursor")

    return branches


def _choose_best_pr(prs: list[PullRequest], default_branch: str) -> Optional[PullRequest]:
    """Prefer merged PRs targeting the default branch, then most-recently merged."""
    if not prs:
        return None

    def sort_key(pr: PullRequest):
        return (pr.merged, pr.base_ref_name == default_branch, pr.merged_at or "")

    return sorted(prs, key=sort_key, reverse=True)[0]


def _work_branch(commit: dict) -> str:
    """Return the meaningful 'work branch' label for a commit.

    Priority:
      1. PR head branch — where the work was actually done
      2. If the commit is on the default branch (no PR): it was committed
         directly there — show the default branch, not an unrelated descendant
      3. Non-tool branches from the set (commit is on a feature branch, no PR)
    """
    pr: Optional[PullRequest] = commit["best_pr"]
    if pr and pr.head_ref_name:
        return pr.head_ref_name

    default = commit["default_branch"]
    branches = commit["branches"]

    # Commit is on the default branch — no PR means it was committed directly
    # to it.  All other branches in the set are merely descendants: noise.
    if default in branches:
        return default

    # Commit is not on the default branch at all (feature branch, no PR).
    # Return the most meaningful branch name.
    candidates = sorted(
        b for b in branches
        if not any(b.startswith(p) for p in _TOOL_BRANCH_PREFIXES)
    )
    return candidates[0] if candidates else sorted(branches)[0]


def _merged_to(commit: dict) -> str:
    """Return the target branch if the commit's PR was merged, else empty string."""
    pr: Optional[PullRequest] = commit["best_pr"]
    if pr and pr.merged and pr.base_ref_name:
        return pr.base_ref_name
    return ""


# ---------------------------------------------------------------------------
# Core worklog builder
# ---------------------------------------------------------------------------

def build_worklog(
    token: str,
    owner: str,
    repo: str,
    since: str,
    until: str,
    author_login: Optional[str] = None,
    progress_callback=None,
) -> list[dict]:
    """Fetch and deduplicate commits across all branches for the given date range.

    Uses GitHubGraphQLClient (with built-in 24 h caching).

    Returns a list of commit dicts sorted by date.  Each dict contains:
      sha, date, author_name, author_login, message,
      branches (set), best_pr (PullRequest|None), default_branch (str)
    """
    client = GitHubGraphQLClient(token)

    def _progress(msg: str):
        if progress_callback:
            progress_callback(msg)

    repo_info = client.get_repository_info(owner, repo)
    default_branch = (repo_info.get("defaultBranchRef") or {}).get("name", "main")

    _progress("Fetching branch list…")
    all_branches = _list_all_branches(client, owner, repo)

    commits_by_sha: dict[str, dict] = {}

    for branch_name in all_branches:
        _progress(f"Scanning {branch_name}…")
        branch_commits = client.get_all_commits_with_prs(
            owner=owner,
            repo=repo,
            branch=branch_name,
            since=since,
            until=until,
            progress_callback=lambda n: _progress(f"  {branch_name}: {n} commits"),
        )

        for cc in branch_commits:
            if author_login:
                login = cc.author.login if cc.author else None
                if login != author_login:
                    continue

            if cc.sha not in commits_by_sha:
                best_pr = _choose_best_pr(cc.associated_prs, default_branch)
                commits_by_sha[cc.sha] = {
                    "sha": cc.sha,
                    "date": cc.committed_date or "",
                    "author_name": (cc.author.name or cc.author.login) if cc.author else "",
                    "author_login": cc.author.login if cc.author else None,
                    "message": cc.message.splitlines()[0] if cc.message else "",
                    "branches": set(),
                    "best_pr": best_pr,
                    "default_branch": default_branch,
                }
            commits_by_sha[cc.sha]["branches"].add(branch_name)

    return sorted(commits_by_sha.values(), key=lambda c: c["date"])


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def render_markdown(commits: list[dict], repo_label: str = "", since: str = "", until: str = "") -> str:
    """Render a clean daily-table markdown work log."""
    lines = []
    header = f"# Work Log — {repo_label}" if repo_label else "# Work Log"
    if since and until:
        header += f"  |  {since[:10]} → {until[:10]}"
    lines.append(header)
    lines.append("")

    current_day = None
    day_rows: list[tuple] = []

    def _flush_day(day: str, rows: list[tuple]):
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
        branch = _work_branch(c)
        merged = _merged_to(c) or "—"
        pr: Optional[PullRequest] = c["best_pr"]
        pr_cell = f"#{pr.number}" if pr else "—"

        if day != current_day:
            if current_day is not None:
                _flush_day(current_day, day_rows)
            current_day = day
            day_rows = []

        day_rows.append((
            f"`{c['sha'][:7]}`",
            time,
            author,
            c["message"].replace("|", "\\|"),
            branch,
            merged,
            pr_cell,
        ))

    if current_day is not None:
        _flush_day(current_day, day_rows)

    return "\n".join(lines)


def render_csv(commits: list[dict]) -> str:
    """Render a CSV work log suitable for billing/import."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "date", "time", "sha", "author",
        "message", "work_branch", "merged_to",
        "pr_number", "pr_title", "pr_merged_at",
    ])
    for c in commits:
        pr: Optional[PullRequest] = c["best_pr"]
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
        if not self.get_option("github_token"):
            raise click.UsageError(
                "A GitHub token is required. Pass --github-token or set GITHUB_TOKEN."
            )
        if not self.get_option("since"):
            raise click.UsageError("--since is required (start date, YYYY-MM-DD).")
        if not self.get_option("until"):
            raise click.UsageError("--until is required (end date, YYYY-MM-DD).")

    def execute(self):
        token = self.get_option("github_token")
        repo_spec = self.get_option("repo", ".")
        since = _to_iso(self.get_option("since"))
        until = _to_iso(self.get_option("until"), end_of_day=True)
        author = self.get_option("author")
        fmt = self.get_option("fmt", "markdown")
        output = self.get_option("output")

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
            task = progress.add_task("Fetching commits…", total=None)

            def update(msg: str):
                progress.update(task, description=msg)

            try:
                commits = build_worklog(
                    token=token,
                    owner=owner,
                    repo=repo_name,
                    since=since,
                    until=until,
                    author_login=author,
                    progress_callback=update,
                )
            except GitHubRateLimitError as exc:
                self.print_error(f"GitHub rate limit exceeded. Resets at {exc.reset_at}")
                sys.exit(1)
            except GitHubGraphQLError as exc:
                self.print_error(f"GitHub API error: {exc}")
                sys.exit(1)

        self.print_success(f"Found {len(commits)} unique commit(s).")

        if fmt == "csv":
            content = render_csv(commits)
            default_filename = f"worklog_{owner}_{repo_name}_{since[:10]}_{until[:10]}.csv"
        else:
            content = render_markdown(
                commits, repo_label=repo_label,
                since=since, until=until,
            )
            default_filename = f"worklog_{owner}_{repo_name}_{since[:10]}_{until[:10]}.md"

        if output:
            dest = output
        elif not sys.stdout.isatty():
            print(content)
            return
        else:
            dest = default_filename

        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(content)
        self.print_success(f"Work log written to: {dest}")
