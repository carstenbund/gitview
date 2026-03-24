"""Worklog command — GitHub-based work log for billing and reporting."""

import os
import sys
from datetime import datetime, timezone

import click
import requests

from .base import BaseCommand
from ..github_worklog import build_worklog, render_markdown, render_csv


def _to_iso(date_str: str, end_of_day: bool = False) -> str:
    """Normalise a YYYY-MM-DD string to a full ISO-8601 UTC timestamp."""
    if "T" in date_str:
        return date_str  # already a full timestamp
    if end_of_day:
        return f"{date_str}T23:59:59Z"
    return f"{date_str}T00:00:00Z"


class WorklogCommand(BaseCommand):
    """Generate a work log from GitHub commit history across all branches.

    Fetches every commit visible on any branch within the date range, deduplicates
    by SHA so merged commits are only counted once, resolves the best associated
    PR for each commit, and renders the result as Markdown or CSV.
    """

    def validate(self) -> None:
        if not self.get_option("github_token"):
            raise click.UsageError(
                "A GitHub token is required. Pass --github-token or set GITHUB_TOKEN."
            )
        if not self.get_option("owner"):
            raise click.UsageError("--owner is required (GitHub organisation or user).")
        if not self.get_option("repo"):
            raise click.UsageError("--repo is required (repository name).")
        if not self.get_option("since"):
            raise click.UsageError("--since is required (start date, YYYY-MM-DD).")
        if not self.get_option("until"):
            raise click.UsageError("--until is required (end date, YYYY-MM-DD).")

    def execute(self):
        token = self.get_option("github_token")
        owner = self.get_option("owner")
        repo = self.get_option("repo")
        since = _to_iso(self.get_option("since"), end_of_day=False)
        until = _to_iso(self.get_option("until"), end_of_day=True)
        author = self.get_option("author")
        fmt = self.get_option("fmt", "markdown")
        output = self.get_option("output")

        self.print_header(
            f"GitHub Work Log — {owner}/{repo}  |  {since[:10]} → {until[:10]}"
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
                    repo=repo,
                    since=since,
                    until=until,
                    author_login=author,
                    progress_callback=update,
                )
            except requests.HTTPError as exc:
                self.print_error(f"GitHub API error: {exc}")
                sys.exit(1)

        self.print_success(f"Found {len(commits)} unique commit(s) across all branches.")

        # Render
        repo_label = f"{owner}/{repo}"
        if fmt == "csv":
            content = render_csv(commits)
            default_filename = f"worklog_{owner}_{repo}_{since[:10]}_{until[:10]}.csv"
        else:
            content = render_markdown(commits, repo_label=repo_label)
            default_filename = f"worklog_{owner}_{repo}_{since[:10]}_{until[:10]}.md"

        if output:
            dest = output
        elif not sys.stdout.isatty():
            # Piped output — write to stdout without decoration
            print(content)
            return
        else:
            dest = default_filename

        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(content)
        self.print_success(f"Work log written to: {dest}")
