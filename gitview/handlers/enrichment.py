"""GitHub enrichment handler.

Enriches commit records with GitHub PR/review context using the
GitHub GraphQL API.
"""

from typing import Optional
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..analyzer import AnalysisContext
from .base import BaseHandler, HandlerError


class GitHubEnrichmentHandler(BaseHandler):
    """Handles GitHub PR/review enrichment.

    This handler enriches commits with additional context from GitHub:
    - Pull request titles and descriptions
    - Review comments and feedback
    - Reviewer attribution
    - PR labels and metadata
    """

    def should_execute(self, context: AnalysisContext) -> bool:
        """Only execute if GitHub token is provided and records exist.

        Args:
            context: Analysis context

        Returns:
            True if handler should execute
        """
        return (self.config.github_token is not None and
                len(context.records) > 0)

    def execute(self, context: AnalysisContext) -> None:
        """Enrich commits with GitHub context.

        Args:
            context: Analysis context with records to enrich

        Raises:
            HandlerError: If enrichment fails critically (non-fatal)
        """
        self._log_bold("Step 1.5: Enriching with GitHub context...")

        # 1. Determine GitHub repo URL
        github_repo_url = self._determine_github_url()

        if not github_repo_url:
            self._log_warning("Could not determine GitHub repository URL")
            self._log_warning("GitHub enrichment skipped\n")
            return

        # 2. Fetch GitHub contexts
        try:
            github_contexts = self._fetch_github_contexts(context, github_repo_url)

            # 3. Attach contexts to records
            self._attach_contexts(context, github_contexts)

            # 4. Re-save with GitHub context
            self._save_enriched_history(context)

            self._log_success(
                f"Enriched {context.enriched_commit_count} commits with GitHub PR context\n"
            )

        except Exception as e:
            # GitHub enrichment is non-fatal - log and continue
            self._log_warning(f"GitHub enrichment failed: {e}")
            self._log_warning("Continuing without GitHub context...\n")

    def _determine_github_url(self) -> Optional[str]:
        """Determine GitHub repo URL from config or repo remotes.

        Returns:
            GitHub repo URL in format "owner/repo" or None
        """
        # If provided directly, use it
        if self.config.github_repo_url:
            return self.config.github_repo_url

        # Try to detect from git remotes
        try:
            from git import Repo as GitRepo
            git_repo = GitRepo(str(self.config.repo_path))

            for remote in git_repo.remotes:
                for url in remote.urls:
                    if 'github.com' in url:
                        from ..github_graphql import parse_github_url
                        owner, repo_name = parse_github_url(url)
                        return f"{owner}/{repo_name}"
        except Exception:
            pass

        return None

    def _fetch_github_contexts(self, context: AnalysisContext, repo_url: str) -> dict:
        """Fetch GitHub contexts for commits.

        Args:
            context: Analysis context with commits
            repo_url: GitHub repository URL (owner/repo)

        Returns:
            Dictionary mapping commit hashes to GitHub contexts
        """
        from ..github_enricher import enrich_commits_with_github

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Fetching GitHub PR/review data...", total=None)

            github_contexts = enrich_commits_with_github(
                commits=context.records,
                github_token=self.config.github_token,
                repo_url=repo_url,
                branch=self.config.branch,
            )

            progress.update(task, completed=True)

        return github_contexts

    def _attach_contexts(self, context: AnalysisContext, github_contexts: dict) -> None:
        """Attach GitHub contexts to commit records.

        Args:
            context: Analysis context with records
            github_contexts: Dictionary of GitHub contexts by commit hash
        """
        context.enriched_commit_count = 0

        for record in context.records:
            if record.commit_hash in github_contexts:
                ctx = github_contexts[record.commit_hash]
                record.github_context = ctx.to_dict()
                if ctx.pr_number:
                    context.enriched_commit_count += 1

    def _save_enriched_history(self, context: AnalysisContext) -> None:
        """Re-save history with GitHub context attached.

        Args:
            context: Analysis context with enriched records
        """
        from ..extractor import GitHistoryExtractor

        extractor = GitHistoryExtractor(str(self.config.repo_path))
        extractor.save_to_jsonl(context.records, str(context.history_file))
