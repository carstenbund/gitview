"""Brief command — a compiled-once, agent-readable project history digest.

Unlike 'analyze', this never calls an LLM. It is meant to be generated once
(and regenerated cheaply after a batch of new commits) and then just read —
a substitute for an AI coding agent re-deriving project history from
`git log` and file exploration at the start of every session.
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from git import Repo

from .base import BaseCommand
from .. import __version__
from ..extractor import CommitRecord, GitHistoryExtractor
from ..chunker import HistoryChunker, Phase
from ..significance_analyzer import SignificanceAnalyzer
from ..storyline import StorylineReporter, StorylineTracker

_LOOPBACK_MARKERS = ("127.0.0.1", "localhost", "0.0.0.0", "local_proxy")


def _looks_like_public_remote(url: str) -> bool:
    """Filter out proxy/loopback rewrites some sandboxed environments apply to 'origin'."""
    return not any(marker in url for marker in _LOOPBACK_MARKERS)


STAMP_PATTERN = re.compile(
    r'<!--\s*gitview:brief\s+head=(?P<head>[0-9a-f]+)\s+generated=(?P<generated>\S+)\s+'
    r'commits=(?P<commits>\d+)\s+gitview_version=(?P<version>\S+)\s*-->'
)

HEADING_PATTERN = re.compile(r'(?m)^(#+)( .*)$')


# ---------------------------------------------------------------------------
# Repo identity
# ---------------------------------------------------------------------------

def _find_readme(repo_path: Path) -> Optional[Path]:
    """Find a README file using common naming conventions."""
    for name in ("README.md", "README.rst", "README.txt", "README"):
        candidate = repo_path / name
        if candidate.exists():
            return candidate
    return None


def _first_paragraph(readme_path: Path, max_len: int = 400) -> str:
    """Best-effort tagline: first substantial line, skipping headings/badges."""
    try:
        lines = readme_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return ""

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("[!["):
            continue
        text = stripped.strip("*_ ")
        if len(text) > 20:
            return text[:max_len]

    return ""


def _detect_repo_identity(repo_path: Path) -> Dict[str, str]:
    """Best-effort project name/description from common manifest files."""
    name = repo_path.resolve().name
    description = ""

    pyproject = repo_path / "pyproject.toml"
    if pyproject.exists():
        text = pyproject.read_text(encoding="utf-8", errors="ignore")
        name_match = re.search(r'(?m)^\s*name\s*=\s*"([^"]+)"', text)
        desc_match = re.search(r'(?m)^\s*description\s*=\s*"([^"]+)"', text)
        if name_match:
            name = name_match.group(1)
        if desc_match:
            description = desc_match.group(1)

    if not description:
        package_json = repo_path / "package.json"
        if package_json.exists():
            try:
                data = json.loads(package_json.read_text(encoding="utf-8", errors="ignore"))
                name = data.get("name", name)
                description = data.get("description", description)
            except (json.JSONDecodeError, OSError):
                pass

    if not description:
        readme = _find_readme(repo_path)
        if readme:
            description = _first_paragraph(readme)

    return {"name": name, "description": description}


# ---------------------------------------------------------------------------
# Data gathering (all deterministic — no LLM calls)
# ---------------------------------------------------------------------------

def _phase_highlight(phase: Phase, analyzer: SignificanceAnalyzer) -> str:
    """One-line label for a phase, from its most significant commit cluster."""
    if not phase.commits:
        return "—"

    clusters = analyzer.cluster_commits(phase.commits)
    if not clusters:
        return "—"

    dominant = max(clusters, key=lambda c: len(c.commits))
    key_commit = dominant.key_commit
    text = key_commit.get_pr_title() or key_commit.commit_subject
    return text[:70] + "…" if len(text) > 70 else text


def _top_churn_files(records: List[CommitRecord], limit: int = 15) -> List[Tuple[str, int]]:
    """Files touched by the most commits — a language-agnostic proxy for
    where the architecture is concentrated."""
    counts: Dict[str, int] = {}
    for record in records:
        for file_path in record.files_stats.keys():
            counts[file_path] = counts.get(file_path, 0) + 1
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:limit]


def _all_contributors(records: List[CommitRecord]) -> List[Tuple[str, int]]:
    """All authors sorted by commit count, descending."""
    counts: Dict[str, int] = {}
    for record in records:
        counts[record.author] = counts.get(record.author, 0) + 1
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)


def _language_mix(records: List[CommitRecord]) -> List[Tuple[str, float]]:
    """Language breakdown snapshot at the most recent commit."""
    if not records:
        return []
    breakdown = records[-1].language_breakdown or {}
    total = sum(breakdown.values())
    if not total:
        return []
    return sorted(
        [(lang, count / total) for lang, count in breakdown.items()],
        key=lambda x: x[1],
        reverse=True,
    )


def _build_storyline_reporter(phases: List[Phase]) -> StorylineReporter:
    """Run non-LLM multi-signal storyline detection over phases, in-memory only."""
    tracker = StorylineTracker(persist_path=None, confidence_threshold=0.6, min_signals=1)
    for phase in phases:
        tracker.process_phase(phase, llm_storylines=None)
    return StorylineReporter(database=tracker.database)


def _increase_heading_level(markdown: str, levels: int) -> str:
    """Shift ATX heading levels down so pre-rendered fragments nest correctly."""
    return HEADING_PATTERN.sub(lambda m: "#" * (len(m.group(1)) + levels) + m.group(2), markdown)


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------

def _read_stamp(output_path: Path) -> Optional[Dict[str, str]]:
    """Read the head-SHA stamp from a previously generated brief, if any."""
    if not output_path.exists():
        return None
    try:
        head_chunk = output_path.read_text(encoding="utf-8", errors="ignore")[:500]
    except OSError:
        return None
    match = STAMP_PATTERN.search(head_chunk)
    return match.groupdict() if match else None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_brief(
    repo_path: Path,
    records: List[CommitRecord],
    phases: List[Phase],
    head_sha: str,
    is_shallow: bool,
    remote_url: Optional[str],
    top_files_limit: int = 15,
) -> str:
    """Render the full agent-brief markdown document."""
    identity = _detect_repo_identity(repo_path)
    analyzer = SignificanceAnalyzer()
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    lines: List[str] = []
    lines.append(
        f"<!-- gitview:brief head={head_sha} generated={generated_at} "
        f"commits={len(records)} gitview_version={__version__} -->"
    )
    lines.append(f"# Agent Brief — {identity['name']}")
    lines.append("")
    lines.append(
        "> Auto-generated, token-efficient history digest for AI coding agents.\n"
        "> Read this once at the start of a session instead of re-deriving project\n"
        "> history from `git log` and file exploration.\n"
        ">\n"
        "> Regenerate after new commits: `gitview brief` (skips automatically if\n"
        "> already up to date; add `--force` to always regenerate). Check freshness\n"
        "> only, no write: `gitview brief --check`."
    )
    lines.append("")
    lines.append(
        f"**Generated:** {generated_at} at commit `{head_sha[:8]}` "
        f"({len(records)} commits analyzed)"
    )
    if (repo_path / "AGENTS.md").exists():
        lines.append("")
        lines.append(
            "For a hand-maintained architecture map and known-gaps list, "
            "see [AGENTS.md](AGENTS.md)."
        )
    lines.append("")
    lines.append("---")
    lines.append("")

    # Project identity
    lines.append("## Project")
    lines.append("")
    lines.append(f"- **Name:** {identity['name']}")
    if identity["description"]:
        lines.append(f"- **Description:** {identity['description']}")
    if remote_url:
        lines.append(f"- **Remote:** {remote_url}")
    lines.append("")

    # At a glance
    contributors = _all_contributors(records)
    lines.append("## At a Glance")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    shallow_note = " (⚠ shallow clone — full history is longer)" if is_shallow else ""
    lines.append(f"| Commits analyzed | {len(records)}{shallow_note} |")
    lines.append(f"| Contributors | {len(contributors)} |")
    if records:
        lines.append(f"| First commit | {records[0].timestamp[:10]} |")
        lines.append(f"| Last commit | {records[-1].timestamp[:10]} |")
    lines.append(f"| Phases | {len(phases)} |")
    lines.append("")

    if contributors:
        top = ", ".join(f"{author} ({count})" for author, count in contributors[:10])
        lines.append(f"**Top contributors:** {top}")
        lines.append("")

    langs = _language_mix(records)
    if langs:
        lang_str = ", ".join(f"{lang} {pct:.0%}" for lang, pct in langs[:6])
        lines.append(f"**Language mix (at HEAD):** {lang_str}")
        lines.append("")

    # Timeline
    lines.append("## Timeline")
    lines.append("")
    lines.append("| Phase | Period | Commits | LOC Δ | Highlight |")
    lines.append("|---|---|---|---|---|")
    for phase in phases:
        highlight = _phase_highlight(phase, analyzer).replace("|", "\\|")
        lines.append(
            f"| {phase.phase_number} | {phase.start_date[:10]} → {phase.end_date[:10]} "
            f"| {phase.commit_count} | {phase.loc_delta:+,d} | {highlight} |"
        )
    lines.append("")

    # Storylines (reuses the existing StorylineReporter almost verbatim)
    lines.append("## Storylines")
    lines.append("")
    lines.append(
        "*Detected from PR labels/titles, file-change clusters, commit-message "
        "patterns, and this project's `Storyline:` commit trailer — no LLM used.*"
    )
    lines.append("")
    reporter = _build_storyline_reporter(phases)
    lines.append(_increase_heading_level(reporter.generate_storyline_index(), 2))
    lines.append("")
    lines.append(_increase_heading_level(reporter.generate_timeline_view(), 2))
    lines.append("")

    # Most-changed files
    lines.append("## Most-Changed Files")
    lines.append("")
    lines.append("| File | Commits touching it |")
    lines.append("|---|---|")
    for file_path, count in _top_churn_files(records, limit=top_files_limit):
        lines.append(f"| `{file_path}` | {count} |")
    lines.append("")

    # Limitations
    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "This digest is fully mechanical — no LLM calls. Dates, counts, and "
        "storyline signals come directly from git metadata and commit/PR patterns; "
        "it does not explain *why* decisions were made. Run `gitview analyze` for "
        "full LLM-generated narrative prose."
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------

class BriefCommand(BaseCommand):
    """Generate a compact, agent-oriented project history digest.

    Unlike 'analyze', this never calls an LLM: it's meant to be compiled
    once (or after a batch of new commits) and then just read — a cheap
    substitute for an agent re-deriving project history from git log /
    file exploration each session.

    Note: --repo accepts a local path only (unlike 'analyze'/'worklog',
    which also accept GitHub shortcuts/URLs) — this command is about
    bootstrapping context for agents working in an already-checked-out
    repository.
    """

    def validate(self) -> None:
        pass  # No required options; repo-path validity is checked in execute()

    def _extract_records(self, extractor, repo, repo_path, branch, max_commits, head_sha):
        """Extract commit records, reusing a cached extraction when possible.

        `brief` is designed to be regenerated repeatedly ("after a batch of new
        commits"), so re-extracting the entire history each time is wasteful —
        and on very large repositories, extracting from scratch is exactly what
        exhausts memory. We persist extracted records to a per-repo JSONL cache
        and, on a subsequent run whose HEAD descends from the cached HEAD, only
        extract the new commits and append them.

        Every failure path falls back to a full extraction, so a missing,
        stale, or unreadable cache is never fatal — worst case is the original
        behaviour (now itself far lighter on memory).
        """
        # A bounded run (--max-commits) is a partial view; don't cache it or
        # serve one from cache.
        if max_commits:
            return extractor.extract_history(max_commits=max_commits, branch=branch)

        cache_path = repo_path / ".gitview" / "brief_history.jsonl"

        cached = None
        if cache_path.exists():
            try:
                cached = extractor.load_from_jsonl(str(cache_path))
            except Exception:
                cached = None

        records = None
        if cached:
            cached_head = cached[-1].commit_hash
            if cached_head == head_sha:
                records = cached
            else:
                try:
                    descends = repo.is_ancestor(cached_head, head_sha)
                except Exception:
                    descends = False
                if descends:
                    new_records = extractor.extract_incremental(
                        since_commit=cached_head, branch=branch
                    )
                    # extract_incremental leaves loc_total at 0 for the range;
                    # continue the cumulative count from where the cache left off.
                    extractor._calculate_cumulative_loc(
                        new_records, starting_loc=cached[-1].loc_total
                    )
                    records = cached + new_records

        if records is None:
            records = extractor.extract_history(branch=branch)

        # Refresh the cache (best-effort; never fail the command over it).
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            # Keep the cache dir out of the user's git status.
            gitignore = cache_path.parent / ".gitignore"
            if not gitignore.exists():
                gitignore.write_text("*\n", encoding="utf-8")
            extractor.save_to_jsonl(records, str(cache_path))
        except Exception:
            pass

        return records

    def execute(self):
        repo_spec = self.get_option("repo", ".")
        output = self.get_option("output") or "AGENT_BRIEF.md"
        branch = self.get_option("branch", "HEAD")
        max_commits = self.get_option("max_commits")
        force = self.get_option("force", False)
        check_only = self.get_option("check", False)
        top_files_limit = self.get_option("top_files", 15)

        repo_path = Path(repo_spec).resolve()
        if not (repo_path / ".git").exists():
            self.print_error(f"Error: {repo_path} is not a git repository")
            sys.exit(1)

        output_path = Path(output)
        if not output_path.is_absolute():
            output_path = repo_path / output_path

        try:
            repo = Repo(str(repo_path))
            head_sha = repo.head.commit.hexsha
            is_shallow = repo.git.rev_parse("--is-shallow-repository").strip() == "true"
        except Exception as exc:
            self.print_error(f"Error reading git repository: {exc}")
            sys.exit(1)

        remote_url = None
        try:
            candidate_url = repo.remotes.origin.url
            if _looks_like_public_remote(candidate_url):
                remote_url = candidate_url
        except Exception:
            pass

        stamp = _read_stamp(output_path)

        if check_only:
            if stamp and stamp["head"] == head_sha:
                self.print_success(f"Up to date (HEAD {head_sha[:8]}).")
                return
            if stamp:
                try:
                    new_commits = sum(
                        1 for _ in repo.iter_commits(f"{stamp['head']}..{head_sha}")
                    )
                    self.print_warning(
                        f"Stale: {new_commits} new commit(s) since brief was "
                        f"generated at {stamp['head'][:8]}."
                    )
                except Exception:
                    self.print_warning("Stale: brief's stamped commit is no longer in history.")
            else:
                self.print_warning(
                    f"No brief found at {output_path}. Run 'gitview brief' to generate one."
                )
            sys.exit(1)

        if stamp and stamp["head"] == head_sha and not force:
            self.print_success(f"Already up to date (HEAD {head_sha[:8]}).")
            self.print_info("Use --force to regenerate anyway.")
            return

        self.print_header(f"Generating Agent Brief for {repo_path.name}")

        with self.create_progress() as progress:
            task = progress.add_task("Extracting git history...", total=None)
            extractor = GitHistoryExtractor(str(repo_path))
            records = self._extract_records(
                extractor, repo, repo_path, branch, max_commits, head_sha
            )
            progress.update(task, completed=True)

        if not records:
            self.print_warning("No commits found; nothing to brief.")
            return

        self.print_success(f"Extracted {len(records)} commits")

        with self.create_progress() as progress:
            task = progress.add_task("Chunking into phases...", total=None)
            chunker = HistoryChunker("adaptive")
            phases = chunker.chunk(records)
            progress.update(task, completed=True)

        with self.create_progress() as progress:
            task = progress.add_task("Detecting storylines (no LLM)...", total=None)
            content = render_brief(
                repo_path=repo_path,
                records=records,
                phases=phases,
                head_sha=head_sha,
                is_shallow=is_shallow,
                remote_url=remote_url,
                top_files_limit=top_files_limit,
            )
            progress.update(task, completed=True)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")

        self.print_success(f"\nWrote {output_path} ({len(records)} commits, {len(phases)} phases)")
