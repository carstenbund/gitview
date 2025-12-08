# GitView Analyzer Refactoring Design

**Version:** 1.0
**Date:** 2025-12-08
**Branch:** claude/refactor-cli-analyzer-0136WamNWNDn3rrY8ys6XGge
**Status:** Design Phase

## Executive Summary

This document outlines the refactoring of GitView's CLI from a monolithic 1,429-line `cli.py` into a modular, testable architecture using an `AnalysisPipeline` orchestrator with specialized handler classes.

### Current Problems
- 578-line `analyze()` function with too many responsibilities
- 308-line `_analyze_single_branch()` helper with 20+ parameters
- Significant code duplication between single/multi-branch paths
- Difficult to test individual components in isolation
- Complex nested conditionals for feature combinations

### Proposed Solution
- Separate concerns into specialized handlers (6 handlers)
- Introduce orchestration layer (`AnalysisPipeline`)
- Clear config/context/result data structures
- Eliminate duplication through shared handlers
- Enable unit testing at handler level

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI Layer                            │
│                      (cli.py: commands)                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   AnalysisPipeline                           │
│              (Orchestrates analysis flow)                    │
├─────────────────────────────────────────────────────────────┤
│  • Initializes handlers                                      │
│  • Manages execution flow                                    │
│  • Coordinates context                                       │
│  • Error handling & progress                                 │
└───┬───────┬────────┬────────┬────────┬────────┬─────────────┘
    │       │        │        │        │        │
    ▼       ▼        ▼        ▼        ▼        ▼
┌──────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐
│Extract│ │Enrich│Chunk││Summ││Story││Write│
│Handler│ │Handler│Handler│Handler│Handler│Handler│
└──────┘ └────┘ └────┘ └────┘ └────┘ └────┘

Each handler:
  • Single responsibility
  • Operates on shared context
  • Delegates to existing modules (extractor, chunker, etc.)
  • Returns status/results
```

---

## Core Data Structures

### 1. AnalysisConfig

Immutable configuration for an analysis run. Replaces 20+ function parameters.

```python
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

@dataclass(frozen=True)
class AnalysisConfig:
    """Immutable configuration for a single analysis run."""

    # Repository
    repo_path: Path
    branch: str = "HEAD"

    # Output
    output_dir: Path
    repo_name: str

    # Extraction options
    max_commits: Optional[int] = None
    incremental: bool = False
    since_commit: Optional[str] = None
    since_date: Optional[str] = None

    # Chunking strategy
    strategy: str = "adaptive"  # 'fixed', 'time', 'adaptive'
    chunk_size: int = 50

    # LLM configuration
    skip_llm: bool = False
    backend: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    ollama_url: str = "http://localhost:11434"

    # Critical mode
    critical_mode: bool = False
    todo_content: Optional[str] = None
    directives: Optional[str] = None

    # GitHub enrichment
    github_token: Optional[str] = None
    github_repo_url: Optional[str] = None

    @classmethod
    def from_cli_args(cls, **kwargs) -> 'AnalysisConfig':
        """Factory method to create config from CLI arguments."""
        # Convert and validate CLI args to config
        pass

    def validate(self) -> None:
        """Validate configuration consistency."""
        if self.strategy == 'fixed' and self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive for fixed strategy")
        # Additional validation...
```

### 2. AnalysisContext

Mutable runtime state shared across handlers during execution.

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class AnalysisContext:
    """Runtime context for a single analysis execution."""

    # Input configuration (immutable)
    config: AnalysisConfig

    # Extracted data (populated during execution)
    records: List[Any] = field(default_factory=list)  # CommitRecord objects
    phases: List[Any] = field(default_factory=list)   # Phase objects
    stories: Dict[str, str] = field(default_factory=dict)

    # Caching and incremental support
    cached_records: Optional[List[Any]] = None
    cached_phases: Optional[List[Any]] = None
    previous_analysis: Optional[Dict[str, Any]] = None
    existing_phases: List[Any] = field(default_factory=list)
    starting_loc: int = 0

    # Metrics and stats
    enriched_commit_count: int = 0
    phases_summarized: int = 0

    # Paths (computed)
    history_file: Optional[Path] = None
    phases_dir: Optional[Path] = None

    def __post_init__(self):
        """Initialize computed paths."""
        self.history_file = self.config.output_dir / "repo_history.jsonl"
        self.phases_dir = self.config.output_dir / "phases"

    def has_cached_data(self) -> bool:
        """Check if cached data is available."""
        return self.cached_records is not None or self.cached_phases is not None

    def is_incremental(self) -> bool:
        """Check if this is an incremental analysis."""
        return (self.config.incremental or
                self.config.since_commit is not None or
                self.config.since_date is not None)
```

### 3. AnalysisResult

Immutable result of an analysis run.

```python
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

@dataclass(frozen=True)
class AnalysisResult:
    """Result of a completed analysis."""

    # Success status
    success: bool
    error_message: Optional[str] = None

    # Output paths
    markdown_path: Optional[Path] = None
    json_path: Optional[Path] = None
    timeline_path: Optional[Path] = None

    # Statistics
    commits_analyzed: int = 0
    phases_created: int = 0
    commits_enriched: int = 0
    llm_calls_made: int = 0

    # Timing
    duration_seconds: float = 0.0

    @classmethod
    def success_result(cls, context: AnalysisContext, duration: float) -> 'AnalysisResult':
        """Create a successful result from context."""
        return cls(
            success=True,
            markdown_path=context.config.output_dir / "history_story.md",
            json_path=context.config.output_dir / "history_data.json",
            timeline_path=context.config.output_dir / "timeline.md",
            commits_analyzed=len(context.records),
            phases_created=len(context.phases),
            commits_enriched=context.enriched_commit_count,
            llm_calls_made=context.phases_summarized,
            duration_seconds=duration
        )

    @classmethod
    def failure_result(cls, error: Exception) -> 'AnalysisResult':
        """Create a failure result."""
        return cls(
            success=False,
            error_message=str(error)
        )
```

---

## Handler Interface Design

### Base Handler

All handlers inherit from this base class.

```python
from abc import ABC, abstractmethod
from rich.console import Console

class BaseHandler(ABC):
    """Base class for all analysis handlers."""

    def __init__(self, config: AnalysisConfig, console: Console):
        self.config = config
        self.console = console

    @abstractmethod
    def execute(self, context: AnalysisContext) -> None:
        """Execute this handler's responsibility.

        Args:
            context: Shared analysis context (modified in-place)

        Raises:
            HandlerError: If handler execution fails
        """
        pass

    def should_execute(self, context: AnalysisContext) -> bool:
        """Check if this handler should execute.

        Default: always execute. Override for conditional handlers.
        """
        return True

    def _log_info(self, message: str):
        """Helper to log info messages."""
        self.console.print(f"[cyan]{message}[/cyan]")

    def _log_success(self, message: str):
        """Helper to log success messages."""
        self.console.print(f"[green]{message}[/green]")

    def _log_warning(self, message: str):
        """Helper to log warnings."""
        self.console.print(f"[yellow]{message}[/yellow]")
```

---

## Handler Specifications

### 1. ExtractionHandler

**Responsibility:** Extract git commit history with caching and incremental support.

**Extracted from:** Lines 345-387, 945-988 in cli.py

```python
class ExtractionHandler(BaseHandler):
    """Handles git history extraction with caching and incremental support."""

    def execute(self, context: AnalysisContext) -> None:
        """Extract git history into context.records."""
        self.console.print("[bold]Step 1: Extracting git history...[/bold]")

        # 1. Try to load cached records (if not incremental)
        if not context.is_incremental():
            self._try_load_cached_records(context)

        # 2. If cached, use them; otherwise extract
        if context.cached_records is not None:
            context.records = context.cached_records
            self._log_info("Using cached commit history from previous run.")
        else:
            self._extract_commits(context)

        # 3. Save to JSONL
        self._save_history(context)

        # 4. Display summary
        self._display_summary(context)

    def _try_load_cached_records(self, context: AnalysisContext) -> None:
        """Attempt to load cached records from previous run."""
        if context.history_file and context.history_file.exists():
            try:
                from .extractor import GitHistoryExtractor
                context.cached_records = GitHistoryExtractor.load_from_jsonl(
                    str(context.history_file)
                )
            except Exception as e:
                self._log_warning(f"Failed to load cached history: {e}")

    def _extract_commits(self, context: AnalysisContext) -> None:
        """Extract commits from repository."""
        from .extractor import GitHistoryExtractor
        from rich.progress import Progress, SpinnerColumn, TextColumn

        extractor = GitHistoryExtractor(str(self.config.repo_path))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Extracting commits...", total=None)

            if context.is_incremental():
                records = extractor.extract_incremental(
                    since_commit=self.config.since_commit,
                    since_date=self.config.since_date,
                    branch=self.config.branch
                )
                # Adjust LOC for continuation
                if context.starting_loc > 0:
                    extractor._calculate_cumulative_loc(records, context.starting_loc)
            else:
                records = extractor.extract_history(
                    max_commits=self.config.max_commits,
                    branch=self.config.branch
                )

            context.records = records
            progress.update(task, completed=True)

    def _save_history(self, context: AnalysisContext) -> None:
        """Save extracted history to JSONL file."""
        from .extractor import GitHistoryExtractor

        context.config.output_dir.mkdir(parents=True, exist_ok=True)

        extractor = GitHistoryExtractor(str(self.config.repo_path))
        extractor.save_to_jsonl(context.records, str(context.history_file))

    def _display_summary(self, context: AnalysisContext) -> None:
        """Display extraction summary."""
        count = len(context.records)
        if context.is_incremental():
            self._log_success(f"Extracted {count} new commits\n")
            if count == 0:
                self._log_info("No new commits since last analysis.")
        else:
            self._log_success(f"Extracted {count} commits\n")
```

---

### 2. GitHubEnrichmentHandler

**Responsibility:** Enrich commits with GitHub PR/review context.

**Extracted from:** Lines 390-426, 990-1053 in cli.py

```python
class GitHubEnrichmentHandler(BaseHandler):
    """Handles GitHub PR/review enrichment."""

    def should_execute(self, context: AnalysisContext) -> bool:
        """Only execute if GitHub token is provided and records exist."""
        return (self.config.github_token is not None and
                len(context.records) > 0)

    def execute(self, context: AnalysisContext) -> None:
        """Enrich commits with GitHub context."""
        self.console.print("[bold]Step 1.5: Enriching with GitHub context...[/bold]")

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
            self._log_warning(f"GitHub enrichment failed: {e}")
            self._log_warning("Continuing without GitHub context...\n")

    def _determine_github_url(self) -> Optional[str]:
        """Determine GitHub repo URL from config or repo remotes."""
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
                        from .github_graphql import parse_github_url
                        owner, repo_name = parse_github_url(url)
                        return f"{owner}/{repo_name}"
        except Exception:
            pass

        return None

    def _fetch_github_contexts(self, context: AnalysisContext, repo_url: str):
        """Fetch GitHub contexts for commits."""
        from .github_enricher import enrich_commits_with_github
        from rich.progress import Progress, SpinnerColumn, TextColumn

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

    def _attach_contexts(self, context: AnalysisContext, github_contexts: dict):
        """Attach GitHub contexts to commit records."""
        context.enriched_commit_count = 0

        for record in context.records:
            if record.commit_hash in github_contexts:
                ctx = github_contexts[record.commit_hash]
                record.github_context = ctx.to_dict()
                if ctx.pr_number:
                    context.enriched_commit_count += 1

    def _save_enriched_history(self, context: AnalysisContext):
        """Re-save history with GitHub context attached."""
        from .extractor import GitHistoryExtractor

        extractor = GitHistoryExtractor(str(self.config.repo_path))
        extractor.save_to_jsonl(context.records, str(context.history_file))
```

---

### 3. ChunkerHandler

**Responsibility:** Chunk commits into phases with incremental support.

**Extracted from:** Lines 428-486, 1055-1113 in cli.py

```python
class ChunkerHandler(BaseHandler):
    """Handles phase chunking with incremental support."""

    def execute(self, context: AnalysisContext) -> None:
        """Chunk commits into phases."""
        self.console.print("[bold]Step 2: Chunking into phases...[/bold]")

        # 1. Try to load cached phases
        if not context.is_incremental():
            self._try_load_cached_phases(context)

        # 2. Handle different chunking scenarios
        if context.cached_phases is not None:
            context.phases = context.cached_phases
            self._log_info(f"Reusing {len(context.phases)} cached phases.\n")
        elif context.existing_phases and len(context.records) > 0:
            self._handle_incremental_chunking(context)
        else:
            self._create_new_phases(context)

        # 3. Display overview
        self._display_phase_overview(context)

        # 4. Save phases
        self._save_phases(context)

    def _try_load_cached_phases(self, context: AnalysisContext):
        """Try to load cached phases from previous run."""
        if context.phases_dir and context.phases_dir.exists():
            try:
                from .chunker import HistoryChunker
                context.cached_phases = HistoryChunker.load_phases(
                    str(context.phases_dir)
                )
            except Exception as e:
                self._log_warning(f"Failed to load cached phases: {e}")

    def _create_new_phases(self, context: AnalysisContext):
        """Create new phases from records."""
        from .chunker import HistoryChunker

        chunker = HistoryChunker(self.config.strategy)

        kwargs = {}
        if self.config.strategy == 'fixed':
            kwargs['chunk_size'] = self.config.chunk_size

        context.phases = chunker.chunk(context.records, **kwargs)
        self._log_success(f"Created {len(context.phases)} phases\n")

    def _handle_incremental_chunking(self, context: AnalysisContext):
        """Handle incremental phase creation/merging."""
        from .chunker import HistoryChunker, Phase

        merge_threshold = 10  # commits

        if len(context.records) < merge_threshold:
            # Merge into last phase
            self._merge_into_last_phase(context)
        else:
            # Create new phases
            self._create_and_append_phases(context)

    def _merge_into_last_phase(self, context: AnalysisContext):
        """Merge new commits into the last existing phase."""
        from .chunker import Phase

        self._log_info(f"Merging {len(context.records)} commits into last phase...")

        last_phase = context.existing_phases[-1]
        last_phase.commits.extend(context.records)

        # Recalculate stats
        last_phase.commit_count = len(last_phase.commits)
        last_phase.end_date = context.records[-1].timestamp
        last_phase.total_insertions = sum(c.insertions for c in last_phase.commits)
        last_phase.total_deletions = sum(c.deletions for c in last_phase.commits)
        last_phase.loc_end = context.records[-1].loc_total
        last_phase.loc_delta = last_phase.loc_end - last_phase.loc_start
        if last_phase.loc_start > 0:
            last_phase.loc_delta_percent = (last_phase.loc_delta / last_phase.loc_start) * 100

        # Clear summary for regeneration
        last_phase.summary = None

        context.phases = context.existing_phases
        self._log_success(f"Updated last phase (now {last_phase.commit_count} commits)\n")

    def _create_and_append_phases(self, context: AnalysisContext):
        """Create new phases and append to existing."""
        from .chunker import HistoryChunker

        chunker = HistoryChunker(self.config.strategy)

        kwargs = {}
        if self.config.strategy == 'fixed':
            kwargs['chunk_size'] = self.config.chunk_size

        new_phases = chunker.chunk(context.records, **kwargs)

        # Renumber to continue from existing
        for phase in new_phases:
            phase.phase_number = len(context.existing_phases) + phase.phase_number

        context.phases = context.existing_phases + new_phases
        self._log_success(
            f"Created {len(new_phases)} new phases (total: {len(context.phases)})\n"
        )

    def _display_phase_overview(self, context: AnalysisContext):
        """Display phase overview table."""
        from rich.table import Table

        table = Table(title="Phase Overview")
        table.add_column("Phase", style="cyan", justify="right")
        table.add_column("Period", style="magenta")
        table.add_column("Commits", justify="right")
        table.add_column("LOC Δ", justify="right")
        table.add_column("Events", style="yellow")

        for phase in context.phases:
            events = []
            if phase.has_large_deletion:
                events.append("×")
            if phase.has_large_addition:
                events.append("+")
            if phase.has_refactor:
                events.append(">>")
            if phase.readme_changed:
                events.append(">")

            table.add_row(
                str(phase.phase_number),
                f"{phase.start_date[:10]} to {phase.end_date[:10]}",
                str(phase.commit_count),
                f"{phase.loc_delta:+,d}",
                " ".join(events)
            )

        self.console.print(table)

    def _save_phases(self, context: AnalysisContext):
        """Save phases to disk."""
        from .chunker import HistoryChunker

        chunker = HistoryChunker(self.config.strategy)
        chunker.save_phases(context.phases, str(context.phases_dir))
```

---

### 4. SummarizerHandler

**Responsibility:** Summarize phases using LLM.

**Extracted from:** Lines 496-544, 1122-1171 in cli.py

```python
class SummarizerHandler(BaseHandler):
    """Handles phase summarization with LLM."""

    def should_execute(self, context: AnalysisContext) -> bool:
        """Only execute if LLM is not skipped."""
        return not self.config.skip_llm

    def execute(self, context: AnalysisContext) -> None:
        """Summarize phases with LLM."""
        self.console.print("[bold]Step 3: Summarizing phases with LLM...[/bold]")

        # 1. Initialize summarizer
        summarizer = self._create_summarizer()

        # 2. Identify phases needing summarization
        phases_to_summarize = [p for p in context.phases if p.summary is None]

        if len(phases_to_summarize) < len(context.phases):
            self._log_info(
                f"{len(phases_to_summarize)} phases need summarization "
                f"({len(context.phases) - len(phases_to_summarize)} already summarized)"
            )

        # 3. Summarize phases
        self._summarize_phases(context, summarizer, phases_to_summarize)

        # 4. Display summary
        if len(phases_to_summarize) > 0:
            self._log_success(f"Summarized {len(phases_to_summarize)} phase(s)\n")
        else:
            self._log_success("All phases already summarized\n")

        context.phases_summarized = len(phases_to_summarize)

    def _create_summarizer(self):
        """Create and configure PhaseSummarizer."""
        from .summarizer import PhaseSummarizer

        return PhaseSummarizer(
            backend=self.config.backend,
            model=self.config.model,
            api_key=self.config.api_key,
            ollama_url=self.config.ollama_url,
            todo_content=self.config.todo_content,
            critical_mode=self.config.critical_mode,
            directives=self.config.directives
        )

    def _summarize_phases(self, context: AnalysisContext, summarizer, phases_to_summarize):
        """Summarize all phases, reusing existing summaries."""
        from rich.progress import Progress, SpinnerColumn, TextColumn

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task(
                "Summarizing phases...",
                total=len(phases_to_summarize)
            )

            previous_summaries = []

            for i, phase in enumerate(context.phases):
                progress.update(
                    task,
                    description=f"Processing phase {i+1}/{len(context.phases)}..."
                )

                if phase.summary is None:
                    # Summarize this phase
                    context_str = summarizer._build_context(previous_summaries)
                    summary = summarizer.summarize_phase(phase, context_str)
                    phase.summary = summary
                    progress.update(task, advance=1)

                # Build context for next phase
                previous_summaries.append({
                    'phase_number': phase.phase_number,
                    'summary': phase.summary,
                    'loc_delta': phase.loc_delta,
                })

                # Save phase with summary
                summarizer._save_phase_with_summary(phase, str(context.phases_dir))
```

---

### 5. StorytellerHandler

**Responsibility:** Generate global narrative from phase summaries.

**Extracted from:** Lines 547-567, 1173-1194 in cli.py

```python
class StorytellerHandler(BaseHandler):
    """Handles global narrative generation."""

    def should_execute(self, context: AnalysisContext) -> bool:
        """Only execute if LLM is not skipped."""
        return not self.config.skip_llm

    def execute(self, context: AnalysisContext) -> None:
        """Generate global narrative stories."""
        self.console.print("[bold]Step 4: Generating global narrative...[/bold]")

        # 1. Create storyteller
        storyteller = self._create_storyteller()

        # 2. Generate stories
        context.stories = self._generate_stories(context, storyteller)

        self._log_success("Generated global narrative\n")

    def _create_storyteller(self):
        """Create and configure StoryTeller."""
        from .storyteller import StoryTeller

        return StoryTeller(
            backend=self.config.backend,
            model=self.config.model,
            api_key=self.config.api_key,
            ollama_url=self.config.ollama_url,
            todo_content=self.config.todo_content,
            critical_mode=self.config.critical_mode,
            directives=self.config.directives
        )

    def _generate_stories(self, context: AnalysisContext, storyteller):
        """Generate global stories with progress indicator."""
        from rich.progress import Progress, SpinnerColumn, TextColumn

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Generating story...", total=None)

            stories = storyteller.generate_global_story(
                context.phases,
                self.config.repo_name
            )

            progress.update(task, completed=True)

        return stories
```

---

### 6. OutputWriterHandler

**Responsibility:** Write all output files (markdown, JSON, timeline).

**Extracted from:** Lines 569-591, 1196-1218 in cli.py

```python
class OutputWriterHandler(BaseHandler):
    """Handles writing output files."""

    def execute(self, context: AnalysisContext) -> None:
        """Write all output files."""
        self.console.print("[bold]Step 5: Writing output files...[/bold]")

        if self.config.skip_llm:
            self._write_timeline_only(context)
        else:
            self._write_all_outputs(context)

    def _write_timeline_only(self, context: AnalysisContext):
        """Write only timeline (when LLM is skipped)."""
        from .writer import OutputWriter

        self._log_info("Skipping LLM summarization. Writing basic timeline...")

        timeline_path = self.config.output_dir / "timeline.md"
        OutputWriter.write_simple_timeline(context.phases, str(timeline_path))

        self._log_success(f"Wrote timeline to {timeline_path}\n")

    def _write_all_outputs(self, context: AnalysisContext):
        """Write all output files (markdown, JSON, timeline)."""
        from .writer import OutputWriter

        # Write markdown report
        markdown_path = self.config.output_dir / "history_story.md"
        OutputWriter.write_markdown(
            context.stories,
            context.phases,
            str(markdown_path),
            self.config.repo_name
        )
        self._log_success(f"Wrote {markdown_path}")

        # Write JSON data
        json_path = self.config.output_dir / "history_data.json"
        OutputWriter.write_json(
            context.stories,
            context.phases,
            str(json_path),
            repo_path=str(self.config.repo_path)
        )
        self._log_success(f"Wrote {json_path}")

        # Write timeline
        timeline_path = self.config.output_dir / "timeline.md"
        OutputWriter.write_simple_timeline(context.phases, str(timeline_path))
        self._log_success(f"Wrote {timeline_path}\n")
```

---

## Pipeline Orchestration

### AnalysisPipeline Class

The main orchestrator that coordinates all handlers.

```python
import time
from typing import List, Optional
from rich.console import Console

class AnalysisPipeline:
    """Orchestrates the complete analysis pipeline."""

    def __init__(self, config: AnalysisConfig, console: Optional[Console] = None):
        """Initialize pipeline with configuration.

        Args:
            config: Analysis configuration
            console: Rich console for output (creates default if None)
        """
        self.config = config
        self.console = console or Console()

        # Initialize handlers
        self._handlers: List[BaseHandler] = []
        self._setup_handlers()

    def _setup_handlers(self):
        """Initialize all handlers in execution order."""
        self._handlers = [
            ExtractionHandler(self.config, self.console),
            GitHubEnrichmentHandler(self.config, self.console),
            ChunkerHandler(self.config, self.console),
            SummarizerHandler(self.config, self.console),
            StorytellerHandler(self.config, self.console),
            OutputWriterHandler(self.config, self.console),
        ]

    def run(self) -> AnalysisResult:
        """Execute the complete analysis pipeline.

        Returns:
            AnalysisResult with execution details
        """
        start_time = time.time()

        try:
            # 1. Validate configuration
            self.config.validate()

            # 2. Create context
            context = self._create_context()

            # 3. Load previous analysis (for incremental)
            self._load_previous_analysis(context)

            # 4. Execute handlers
            self._execute_handlers(context)

            # 5. Check for early exit
            if self._should_exit_early(context):
                return self._create_early_exit_result(context)

            # 6. Create success result
            duration = time.time() - start_time
            result = AnalysisResult.success_result(context, duration)

            # 7. Display success message
            self._display_success(context, result)

            return result

        except Exception as e:
            # Handle failure
            self.console.print(f"\n[red]Error: {e}[/red]")
            import traceback
            traceback.print_exc()
            return AnalysisResult.failure_result(e)

    def _create_context(self) -> AnalysisContext:
        """Create initial analysis context."""
        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        return AnalysisContext(config=self.config)

    def _load_previous_analysis(self, context: AnalysisContext):
        """Load previous analysis for incremental runs."""
        if not context.is_incremental():
            return

        from .writer import OutputWriter

        context.previous_analysis = OutputWriter.load_previous_analysis(
            str(self.config.output_dir)
        )

        if self.config.incremental and not context.previous_analysis:
            self.console.print(
                "[yellow]Warning: --incremental specified but no previous analysis found.[/yellow]"
            )
            self.console.print("[yellow]Running full analysis instead...[/yellow]\n")
            # Reset incremental flag in config (but config is frozen!)
            # Need to handle this gracefully
            return

        if context.previous_analysis:
            self._load_existing_phases(context)

    def _load_existing_phases(self, context: AnalysisContext):
        """Load existing phases from previous analysis."""
        from .chunker import Phase

        metadata = context.previous_analysis.get('metadata', {})

        if self.config.incremental:
            last_hash = metadata.get('last_commit_hash')
            self.console.print(
                f"[cyan]Incremental mode:[/cyan] Analyzing commits since {last_hash[:8]}"
            )
            self.console.print(
                f"[cyan]Last analysis:[/cyan] {metadata.get('generated_at', 'unknown')}\n"
            )

        # Load phases
        context.existing_phases = [
            Phase.from_dict(p)
            for p in context.previous_analysis.get('phases', [])
        ]

        # Calculate starting LOC
        if context.existing_phases and context.existing_phases[-1].commits:
            context.starting_loc = context.existing_phases[-1].commits[-1].loc_total

    def _execute_handlers(self, context: AnalysisContext):
        """Execute all handlers in sequence."""
        for handler in self._handlers:
            if handler.should_execute(context):
                handler.execute(context)

    def _should_exit_early(self, context: AnalysisContext) -> bool:
        """Check if we should exit early (e.g., no new commits in incremental)."""
        if context.is_incremental() and len(context.records) == 0:
            self.console.print("[yellow]No new commits found since last analysis.[/yellow]")
            self.console.print("[green]Repository is up to date![/green]\n")
            return True
        return False

    def _create_early_exit_result(self, context: AnalysisContext) -> AnalysisResult:
        """Create result for early exit scenarios."""
        return AnalysisResult(
            success=True,
            commits_analyzed=0,
            phases_created=len(context.phases),
        )

    def _display_success(self, context: AnalysisContext, result: AnalysisResult):
        """Display success message with summary."""
        self.console.print("[bold green]Analysis complete![/bold green]\n")
        self.console.print(
            f"Analyzed {result.commits_analyzed} commits across {result.phases_created} phases"
        )
        self.console.print(f"Output written to: {self.config.output_dir.resolve()}\n")
```

---

## Integration with CLI

### Simplified CLI Command

The `analyze()` command becomes much simpler:

```python
@cli.command(help=ANALYZE_HELP)
@click.option(...)  # All existing options
def analyze(**kwargs):
    """Analyze git repository and generate narrative history."""

    console.print("\n[bold blue]GitView - Repository History Analyzer[/bold blue]\n")

    try:
        # 1. Handle repository detection and cloning
        repo_handler = RemoteRepoHandler(kwargs['repo'])
        repo_path, output_dir, repo_name = _setup_repository(repo_handler, kwargs)

        # 2. Handle branch selection
        branch_manager = BranchManager(str(repo_path))
        branches_to_analyze = _determine_branches(branch_manager, kwargs)

        # 3. Handle multi-branch vs single-branch
        if len(branches_to_analyze) > 1:
            _analyze_multiple_branches(
                branches_to_analyze,
                repo_path,
                output_dir,
                repo_name,
                kwargs
            )
        else:
            # Single branch - use new pipeline
            _analyze_single_branch_with_pipeline(
                branches_to_analyze[0] if branches_to_analyze else kwargs['branch'],
                repo_path,
                output_dir,
                repo_name,
                kwargs
            )

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)
    finally:
        # Cleanup
        if repo_handler and not kwargs.get('keep_clone'):
            repo_handler.cleanup()


def _analyze_single_branch_with_pipeline(branch, repo_path, output_dir, repo_name, cli_args):
    """Analyze a single branch using the new pipeline architecture."""

    # 1. Create configuration
    config = AnalysisConfig.from_cli_args(
        repo_path=repo_path,
        branch=branch,
        output_dir=output_dir,
        repo_name=repo_name,
        **cli_args
    )

    # 2. Display configuration
    _display_config(config)

    # 3. Create and run pipeline
    pipeline = AnalysisPipeline(config, console)
    result = pipeline.run()

    # 4. Handle result
    if not result.success:
        sys.exit(1)
```

---

## File Structure

New files to create:

```
gitview/
├── analyzer.py                    # Core data structures (Config, Context, Result)
├── pipeline.py                    # AnalysisPipeline orchestrator
├── handlers/
│   ├── __init__.py               # Exports all handlers
│   ├── base.py                   # BaseHandler abstract class
│   ├── extraction.py             # ExtractionHandler
│   ├── enrichment.py             # GitHubEnrichmentHandler
│   ├── chunker.py                # ChunkerHandler
│   ├── summarizer.py             # SummarizerHandler
│   ├── storyteller.py            # StorytellerHandler
│   └── writer.py                 # OutputWriterHandler
├── cli.py                        # Simplified CLI (uses pipeline)
└── ...existing files...
```

---

## Migration Strategy

### Phase 1: Build Infrastructure (No Breaking Changes)

1. Create `gitview/analyzer.py` with data structures
2. Create `gitview/handlers/` directory structure
3. Create `gitview/handlers/base.py` with BaseHandler
4. Add comprehensive unit tests for data structures

**Risk:** None - no existing code modified
**Duration:** 1-2 hours

### Phase 2: Implement Handlers One-by-One

Implement each handler in order, with tests:

1. `ExtractionHandler` + tests
2. `GitHubEnrichmentHandler` + tests
3. `ChunkerHandler` + tests
4. `SummarizerHandler` + tests
5. `StorytellerHandler` + tests
6. `OutputWriterHandler` + tests

For each handler:
- Extract logic from existing CLI
- Add comprehensive unit tests
- Test with real repository data

**Risk:** Low - handlers are isolated, old code still works
**Duration:** 4-6 hours

### Phase 3: Build Pipeline Orchestrator

1. Create `gitview/pipeline.py` with AnalysisPipeline
2. Add integration tests for full pipeline
3. Test against multiple repositories

**Risk:** Low - pipeline is new code, old path still exists
**Duration:** 2-3 hours

### Phase 4: Integrate with CLI (Dual Path)

1. Add new helper function: `_analyze_single_branch_with_pipeline()`
2. Add CLI flag: `--use-new-pipeline` (opt-in)
3. Keep old `_analyze_single_branch()` function
4. Run both paths in test suite, compare outputs
5. Fix any discrepancies

**Risk:** Medium - integration with CLI, but old path preserved
**Duration:** 2-3 hours

### Phase 5: Switch Default and Deprecate Old Path

1. Make new pipeline the default
2. Add flag: `--use-legacy-pipeline` for old path
3. Run extensive testing on various repositories
4. Deprecation warning for legacy path

**Risk:** Medium - changes default behavior
**Duration:** 1-2 hours

### Phase 6: Remove Legacy Code

1. Remove `_analyze_single_branch()` function
2. Remove legacy flag
3. Clean up unused code
4. Update documentation

**Risk:** Low - new system proven at this point
**Duration:** 1 hour

**Total Estimated Duration:** 11-17 hours of development

---

## Testing Strategy

### Unit Tests

Each handler gets comprehensive unit tests:

```python
# tests/handlers/test_extraction.py
def test_extraction_handler_basic():
    """Test basic extraction without caching."""
    config = AnalysisConfig(...)
    context = AnalysisContext(config)
    handler = ExtractionHandler(config, console)

    handler.execute(context)

    assert len(context.records) > 0
    assert context.history_file.exists()

def test_extraction_handler_with_cache():
    """Test extraction with cached data."""
    # Setup cached data
    # Run handler
    # Assert cached data was used

def test_extraction_handler_incremental():
    """Test incremental extraction."""
    # Setup previous analysis
    # Run handler
    # Assert only new commits extracted
```

### Integration Tests

Test the full pipeline:

```python
# tests/test_pipeline_integration.py
def test_full_pipeline_basic(tmp_repo):
    """Test full pipeline on a basic repository."""
    config = AnalysisConfig(
        repo_path=tmp_repo,
        output_dir=tmp_output,
        ...
    )
    pipeline = AnalysisPipeline(config)
    result = pipeline.run()

    assert result.success
    assert result.commits_analyzed > 0
    assert (config.output_dir / "history_story.md").exists()

def test_pipeline_incremental(tmp_repo):
    """Test incremental analysis."""
    # Run initial analysis
    # Add new commits
    # Run incremental analysis
    # Assert only new commits processed
```

### Regression Tests

Compare outputs of old vs new pipeline:

```python
def test_legacy_vs_new_pipeline_equivalence(test_repo):
    """Ensure new pipeline produces same results as legacy."""
    # Run legacy path
    legacy_result = _analyze_single_branch(...)

    # Run new pipeline
    new_result = pipeline.run()

    # Compare outputs (commits, phases, stories)
    assert_outputs_equivalent(legacy_result, new_result)
```

---

## Benefits Summary

### Modularity
- Each handler has a single, clear responsibility
- Easy to modify one aspect without affecting others
- New features can be added as new handlers

### Testability
- Unit test each handler in isolation
- Mock dependencies easily with clear interfaces
- Integration tests validate the full flow

### Maintainability
- Code is organized by concern, not by flow
- Reduces cognitive load when reading code
- Clear boundaries between components

### Extensibility
- New handlers can be plugged in
- Pipeline can be customized or reordered
- Support for hooks and plugins in future

### Performance
- Caching logic centralized in handlers
- Easy to add profiling per handler
- Can parallelize independent handlers in future

### Code Quality
- Eliminates 300+ lines of duplication
- Reduces parameter passing from 20+ to 2 (config, context)
- Clearer error handling boundaries

---

## Open Questions

1. **Handler Dependencies:** Should handlers be able to declare dependencies on other handlers?
   - Current: Implicit through execution order
   - Alternative: Explicit dependency graph

2. **Error Handling:** How should partial failures be handled?
   - Current: Exception stops pipeline
   - Alternative: Continue with warnings, collect errors

3. **Hooks/Plugins:** Should we add a plugin system for custom handlers?
   - Future enhancement
   - Would allow users to add custom analysis steps

4. **Parallel Execution:** Should independent handlers run in parallel?
   - Future optimization
   - Extraction and GitHub enrichment could potentially overlap

5. **Progress Reporting:** Should progress be unified across handlers?
   - Current: Each handler manages its own progress
   - Alternative: Centralized progress tracking in pipeline

6. **Configuration Validation:** Where should complex validation live?
   - Current: AnalysisConfig.validate()
   - Alternative: Each handler validates what it needs

---

## Success Criteria

This refactoring will be considered successful when:

1. ✅ All existing CLI functionality works with new pipeline
2. ✅ Test coverage increases (target: >80% for handlers)
3. ✅ CLI function is <200 lines (from current 578)
4. ✅ No duplicate code between single/multi-branch paths
5. ✅ New features can be added as handlers without modifying pipeline
6. ✅ Handler logic can be unit tested without mocking git repos
7. ✅ Code review feedback positive on maintainability

---

## References

- Current CLI implementation: `gitview/cli.py` lines 284-1232
- Existing modules:
  - `gitview/extractor.py` - GitHistoryExtractor
  - `gitview/chunker.py` - HistoryChunker
  - `gitview/summarizer.py` - PhaseSummarizer
  - `gitview/storyteller.py` - StoryTeller
  - `gitview/writer.py` - OutputWriter
  - `gitview/github_enricher.py` - enrich_commits_with_github

---

## Next Steps

1. Review this design document
2. Gather feedback and address concerns
3. Create tracking issues for each phase
4. Begin Phase 1 implementation
5. Iterate based on lessons learned

---

**Document Status:** Draft for Review
**Last Updated:** 2025-12-08
**Author:** Claude (AI Assistant)
**Reviewers:** [To be filled]
