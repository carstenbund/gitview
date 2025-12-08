"""Core data structures for the analysis pipeline.

This module defines the configuration, context, and result structures
used throughout the modular analysis pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass(frozen=True)
class AnalysisConfig:
    """Immutable configuration for a single analysis run.

    This replaces the 20+ parameters previously passed between functions,
    providing a clean, validated configuration object.
    """

    # Repository (required fields first)
    repo_path: Path
    output_dir: Path

    # Repository (optional fields)
    branch: str = "HEAD"
    repo_name: str = ""

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
    summarization_strategy: str = "simple"  # 'simple' or 'hierarchical'

    # Critical mode
    critical_mode: bool = False
    todo_content: Optional[str] = None
    directives: Optional[str] = None

    # GitHub enrichment
    github_token: Optional[str] = None
    github_repo_url: Optional[str] = None

    @classmethod
    def from_cli_args(
        cls,
        repo_path: Path,
        branch: str,
        output_dir: Path,
        repo_name: str,
        strategy: str = "adaptive",
        chunk_size: int = 50,
        max_commits: Optional[int] = None,
        skip_llm: bool = False,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        ollama_url: str = "http://localhost:11434",
        summarization_strategy: str = "simple",
        incremental: bool = False,
        since_commit: Optional[str] = None,
        since_date: Optional[str] = None,
        critical_mode: bool = False,
        todo_content: Optional[str] = None,
        directives: Optional[str] = None,
        github_token: Optional[str] = None,
        github_repo_url: Optional[str] = None,
    ) -> 'AnalysisConfig':
        """Factory method to create config from CLI arguments.

        Args:
            repo_path: Path to git repository
            branch: Branch to analyze
            output_dir: Output directory for results
            repo_name: Repository name for reports
            strategy: Chunking strategy
            chunk_size: Chunk size for fixed strategy
            max_commits: Maximum commits to analyze
            skip_llm: Skip LLM summarization
            backend: LLM backend
            model: LLM model
            api_key: API key for LLM
            ollama_url: Ollama server URL
            incremental: Incremental analysis flag
            since_commit: Start commit for incremental
            since_date: Start date for incremental
            critical_mode: Enable critical examination mode
            todo_content: TODO/goals content
            directives: Additional directives
            github_token: GitHub API token
            github_repo_url: GitHub repository URL

        Returns:
            Configured AnalysisConfig instance
        """
        return cls(
            repo_path=Path(repo_path),
            branch=branch,
            output_dir=Path(output_dir),
            repo_name=repo_name or Path(repo_path).name,
            strategy=strategy,
            chunk_size=chunk_size,
            max_commits=max_commits,
            skip_llm=skip_llm,
            backend=backend,
            model=model,
            api_key=api_key,
            ollama_url=ollama_url,
            summarization_strategy=summarization_strategy,
            incremental=incremental,
            since_commit=since_commit,
            since_date=since_date,
            critical_mode=critical_mode,
            todo_content=todo_content,
            directives=directives,
            github_token=github_token,
            github_repo_url=github_repo_url,
        )

    def validate(self) -> None:
        """Validate configuration consistency.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.strategy == 'fixed' and self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive for fixed strategy")

        if not self.repo_path.exists():
            raise ValueError(f"Repository path does not exist: {self.repo_path}")

        if not (self.repo_path / '.git').exists():
            raise ValueError(f"Not a git repository: {self.repo_path}")

        if self.strategy not in ('fixed', 'time', 'adaptive'):
            raise ValueError(f"Invalid strategy: {self.strategy}")

        if self.summarization_strategy not in ('simple', 'hierarchical'):
            raise ValueError(f"Invalid summarization_strategy: {self.summarization_strategy}")


@dataclass
class AnalysisContext:
    """Runtime context for a single analysis execution.

    This mutable object is shared across all handlers during pipeline
    execution, accumulating data and state as the analysis progresses.
    """

    # Input configuration (immutable reference)
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

    # Paths (computed in __post_init__)
    history_file: Optional[Path] = None
    phases_dir: Optional[Path] = None

    def __post_init__(self):
        """Initialize computed paths."""
        self.history_file = self.config.output_dir / "repo_history.jsonl"
        self.phases_dir = self.config.output_dir / "phases"

    def has_cached_data(self) -> bool:
        """Check if cached data is available.

        Returns:
            True if either cached records or phases exist
        """
        return self.cached_records is not None or self.cached_phases is not None

    def is_incremental(self) -> bool:
        """Check if this is an incremental analysis.

        Returns:
            True if any incremental flag is set
        """
        return (self.config.incremental or
                self.config.since_commit is not None or
                self.config.since_date is not None)


@dataclass(frozen=True)
class AnalysisResult:
    """Result of a completed analysis.

    This immutable object captures the outcome of a pipeline execution,
    including success status, output paths, and statistics.
    """

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
        """Create a successful result from context.

        Args:
            context: Analysis context with execution data
            duration: Execution duration in seconds

        Returns:
            AnalysisResult indicating success
        """
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
        """Create a failure result from an exception.

        Args:
            error: The exception that caused the failure

        Returns:
            AnalysisResult indicating failure
        """
        return cls(
            success=False,
            error_message=str(error)
        )

    @classmethod
    def early_exit_result(cls, context: AnalysisContext) -> 'AnalysisResult':
        """Create a result for early exit (e.g., no new commits).

        Args:
            context: Analysis context

        Returns:
            AnalysisResult for early exit scenario
        """
        return cls(
            success=True,
            commits_analyzed=0,
            phases_created=len(context.phases),
        )
