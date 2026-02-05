"""Analyze command implementation."""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

from .base import BaseCommand
from ..extractor import GitHistoryExtractor
from ..chunker import HistoryChunker, Phase
from ..summarizer import PhaseSummarizer
from ..storyteller import StoryTeller
from ..hierarchical_summarizer import HierarchicalPhaseSummarizer
from ..hierarchical_storyteller import HierarchicalStoryTeller
from ..writer import OutputWriter
from ..remote import RemoteRepoHandler
from ..branches import BranchManager, parse_branch_spec
from ..index_writer import IndexWriter
from ..github_enricher import enrich_commits_with_github
from ..storyline import StorylineTracker, StorylineDatabase


class AnalyzeCommand(BaseCommand):
    """Analyze git repository and generate narrative history.

    This command runs the full pipeline:
    1. Extract git history with detailed metadata
    2. Enrich with GitHub PR/review context (optional)
    3. Chunk commits into meaningful phases/epochs
    4. Summarize each phase using LLM
    5. Track storylines using multi-signal detection
    6. Generate global narrative stories
    7. Write markdown reports and JSON data
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._repo_path = None
        self._repo_name = None
        self._output_path = None
        self._cloned_repo_path = None
        self._repo_handler = None
        self._todo_content = None

    def validate(self) -> None:
        """Validate command options."""
        # Load todo/goals file if specified
        todo = self.get_option('todo')
        if todo:
            todo_path = Path(todo)
            if not todo_path.exists():
                self.print_error(f"Error: Todo file not found: {todo}")
                sys.exit(1)
            with open(todo_path, 'r') as f:
                self._todo_content = f.read()
            self.print_info(f"Loaded goals from: {todo}")

        # Validate critical mode
        critical = self.get_option('critical', False)
        directives = self.get_option('directives')
        if critical and not todo and not directives:
            self.print_warning("Warning: --critical mode enabled without --todo or --directives.")
            self.print_warning("Critical mode works best with goals/directives to measure against.\n")

    def execute(self):
        """Execute repository analysis."""
        self.console.print("\n[bold blue]GitView - Repository History Analyzer[/bold blue]\n")

        # Check for adaptive mode
        adaptive_mode = self.get_option('adaptive', False)
        if adaptive_mode:
            self.print_info("Adaptive agent mode enabled - discovery-driven analysis")
            return self._execute_adaptive()

        # Normalize summarization strategy
        summarization_strategy = self.get_option('summarization_strategy', 'simple')
        if self.get_option('hierarchical', False):
            summarization_strategy = 'hierarchical'

        if summarization_strategy == 'hierarchical':
            self.print_info("Using hierarchical summarization strategy")

        repo = self.get_option('repo', '.')
        output = self.get_option('output')
        list_branches = self.get_option('list_branches', False)

        # Handle remote repository detection and cloning
        self._repo_handler = RemoteRepoHandler(repo)

        try:
            if self._repo_handler.is_local:
                self._setup_local_repo(output)
            else:
                self._setup_remote_repo(output)

            # Initialize branch manager
            branch_manager = BranchManager(str(self._repo_path))

            # Handle --list-branches flag
            if list_branches:
                self._list_branches(branch_manager)
                return

            # Determine which branches to analyze
            branches_to_analyze = self._get_branches_to_analyze(branch_manager)

            # Check if multi-branch mode
            is_multi_branch = len(branches_to_analyze) > 1

            if is_multi_branch:
                self._analyze_multiple_branches(branches_to_analyze, branch_manager, summarization_strategy)
            else:
                self._analyze_single_branch_mode(branches_to_analyze, summarization_strategy)

        except Exception as e:
            self.print_error(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        finally:
            self._cleanup()

    def _setup_local_repo(self, output: Optional[str]):
        """Setup local repository paths."""
        self._repo_path = self._repo_handler.get_local_path()
        if not (self._repo_path / '.git').exists():
            self.print_error(f"Error: {self._repo_path} is not a git repository")
            sys.exit(1)

        repo_name = self.get_option('repo_name')
        if not repo_name:
            repo_name = self._repo_path.name
        self._repo_name = repo_name

        if output is None:
            output = "output"
        self._output_path = Path(output)

    def _setup_remote_repo(self, output: Optional[str]):
        """Setup remote repository (clone and configure paths)."""
        self.print_info(f"Remote repository detected: {self._repo_handler.repo_info.short_name}\n")

        self._cloned_repo_path = self._repo_handler.clone()
        self._repo_path = self._cloned_repo_path

        repo_name = self.get_option('repo_name')
        if not repo_name:
            repo_name = self._repo_handler.repo_info.repo
        self._repo_name = repo_name

        if output is None:
            output = str(self._repo_handler.get_default_output_path())
            self.print_info(f"Output will be saved to: {output}\n")
        self._output_path = Path(output)

    def _cleanup(self):
        """Cleanup temporary resources."""
        keep_clone = self.get_option('keep_clone', False)
        if self._cloned_repo_path and not keep_clone:
            self._repo_handler.cleanup()
        elif self._cloned_repo_path and keep_clone:
            self.print_info(f"\nTemporary clone preserved at: {self._cloned_repo_path}")

    def _execute_adaptive(self):
        """Execute analysis using the adaptive review agent."""
        from ..adaptive import AdaptiveReviewAgent
        from ..extractor import GitHistoryExtractor
        from ..chunker import HistoryChunker
        from ..summarizer import PhaseSummarizer
        from ..storyteller import StoryTeller
        from ..writer import OutputWriter

        repo = self.get_option('repo', '.')
        output = self.get_option('output')

        # Handle remote repository
        self._repo_handler = RemoteRepoHandler(repo)

        try:
            if self._repo_handler.is_local:
                self._setup_local_repo(output)
            else:
                self._setup_remote_repo(output)

            # Get options
            branch = self.get_option('branch', 'HEAD')
            max_commits = self.get_option('max_commits')
            backend = self.get_option('backend')
            model = self.get_option('model')
            api_key = self.get_option('api_key')
            ollama_url = self.get_option('ollama_url', 'http://localhost:11434')
            skip_llm = self.get_option('skip_llm', False)
            critical = self.get_option('critical', False)
            directives = self.get_option('directives')
            github_token = self.get_option('github_token')

            # Load goals if provided
            user_goals = []
            if self._todo_content:
                # Parse goals from todo content (one per line, skip empty/comments)
                for line in self._todo_content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('-'):
                        user_goals.append(line)
                    elif line.startswith('- '):
                        user_goals.append(line[2:])

            # Create components
            extractor = GitHistoryExtractor(str(self._repo_path))
            chunker = HistoryChunker('adaptive')

            summarizer = None
            storyteller = None
            if not skip_llm:
                summarizer = PhaseSummarizer(
                    backend=backend,
                    model=model,
                    api_key=api_key,
                    ollama_url=ollama_url,
                    todo_content=self._todo_content,
                    critical_mode=critical,
                    directives=directives
                )
                storyteller = StoryTeller(
                    backend=backend,
                    model=model,
                    api_key=api_key,
                    ollama_url=ollama_url,
                    todo_content=self._todo_content,
                    critical_mode=critical,
                    directives=directives
                )

            # Create adaptive agent
            agent = AdaptiveReviewAgent(
                extractor=extractor,
                chunker=chunker,
                summarizer=summarizer,
                storyteller=storyteller,
                logger=lambda msg: self.print_info(msg)
            )

            # Run adaptive analysis
            self.print_info(f"Repository: {self._repo_path}")
            self.print_info(f"Output: {self._output_path}")
            if user_goals:
                self.print_info(f"Goals: {len(user_goals)} objectives loaded\n")

            result = agent.analyze(
                repo_path=str(self._repo_path),
                output_path=str(self._output_path),
                branch=branch,
                user_goals=user_goals,
                directives=directives,
                critical_mode=critical,
                max_commits=max_commits,
                skip_llm=skip_llm,
                github_token=github_token,
            )

            # Write output files
            if result.narrative and not skip_llm:
                self.console.print("\n[bold]Writing output files...[/bold]")
                markdown_path = self._output_path / "history_story.md"
                OutputWriter.write_markdown(
                    result.narrative, result.phases, str(markdown_path), self._repo_name
                )
                self.print_success(f"Wrote {markdown_path}")

                json_path = self._output_path / "history_data.json"
                OutputWriter.write_json(
                    result.narrative, result.phases, str(json_path),
                    repo_path=str(self._repo_path)
                )
                self.print_success(f"Wrote {json_path}")

            # Report summary
            self.print_success("\nAdaptive analysis complete!")
            self.console.print(f"  Phases analyzed: {len(result.phases)}")
            self.console.print(f"  Discoveries: {len(result.discoveries)}")
            self.console.print(f"  Decisions made: {result.total_decisions}")
            self.console.print(f"  Iterations: {result.total_iterations}")

            # Show key findings
            if result.key_findings:
                self.console.print("\n[bold]Key Findings:[/bold]")
                for finding in result.key_findings[:5]:
                    self.console.print(f"  - {finding}")

            # Show risks
            if result.risk_summary:
                self.console.print("\n[bold yellow]Risks Identified:[/bold yellow]")
                for risk in result.risk_summary[:3]:
                    self.console.print(f"  [yellow]! {risk[:80]}...[/yellow]" if len(risk) > 80 else f"  [yellow]! {risk}[/yellow]")

            self.console.print(f"\nOutput written to: {self._output_path.resolve()}\n")

        except Exception as e:
            self.print_error(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            self._cleanup()

    def _list_branches(self, branch_manager: BranchManager):
        """List all available branches."""
        self.console.print("[bold]Available Branches:[/bold]\n")
        all_branches = branch_manager.list_all_branches(include_remote=True)

        local_branches = [b for b in all_branches if not b.is_remote]
        remote_branches = [b for b in all_branches if b.is_remote]

        if local_branches:
            self.print_info("Local Branches:")
            for b in sorted(local_branches, key=lambda x: x.name):
                self.console.print(f"  - {b.name} ({b.commit_count:,} commits)")

        if remote_branches:
            self.print_info("\nRemote Branches:")
            for b in sorted(remote_branches, key=lambda x: x.name):
                self.console.print(f"  - {b.name} ({b.commit_count:,} commits)")

        self.print_success(f"\nTotal: {len(all_branches)} branches")

    def _get_branches_to_analyze(self, branch_manager: BranchManager) -> List:
        """Determine which branches to analyze."""
        branches = self.get_option('branches')
        all_branches_flag = self.get_option('all_branches', False)
        exclude_branches = self.get_option('exclude_branches')
        branch = self.get_option('branch', 'HEAD')

        branches_to_analyze = []

        if all_branches_flag:
            branches_to_analyze = branch_manager.list_all_branches(include_remote=True)
            self.print_info(f"Analyzing all branches: {len(branches_to_analyze)} branches")

        elif branches:
            patterns = parse_branch_spec(branches)
            all_available = branch_manager.list_all_branches(include_remote=True)
            branches_to_analyze = branch_manager.filter_branches(all_available, include_patterns=patterns)

            if not branches_to_analyze:
                self.print_error(f"Error: No branches match pattern(s): {branches}")
                sys.exit(1)

            self.print_info(f"Analyzing {len(branches_to_analyze)} branch(es) matching '{branches}'")

        else:
            branch_info = branch_manager.get_branch_by_name(branch)
            if branch_info:
                branches_to_analyze = [branch_info]
            else:
                self.print_warning(f"Warning: Branch '{branch}' not found in branch list, proceeding anyway...")
                branches_to_analyze = []

        # Apply exclusion patterns
        if exclude_branches and branches_to_analyze:
            exclude_patterns = parse_branch_spec(exclude_branches)
            before_count = len(branches_to_analyze)
            branches_to_analyze = branch_manager.filter_branches(
                branches_to_analyze,
                exclude_patterns=exclude_patterns
            )
            excluded_count = before_count - len(branches_to_analyze)
            if excluded_count > 0:
                self.print_warning(f"Excluded {excluded_count} branch(es) matching exclusion patterns")

        return branches_to_analyze

    def _analyze_multiple_branches(self, branches_to_analyze, branch_manager, summarization_strategy):
        """Analyze multiple branches."""
        import click

        skip_llm = self.get_option('skip_llm', False)
        stats = branch_manager.get_branch_statistics(branches_to_analyze)
        total_commits = stats['total_commits']

        self.console.print(f"\n[bold yellow]Multi-Branch Analysis[/bold yellow]")
        self.console.print(f"  Branches: {len(branches_to_analyze)}")
        self.console.print(f"  Estimated total commits: {total_commits:,}")

        if not skip_llm:
            estimated_llm_calls = (total_commits // 50) * len(branches_to_analyze) + len(branches_to_analyze) * 5
            self.console.print(f"  Estimated LLM calls: ~{estimated_llm_calls:,}")
            self.print_warning("\nThis will incur API costs. Use --skip-llm to avoid LLM costs.")

        if total_commits > 10000 and not skip_llm:
            self.print_error(f"\nWARNING: Large analysis detected ({total_commits:,} commits)")
            self.print_error("This may take a long time and incur significant API costs.")
            self.print_warning("Consider using --skip-llm or limiting branches.\n")

            if not click.confirm("Do you want to continue?", default=False):
                self.print_warning("Analysis cancelled.")
                return

        self.console.print()
        self._print_analysis_info()

        base_output_dir = self._output_path
        analyzed_branches = []

        for idx, branch_info in enumerate(branches_to_analyze, 1):
            self.console.print(f"\n[bold cyan]=== Analyzing Branch {idx}/{len(branches_to_analyze)}: {branch_info.name} ===[/bold cyan]\n")

            branch_output = base_output_dir / branch_info.sanitized_name
            self._analyze_branch(
                branch=branch_info.name,
                output=str(branch_output),
                repo_name=f"{self._repo_name} ({branch_info.short_name})",
                summarization_strategy=summarization_strategy
            )
            analyzed_branches.append(branch_info)

        # Generate index report
        self.console.print(f"\n[bold]Generating multi-branch index...[/bold]")
        IndexWriter.write_branch_index(analyzed_branches, base_output_dir, self._repo_name)
        IndexWriter.write_branch_metadata(analyzed_branches, base_output_dir)
        IndexWriter.write_simple_branch_list(analyzed_branches, base_output_dir)

        self.print_success(f"\nMulti-branch analysis complete!")
        self.console.print(f"Analyzed {len(analyzed_branches)} branches")
        self.console.print(f"Index report: {base_output_dir / 'index.md'}\n")

    def _analyze_single_branch_mode(self, branches_to_analyze, summarization_strategy):
        """Analyze a single branch (backward compatible mode)."""
        branch = self.get_option('branch', 'HEAD')

        if branches_to_analyze:
            branch = branches_to_analyze[0].name

        self._print_analysis_info()
        self._analyze_branch(
            branch=branch,
            output=str(self._output_path),
            repo_name=self._repo_name,
            summarization_strategy=summarization_strategy
        )

    def _print_analysis_info(self):
        """Print analysis configuration info."""
        skip_llm = self.get_option('skip_llm', False)
        backend = self.get_option('backend')
        model = self.get_option('model')
        api_key = self.get_option('api_key')
        ollama_url = self.get_option('ollama_url', 'http://localhost:11434')
        strategy = self.get_option('strategy', 'adaptive')

        self.print_info(f"Repository: {self._repo_path}")
        self.print_info(f"Output: {self._output_path}")
        self.print_info(f"Strategy: {strategy}")

        if not skip_llm:
            from ..backends import LLMRouter
            router = LLMRouter(backend=backend, model=model, api_key=api_key, ollama_url=ollama_url)
            self.print_info(f"Backend: {router.backend_type.value}")
            self.print_info(f"Model: {router.model}\n")
        else:
            self.print_warning("Skipping LLM summarization\n")

    def _analyze_branch(self, branch: str, output: str, repo_name: str, summarization_strategy: str):
        """Analyze a single branch."""
        strategy = self.get_option('strategy', 'adaptive')
        chunk_size = self.get_option('chunk_size', 50)
        max_commits = self.get_option('max_commits')
        backend = self.get_option('backend')
        model = self.get_option('model')
        api_key = self.get_option('api_key')
        ollama_url = self.get_option('ollama_url', 'http://localhost:11434')
        skip_llm = self.get_option('skip_llm', False)
        incremental = self.get_option('incremental', False)
        since_commit = self.get_option('since_commit')
        since_date = self.get_option('since_date')
        critical = self.get_option('critical', False)
        directives = self.get_option('directives')
        github_token = self.get_option('github_token')

        # Smart incremental detection and loading
        previous_analysis = None
        existing_phases = []
        starting_loc = 0
        cached_records = None
        cached_phases = None
        auto_incremental = False

        if not incremental and not since_commit and not since_date:
            cached_records, cached_phases = self._load_cached_analysis(output)
            if cached_records and cached_phases:
                previous_analysis = OutputWriter.load_previous_analysis(output)
                if previous_analysis:
                    auto_incremental = True
                    incremental = True
                    metadata = previous_analysis.get('metadata', {})
                    since_commit = metadata.get('last_commit_hash')
                    self.print_info("Smart cache detected: Auto-enabling incremental mode to save costs")
                    self.print_info(f"Analyzing commits since: {since_commit[:8] if since_commit else 'last analysis'}")
                    self.print_info(f"Last analysis: {metadata.get('generated_at', 'unknown')}\n")

        if incremental or since_commit or since_date:
            if not previous_analysis:
                previous_analysis = OutputWriter.load_previous_analysis(output)

            if incremental and not previous_analysis:
                if auto_incremental:
                    self.print_warning("Cache exists but incomplete. Running full analysis...\n")
                    incremental = False
                    auto_incremental = False
                    since_commit = None
                else:
                    self.print_warning("Warning: --incremental specified but no previous analysis found.")
                    self.print_warning("Running full analysis instead...\n")
                    incremental = False
            elif previous_analysis:
                metadata = previous_analysis.get('metadata', {})

                if incremental and not since_commit:
                    since_commit = metadata.get('last_commit_hash')
                    self.print_info(f"Incremental mode: Analyzing commits since {since_commit[:8] if since_commit else 'unknown'}")
                    self.print_info(f"Last analysis: {metadata.get('generated_at', 'unknown')}\n")

                existing_phases = [Phase.from_dict(p) for p in previous_analysis.get('phases', [])]

            if existing_phases and existing_phases[-1].commits:
                starting_loc = existing_phases[-1].commits[-1].loc_total

        # Reload cached data if needed
        if not since_commit and not since_date and not auto_incremental:
            cached_records, cached_phases = self._load_cached_analysis(output)

        # Step 1: Extract git history
        self.console.print("[bold]Step 1: Extracting git history...[/bold]")
        extractor = GitHistoryExtractor(str(self._repo_path))

        if cached_records is not None and not since_commit and not since_date:
            records = cached_records
            self.print_info("Found cached commit history; reusing repo_history.jsonl from previous run.\n")
        else:
            with self.create_progress() as progress:
                task = progress.add_task("Extracting commits...", total=None)

                if since_commit or since_date:
                    records = extractor.extract_incremental(
                        since_commit=since_commit,
                        since_date=since_date,
                        branch=branch
                    )
                    if starting_loc > 0:
                        extractor._calculate_cumulative_loc(records, starting_loc)
                else:
                    records = extractor.extract_history(max_commits=max_commits, branch=branch)

                progress.update(task, completed=True)

        if since_commit or since_date:
            self.print_success(f"Extracted {len(records)} new commits\n")
            if len(records) == 0:
                self.print_warning("No new commits found since last analysis.")
                self.print_success("Repository is up to date!\n")
                return
        else:
            self.print_success(f"Extracted {len(records)} commits\n")

        # Save raw history
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        history_file = output_path / "repo_history.jsonl"
        extractor.save_to_jsonl(records, str(history_file))

        # Step 1.5: Enrich with GitHub context
        github_repo_url = self._get_github_repo_url()
        if github_token and github_repo_url and records:
            self._enrich_with_github(records, github_token, github_repo_url, branch, str(history_file), extractor)

        # Step 2: Chunk into phases
        self.console.print("[bold]Step 2: Chunking into phases...[/bold]")
        chunker = HistoryChunker(strategy)
        kwargs = {}
        if strategy == 'fixed':
            kwargs['chunk_size'] = chunk_size

        phases = self._chunk_phases(
            chunker, records, existing_phases, kwargs,
            cached_phases, since_commit, since_date
        )

        self._display_phase_overview(phases)

        # Save phases
        phases_dir = output_path / "phases"
        chunker.save_phases(phases, str(phases_dir))

        if skip_llm:
            self.print_warning("\nSkipping LLM summarization. Writing basic timeline...")
            timeline_file = output_path / "timeline.md"
            OutputWriter.write_simple_timeline(phases, str(timeline_file))
            self.print_success(f"Wrote timeline to {timeline_file}\n")
            return

        # Show cost estimate
        self._show_cost_estimate(records, backend, model, api_key)

        # Step 3: Summarize phases
        self.console.print("[bold]Step 3: Summarizing phases with LLM...[/bold]")

        use_hierarchical = summarization_strategy == 'hierarchical'
        phases = self._summarize_phases(
            phases, previous_analysis, use_hierarchical,
            backend, model, api_key, ollama_url, critical, directives, str(phases_dir)
        )

        # Step 4: Track storylines with StorylineTracker (NEW - Phase 1 integration)
        self.console.print("[bold]Step 4: Tracking storylines...[/bold]")
        storyline_tracker = self._track_storylines(phases, str(phases_dir), github_token)

        # Step 5: Generate global story
        self.console.print("[bold]Step 5: Generating global narrative...[/bold]")
        stories = self._generate_stories(
            phases, use_hierarchical, backend, model, api_key, ollama_url,
            critical, directives, storyline_tracker, repo_name, output
        )

        # Step 6: Write output
        self.console.print("[bold]Step 6: Writing output files...[/bold]")
        self._write_output(stories, phases, output_path, repo_name, use_hierarchical, storyline_tracker)

        # Success summary
        self.print_success("Analysis complete!\n")
        self.console.print(f"Analyzed {len(records)} commits across {len(phases)} phases")
        self.console.print(f"Output written to: {output_path.resolve()}\n")

    def _load_cached_analysis(self, output_dir: str):
        """Load cached commit history and phases if available."""
        history_file = Path(output_dir) / "repo_history.jsonl"
        phases_dir = Path(output_dir) / "phases"

        cached_records = None
        cached_phases = None

        if history_file.exists():
            try:
                cached_records = GitHistoryExtractor.load_from_jsonl(str(history_file))
            except Exception as exc:
                self.print_warning(f"Warning: Failed to load cached commit history: {exc}")

        if phases_dir.exists():
            try:
                cached_phases = HistoryChunker.load_phases(str(phases_dir))
            except Exception as exc:
                self.print_warning(f"Warning: Failed to load cached phases: {exc}")

        return cached_records, cached_phases

    def _get_github_repo_url(self) -> Optional[str]:
        """Determine GitHub repository URL for enrichment."""
        if not self._repo_handler.is_local and self._repo_handler.repo_info:
            return f"{self._repo_handler.repo_info.org}/{self._repo_handler.repo_info.repo}"
        elif self._repo_handler.is_local:
            try:
                from git import Repo as GitRepo
                git_repo = GitRepo(str(self._repo_path))
                for remote in git_repo.remotes:
                    for url in remote.urls:
                        if 'github.com' in url:
                            from ..github_graphql import parse_github_url
                            owner, repo_n = parse_github_url(url)
                            return f"{owner}/{repo_n}"
            except Exception:
                pass
        return None

    def _enrich_with_github(self, records, github_token, github_repo_url, branch, history_file, extractor):
        """Enrich commits with GitHub PR/review context."""
        self.console.print("[bold]Step 1.5: Enriching with GitHub context...[/bold]")
        try:
            with self.create_progress() as progress:
                task = progress.add_task("Fetching GitHub PR/review data...", total=None)

                github_contexts = enrich_commits_with_github(
                    commits=records,
                    github_token=github_token,
                    repo_url=github_repo_url,
                    branch=branch,
                )

                enriched_count = 0
                for record in records:
                    if record.commit_hash in github_contexts:
                        ctx = github_contexts[record.commit_hash]
                        record.github_context = ctx.to_dict()
                        if ctx.pr_number:
                            enriched_count += 1

                progress.update(task, completed=True)

            self.print_success(f"Enriched {enriched_count} commits with GitHub PR context\n")
            extractor.save_to_jsonl(records, history_file)

        except Exception as e:
            self.print_warning(f"Warning: GitHub enrichment failed: {e}")
            self.print_warning("Continuing without GitHub context...\n")

    def _chunk_phases(self, chunker, records, existing_phases, kwargs, cached_phases, since_commit, since_date):
        """Chunk records into phases, handling incremental mode."""
        if existing_phases and len(records) > 0:
            merge_threshold = 10

            if len(records) < merge_threshold:
                self.print_warning(f"Merging {len(records)} new commits into last phase...")
                last_phase = existing_phases[-1]
                last_phase.commits.extend(records)

                last_phase.commit_count = len(last_phase.commits)
                last_phase.end_date = records[-1].timestamp
                last_phase.total_insertions = sum(c.insertions for c in last_phase.commits)
                last_phase.total_deletions = sum(c.deletions for c in last_phase.commits)
                last_phase.loc_end = records[-1].loc_total
                last_phase.loc_delta = last_phase.loc_end - last_phase.loc_start
                if last_phase.loc_start > 0:
                    last_phase.loc_delta_percent = (last_phase.loc_delta / last_phase.loc_start) * 100

                last_phase.summary = None
                phases = existing_phases
                self.print_success(f"Updated last phase (now {last_phase.commit_count} commits)\n")
            else:
                new_phases = chunker.chunk(records, **kwargs)

                for phase in new_phases:
                    phase.phase_number = len(existing_phases) + phase.phase_number

                phases = existing_phases + new_phases
                self.print_success(f"Created {len(new_phases)} new phases (total: {len(phases)})\n")

        elif cached_phases is not None and not since_commit and not since_date:
            phases = cached_phases
            self.print_info(f"Reusing {len(phases)} cached phases from previous run.\n")
        else:
            phases = chunker.chunk(records, **kwargs)
            self.print_success(f"Created {len(phases)} phases\n")

        return phases

    def _display_phase_overview(self, phases):
        """Display phase overview table."""
        table = self.create_table("Phase Overview")

        table.add_column("Phase", style="cyan", justify="right")
        table.add_column("Period", style="magenta")
        table.add_column("Commits", justify="right")
        table.add_column("LOC Δ", justify="right")
        table.add_column("Events", style="yellow")

        for phase in phases:
            events = []
            if phase.has_large_deletion:
                events.append("×")
            if phase.has_large_addition:
                events.append("+")
            if phase.has_refactor:
                events.append("»")
            if phase.readme_changed:
                events.append("▶")

            table.add_row(
                str(phase.phase_number),
                f"{phase.start_date[:10]} to {phase.end_date[:10]}",
                str(phase.commit_count),
                f"{phase.loc_delta:+,d}",
                " ".join(events)
            )

        self.console.print(table)

    def _show_cost_estimate(self, records, backend, model, api_key):
        """Show estimated LLM cost before analysis."""
        avg_msg_length = sum(len(r.commit_message) for r in records) // max(1, len(records))

        from ..backends.router import LLMRouter
        router_for_estimate = LLMRouter(backend=backend, model=model, api_key=api_key)
        estimate = self._estimate_analysis_cost(
            commit_count=len(records),
            avg_msg_length=avg_msg_length,
            backend=router_for_estimate.backend_type.value,
            model=router_for_estimate.model
        )

        if estimate['cost_usd'] > 0:
            self.console.print(f"\n[bold]Cost Estimate:[/bold]")
            self.console.print(f"  Backend: {estimate['backend']} / {estimate['model']}")
            self.console.print(f"  Estimated tokens: ~{estimate['input_tokens']:,} input + ~{estimate['output_tokens']:,} output")
            self.console.print(f"  Estimated cost: [yellow]${estimate['cost_usd']:.2f}[/yellow]")
            self.console.print(f"  ({estimate['num_phases']} phases to summarize + story generation)")

            if estimate['cost_usd'] > 2.0:
                self.print_info("\n💡 To reduce costs:")
                if estimate['backend'] == 'anthropic':
                    self.console.print("  • Use OpenAI gpt-4o-mini: --backend openai (~4-10x cheaper)")
                self.console.print("  • Use larger chunks: --strategy fixed --chunk-size 100")
                self.console.print("  • Limit commits: --max-commits 500")
            self.console.print()

    def _estimate_analysis_cost(self, commit_count: int, avg_msg_length: int, backend: str, model: str) -> dict:
        """Estimate LLM API cost before analysis."""
        avg_commits_per_phase = 40
        num_phases = max(1, commit_count // avg_commits_per_phase)

        tokens_per_commit = 50 + (avg_msg_length // 4)
        commits_shown_per_phase = min(20, commit_count // num_phases)

        input_tokens_per_phase = commits_shown_per_phase * tokens_per_commit + 500
        output_tokens_per_phase = 600

        phase_summarization_input = num_phases * input_tokens_per_phase
        phase_summarization_output = num_phases * output_tokens_per_phase

        story_input_tokens = num_phases * 400 + 2000
        story_output_tokens = 10000

        total_input_tokens = phase_summarization_input + story_input_tokens
        total_output_tokens = phase_summarization_output + story_output_tokens

        cost_table = {
            ('openai', 'gpt-4o-mini'): (0.150, 0.600),
            ('openai', 'gpt-4o'): (2.50, 10.00),
            ('anthropic', 'claude-sonnet-4-5-20250929'): (3.00, 15.00),
            ('anthropic', 'claude-sonnet-3-5-20240229'): (3.00, 15.00),
            ('anthropic', 'claude-haiku-3-5-20241022'): (0.25, 1.25),
            ('ollama', None): (0, 0),
        }

        input_cost_per_m, output_cost_per_m = cost_table.get((backend, model), (1.0, 5.0))

        estimated_cost = (
            (total_input_tokens / 1_000_000) * input_cost_per_m +
            (total_output_tokens / 1_000_000) * output_cost_per_m
        )

        return {
            'cost_usd': estimated_cost,
            'num_phases': num_phases,
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
            'backend': backend,
            'model': model
        }

    def _summarize_phases(self, phases, previous_analysis, use_hierarchical,
                          backend, model, api_key, ollama_url, critical, directives, phases_dir):
        """Summarize phases with LLM."""
        if use_hierarchical:
            self.print_info("Using hierarchical summarization strategy")
            self.print_warning("Note: This makes more API calls but preserves more details\n")
            summarizer = HierarchicalPhaseSummarizer(
                backend=backend,
                model=model,
                api_key=api_key,
                ollama_url=ollama_url,
            )
            phases_to_summarize = [p for p in phases if p.summary is None or
                                   not getattr(p, 'metadata', {}).get('hierarchical_summary')]
        else:
            summarizer = PhaseSummarizer(
                backend=backend,
                model=model,
                api_key=api_key,
                ollama_url=ollama_url,
                todo_content=self._todo_content,
                critical_mode=critical,
                directives=directives
            )
            phases_to_summarize = [p for p in phases if p.summary is None]

        if previous_analysis and len(phases_to_summarize) < len(phases):
            self.print_info(f"Incremental mode: {len(phases_to_summarize)} phases need summarization "
                           f"({len(phases) - len(phases_to_summarize)} already summarized)")

        with self.create_progress() as progress:
            task = progress.add_task("Summarizing phases...", total=len(phases_to_summarize))

            previous_summaries = []
            for i, phase in enumerate(phases):
                progress.update(task, description=f"Processing phase {i+1}/{len(phases)}...")

                if phase.summary is None:
                    if use_hierarchical:
                        result = summarizer.summarize_phase(phase)
                        phase.summary = result['full_summary']
                        if not hasattr(phase, 'metadata'):
                            phase.metadata = {}
                        phase.metadata['hierarchical_summary'] = result
                        summarizer._save_phase_summary(phase, result, phases_dir)
                    else:
                        context = summarizer._build_context(previous_summaries)
                        summary = summarizer.summarize_phase(phase, context)
                        phase.summary = summary
                        summarizer._save_phase_with_summary(phase, phases_dir)
                    progress.update(task, advance=1)

                previous_summaries.append({
                    'phase_number': phase.phase_number,
                    'summary': phase.summary,
                    'loc_delta': phase.loc_delta,
                })

        if len(phases_to_summarize) > 0:
            self.print_success(f"Summarized {len(phases_to_summarize)} phase(s)\n")
        else:
            self.print_success("All phases already summarized\n")

        return phases

    def _track_storylines(self, phases: List[Phase], phases_dir: str, github_token: Optional[str]) -> StorylineTracker:
        """Track storylines using multi-signal detection.

        This integrates the new StorylineTracker system to replace the
        old simple regex-based approach in PhaseSummarizer.
        """
        from ..summarizer import _parse_storylines

        db_path = str(Path(phases_dir) / "storylines.json")

        tracker = StorylineTracker(
            persist_path=db_path,
            confidence_threshold=0.6,
            min_signals=1,  # Lower threshold for better detection
        )

        total_signals = 0
        total_updated = 0
        total_created = 0
        transitions = []

        with self.create_progress() as progress:
            task = progress.add_task("Processing storylines...", total=len(phases))

            for phase in phases:
                # Extract LLM storylines from summary (if available)
                llm_storylines = None
                if phase.summary:
                    llm_storylines = _parse_storylines(phase.summary)

                # Process phase with multi-signal detection
                result = tracker.process_phase(phase, llm_storylines)

                total_signals += result['signals_detected']
                total_updated += result['storylines_updated']
                total_created += result['storylines_created']
                transitions.extend(result['transitions'])

                progress.update(task, advance=1)

        # Report results
        self.print_success(f"Processed {len(phases)} phases")
        self.console.print(f"  Signals detected: {total_signals}")
        self.console.print(f"  Storylines created: {total_created}")
        self.console.print(f"  Storylines updated: {total_updated}")

        if transitions:
            self.console.print(f"  State transitions: {len(transitions)}")
            for t in transitions[:5]:  # Show first 5 transitions
                self.console.print(f"    - {t['title'][:30]}: {t['from']} → {t['to']}")
            if len(transitions) > 5:
                self.console.print(f"    ... and {len(transitions) - 5} more")

        active_count = len(tracker.database.get_active())
        total_count = len(tracker.database.storylines)
        self.print_info(f"Total storylines: {total_count} ({active_count} active)\n")

        return tracker

    def _generate_stories(self, phases, use_hierarchical, backend, model, api_key, ollama_url,
                          critical, directives, storyline_tracker, repo_name, cache_dir):
        """Generate global narrative stories."""
        # Get storylines for narrative continuity
        storylines = storyline_tracker.get_storylines_for_prompt(limit=10)

        if use_hierarchical:
            storyteller = HierarchicalStoryTeller(
                backend=backend,
                model=model,
                api_key=api_key,
                ollama_url=ollama_url,
            )

            with self.create_progress() as progress:
                task = progress.add_task("Generating hierarchical timeline...", total=None)
                timeline = storyteller.generate_timeline(phases, repo_name=repo_name)
                progress.update(task, completed=True)

            stories = {
                'timeline': timeline,
                'executive_summary': 'See timeline for detailed evolution',
                'technical_evolution': 'See timeline for technical details',
                'deletion_story': '',
                'full_narrative': timeline,
            }

            self.print_success("Generated hierarchical timeline\n")
        else:
            storyteller = StoryTeller(
                backend=backend,
                model=model,
                api_key=api_key,
                ollama_url=ollama_url,
                todo_content=self._todo_content,
                critical_mode=critical,
                directives=directives,
                storylines=storylines
            )

            with self.create_progress() as progress:
                task = progress.add_task("Generating story...", total=None)
                stories = storyteller.generate_global_story(phases, repo_name, cache_dir=cache_dir)
                progress.update(task, completed=True)

            self.print_success("Generated global narrative\n")

        return stories

    def _write_output(self, stories, phases, output_path, repo_name, use_hierarchical, storyline_tracker):
        """Write output files."""
        # Write markdown report
        markdown_path = output_path / "history_story.md"
        OutputWriter.write_markdown(stories, phases, str(markdown_path), repo_name)
        self.print_success(f"Wrote {markdown_path}")

        # Write JSON data with storylines
        json_path = output_path / "history_data.json"
        OutputWriter.write_json(
            stories, phases, str(json_path),
            repo_path=str(self._repo_path),
            storylines=storyline_tracker.get_all_storylines_for_report()
        )
        self.print_success(f"Wrote {json_path}")

        # Write timeline
        timeline_path = output_path / "timeline.md"
        if use_hierarchical:
            timeline_path.parent.mkdir(parents=True, exist_ok=True)
            timeline_path.write_text(stories['timeline'])
        else:
            OutputWriter.write_simple_timeline(phases, str(timeline_path))
        self.print_success(f"Wrote {timeline_path}\n")
