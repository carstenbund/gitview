"""Command-line interface for GitView.

This module provides the CLI entry point and command registration.
Actual command logic is delegated to classes in the commands/ module.
"""

import click
from rich.console import Console

from . import __version__
from .commands import (
    AnalyzeCommand,
    ExtractCommand,
    ChunkCommand,
    TrackFilesCommand,
    FileHistoryCommand,
    InjectHistoryCommand,
    RemoveHistoryCommand,
    CompareBranchesCommand,
)
from .commands.storyline import (
    ListStorylineCommand,
    ShowStorylineCommand,
    ReportStorylineCommand,
    ExportStorylineCommand,
    TimelineStorylineCommand,
)

console = Console()


@click.group()
@click.version_option(version=__version__)
def cli():
    """GitView - Git history analyzer with LLM-powered narrative generation.

    \b
    Extract, chunk, and use LLMs to generate compelling narratives from your
    git repository's history.

    \b
    Quick Start:
      # Using OpenAI GPT (default, most cost-effective)
      export OPENAI_API_KEY="your-key"
      gitview analyze

      # Using Anthropic Claude
      export ANTHROPIC_API_KEY="your-key"
      gitview analyze --backend anthropic

      # Using local Ollama (no API key needed)
      gitview analyze --backend ollama --model llama3

    \b
    See 'gitview analyze --help' for detailed LLM configuration options.
    """
    pass


# ============================================================================
# Main Analysis Commands
# ============================================================================

ANALYZE_HELP = """Analyze git repository and generate narrative history.

\b
This command runs the full pipeline:
  1. Extract git history with detailed metadata
  2. Enrich with GitHub PR/review context (optional, with --github-token)
  3. Chunk commits into meaningful phases/epochs
  4. Summarize each phase using LLM
  5. Track storylines using multi-signal detection
  6. Generate global narrative stories
  7. Write markdown reports and JSON data

\b
REPOSITORY SOURCES:
  Local repository:     gitview analyze --repo /path/to/repo
  GitHub shortcut:      gitview analyze --repo org/repo
  Full HTTPS URL:       gitview analyze --repo https://github.com/org/repo.git

\b
LLM BACKENDS:
  OpenAI (default):     export OPENAI_API_KEY="your-key"
  Anthropic Claude:     gitview analyze --backend anthropic
  Local Ollama (free):  gitview analyze --backend ollama --model llama3

\b
EXAMPLES:
  gitview analyze                           # Analyze current directory
  gitview analyze --repo org/repo           # Analyze GitHub repo
  gitview analyze --skip-llm                # Quick analysis without LLM
  gitview analyze --hierarchical            # Detailed hierarchical mode
  gitview analyze --critical --todo GOALS.md  # Critical examination mode
"""


@cli.command(help=ANALYZE_HELP)
@click.option('--repo', '-r', default=".",
              help="Repository: local path, GitHub shortcut (org/repo), or full URL")
@click.option('--output', '-o', default=None,
              help="Output directory (default: auto-generated based on repo)")
@click.option('--strategy', '-s', type=click.Choice(['fixed', 'time', 'adaptive']),
              default='adaptive',
              help="Chunking strategy: 'adaptive' (default), 'fixed', 'time'")
@click.option('--chunk-size', type=int, default=50,
              help="Commits per chunk when using 'fixed' strategy")
@click.option('--max-commits', type=int,
              help="Maximum commits to analyze (default: all commits)")
@click.option('--branch', default='HEAD',
              help="Branch to analyze (default: HEAD). Use --branches for multiple.")
@click.option('--list-branches', is_flag=True,
              help="List all available branches and exit")
@click.option('--branches',
              help="Analyze specific branches (comma-separated or patterns)")
@click.option('--all-branches', is_flag=True,
              help="Analyze all branches (local and remote)")
@click.option('--exclude-branches',
              help="Exclude branches matching patterns (comma-separated)")
@click.option('--backend', '-b', type=click.Choice(['anthropic', 'openai', 'ollama']),
              help="LLM backend (auto-detected from env vars if not specified)")
@click.option('--model', '-m',
              help="Model identifier (uses backend defaults if not specified)")
@click.option('--api-key',
              help="API key (defaults to ANTHROPIC_API_KEY or OPENAI_API_KEY)")
@click.option('--ollama-url', default='http://localhost:11434',
              help="Ollama server URL (only for --backend ollama)")
@click.option('--repo-name',
              help="Repository name for output (default: directory name)")
@click.option('--summarization-strategy',
              type=click.Choice(['simple', 'hierarchical']),
              default='simple',
              help="Summarization strategy: 'simple' or 'hierarchical'")
@click.option('--hierarchical', is_flag=True,
              help="Shortcut for --summarization-strategy hierarchical")
@click.option('--skip-llm', is_flag=True,
              help="Skip LLM summarization (no API costs)")
@click.option('--incremental', is_flag=True,
              help="Incremental analysis: only process commits since last run")
@click.option('--since-commit',
              help="Extract commits since this commit hash")
@click.option('--since-date',
              help="Extract commits since this date (ISO format: YYYY-MM-DD)")
@click.option('--keep-clone', is_flag=True,
              help="Keep any temporary clone copies")
@click.option('--todo',
              help="Path to todo/goals file for critical examination mode")
@click.option('--critical', is_flag=True,
              help="Enable critical examination mode: focus on gaps and issues")
@click.option('--directives',
              help="Additional directives for custom analysis focus")
@click.option('--github-token',
              envvar='GITHUB_TOKEN',
              help="GitHub token for PR/review context enrichment")
def analyze(**kwargs):
    """Analyze git repository and generate narrative history."""
    cmd = AnalyzeCommand(**kwargs)
    cmd.run()


EXTRACT_HELP = """Extract git history to JSONL file (no LLM needed).

\b
Extracts detailed metadata from git commits without using an LLM.
Useful for quick history extraction or pre-processing.

\b
EXAMPLES:
  gitview extract                          # Extract full history
  gitview extract --output my_history.jsonl
  gitview extract --max-commits 100
  gitview extract --branch develop
"""


@cli.command(help=EXTRACT_HELP)
@click.option('--repo', '-r', default=".",
              help="Path to git repository (default: current directory)")
@click.option('--output', '-o', default="output/repo_history.jsonl",
              help="Output JSONL file path")
@click.option('--max-commits', type=int,
              help="Maximum commits to extract (default: all)")
@click.option('--branch', default='HEAD',
              help="Branch to extract from (default: HEAD)")
def extract(**kwargs):
    """Extract git history to JSONL file."""
    cmd = ExtractCommand(**kwargs)
    cmd.run()


CHUNK_HELP = """Chunk extracted history into phases (no LLM needed).

\b
Takes a JSONL file from 'gitview extract' and splits it into phases.

\b
STRATEGIES:
  adaptive (default): Splits on significant changes
  fixed:              N commits per phase
  time:               Split by time periods

\b
EXAMPLES:
  gitview chunk repo_history.jsonl
  gitview chunk repo_history.jsonl --strategy fixed --chunk-size 25
"""


@cli.command(help=CHUNK_HELP)
@click.argument('history_file', type=click.Path(exists=True))
@click.option('--output', '-o', default="output/phases",
              help="Output directory for phase JSON files")
@click.option('--strategy', '-s', type=click.Choice(['fixed', 'time', 'adaptive']),
              default='adaptive',
              help="Chunking strategy")
@click.option('--chunk-size', type=int, default=50,
              help="Commits per chunk (for 'fixed' strategy)")
def chunk(history_file, **kwargs):
    """Chunk extracted history into phases."""
    kwargs['history_file'] = history_file
    cmd = ChunkCommand(**kwargs)
    cmd.run()


# ============================================================================
# File Tracking Commands
# ============================================================================

@cli.command('track-files')
@click.option('--repo', default='.', help='Path to repository')
@click.option('--output', default='output/file_histories',
              help='Output directory for file histories')
@click.option('--patterns', default=None,
              help='File patterns to track (e.g., "*.py,*.js")')
@click.option('--exclude', default=None,
              help='Exclude patterns (e.g., "*_test.py")')
@click.option('--since-commit', default=None,
              help='Only track changes since this commit')
@click.option('--incremental/--no-incremental', default=True,
              help='Use checkpoint for incremental processing')
@click.option('--max-entries', default=100, type=int,
              help='Maximum history entries per file')
@click.option('--with-ai/--no-ai', default=False,
              help='Generate AI summaries (requires API key)')
@click.option('--backend', default=None,
              help='LLM backend for AI summaries')
@click.option('--model', default=None,
              help='LLM model to use')
def track_files(**kwargs):
    """Track detailed change history for all files in repository."""
    cmd = TrackFilesCommand(**kwargs)
    cmd.run()


@cli.command('file-history')
@click.argument('file_path')
@click.option('--repo', default='.', help='Path to repository')
@click.option('--output', default='output/file_histories',
              help='Output directory where histories are stored')
@click.option('--format', 'output_format', default='text',
              type=click.Choice(['text', 'json']),
              help='Output format (text or json)')
@click.option('--recent', default=None, type=int,
              help='Show only N most recent changes')
def file_history(file_path, **kwargs):
    """Display change history for a specific file."""
    kwargs['file_path'] = file_path
    cmd = FileHistoryCommand(**kwargs)
    cmd.run()


@cli.command('inject-history')
@click.argument('paths', nargs=-1, required=False)
@click.option('--repo', default='.', help='Path to repository')
@click.option('--output', default='output/file_histories',
              help='Output directory where histories are stored')
@click.option('--max-entries', default=10, type=int,
              help='Maximum number of recent changes to include')
@click.option('--dry-run', is_flag=True,
              help='Preview changes without modifying files')
@click.option('--all', 'inject_all', is_flag=True,
              help='Inject headers into all tracked files')
def inject_history(paths, **kwargs):
    """Inject file change history as header comments."""
    kwargs['paths'] = paths
    cmd = InjectHistoryCommand(**kwargs)
    cmd.run()


@cli.command('remove-history')
@click.argument('paths', nargs=-1, required=False)
@click.option('--repo', default='.', help='Path to repository')
@click.option('--dry-run', is_flag=True,
              help='Preview changes without modifying files')
@click.option('--all', 'remove_all', is_flag=True,
              help='Remove headers from all files')
def remove_history(paths, **kwargs):
    """Remove injected file change history headers."""
    kwargs['paths'] = paths
    cmd = RemoveHistoryCommand(**kwargs)
    cmd.run()


@cli.command('compare-branches')
@click.argument('branch_a')
@click.argument('branch_b')
@click.option('--repo', default='.', help='Path to repository')
@click.option('--output', default='output/branch_comparisons',
              help='Output directory for comparison results')
@click.option('--patterns', default=None,
              help='File patterns to compare (e.g., "*.py,*.js")')
@click.option('--exclude', default=None,
              help='Exclude patterns')
@click.option('--top-n', default=10, type=int,
              help='Number of top divergent files to show')
def compare_branches(branch_a, branch_b, **kwargs):
    """Compare file histories between two git branches."""
    kwargs['branch_a'] = branch_a
    kwargs['branch_b'] = branch_b
    cmd = CompareBranchesCommand(**kwargs)
    cmd.run()


# ============================================================================
# Storyline Commands
# ============================================================================

@cli.group('storyline')
def storyline_group():
    """Manage and explore storylines tracked across phases.

    \b
    Storylines are narrative threads that span multiple phases of development,
    such as feature implementations, refactoring efforts, or bug fix campaigns.

    \b
    Examples:
      gitview storyline list              # List all storylines
      gitview storyline show <id>         # Show details for a storyline
      gitview storyline report            # Generate comprehensive report
      gitview storyline export --json     # Export storylines to JSON
    """
    pass


@storyline_group.command('list')
@click.option('--output', default='output', help='Output directory')
@click.option('--status', type=click.Choice(['all', 'active', 'completed', 'stalled', 'abandoned']),
              default='all', help='Filter by status')
@click.option('--category', default=None, help='Filter by category')
@click.option('--limit', default=20, type=int, help='Maximum storylines to show')
def storyline_list(**kwargs):
    """List all tracked storylines."""
    cmd = ListStorylineCommand(**kwargs)
    cmd.run()


@storyline_group.command('show')
@click.argument('storyline_id')
@click.option('--output', default='output', help='Output directory')
def storyline_show(storyline_id, **kwargs):
    """Show detailed information about a specific storyline."""
    kwargs['storyline_id'] = storyline_id
    cmd = ShowStorylineCommand(**kwargs)
    cmd.run()


@storyline_group.command('report')
@click.option('--output', default='output', help='Output directory')
@click.option('--format', 'fmt', type=click.Choice(['markdown', 'terminal']),
              default='terminal', help='Output format')
@click.option('--include-timeline/--no-timeline', default=True,
              help='Include ASCII timeline visualization')
@click.option('--include-themes/--no-themes', default=True,
              help='Include cross-phase theme analysis')
@click.option('--save', default=None, help='Save report to file')
def storyline_report(**kwargs):
    """Generate a comprehensive storyline report."""
    cmd = ReportStorylineCommand(**kwargs)
    cmd.run()


@storyline_group.command('export')
@click.option('--output', default='output', help='Output directory')
@click.option('--format', 'fmt', type=click.Choice(['json', 'csv']),
              default='json', help='Export format')
@click.option('--dest', default=None, help='Destination file path')
@click.option('--status', type=click.Choice(['all', 'active', 'completed', 'stalled']),
              default='all', help='Filter by status')
def storyline_export(**kwargs):
    """Export storylines to JSON or CSV format."""
    cmd = ExportStorylineCommand(**kwargs)
    cmd.run()


@storyline_group.command('timeline')
@click.option('--output', default='output', help='Output directory')
def storyline_timeline(**kwargs):
    """Display ASCII timeline of storyline arcs."""
    cmd = TimelineStorylineCommand(**kwargs)
    cmd.run()


def main():
    """Main entry point."""
    cli()


if __name__ == '__main__':
    main()
