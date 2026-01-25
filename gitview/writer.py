"""Output writers for git history stories."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from .chunker import Phase


class OutputWriter:
    """Write git history stories to various formats."""

    @staticmethod
    def load_previous_analysis(output_path: str) -> Dict[str, Any]:
        """
        Load previous analysis from JSON file.

        Args:
            output_path: Path to output directory containing history_data.json

        Returns:
            Dict with previous analysis data, or None if not found
        """
        json_file = Path(output_path) / "history_data.json"
        if not json_file.exists():
            return None

        try:
            with open(json_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    @staticmethod
    def write_markdown(stories: Dict[str, str], phases: List[Phase],
                      output_path: str, repo_name: str = "Repository"):
        """
        Write comprehensive markdown report.

        Args:
            stories: Dict of story sections from storyteller
            phases: List of Phase objects
            output_path: Path to output markdown file
            repo_name: Repository name for title
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            # Header
            f.write(f"# Evolution of {repo_name}\n\n")
            f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            f.write("---\n\n")

            # Table of Contents
            f.write("## Table of Contents\n\n")
            f.write("1. [Executive Summary](#executive-summary)\n")
            f.write("2. [Timeline](#timeline)\n")
            f.write("3. [Full Narrative](#full-narrative)\n")
            f.write("4. [Technical Evolution](#technical-evolution)\n")
            f.write("5. [Story of Deletions](#story-of-deletions)\n")
            f.write("6. [Storylines](#storylines)\n")
            f.write("7. [Phase Details](#phase-details)\n")
            f.write("8. [Statistics](#statistics)\n\n")
            f.write("---\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(stories['executive_summary'])
            f.write("\n\n---\n\n")

            # Timeline
            f.write("## Timeline\n\n")
            f.write(stories['timeline'])
            f.write("\n\n---\n\n")

            # Full Narrative
            f.write("## Full Narrative\n\n")
            f.write(stories['full_narrative'])
            f.write("\n\n---\n\n")

            # Technical Evolution
            f.write("## Technical Evolution\n\n")
            f.write(stories['technical_evolution'])
            f.write("\n\n---\n\n")

            # Deletion Story
            f.write("## Story of Deletions\n\n")
            f.write(stories['deletion_story'])
            f.write("\n\n---\n\n")

            # Storylines section (if available)
            if stories.get('storylines'):
                f.write("## Storylines\n\n")
                f.write(stories['storylines'])
                f.write("\n\n---\n\n")

            # Phase Details
            f.write("## Phase Details\n\n")

            for phase in phases:
                f.write(f"### Phase {phase.phase_number}\n\n")
                f.write(f"**Period:** {phase.start_date[:10]} to {phase.end_date[:10]}\n\n")

                # Stats table
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                f.write(f"| Commits | {phase.commit_count} |\n")
                f.write(f"| LOC Start | {phase.loc_start:,} |\n")
                f.write(f"| LOC End | {phase.loc_end:,} |\n")
                f.write(f"| LOC Delta | {phase.loc_delta:+,d} ({phase.loc_delta_percent:+.1f}%) |\n")
                f.write(f"| Insertions | +{phase.total_insertions:,} |\n")
                f.write(f"| Deletions | -{phase.total_deletions:,} |\n")
                f.write(f"| Authors | {', '.join(phase.authors)} |\n")
                f.write(f"| Primary Author | {phase.primary_author} |\n\n")

                # Events
                events = []
                if phase.has_large_deletion:
                    events.append("Large Deletion")
                if phase.has_large_addition:
                    events.append("Large Addition")
                if phase.has_refactor:
                    events.append("Refactoring")
                if phase.readme_changed:
                    events.append("README Changed")

                if events:
                    f.write(f"**Events:** {' | '.join(events)}\n\n")

                # Summary
                if phase.summary:
                    f.write("**Summary:**\n\n")
                    f.write(phase.summary)
                    f.write("\n\n")

                f.write("---\n\n")

            # Statistics
            f.write("## Statistics\n\n")
            OutputWriter._write_statistics(f, phases)

    @staticmethod
    def _write_statistics(f, phases: List[Phase]):
        """Write statistics section to markdown file."""
        total_commits = sum(p.commit_count for p in phases)
        total_insertions = sum(p.total_insertions for p in phases)
        total_deletions = sum(p.total_deletions for p in phases)

        all_authors = set()
        for p in phases:
            all_authors.update(p.authors)

        # Overall stats
        f.write("### Overall Statistics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Total Phases | {len(phases)} |\n")
        f.write(f"| Total Commits | {total_commits:,} |\n")
        f.write(f"| Total Insertions | +{total_insertions:,} |\n")
        f.write(f"| Total Deletions | -{total_deletions:,} |\n")
        f.write(f"| Net Change | {total_insertions - total_deletions:+,d} |\n")
        f.write(f"| Contributors | {len(all_authors)} |\n")
        f.write(f"| Time Span | {phases[0].start_date[:10]} to {phases[-1].end_date[:10]} |\n\n")

        # Phase-by-phase stats
        f.write("### Phase-by-Phase Statistics\n\n")
        f.write("| Phase | Period | Commits | LOC D | D% | Insertions | Deletions |\n")
        f.write("|-------|--------|---------|-------|-----|------------|------------|\n")

        for p in phases:
            f.write(f"| {p.phase_number} | {p.start_date[:10]} to {p.end_date[:10]} | "
                   f"{p.commit_count} | {p.loc_delta:+,d} | {p.loc_delta_percent:+.1f}% | "
                   f"+{p.total_insertions:,} | -{p.total_deletions:,} |\n")

        f.write("\n")

        # Language evolution
        f.write("### Language Evolution\n\n")

        # Collect all languages
        all_languages = set()
        for p in phases:
            all_languages.update(p.languages_start.keys())
            all_languages.update(p.languages_end.keys())

        if all_languages:
            f.write("Languages detected across phases:\n\n")
            for lang in sorted(all_languages):
                f.write(f"- {lang}\n")
            f.write("\n")

    @staticmethod
    def write_json(stories: Dict[str, str], phases: List[Phase], output_path: str,
                   repo_path: str = None, storyline_data: Optional[Dict[str, Any]] = None):
        """
        Write complete data to JSON file with metadata for incremental analysis.

        Args:
            stories: Dict of story sections
            phases: List of Phase objects
            output_path: Path to output JSON file
            repo_path: Path to git repository (for metadata)
            storyline_data: Optional storyline tracking data
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Calculate metadata from phases
        metadata = {
            'generated_at': datetime.now().isoformat(),
            'total_commits_analyzed': sum(p.commit_count for p in phases),
        }

        # Add last commit info if phases exist
        if phases:
            last_phase = phases[-1]
            if last_phase.commits:
                last_commit = last_phase.commits[-1]
                metadata['last_commit_hash'] = last_commit.commit_hash
                metadata['last_commit_date'] = last_commit.timestamp

        if repo_path:
            metadata['repository_path'] = str(Path(repo_path).resolve())

        # Add storyline summary to metadata
        if storyline_data and 'summary' in storyline_data:
            metadata['storylines'] = storyline_data['summary']

        data = {
            'metadata': metadata,
            'total_phases': len(phases),
            'total_commits': sum(p.commit_count for p in phases),
            'stories': stories,
            'phases': [p.to_dict() for p in phases],
        }

        # Include full storyline data if available
        if storyline_data:
            data['storylines'] = storyline_data

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def write_storyline_report(storyline_data: Dict[str, Any], output_path: str):
        """
        Write a dedicated storyline report.

        Args:
            storyline_data: Storyline data from tracker or reporter
            output_path: Path to output markdown file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            f.write("# Storyline Report\n\n")
            f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            f.write("---\n\n")

            # Summary
            if 'summary' in storyline_data:
                summary = storyline_data['summary']
                f.write("## Summary\n\n")
                f.write(f"- **Total Storylines:** {summary.get('total', 0)}\n")
                f.write(f"- **Completed:** {summary.get('completed', 0)}\n")
                f.write(f"- **Active:** {summary.get('active', 0)}\n")
                f.write(f"- **Stalled:** {summary.get('stalled', 0)}\n\n")
                f.write("---\n\n")

            # Storylines by status
            storylines = storyline_data.get('storylines', [])

            if storylines:
                # Group by status
                completed = [s for s in storylines if s.get('status') == 'completed']
                active = [s for s in storylines if s.get('status') in ('active', 'progressing')]
                stalled = [s for s in storylines if s.get('status') == 'stalled']

                if completed:
                    f.write("## Completed Storylines\n\n")
                    for sl in completed:
                        phases = sl.get('phases_involved', [])
                        phase_str = f"Phases {phases[0]}-{phases[-1]}" if len(phases) > 1 else f"Phase {phases[0]}" if phases else ""
                        f.write(f"### {sl['title']}\n\n")
                        f.write(f"**Category:** {sl.get('category', 'unknown')} | **{phase_str}**\n\n")
                        f.write(f"{sl.get('description', sl.get('last_update', ''))}\n\n")

                if active:
                    f.write("## Active Storylines\n\n")
                    for sl in active:
                        phases = sl.get('phases_involved', [])
                        phase_str = f"Phases {phases[0]}-{phases[-1]}" if len(phases) > 1 else f"Phase {phases[0]}" if phases else ""
                        f.write(f"### {sl['title']}\n\n")
                        f.write(f"**Category:** {sl.get('category', 'unknown')} | **{phase_str}**\n\n")
                        f.write(f"{sl.get('last_update', sl.get('description', ''))}\n\n")

                if stalled:
                    f.write("## Stalled Storylines\n\n")
                    for sl in stalled:
                        phases = sl.get('phases_involved', [])
                        phase_str = f"Phases {phases[0]}-{phases[-1]}" if len(phases) > 1 else f"Phase {phases[0]}" if phases else ""
                        f.write(f"### {sl['title']}\n\n")
                        f.write(f"**Category:** {sl.get('category', 'unknown')} | **{phase_str}**\n\n")
                        f.write(f"{sl.get('last_update', sl.get('description', ''))}\n\n")

    @staticmethod
    def write_simple_timeline(phases: List[Phase], output_path: str):
        """
        Write a simple timeline markdown (without LLM-generated content).

        Args:
            phases: List of Phase objects
            output_path: Path to output markdown file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            f.write("# Repository Timeline\n\n")
            f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

            for phase in phases:
                f.write(f"## Phase {phase.phase_number}: "
                       f"{phase.start_date[:10]} to {phase.end_date[:10]}\n\n")

                f.write(f"- **Commits:** {phase.commit_count}\n")
                f.write(f"- **LOC Change:** {phase.loc_delta:+,d} "
                       f"({phase.loc_delta_percent:+.1f}%)\n")
                f.write(f"- **Authors:** {', '.join(phase.authors)}\n")

                events = []
                if phase.has_large_deletion:
                    events.append("Large Deletion")
                if phase.has_large_addition:
                    events.append("Large Addition")
                if phase.has_refactor:
                    events.append("Refactoring")
                if phase.readme_changed:
                    events.append("README Changed")

                if events:
                    f.write(f"- **Events:** {', '.join(events)}\n")

                f.write("\n")

                # Key commits
                significant = [c for c in phase.commits
                             if c.is_large_deletion or c.is_large_addition or c.is_refactor]

                if significant:
                    f.write("**Significant Commits:**\n\n")
                    for commit in significant[:5]:
                        f.write(f"- `{commit.short_hash}` - {commit.commit_subject} "
                               f"(+{commit.insertions}/-{commit.deletions})\n")
                    f.write("\n")

                f.write("---\n\n")


def write_output(stories: Dict[str, str], phases: List[Phase],
                output_dir: str = "docs", repo_name: str = "Repository"):
    """
    Write all output formats.

    Args:
        stories: Dict of story sections
        phases: List of Phase objects
        output_dir: Directory for output files
        repo_name: Repository name
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Write main markdown report
    markdown_path = output_path / "history_story.md"
    OutputWriter.write_markdown(stories, phases, str(markdown_path), repo_name)
    print(f"Wrote markdown report to: {markdown_path}")

    # Write JSON data
    json_path = output_path / "history_data.json"
    OutputWriter.write_json(stories, phases, str(json_path))
    print(f"Wrote JSON data to: {json_path}")

    # Write simple timeline
    timeline_path = output_path / "timeline.md"
    OutputWriter.write_simple_timeline(phases, str(timeline_path))
    print(f"Wrote timeline to: {timeline_path}")
