"""LLM-based phase summarization."""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from .chunker import Phase
from .backends import LLMRouter, LLMMessage


class PhaseSummarizer:
    """Summarize git history phases using LLM."""

    def __init__(self, backend: Optional[str] = None, model: Optional[str] = None,
                 api_key: Optional[str] = None, **kwargs):
        """
        Initialize summarizer with LLM backend.

        Args:
            backend: LLM backend ('anthropic', 'openai', 'ollama')
            model: Model identifier (uses backend defaults if not specified)
            api_key: API key for the backend (if required)
            **kwargs: Additional backend parameters
        """
        self.router = LLMRouter(backend=backend, model=model, api_key=api_key, **kwargs)
        self.model = self.router.model

    def summarize_phase(self, phase: Phase, context: Optional[str] = None) -> str:
        """
        Generate a narrative summary for a single phase.

        Args:
            phase: Phase object to summarize
            context: Optional context from previous phases

        Returns:
            Narrative summary string
        """
        # Prepare phase data for LLM
        phase_data = self._prepare_phase_data(phase)

        # Build prompt
        prompt = self._build_phase_prompt(phase_data, context)

        # Call LLM backend
        messages = [LLMMessage(role="user", content=prompt)]
        response = self.router.generate(messages, max_tokens=2000)

        return response.content.strip()

    def summarize_all_phases(self, phases: List[Phase],
                            output_dir: Optional[str] = None) -> List[Phase]:
        """
        Summarize all phases with context from previous phases.

        Args:
            phases: List of Phase objects
            output_dir: Optional directory to save updated phases

        Returns:
            List of Phase objects with summaries filled in
        """
        previous_summaries = []

        for i, phase in enumerate(phases):
            print(f"Summarizing phase {phase.phase_number}/{len(phases)}...")

            # Build context from previous phases
            context = self._build_context(previous_summaries)

            # Generate summary
            summary = self.summarize_phase(phase, context)
            phase.summary = summary

            # Store for next iteration
            previous_summaries.append({
                'phase_number': phase.phase_number,
                'summary': summary,
                'loc_delta': phase.loc_delta,
            })

            # Save updated phase if output_dir provided
            if output_dir:
                self._save_phase_with_summary(phase, output_dir)

        return phases

    def _prepare_phase_data(self, phase: Phase) -> Dict[str, Any]:
        """Prepare phase data for LLM prompt."""
        # Get commit details
        commits_summary = []
        for commit in phase.commits[:20]:  # Limit to first 20 commits
            commits_summary.append({
                'hash': commit.short_hash,
                'date': commit.timestamp[:10],  # Just the date
                'author': commit.author,
                'message': commit.commit_subject,
                'insertions': commit.insertions,
                'deletions': commit.deletions,
                'files_changed': commit.files_changed,
                'is_refactor': commit.is_refactor,
                'is_large_deletion': commit.is_large_deletion,
                'is_large_addition': commit.is_large_addition,
            })

        # Get significant commits (large changes, refactors)
        significant_commits = []
        for commit in phase.commits:
            if commit.is_large_deletion or commit.is_large_addition or commit.is_refactor:
                significant_commits.append({
                    'hash': commit.short_hash,
                    'message': commit.commit_message,
                    'insertions': commit.insertions,
                    'deletions': commit.deletions,
                    'is_refactor': commit.is_refactor,
                    'is_large_deletion': commit.is_large_deletion,
                    'is_large_addition': commit.is_large_addition,
                })

        # Get README changes
        readme_changes = []
        for i, commit in enumerate(phase.commits):
            if commit.readme_exists and commit.readme_excerpt:
                if i == 0 or i == len(phase.commits) - 1:
                    readme_changes.append({
                        'hash': commit.short_hash,
                        'excerpt': commit.readme_excerpt,
                        'position': 'start' if i == 0 else 'end'
                    })

        # Get comment samples
        comment_samples = []
        for commit in phase.commits:
            if commit.comment_samples:
                comment_samples.extend(commit.comment_samples[:2])
        comment_samples = comment_samples[:5]  # Limit total

        return {
            'phase_number': phase.phase_number,
            'start_date': phase.start_date[:10],
            'end_date': phase.end_date[:10],
            'commit_count': phase.commit_count,
            'loc_start': phase.loc_start,
            'loc_end': phase.loc_end,
            'loc_delta': phase.loc_delta,
            'loc_delta_percent': phase.loc_delta_percent,
            'total_insertions': phase.total_insertions,
            'total_deletions': phase.total_deletions,
            'languages_start': phase.languages_start,
            'languages_end': phase.languages_end,
            'authors': phase.authors,
            'primary_author': phase.primary_author,
            'has_large_deletion': phase.has_large_deletion,
            'has_large_addition': phase.has_large_addition,
            'has_refactor': phase.has_refactor,
            'readme_changed': phase.readme_changed,
            'commits': commits_summary,
            'significant_commits': significant_commits,
            'readme_changes': readme_changes,
            'comment_samples': comment_samples,
        }

    def _build_phase_prompt(self, phase_data: Dict[str, Any],
                           context: Optional[str] = None) -> str:
        """Build prompt for phase summarization."""
        prompt = f"""You are analyzing a phase in a git repository's history. Your task is to write a concise narrative summary of what happened during this phase.

**Phase Overview:**
- Phase Number: {phase_data['phase_number']}
- Time Period: {phase_data['start_date']} to {phase_data['end_date']}
- Commits: {phase_data['commit_count']}
- LOC Change: {phase_data['loc_delta']:+,d} ({phase_data['loc_delta_percent']:+.1f}%)
  - Start: {phase_data['loc_start']:,} LOC
  - End: {phase_data['loc_end']:,} LOC
- Total Changes: +{phase_data['total_insertions']:,} / -{phase_data['total_deletions']:,} lines
- Authors: {', '.join(phase_data['authors'])}
- Primary Author: {phase_data['primary_author']}

**Language Breakdown:**
- Start: {phase_data['languages_start']}
- End: {phase_data['languages_end']}

**Major Events:**
- Large Deletion: {phase_data['has_large_deletion']}
- Large Addition: {phase_data['has_large_addition']}
- Refactoring: {phase_data['has_refactor']}
- README Changed: {phase_data['readme_changed']}

**Commits Summary:**
{json.dumps(phase_data['commits'], indent=2)}

**Significant Commits (Large Changes/Refactors):**
{json.dumps(phase_data['significant_commits'], indent=2)}

**README Changes:**
{json.dumps(phase_data['readme_changes'], indent=2)}

**Comment Samples:**
{json.dumps(phase_data['comment_samples'], indent=2)}
"""

        if context:
            prompt += f"\n**Context from Previous Phases:**\n{context}\n"

        prompt += """
**Your Task:**
Write a concise narrative summary (3-5 paragraphs) that:

1. Describes the main activities during this phase
2. Explains major code additions, deletions, migrations, and cleanups
3. Highlights how the README or documentation evolved
4. Identifies themes from commit messages and comments (TODOs, deprecations, commentary)
5. Explains the intent behind large diffs or refactorings
6. Notes any significant architectural or technical decisions
7. Maintains chronological flow while being concise

Focus on the "why" and "what changed" rather than just listing commits. Make it read like a story of the codebase's evolution.

Write the summary now:"""

        return prompt

    def _build_context(self, previous_summaries: List[Dict[str, Any]]) -> str:
        """Build context string from previous phase summaries."""
        if not previous_summaries:
            return ""

        context_parts = []
        for summary_info in previous_summaries[-3:]:  # Last 3 phases
            context_parts.append(
                f"Phase {summary_info['phase_number']} "
                f"(LOC Δ: {summary_info['loc_delta']:+,d}): "
                f"{summary_info['summary'][:200]}..."
            )

        return "\n\n".join(context_parts)

    def _save_phase_with_summary(self, phase: Phase, output_dir: str):
        """Save phase with updated summary."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        phase_file = output_path / f"phase_{phase.phase_number:02d}.json"
        with open(phase_file, 'w') as f:
            json.dump(phase.to_dict(), f, indent=2)


def summarize_phases(phases: List[Phase],
                     output_dir: str = "output/phases",
                     backend: Optional[str] = None,
                     model: Optional[str] = None,
                     api_key: Optional[str] = None,
                     **kwargs) -> List[Phase]:
    """
    Summarize all phases using LLM backend.

    Args:
        phases: List of Phase objects
        output_dir: Directory to save updated phases
        backend: LLM backend ('anthropic', 'openai', 'ollama')
        model: Model identifier (uses backend defaults if not specified)
        api_key: API key for the backend (if required)
        **kwargs: Additional backend parameters

    Returns:
        List of Phase objects with summaries
    """
    summarizer = PhaseSummarizer(backend=backend, model=model, api_key=api_key, **kwargs)
    return summarizer.summarize_all_phases(phases, output_dir)
