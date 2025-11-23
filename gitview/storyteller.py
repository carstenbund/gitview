"""Generate global narrative from phase summaries."""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from .chunker import Phase
from .backends import LLMRouter, LLMMessage


class StoryTeller:
    """Generate global repository story from phase summaries."""

    def __init__(self, backend: Optional[str] = None, model: Optional[str] = None,
                 api_key: Optional[str] = None, todo_content: Optional[str] = None,
                 critical_mode: bool = False, directives: Optional[str] = None, **kwargs):
        """
        Initialize storyteller with LLM backend.

        Args:
            backend: LLM backend ('anthropic', 'openai', 'ollama')
            model: Model identifier (uses backend defaults if not specified)
            api_key: API key for the backend (if required)
            todo_content: Optional content from todo/goals file for critical examination
            critical_mode: Enable critical examination mode (focus on gaps and issues)
            directives: Additional directives to inject into prompts
            **kwargs: Additional backend parameters
        """
        self.router = LLMRouter(backend=backend, model=model, api_key=api_key, **kwargs)
        self.model = self.router.model
        self.todo_content = todo_content
        self.critical_mode = critical_mode
        self.directives = directives

    def generate_global_story(self, phases: List[Phase],
                             repo_name: Optional[str] = None) -> Dict[str, str]:
        """
        Generate comprehensive repository story from phases.

        Args:
            phases: List of Phase objects with summaries
            repo_name: Optional repository name

        Returns:
            Dict with different story sections:
            - 'executive_summary': High-level overview
            - 'timeline': Chronological timeline with headings
            - 'technical_evolution': Technical architecture evolution
            - 'deletion_story': Story of what was removed and why
            - 'full_narrative': Complete detailed narrative
        """
        # Ensure all phases have summaries
        if any(p.summary is None for p in phases):
            raise ValueError("All phases must have summaries before generating global story")

        print("Generating global story...")

        # Prepare phase summaries data
        phase_summaries = self._prepare_phase_summaries(phases)

        # Generate different story sections
        stories = {}

        print("  - Executive summary...")
        stories['executive_summary'] = self._generate_executive_summary(
            phase_summaries, repo_name
        )

        print("  - Timeline...")
        stories['timeline'] = self._generate_timeline(phase_summaries, repo_name)

        print("  - Technical evolution...")
        stories['technical_evolution'] = self._generate_technical_evolution(
            phase_summaries, repo_name
        )

        print("  - Deletion story...")
        stories['deletion_story'] = self._generate_deletion_story(
            phase_summaries, repo_name
        )

        print("  - Full narrative...")
        stories['full_narrative'] = self._generate_full_narrative(
            phase_summaries, repo_name
        )

        return stories

    def _prepare_phase_summaries(self, phases: List[Phase]) -> List[Dict[str, Any]]:
        """Prepare phase summaries for LLM prompts."""
        summaries = []

        for phase in phases:
            summaries.append({
                'phase_number': phase.phase_number,
                'start_date': phase.start_date[:10],
                'end_date': phase.end_date[:10],
                'commit_count': phase.commit_count,
                'loc_delta': phase.loc_delta,
                'loc_delta_percent': phase.loc_delta_percent,
                'total_insertions': phase.total_insertions,
                'total_deletions': phase.total_deletions,
                'authors': phase.authors,
                'primary_author': phase.primary_author,
                'has_large_deletion': phase.has_large_deletion,
                'has_large_addition': phase.has_large_addition,
                'has_refactor': phase.has_refactor,
                'readme_changed': phase.readme_changed,
                'summary': phase.summary,
            })

        return summaries

    def _generate_executive_summary(self, phase_summaries: List[Dict[str, Any]],
                                   repo_name: Optional[str] = None) -> str:
        """Generate high-level executive summary."""
        prompt = self._build_executive_summary_prompt(phase_summaries, repo_name)

        messages = [LLMMessage(role="user", content=prompt)]
        response = self.router.generate(messages, max_tokens=1500)

        return response.content.strip()

    def _generate_timeline(self, phase_summaries: List[Dict[str, Any]],
                          repo_name: Optional[str] = None) -> str:
        """Generate chronological timeline."""
        prompt = self._build_timeline_prompt(phase_summaries, repo_name)

        messages = [LLMMessage(role="user", content=prompt)]
        response = self.router.generate(messages, max_tokens=3000)

        return response.content.strip()

    def _generate_technical_evolution(self, phase_summaries: List[Dict[str, Any]],
                                     repo_name: Optional[str] = None) -> str:
        """Generate technical architecture evolution story."""
        prompt = self._build_technical_evolution_prompt(phase_summaries, repo_name)

        messages = [LLMMessage(role="user", content=prompt)]
        response = self.router.generate(messages, max_tokens=3000)

        return response.content.strip()

    def _generate_deletion_story(self, phase_summaries: List[Dict[str, Any]],
                                repo_name: Optional[str] = None) -> str:
        """Generate story of code deletions and cleanups."""
        prompt = self._build_deletion_story_prompt(phase_summaries, repo_name)

        messages = [LLMMessage(role="user", content=prompt)]
        response = self.router.generate(messages, max_tokens=2000)

        return response.content.strip()

    def _generate_full_narrative(self, phase_summaries: List[Dict[str, Any]],
                                repo_name: Optional[str] = None) -> str:
        """Generate complete detailed narrative."""
        prompt = self._build_full_narrative_prompt(phase_summaries, repo_name)

        messages = [LLMMessage(role="user", content=prompt)]
        response = self.router.generate(messages, max_tokens=4000)

        return response.content.strip()

    def _build_executive_summary_prompt(self, phase_summaries: List[Dict[str, Any]],
                                       repo_name: Optional[str]) -> str:
        """Build prompt for executive summary."""
        repo_title = repo_name or "this repository"

        # Calculate totals
        total_commits = sum(p['commit_count'] for p in phase_summaries)
        total_insertions = sum(p['total_insertions'] for p in phase_summaries)
        total_deletions = sum(p['total_deletions'] for p in phase_summaries)
        all_authors = set()
        for p in phase_summaries:
            all_authors.update(p['authors'])

        if self.critical_mode:
            prompt = f"""You are conducting a critical examination of {repo_title}'s evolution.

**Overall Statistics:**
- Total Phases: {len(phase_summaries)}
- Total Commits: {total_commits:,}
- Total Insertions: +{total_insertions:,} lines
- Total Deletions: -{total_deletions:,} lines
- Contributors: {len(all_authors)}
- Time Span: {phase_summaries[0]['start_date']} to {phase_summaries[-1]['end_date']}
"""
        else:
            prompt = f"""You are writing an executive summary of the evolution of {repo_title}.

**Overall Statistics:**
- Total Phases: {len(phase_summaries)}
- Total Commits: {total_commits:,}
- Total Insertions: +{total_insertions:,} lines
- Total Deletions: -{total_deletions:,} lines
- Contributors: {len(all_authors)}
- Time Span: {phase_summaries[0]['start_date']} to {phase_summaries[-1]['end_date']}
"""

        # Add goals/todo content if provided
        if self.todo_content:
            prompt += f"""
**Project Goals and Objectives:**
{self.todo_content}
"""

        # Add custom directives if provided
        if self.directives:
            prompt += f"""
**Additional Analysis Directives:**
{self.directives}
"""

        prompt += """
**Phase Summaries:**
"""

        for p in phase_summaries:
            prompt += f"\n**Phase {p['phase_number']} ({p['start_date']} to {p['end_date']})**\n"
            prompt += f"- LOC Δ: {p['loc_delta']:+,d} ({p['loc_delta_percent']:+.1f}%)\n"
            prompt += f"- Summary: {p['summary']}\n"

        if self.critical_mode:
            prompt += """
**Your Task:**
Write a critical executive summary (2-3 paragraphs) that:
1. Objectively assesses progress against stated goals and objectives
2. Identifies gaps, delays, or areas not addressed
3. Highlights technical debt, incomplete features, or unresolved issues
4. Evaluates whether development effort aligns with project priorities
5. Notes any concerning patterns or risks

Focus on factual assessment rather than celebration. Identify what's missing or incomplete.

Write the critical executive summary now:"""
        else:
            prompt += """
**Your Task:**
Write a concise executive summary (2-3 paragraphs) that:
1. Provides a high-level overview of the repository's evolution
2. Highlights the major milestones and transformations
3. Identifies key themes across the entire history
4. Summarizes the overall trajectory (growth, maturity, focus areas)

Keep it business-friendly and accessible to non-technical readers while maintaining technical accuracy.

Write the executive summary now:"""

        return prompt

    def _build_timeline_prompt(self, phase_summaries: List[Dict[str, Any]],
                              repo_name: Optional[str]) -> str:
        """Build prompt for timeline generation."""
        repo_title = repo_name or "Repository"

        if self.critical_mode:
            prompt = f"""Create a critical timeline examining {repo_title}'s evolution with focus on progress against objectives.
"""
        else:
            prompt = f"""Create a chronological timeline of {repo_title}'s evolution with clear headings for each phase.
"""

        # Add goals/todo content if provided
        if self.todo_content:
            prompt += f"""
**Project Goals and Objectives:**
{self.todo_content}
"""

        # Add custom directives if provided
        if self.directives:
            prompt += f"""
**Additional Analysis Directives:**
{self.directives}
"""

        prompt += """
**Phase Summaries:**
"""

        for p in phase_summaries:
            prompt += f"\n**Phase {p['phase_number']} ({p['start_date']} to {p['end_date']})**\n"
            prompt += f"- Commits: {p['commit_count']}\n"
            prompt += f"- LOC Δ: {p['loc_delta']:+,d} ({p['loc_delta_percent']:+.1f}%)\n"
            prompt += f"- Authors: {', '.join(p['authors'])}\n"
            prompt += f"- Summary: {p['summary']}\n"

        if self.critical_mode:
            prompt += """
**Your Task:**
Create a critical timeline in markdown format with:
1. A descriptive heading for each phase indicating goal alignment or gaps
2. Date range
3. Progress toward stated objectives
4. Incomplete or missing features
5. Technical issues or delays
6. Divergence from expected priorities

Focus on objective assessment of what was accomplished vs. what should have been done.

Write the critical timeline now:"""
        else:
            prompt += """
**Your Task:**
Create a timeline in markdown format with:
1. A descriptive heading for each phase (e.g., "Early Prototyping", "Major Refactoring", "Stabilization")
2. Date range
3. Key highlights in bullet points
4. Major changes and decisions

Format example:
## Phase 1: Early Prototyping (Jan - Mar 2018)
- Initial commit with basic structure
- Rapid experimentation with...
- ...

Write the timeline now:"""

        return prompt

    def _build_technical_evolution_prompt(self, phase_summaries: List[Dict[str, Any]],
                                         repo_name: Optional[str]) -> str:
        """Build prompt for technical evolution."""
        repo_title = repo_name or "this codebase"

        if self.critical_mode:
            prompt = f"""Critically analyze the technical and architectural evolution of {repo_title}.
"""
        else:
            prompt = f"""Analyze the technical and architectural evolution of {repo_title}.
"""

        # Add goals/todo content if provided
        if self.todo_content:
            prompt += f"""
**Project Goals and Objectives:**
{self.todo_content}
"""

        # Add custom directives if provided
        if self.directives:
            prompt += f"""
**Additional Analysis Directives:**
{self.directives}
"""

        prompt += """
**Phase Summaries:**
"""

        for p in phase_summaries:
            prompt += f"\n**Phase {p['phase_number']}** ({p['start_date']} to {p['end_date']})\n"
            prompt += f"{p['summary']}\n"

        if self.critical_mode:
            prompt += """
**Your Task:**
Write a critical technical assessment that:
1. Evaluates architectural decisions against best practices and project goals
2. Identifies technical debt, design flaws, or suboptimal choices
3. Assesses whether refactorings addressed root issues or just symptoms
4. Questions technology choices and their necessity
5. Notes missing architectural elements or incomplete implementations
6. Identifies areas requiring future technical work

Write from a technical lead's perspective conducting a code review. Be objective and identify issues.

Write the critical technical assessment now:"""
        else:
            prompt += """
**Your Task:**
Write a technical retrospective that:
1. Traces the architectural evolution across phases
2. Identifies major technical decisions and their motivations
3. Highlights refactorings and their impact
4. Discusses technology choices and migrations
5. Notes patterns in how the codebase matured

Write from a senior engineer's perspective, focusing on the technical journey.

Write the technical evolution now:"""

        return prompt

    def _build_deletion_story_prompt(self, phase_summaries: List[Dict[str, Any]],
                                    repo_name: Optional[str]) -> str:
        """Build prompt for deletion story."""
        # Find phases with significant deletions
        deletion_phases = [p for p in phase_summaries if p['has_large_deletion']]

        if self.critical_mode:
            prompt = f"""Analyze what was removed from the codebase, including what should have been removed but wasn't.
"""
        else:
            prompt = f"""Tell the story of what was removed from the codebase and why.
"""

        # Add goals/todo content if provided
        if self.todo_content:
            prompt += f"""
**Project Goals and Objectives:**
{self.todo_content}
"""

        # Add custom directives if provided
        if self.directives:
            prompt += f"""
**Additional Analysis Directives:**
{self.directives}
"""

        prompt += """
**All Phases:**
"""

        for p in phase_summaries:
            prompt += f"\nPhase {p['phase_number']} ({p['start_date']} to {p['end_date']})\n"
            prompt += f"- Deletions: -{p['total_deletions']:,} lines\n"
            prompt += f"- Large Deletion: {p['has_large_deletion']}\n"
            prompt += f"- Summary: {p['summary']}\n"

        if self.critical_mode:
            prompt += """
**Your Task:**
Write a critical analysis of code removal and cleanup:
1. What was removed and was it sufficient?
2. What obsolete code or technical debt remains unaddressed?
3. Were deletions strategic or just reactive?
4. What should have been removed but is still present?
5. Did cleanup efforts align with project priorities?

Focus on identifying incomplete cleanup and lingering technical debt.

Write the critical deletion analysis now:"""
        else:
            prompt += """
**Your Task:**
Write a narrative about the deletion and cleanup efforts:
1. What major components were removed?
2. Why were they removed? (deprecated, replaced, experimental, etc.)
3. How did these deletions improve the codebase?
4. What does this tell us about the project's evolution?

Focus on the story of simplification, refactoring, and evolution through removal.

Write the deletion story now:"""

        return prompt

    def _build_full_narrative_prompt(self, phase_summaries: List[Dict[str, Any]],
                                    repo_name: Optional[str]) -> str:
        """Build prompt for full narrative."""
        repo_title = repo_name or "this repository"

        if self.critical_mode:
            prompt = f"""Write a comprehensive critical assessment of {repo_title}'s evolution.
"""
        else:
            prompt = f"""Write a comprehensive narrative of {repo_title}'s evolution.
"""

        # Add goals/todo content if provided
        if self.todo_content:
            prompt += f"""
**Project Goals and Objectives:**
{self.todo_content}
"""

        # Add custom directives if provided
        if self.directives:
            prompt += f"""
**Additional Analysis Directives:**
{self.directives}
"""

        prompt += """
**Phase Summaries:**
"""

        for p in phase_summaries:
            prompt += f"\n**Phase {p['phase_number']} ({p['start_date']} to {p['end_date']})**\n"
            prompt += f"- Commits: {p['commit_count']}, LOC Δ: {p['loc_delta']:+,d}\n"
            prompt += f"- Authors: {', '.join(p['authors'])}\n"
            prompt += f"{p['summary']}\n"

        if self.critical_mode:
            prompt += """
**Your Task:**
Write a comprehensive critical assessment (multiple paragraphs) that:
1. Evaluates actual progress against stated objectives throughout the timeline
2. Identifies recurring issues, delays, or patterns of misalignment
3. Assesses resource allocation and whether effort matched priorities
4. Highlights gaps in functionality, missing features, or incomplete implementations
5. Questions strategic decisions and their long-term impact
6. Provides specific, actionable insights on what needs improvement

This should read like a thorough project review identifying both accomplishments and shortcomings.
Do not use flowery or celebratory language. Be factual, direct, and focused on gaps.

Write the critical assessment now:"""
        else:
            prompt += """
**Your Task:**
Write a complete, detailed narrative (multiple paragraphs) that:
1. Tells the full story of the repository from beginning to end
2. Weaves together all phases into a coherent narrative
3. Highlights the evolution, challenges, and successes
4. Maintains chronological flow while identifying themes
5. Makes connections between phases
6. Provides both technical depth and big-picture perspective

This should read like a well-crafted story with a beginning, middle, and current state.

Write the full narrative now:"""

        return prompt


def generate_story(phases: List[Phase],
                  repo_name: Optional[str] = None,
                  backend: Optional[str] = None,
                  model: Optional[str] = None,
                  api_key: Optional[str] = None,
                  **kwargs) -> Dict[str, str]:
    """
    Generate global repository story.

    Args:
        phases: List of Phase objects with summaries
        repo_name: Optional repository name
        backend: LLM backend ('anthropic', 'openai', 'ollama')
        model: Model identifier (uses backend defaults if not specified)
        api_key: API key for the backend (if required)
        **kwargs: Additional backend parameters

    Returns:
        Dict with story sections
    """
    storyteller = StoryTeller(backend=backend, model=model, api_key=api_key, **kwargs)
    return storyteller.generate_global_story(phases, repo_name)
