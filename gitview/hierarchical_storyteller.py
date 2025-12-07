"""Enhanced storyteller that uses hierarchical summaries."""

from typing import List, Dict, Any, Optional

from .chunker import Phase
from .backends import LLMRouter, LLMMessage


class HierarchicalStoryTeller:
    """
    Generate repository story using hierarchical phase summaries.

    Instead of truncating phase summaries to 400 chars, this uses the
    hierarchical context with cluster-level details preserved.
    """

    def __init__(self, backend: Optional[str] = None, model: Optional[str] = None,
                 api_key: Optional[str] = None, max_phases_per_prompt: int = 12,
                 **kwargs):
        """Initialize storyteller with LLM backend."""
        self.router = LLMRouter(backend=backend, model=model, api_key=api_key, **kwargs)
        self.max_phases_per_prompt = max_phases_per_prompt

    def generate_timeline(self, phases: List[Phase],
                         repo_name: Optional[str] = None) -> str:
        """
        Generate chronological timeline with preserved details.

        Args:
            phases: List of Phase objects with hierarchical summaries
            repo_name: Optional repository name

        Returns:
            Markdown timeline with detailed phase descriptions
        """
        print("Generating timeline with hierarchical context...")

        # Check if phases have hierarchical metadata
        if not hasattr(phases[0], 'metadata') or 'hierarchical_summary' not in phases[0].metadata:
            raise ValueError(
                "Phases must be summarized with HierarchicalPhaseSummarizer first. "
                "Call summarize_phases_hierarchical() before generating timeline."
            )

        # Prepare phase data with full context
        phase_data = self._prepare_hierarchical_phase_data(phases)

        # Build timeline prompt
        prompt = self._build_hierarchical_timeline_prompt(phase_data, repo_name)

        # Generate timeline
        messages = [LLMMessage(role="user", content=prompt)]
        response = self.router.generate(messages, max_tokens=4000)

        return response.content.strip()

    def _prepare_hierarchical_phase_data(self, phases: List[Phase]) -> List[Dict[str, Any]]:
        """Prepare phase data preserving hierarchical context."""
        data = []

        for phase in phases:
            hierarchical = phase.metadata['hierarchical_summary']
            timeline_ctx = hierarchical['timeline_context']

            phase_info = {
                'phase_number': phase.phase_number,
                'start_date': phase.start_date[:10],
                'end_date': phase.end_date[:10],
                'commit_count': phase.commit_count,
                'loc_delta': phase.loc_delta,
                'loc_delta_percent': phase.loc_delta_percent,
                'authors': phase.authors,
                'primary_author': phase.primary_author,
                # Hierarchical context
                'full_summary': phase.summary,  # Not truncated!
                'highlights': timeline_ctx['highlights'],
                'cluster_count': timeline_ctx['cluster_count'],
            }

            data.append(phase_info)

        return data

    def _build_hierarchical_timeline_prompt(self, phase_data: List[Dict[str, Any]],
                                           repo_name: Optional[str]) -> str:
        """Build timeline prompt with hierarchical phase context."""
        repo_title = repo_name or "Repository"

        prompt = f"""Create a chronological timeline of {repo_title}'s evolution.

Each phase below includes:
1. A full narrative summary (not truncated)
2. Highlights grouped by activity type (features, bugfixes, refactors, etc.)
3. Key PR titles and descriptions

Use this rich context to create descriptive phase headings and detailed highlights.

**Phases:**
"""

        for p in phase_data:
            prompt += f"\n## Phase {p['phase_number']} ({p['start_date']} to {p['end_date']})\n"
            prompt += f"**Metrics:** {p['commit_count']} commits, LOC {p['loc_delta']:+,d} ({p['loc_delta_percent']:+.1f}%)\n"
            prompt += f"**Authors:** {', '.join(p['authors'])}\n"
            prompt += f"**Activity Clusters:** {p['cluster_count']}\n\n"

            # Add full summary (not truncated!)
            prompt += f"**Summary:**\n{p['full_summary']}\n\n"

            # Add highlights by type
            if p['highlights']:
                prompt += "**Key Activities:**\n"
                for h in p['highlights']:
                    prompt += f"- **{h['type'].title()}** ({h['count']} commits): {h['summary']}\n"
                    if 'pr_title' in h:
                        prompt += f"  PR: {h['pr_title']}\n"
                prompt += "\n"

        prompt += """
**Your Task:**
Create a timeline in markdown format where each phase has:

1. **Descriptive Heading**: Not "Phase 1" but something like "Initial Development" or "Major Refactoring"
   - Base the heading on the actual activities and themes in the summary
   - Make it specific to what happened

2. **Date Range**: (YYYY-MM-DD to YYYY-MM-DD)

3. **Detailed Highlights**: 3-5 bullet points covering:
   - Major features or changes (with PR titles when available)
   - Important refactorings or technical decisions
   - Significant bugfixes or cleanups
   - Documentation or infrastructure changes

**Format Example:**
## Phase 1: Initial Lens Design System (2025-11-21 to 2025-11-21)
- Created foundational mklens_json script with lens geometry calculations for aspheric corneal lenses
- Implemented template-based zone generation system supporting 6-zone microlens designs
- Added JFL file writer aligned with legacy format specifications (PR #1)
- Established Python modules for Gauss-Jordan solver and lens code extraction

**Important:**
- Use specific details from the summaries and PR titles
- Don't just say "added features" - say WHICH features
- Don't lose the significance of individual commits
- Preserve technical context and decisions

Write the complete timeline now:"""

        return prompt


def generate_hierarchical_timeline(phases: List[Phase],
                                   repo_name: Optional[str] = None,
                                   backend: Optional[str] = None,
                                   model: Optional[str] = None,
                                   api_key: Optional[str] = None,
                                   **kwargs) -> str:
    """
    Generate timeline using hierarchical approach.

    Args:
        phases: List of Phase objects (must have hierarchical summaries)
        repo_name: Optional repository name
        backend: LLM backend
        model: Model identifier
        api_key: API key
        **kwargs: Additional backend parameters

    Returns:
        Markdown timeline with preserved details
    """
    storyteller = HierarchicalStoryTeller(
        backend=backend,
        model=model,
        api_key=api_key,
        **kwargs
    )
    return storyteller.generate_timeline(phases, repo_name)
