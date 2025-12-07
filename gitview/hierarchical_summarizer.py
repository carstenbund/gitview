"""Hierarchical summarization strategy that preserves commit significance."""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path

from .chunker import Phase
from .significance_analyzer import SignificanceAnalyzer, CommitCluster
from .backends import LLMRouter, LLMMessage


class HierarchicalPhaseSummarizer:
    """
    Summarize phases using hierarchical approach that preserves significance.

    Instead of:
    1. Taking first 20 commits
    2. Creating one summary
    3. Truncating to 400 chars

    We:
    1. Cluster ALL commits semantically
    2. Create mini-summaries for each cluster
    3. Combine into hierarchical phase summary
    4. Preserve key details (PR titles, file changes) separately
    """

    def __init__(self, backend: Optional[str] = None, model: Optional[str] = None,
                 api_key: Optional[str] = None, **kwargs):
        """Initialize with LLM backend."""
        self.router = LLMRouter(backend=backend, model=model, api_key=api_key, **kwargs)
        self.analyzer = SignificanceAnalyzer()

    def summarize_phase(self, phase: Phase) -> Dict[str, Any]:
        """
        Generate hierarchical summary for a phase.

        Returns:
            Dict with:
            - 'full_summary': Complete narrative summary
            - 'clusters': List of cluster summaries
            - 'key_changes': List of most significant changes
            - 'timeline_context': Condensed context optimized for timeline generation
        """
        print(f"  Analyzing {len(phase.commits)} commits...")

        # Step 1: Cluster commits by significance
        clusters = self.analyzer.cluster_commits(phase.commits)
        print(f"  Identified {len(clusters)} semantic clusters")

        # Step 2: Generate mini-summary for each cluster
        cluster_summaries = []
        for i, cluster in enumerate(clusters, 1):
            print(f"    Summarizing cluster {i}/{len(clusters)} ({cluster.cluster_type})...")
            summary = self._summarize_cluster(cluster)
            cluster.summary = summary
            cluster_summaries.append({
                'type': cluster.cluster_type,
                'summary': summary,
                'key_commit': cluster.key_commit.commit_subject,
                'commit_count': len(cluster.commits),
                'insertions': cluster.total_insertions,
                'deletions': cluster.total_deletions,
            })

        # Step 3: Generate full phase narrative
        print(f"  Generating phase narrative...")
        full_summary = self._generate_phase_narrative(phase, clusters)

        # Step 4: Create timeline-optimized context
        timeline_context = self._create_timeline_context(phase, clusters)

        return {
            'full_summary': full_summary,
            'clusters': cluster_summaries,
            'timeline_context': timeline_context,
            'cluster_details': [c.to_dict() for c in clusters],
        }

    def summarize_all_phases(self, phases: List[Phase],
                            output_dir: Optional[str] = None) -> List[Phase]:
        """Summarize all phases with hierarchical approach."""
        for i, phase in enumerate(phases):
            print(f"Summarizing phase {phase.phase_number}/{len(phases)}...")

            result = self.summarize_phase(phase)

            # Store full summary on phase object
            phase.summary = result['full_summary']

            # Store additional context as metadata
            if not hasattr(phase, 'metadata'):
                phase.metadata = {}
            phase.metadata['hierarchical_summary'] = result

            # Save if output dir provided
            if output_dir:
                self._save_phase_summary(phase, result, output_dir)

        return phases

    def _summarize_cluster(self, cluster: CommitCluster) -> str:
        """Generate concise summary for a commit cluster."""
        details = self.analyzer.extract_significant_details(cluster)

        prompt = f"""Summarize this group of related commits in 1-2 sentences.

**Cluster Type:** {cluster.cluster_type}
**Commits:** {len(cluster.commits)}
**Changes:** +{cluster.total_insertions}/-{cluster.total_deletions} lines

**Key Commit:**
- Message: {cluster.key_commit.commit_subject}
"""

        # Add PR context if available
        if 'pr_title' in details:
            prompt += f"- PR Title: {details['pr_title']}\n"
        if 'pr_description' in details:
            prompt += f"- PR Description: {details['pr_description']}\n"
        if 'labels' in details:
            prompt += f"- Labels: {', '.join(details['labels'])}\n"

        # Add file changes
        if details.get('top_files'):
            files_str = ', '.join(f[0] for f in details['top_files'][:3])
            prompt += f"\n**Top Files Changed:** {files_str}\n"

        # Add other commits if cluster is small
        if 'all_commits' in details:
            prompt += "\n**All Commits:**\n"
            for c in details['all_commits']:
                prompt += f"- {c['hash']}: {c['message']}\n"

        prompt += """
Write a 1-2 sentence summary that:
1. Describes WHAT was changed
2. Explains WHY (based on PR description or commit message)
3. Mentions key files if relevant

Focus on the significance and purpose, not just listing commits."""

        messages = [LLMMessage(role="user", content=prompt)]
        response = self.router.generate(messages, max_tokens=200)

        return response.content.strip()

    def _generate_phase_narrative(self, phase: Phase,
                                  clusters: List[CommitCluster]) -> str:
        """Generate complete narrative for the phase."""
        prompt = f"""Write a narrative summary of this development phase.

**Phase Overview:**
- Time Period: {phase.start_date[:10]} to {phase.end_date[:10]}
- Total Commits: {phase.commit_count}
- LOC Change: {phase.loc_delta:+,d} ({phase.loc_delta_percent:+.1f}%)
- Authors: {', '.join(phase.authors)}

**Development Activities (grouped by purpose):**
"""

        # Add cluster summaries
        for i, cluster in enumerate(clusters, 1):
            prompt += f"\n{i}. **{cluster.cluster_type.title()}** ({len(cluster.commits)} commits, +{cluster.total_insertions}/-{cluster.total_deletions})\n"
            prompt += f"   {cluster.summary}\n"

            # Add PR title for context
            if cluster.key_commit.has_github_context():
                pr_title = cluster.key_commit.get_pr_title()
                if pr_title and pr_title not in cluster.summary:
                    prompt += f"   PR: {pr_title}\n"

        prompt += """
**Your Task:**
Write a 2-3 paragraph narrative that:
1. Describes the main themes and objectives of this phase
2. Explains the major changes and their purpose
3. Connects the activities into a coherent story
4. Highlights any significant technical decisions

Use the cluster summaries above as source material. Focus on the "why" and "what changed" rather than listing every commit.

Write the narrative now:"""

        messages = [LLMMessage(role="user", content=prompt)]
        response = self.router.generate(messages, max_tokens=800)

        return response.content.strip()

    def _create_timeline_context(self, phase: Phase,
                                 clusters: List[CommitCluster]) -> Dict[str, Any]:
        """
        Create condensed context optimized for timeline generation.

        This preserves key details without aggressive truncation.
        """
        # Group clusters by type
        by_type = {}
        for cluster in clusters:
            if cluster.cluster_type not in by_type:
                by_type[cluster.cluster_type] = []
            by_type[cluster.cluster_type].append(cluster)

        # Create condensed highlights
        highlights = []
        for cluster_type, type_clusters in by_type.items():
            total_commits = sum(len(c.commits) for c in type_clusters)

            # Get most significant cluster of this type
            main_cluster = max(
                type_clusters,
                key=lambda c: c.total_insertions + c.total_deletions
            )

            highlight = {
                'type': cluster_type,
                'count': total_commits,
                'summary': main_cluster.summary,
            }

            # Add PR title if available
            if main_cluster.key_commit.has_github_context():
                pr_title = main_cluster.key_commit.get_pr_title()
                if pr_title:
                    highlight['pr_title'] = pr_title

            highlights.append(highlight)

        return {
            'highlights': highlights,
            'cluster_count': len(clusters),
            'total_commits': phase.commit_count,
            'loc_delta': phase.loc_delta,
            'loc_delta_percent': phase.loc_delta_percent,
        }

    def _save_phase_summary(self, phase: Phase, result: Dict[str, Any],
                           output_dir: str):
        """Save detailed summary to file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        summary_file = output_path / f"phase_{phase.phase_number:02d}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(result, f, indent=2)


def summarize_phases_hierarchical(phases: List[Phase],
                                  output_dir: str = "output/phases",
                                  backend: Optional[str] = None,
                                  model: Optional[str] = None,
                                  api_key: Optional[str] = None,
                                  **kwargs) -> List[Phase]:
    """
    Summarize phases using hierarchical approach.

    Args:
        phases: List of Phase objects
        output_dir: Directory to save summaries
        backend: LLM backend
        model: Model identifier
        api_key: API key
        **kwargs: Additional backend parameters

    Returns:
        List of Phase objects with hierarchical summaries
    """
    summarizer = HierarchicalPhaseSummarizer(
        backend=backend,
        model=model,
        api_key=api_key,
        **kwargs
    )
    return summarizer.summarize_all_phases(phases, output_dir)
