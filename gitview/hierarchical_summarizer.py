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
        signals = self._build_cluster_signals(cluster)

        prompt = f"""You are a senior software architect performing a forensic review of this commit cluster. Provide an analytical breakdown, not a story.

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

        if signals:
            prompt += "\n**Structural Signals:**\n"
            for signal in signals:
                prompt += f"- {signal}\n"

        prompt += """
Produce a 3-part investigation:
- **Facts:** Concisely state the concrete changes (files touched, scale, notable additions/removals).
- **Interpretation:** Infer the intent and problem being solved; connect commits to an underlying strategy.
- **Implications:** Call out risks, technical debt, or architectural shifts exposed by this cluster.

Keep the response tight (3-4 sentences total) and make the analysis explicit. Avoid storytelling language."""

        messages = [LLMMessage(role="user", content=prompt)]
        response = self.router.generate(messages, max_tokens=200)

        return response.content.strip()

    def _generate_phase_narrative(self, phase: Phase,
                                  clusters: List[CommitCluster]) -> str:
        """Generate complete narrative for the phase."""
        structural_signals = self._build_structural_signals(clusters)

        prompt = f"""Analyze this development phase like a forensic architecture review.

**Phase Overview:**
- Time Period: {phase.start_date[:10]} to {phase.end_date[:10]}
- Total Commits: {phase.commit_count}
- LOC Change: {phase.loc_delta:+,d} ({phase.loc_delta_percent:+.1f}%)
- Authors: {', '.join(phase.authors)}

**Structural Signals:**
{structural_signals}

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
Deliver a 3-layer investigative report:
1. **Facts:** Key themes, structural movements, and high-churn areas revealed by the clusters.
2. **Interpretation:** The inferred objectives, problem-solving tactics, and architectural direction.
3. **Implications:** Risk surfaces, debt tradeoffs, and downstream consequences for maintainability or stability.

Use the cluster summaries and structural signals as evidence. Be direct and analytical—avoid storytelling language.

Write the investigation now:"""

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

    def _build_structural_signals(self, clusters: List[CommitCluster]) -> str:
        """Build structural signals to guide investigative summaries."""
        top_files: Dict[str, int] = {}
        risk_flags: List[str] = []

        for cluster in clusters:
            for path, count in cluster.file_changes.items():
                top_files[path] = top_files.get(path, 0) + count

            change_volume = cluster.total_insertions + cluster.total_deletions
            if change_volume >= 1500:
                risk_flags.append(
                    f"Cluster '{cluster.cluster_type}' has heavy churn (+{cluster.total_insertions}/-{cluster.total_deletions})."
                )
            if cluster.total_deletions > cluster.total_insertions * 1.5 and cluster.total_deletions > 300:
                risk_flags.append(
                    f"Cluster '{cluster.cluster_type}' is deletion-heavy, suggesting removals or rewrites."
                )

        sorted_files = sorted(top_files.items(), key=lambda x: x[1], reverse=True)[:5]
        file_signal = (
            "High-churn files: " + ", ".join(f"{path} (x{count})" for path, count in sorted_files)
            if sorted_files else "No concentrated file churn detected."
        )

        if not risk_flags:
            risk_flags.append("No major risk flags detected from volume heuristics.")

        return "\n".join([f"- {file_signal}"] + [f"- {flag}" for flag in risk_flags])

    def _build_cluster_signals(self, cluster: CommitCluster) -> List[str]:
        """Derive cluster-level signals for deeper analysis."""
        signals = []
        change_volume = cluster.total_insertions + cluster.total_deletions

        if change_volume >= 800:
            signals.append(
                f"Large change volume: +{cluster.total_insertions}/-{cluster.total_deletions} lines indicates significant movement."
            )
        if cluster.total_deletions > cluster.total_insertions * 1.5 and cluster.total_deletions > 200:
            signals.append("Deletion-heavy changes hint at removals, reversions, or major rewrites.")

        top_files = sorted(cluster.file_changes.items(), key=lambda x: x[1], reverse=True)[:3]
        if top_files:
            signals.append(
                "High-churn files: " + ", ".join(f"{path} (x{count})" for path, count in top_files)
            )

        if not signals:
            signals.append("No anomalous churn detected beyond routine changes.")

        return signals

    def _save_phase_summary(self, phase: Phase, result: Dict[str, Any],
                           output_dir: str):
        """Save detailed summary to file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        summary_file = output_path / f"phase_{phase.phase_number:02d}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(result, f, indent=2)

    def _save_phase_with_summary(self, phase: Phase, output_dir: str):
        """Persist phase data with generated summary."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        phase_file = output_path / f"phase_{phase.phase_number:02d}.json"
        with open(phase_file, 'w') as f:
            json.dump(phase.to_dict(), f, indent=2)


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
