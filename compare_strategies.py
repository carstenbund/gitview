"""Compare old vs new summarization strategies."""

import json
import sys
from pathlib import Path

# Add gitview to path
sys.path.insert(0, str(Path(__file__).parent))

from gitview.extractor import load_commit_records
from gitview.chunker import HistoryChunker
from gitview.summarizer import PhaseSummarizer
from gitview.hierarchical_summarizer import HierarchicalPhaseSummarizer
from gitview.significance_analyzer import analyze_commit_significance


def compare_strategies(commits_file: str, output_dir: str = "output/comparison"):
    """
    Compare old vs new summarization strategies.

    Args:
        commits_file: Path to commits JSON file
        output_dir: Where to save comparison results
    """
    print("Loading commits...")
    commits = load_commit_records(commits_file)
    print(f"Loaded {len(commits)} commits")

    # Create phases
    print("\nChunking history into phases...")
    chunker = HistoryChunker(strategy="adaptive")
    phases = chunker.chunk(commits, min_chunk_size=5, max_chunk_size=30)
    print(f"Created {len(phases)} phases")

    # Analyze first phase in detail
    if not phases:
        print("No phases found!")
        return

    phase = phases[0]
    print(f"\n{'='*80}")
    print(f"ANALYZING PHASE 1: {len(phase.commits)} commits")
    print(f"{'='*80}")

    # ============================================================================
    # OLD STRATEGY
    # ============================================================================
    print("\n" + "="*80)
    print("OLD STRATEGY: First 20 commits, truncated to 400 chars")
    print("="*80)

    # Show what gets processed
    processed_commits = phase.commits[:20]
    ignored_commits = phase.commits[20:]

    print(f"\nProcessed: {len(processed_commits)} commits")
    print(f"Ignored: {len(ignored_commits)} commits")

    if ignored_commits:
        print("\nIgnored commits include:")
        for c in ignored_commits[:5]:
            print(f"  - {c.short_hash}: {c.commit_subject}")
        if len(ignored_commits) > 5:
            print(f"  ... and {len(ignored_commits) - 5} more")

    # Show truncation
    print("\nOld strategy phase data sent to LLM:")
    print("-" * 80)
    old_data = {
        'phase_number': phase.phase_number,
        'commit_count': phase.commit_count,
        'loc_delta': phase.loc_delta,
        'commits_shown': len(processed_commits),
        'commits_ignored': len(ignored_commits),
    }
    print(json.dumps(old_data, indent=2))

    # Simulate old summarizer
    print("\nGenerating summary with old strategy...")
    try:
        old_summarizer = PhaseSummarizer(backend='anthropic')
        old_summary = old_summarizer.summarize_phase(phase)
        print(f"\nOld summary length: {len(old_summary)} chars")
        print(f"Old summary:\n{old_summary}")

        # Show truncation for timeline
        truncated = old_summary[:400] + "…" if len(old_summary) > 400 else old_summary
        print(f"\nTruncated for timeline ({len(truncated)} chars):")
        print(f"{truncated}")
    except Exception as e:
        print(f"Error with old strategy: {e}")
        old_summary = None

    # ============================================================================
    # NEW STRATEGY
    # ============================================================================
    print("\n" + "="*80)
    print("NEW STRATEGY: Hierarchical with semantic clustering")
    print("="*80)

    # Show clustering
    print("\nAnalyzing commit significance...")
    clusters = analyze_commit_significance(phase.commits)

    print(f"\nIdentified {len(clusters)} semantic clusters:")
    for i, cluster in enumerate(clusters, 1):
        print(f"\n  Cluster {i}: {cluster.cluster_type}")
        print(f"    Commits: {len(cluster.commits)}")
        print(f"    Changes: +{cluster.total_insertions}/-{cluster.total_deletions}")
        print(f"    Key commit: {cluster.key_commit.short_hash} - {cluster.key_commit.commit_subject}")

        if cluster.key_commit.has_github_context():
            pr_title = cluster.key_commit.get_pr_title()
            if pr_title:
                print(f"    PR: {pr_title}")

        # Show all commits in small clusters
        if len(cluster.commits) <= 3:
            for c in cluster.commits:
                print(f"      - {c.short_hash}: {c.commit_subject}")

    # Generate hierarchical summary
    print("\nGenerating summary with new strategy...")
    try:
        new_summarizer = HierarchicalPhaseSummarizer(backend='anthropic')
        result = new_summarizer.summarize_phase(phase)

        print(f"\nNew summary length: {len(result['full_summary'])} chars")
        print(f"New summary:\n{result['full_summary']}")

        print(f"\nTimeline context (NOT truncated):")
        print(json.dumps(result['timeline_context'], indent=2))

        print(f"\nCluster summaries:")
        for cs in result['clusters']:
            print(f"\n  {cs['type'].title()} ({cs['commit_count']} commits):")
            print(f"  {cs['summary']}")

        # Save detailed results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / "new_strategy_result.json", 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\nDetailed results saved to {output_path / 'new_strategy_result.json'}")

    except Exception as e:
        print(f"Error with new strategy: {e}")
        import traceback
        traceback.print_exc()

    # ============================================================================
    # COMPARISON
    # ============================================================================
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)

    print(f"""
Old Strategy:
- Commits processed: {len(processed_commits)}/{len(phase.commits)}
- Commits ignored: {len(ignored_commits)}
- Summary length: {len(old_summary) if old_summary else 'N/A'} chars
- Timeline context: ~400 chars (truncated)
- Details preserved: Minimal

New Strategy:
- Commits processed: {len(phase.commits)}/{len(phase.commits)}
- Semantic clusters: {len(clusters)}
- Summary length: {len(result['full_summary']) if 'result' in locals() else 'N/A'} chars
- Timeline context: Full (NOT truncated)
- Details preserved: PR titles, labels, file changes, technical decisions

Key Differences:
1. OLD: Ignores {len(ignored_commits)} commits → NEW: Processes ALL commits
2. OLD: Generic summary → NEW: Clustered by purpose (features, fixes, refactors)
3. OLD: Truncated to 400 chars → NEW: Full context preserved
4. OLD: Loses PR context → NEW: Preserves PR titles and descriptions
5. OLD: No file context → NEW: Top changed files per cluster
    """)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_strategies.py <commits.json>")
        sys.exit(1)

    commits_file = sys.argv[1]
    compare_strategies(commits_file)
