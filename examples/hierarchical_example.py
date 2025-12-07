"""Example: Using hierarchical summarization for large repositories."""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gitview.extractor import load_commit_records
from gitview.chunker import HistoryChunker
from gitview.hierarchical_summarizer import summarize_phases_hierarchical
from gitview.hierarchical_storyteller import generate_hierarchical_timeline


def main():
    """Demonstrate hierarchical summarization."""

    # Load your commits JSON file
    commits_file = "output/commits.json"

    print("="*80)
    print("Hierarchical Summarization Example")
    print("="*80)

    print(f"\n1. Loading commits from {commits_file}...")
    commits = load_commit_records(commits_file)
    print(f"   Loaded {len(commits)} commits")

    print("\n2. Chunking history into phases...")
    chunker = HistoryChunker(strategy="adaptive")
    phases = chunker.chunk(
        commits,
        min_chunk_size=5,   # Smaller phases for more granular analysis
        max_chunk_size=30,  # Prevent oversized phases
    )
    print(f"   Created {len(phases)} phases")

    # Show phase breakdown
    print("\n   Phase breakdown:")
    for phase in phases:
        print(f"   - Phase {phase.phase_number}: {phase.commit_count} commits, "
              f"LOC {phase.loc_delta:+,d} ({phase.loc_delta_percent:+.1f}%)")

    print("\n3. Summarizing phases with hierarchical strategy...")
    print("   (This will make multiple API calls per phase)")

    phases = summarize_phases_hierarchical(
        phases,
        output_dir="output/phases_hierarchical",
        backend='anthropic',  # Change to 'openai' or 'ollama' if preferred
        # api_key='your-key-here',  # Optional, uses env var if not provided
    )

    print("\n4. Generating timeline with preserved details...")
    timeline = generate_hierarchical_timeline(
        phases,
        repo_name="Large Repository",
        backend='anthropic',
    )

    # Save timeline
    output_file = "output/timeline_hierarchical.md"
    with open(output_file, 'w') as f:
        f.write(timeline)

    print(f"\n✓ Timeline saved to {output_file}")
    print("\n" + "="*80)
    print("Timeline Preview:")
    print("="*80)
    print(timeline[:1000] + "\n..." if len(timeline) > 1000 else timeline)

    # Show example of hierarchical metadata
    print("\n" + "="*80)
    print("Example: Phase 1 Hierarchical Summary")
    print("="*80)

    if phases and hasattr(phases[0], 'metadata'):
        meta = phases[0].metadata.get('hierarchical_summary', {})

        print("\nCluster Analysis:")
        for cluster in meta.get('clusters', [])[:3]:
            print(f"\n  {cluster['type'].upper()} ({cluster['commit_count']} commits)")
            print(f"  +{cluster['insertions']}/-{cluster['deletions']} lines")
            print(f"  Summary: {cluster['summary']}")

        print("\nTimeline Context:")
        timeline_ctx = meta.get('timeline_context', {})
        print(f"  Clusters: {timeline_ctx.get('cluster_count', 0)}")
        print(f"  Total Commits: {timeline_ctx.get('total_commits', 0)}")

        print("\n  Highlights by Type:")
        for h in timeline_ctx.get('highlights', [])[:3]:
            print(f"    - {h['type'].title()} ({h['count']} commits): {h['summary'][:100]}...")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nYou need to extract commits first:")
        print("  python -m gitview.cli extract /path/to/repo -o output/commits.json")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
