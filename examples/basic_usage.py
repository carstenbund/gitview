#!/usr/bin/env python3
"""Basic usage example for GitView."""

from gitview.extractor import extract_git_history
from gitview.chunker import chunk_history
from gitview.writer import OutputWriter

def main():
    """Run basic example without LLM."""

    print("GitView - Basic Example")
    print("=" * 50)

    # Extract git history
    print("\n1. Extracting git history...")
    records = extract_git_history(
        repo_path=".",
        output_path="output/example_history.jsonl",
        max_commits=100  # Limit for example
    )
    print(f"   ✓ Extracted {len(records)} commits")

    # Chunk into phases
    print("\n2. Chunking into phases...")
    phases = chunk_history(
        records,
        strategy="adaptive",
        output_dir="output/example_phases"
    )
    print(f"   ✓ Created {len(phases)} phases")

    # Display phase overview
    print("\n3. Phase Overview:")
    print("-" * 50)
    for phase in phases:
        print(f"   Phase {phase.phase_number}: "
              f"{phase.start_date[:10]} to {phase.end_date[:10]}")
        print(f"      Commits: {phase.commit_count}, "
              f"LOC Δ: {phase.loc_delta:+,d} ({phase.loc_delta_percent:+.1f}%)")

    # Write simple timeline
    print("\n4. Writing timeline...")
    OutputWriter.write_simple_timeline(
        phases,
        "output/example_timeline.md"
    )
    print("   ✓ Timeline written to output/example_timeline.md")

    print("\n" + "=" * 50)
    print("Example complete!")
    print("\nTo generate LLM-powered narratives, set ANTHROPIC_API_KEY")
    print("and run: gitview analyze")


if __name__ == "__main__":
    main()
