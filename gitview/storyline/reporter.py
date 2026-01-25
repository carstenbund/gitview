"""Storyline-centric report generation."""

from typing import List, Dict, Any, Optional
from datetime import datetime

from .models import Storyline, StorylineStatus, StorylineCategory, StorylineDatabase
from .tracker import StorylineTracker


class StorylineReporter:
    """Generate storyline-centric reports and narratives."""

    def __init__(self, tracker: Optional[StorylineTracker] = None,
                 database: Optional[StorylineDatabase] = None):
        """
        Initialize reporter with tracker or database.

        Args:
            tracker: StorylineTracker instance (preferred)
            database: StorylineDatabase instance (alternative)
        """
        if tracker:
            self.database = tracker.database
        elif database:
            self.database = database
        else:
            self.database = StorylineDatabase()

    def generate_storyline_index(self) -> str:
        """
        Generate a markdown index of all storylines.

        Returns:
            Markdown string with storyline index
        """
        lines = ["# Storyline Index\n"]

        # Summary stats
        stats = self._get_stats()
        lines.append("## Summary\n")
        lines.append(f"- **Total Storylines:** {stats['total']}")
        lines.append(f"- **Completed:** {stats['completed']}")
        lines.append(f"- **Active:** {stats['active']}")
        lines.append(f"- **Stalled:** {stats['stalled']}")
        lines.append(f"- **Abandoned:** {stats['abandoned']}")
        lines.append("")

        # Group by status
        completed = self.database.get_by_status(StorylineStatus.COMPLETED)
        active = [s for s in self.database.storylines.values()
                  if s.status in (StorylineStatus.ACTIVE, StorylineStatus.PROGRESSING)]
        stalled = self.database.get_by_status(StorylineStatus.STALLED)
        emerging = self.database.get_by_status(StorylineStatus.EMERGING)

        # Completed storylines
        if completed:
            lines.append("## Completed Storylines\n")
            lines.append("| Title | Category | Phases | Duration |")
            lines.append("|-------|----------|--------|----------|")
            for sl in sorted(completed, key=lambda x: x.last_phase, reverse=True):
                phases = f"{sl.first_phase}→{sl.last_phase}"
                duration = f"{len(sl.phases_involved)} phases"
                lines.append(f"| {sl.title} | {sl.category.value} | {phases} | {duration} |")
            lines.append("")

        # Active storylines
        if active:
            lines.append("## Active Storylines\n")
            lines.append("| Title | Category | Started | Last Update | Confidence |")
            lines.append("|-------|----------|---------|-------------|------------|")
            for sl in sorted(active, key=lambda x: x.last_phase, reverse=True):
                confidence = f"{sl.confidence:.0%}"
                lines.append(f"| {sl.title} | {sl.category.value} | Phase {sl.first_phase} | Phase {sl.last_phase} | {confidence} |")
            lines.append("")

        # Stalled storylines
        if stalled:
            lines.append("## Stalled Storylines\n")
            lines.append("| Title | Category | Started | Last Activity |")
            lines.append("|-------|----------|---------|---------------|")
            for sl in sorted(stalled, key=lambda x: x.last_phase, reverse=True):
                lines.append(f"| {sl.title} | {sl.category.value} | Phase {sl.first_phase} | Phase {sl.last_phase} |")
            lines.append("")

        # Emerging storylines
        if emerging:
            lines.append("## Emerging Storylines (Unconfirmed)\n")
            for sl in emerging:
                lines.append(f"- **{sl.title}** ({sl.category.value}): {sl.description[:100]}")
            lines.append("")

        return '\n'.join(lines)

    def generate_storyline_narrative(self, storyline_id: str) -> str:
        """
        Generate a detailed narrative for a single storyline.

        Args:
            storyline_id: ID of the storyline

        Returns:
            Markdown narrative string
        """
        storyline = self.database.storylines.get(storyline_id)
        if not storyline:
            return f"Storyline {storyline_id} not found."

        lines = [f"# {storyline.title}\n"]

        # Metadata
        lines.append(f"**Category:** {storyline.category.value}")
        lines.append(f"**Status:** {storyline.status.value}")
        lines.append(f"**Confidence:** {storyline.confidence:.0%}")
        lines.append(f"**Phases:** {storyline.first_phase} → {storyline.last_phase} ({len(storyline.phases_involved)} phases)")
        lines.append("")

        # Description
        lines.append("## Overview\n")
        lines.append(storyline.description or storyline.current_summary or "No description available.")
        lines.append("")

        # Timeline of updates
        if storyline.updates:
            lines.append("## Timeline\n")
            for update in storyline.updates:
                lines.append(f"### Phase {update.phase_number}")
                lines.append(f"*{update.timestamp[:10]}*\n")
                lines.append(update.description)
                if update.key_files:
                    lines.append(f"\n**Key files:** {', '.join(update.key_files[:5])}")
                if update.commit_count:
                    lines.append(f"\n**Changes:** {update.commit_count} commits, +{update.insertions}/-{update.deletions} lines")
                lines.append("")

        # Key files
        if storyline.key_files:
            lines.append("## Key Files\n")
            for f in sorted(storyline.key_files)[:15]:
                lines.append(f"- `{f}`")
            lines.append("")

        # Related PRs
        if storyline.pr_numbers:
            lines.append("## Related Pull Requests\n")
            for pr in storyline.pr_numbers:
                lines.append(f"- PR #{pr}")
            lines.append("")

        # Authors
        if storyline.key_authors:
            lines.append("## Contributors\n")
            for author in storyline.key_authors:
                lines.append(f"- {author}")
            lines.append("")

        return '\n'.join(lines)

    def generate_category_report(self) -> str:
        """
        Generate a report grouped by category.

        Returns:
            Markdown report string
        """
        lines = ["# Storylines by Category\n"]

        # Group by category
        by_category: Dict[StorylineCategory, List[Storyline]] = {}
        for sl in self.database.storylines.values():
            if sl.category not in by_category:
                by_category[sl.category] = []
            by_category[sl.category].append(sl)

        # Report each category
        for category in StorylineCategory:
            storylines = by_category.get(category, [])
            if not storylines:
                continue

            lines.append(f"## {category.value.replace('_', ' ').title()}\n")

            # Count by status
            completed = len([s for s in storylines if s.status == StorylineStatus.COMPLETED])
            active = len([s for s in storylines if s.status in (StorylineStatus.ACTIVE, StorylineStatus.PROGRESSING)])
            stalled = len([s for s in storylines if s.status == StorylineStatus.STALLED])

            lines.append(f"*{len(storylines)} storylines: {completed} completed, {active} active, {stalled} stalled*\n")

            for sl in sorted(storylines, key=lambda x: (-x.last_phase, x.title)):
                status_icon = self._get_status_icon(sl.status)
                phases = f"Phase {sl.first_phase}" if sl.first_phase == sl.last_phase else f"Phases {sl.first_phase}-{sl.last_phase}"
                lines.append(f"- {status_icon} **{sl.title}** ({phases})")

            lines.append("")

        return '\n'.join(lines)

    def generate_timeline_view(self) -> str:
        """
        Generate a timeline view showing storyline arcs across phases.

        Returns:
            Markdown/ASCII timeline string
        """
        lines = ["# Storyline Timeline\n"]

        if not self.database.storylines:
            lines.append("No storylines tracked.")
            return '\n'.join(lines)

        # Determine phase range
        min_phase = min(sl.first_phase for sl in self.database.storylines.values())
        max_phase = max(sl.last_phase for sl in self.database.storylines.values())

        # Sort storylines by first phase
        sorted_storylines = sorted(
            self.database.storylines.values(),
            key=lambda x: (x.first_phase, x.title)
        )

        lines.append("```")
        lines.append(f"{'Storyline':<40} " + ''.join(f"{p:^3}" for p in range(min_phase, max_phase + 1)))
        lines.append("-" * (40 + (max_phase - min_phase + 1) * 3))

        for sl in sorted_storylines:
            title = sl.title[:38] + ".." if len(sl.title) > 40 else sl.title
            row = f"{title:<40} "

            for phase in range(min_phase, max_phase + 1):
                if phase < sl.first_phase:
                    row += "   "
                elif phase > sl.last_phase:
                    if sl.status == StorylineStatus.COMPLETED:
                        row += " ✓ " if phase == sl.last_phase + 1 else "   "
                    else:
                        row += "   "
                elif phase in sl.phases_involved:
                    if phase == sl.first_phase:
                        row += " ┌─"
                    elif phase == sl.last_phase:
                        if sl.status == StorylineStatus.COMPLETED:
                            row += "─┘ "
                        elif sl.status == StorylineStatus.STALLED:
                            row += "─╳ "
                        else:
                            row += "─→ "
                    else:
                        row += "───"
                else:
                    # Phase in range but not in phases_involved (gap)
                    row += " · "

            lines.append(row)

        lines.append("```")
        lines.append("")
        lines.append("Legend: ┌─ start, ─┘ completed, ─→ ongoing, ─╳ stalled, · gap")

        return '\n'.join(lines)

    def generate_cross_phase_themes(self) -> str:
        """
        Identify and describe themes spanning multiple storylines.

        Returns:
            Markdown report of cross-cutting themes
        """
        lines = ["# Cross-Phase Themes\n"]

        # Theme 1: Long-running storylines
        long_running = [
            sl for sl in self.database.storylines.values()
            if len(sl.phases_involved) >= 3
        ]

        if long_running:
            lines.append("## Long-Running Initiatives\n")
            lines.append("Storylines that span 3 or more phases:\n")
            for sl in sorted(long_running, key=lambda x: -len(x.phases_involved)):
                duration = len(sl.phases_involved)
                lines.append(f"- **{sl.title}** ({duration} phases): {sl.current_summary[:100]}")
            lines.append("")

        # Theme 2: Recently completed
        completed = [
            sl for sl in self.database.storylines.values()
            if sl.status == StorylineStatus.COMPLETED
        ]
        if completed:
            recent = sorted(completed, key=lambda x: -x.last_phase)[:5]
            lines.append("## Recently Completed\n")
            for sl in recent:
                lines.append(f"- **{sl.title}** (Phase {sl.last_phase}): {sl.current_summary[:80]}")
            lines.append("")

        # Theme 3: Stalled work
        stalled = self.database.get_by_status(StorylineStatus.STALLED)
        if stalled:
            lines.append("## Stalled Work Requiring Attention\n")
            for sl in stalled:
                phases_stalled = self.database.last_phase_analyzed - sl.last_phase
                lines.append(f"- **{sl.title}** (stalled for {phases_stalled} phases): {sl.current_summary[:80]}")
            lines.append("")

        # Theme 4: High-confidence active work
        active = [
            sl for sl in self.database.storylines.values()
            if sl.status in (StorylineStatus.ACTIVE, StorylineStatus.PROGRESSING)
            and sl.confidence >= 0.7
        ]
        if active:
            lines.append("## High-Priority Active Work\n")
            for sl in sorted(active, key=lambda x: -x.confidence)[:5]:
                lines.append(f"- **{sl.title}** ({sl.confidence:.0%} confidence): {sl.current_summary[:80]}")
            lines.append("")

        # Theme 5: Category distribution
        lines.append("## Work Distribution by Category\n")
        category_counts: Dict[str, int] = {}
        for sl in self.database.storylines.values():
            cat = sl.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1

        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            bar = "█" * min(count, 20)
            lines.append(f"- {cat}: {bar} ({count})")
        lines.append("")

        return '\n'.join(lines)

    def generate_full_report(self) -> str:
        """
        Generate a comprehensive storyline report.

        Returns:
            Full markdown report
        """
        sections = [
            self.generate_storyline_index(),
            self.generate_timeline_view(),
            self.generate_cross_phase_themes(),
            self.generate_category_report(),
        ]

        return '\n---\n\n'.join(sections)

    def generate_summary_for_narrative(self) -> str:
        """
        Generate a brief summary suitable for injection into global narratives.

        Returns:
            Concise summary string
        """
        stats = self._get_stats()

        lines = []

        if stats['completed'] > 0:
            completed = self.database.get_by_status(StorylineStatus.COMPLETED)
            titles = [sl.title for sl in completed[:3]]
            lines.append(f"**Completed:** {', '.join(titles)}")

        active = [s for s in self.database.storylines.values()
                  if s.status in (StorylineStatus.ACTIVE, StorylineStatus.PROGRESSING)]
        if active:
            titles = [sl.title for sl in sorted(active, key=lambda x: -x.confidence)[:3]]
            lines.append(f"**Active:** {', '.join(titles)}")

        if stats['stalled'] > 0:
            stalled = self.database.get_by_status(StorylineStatus.STALLED)
            titles = [sl.title for sl in stalled[:2]]
            lines.append(f"**Stalled:** {', '.join(titles)}")

        return '\n'.join(lines) if lines else "No storylines tracked."

    def _get_stats(self) -> Dict[str, int]:
        """Get storyline statistics."""
        return {
            'total': len(self.database.storylines),
            'completed': len(self.database.get_by_status(StorylineStatus.COMPLETED)),
            'active': len([s for s in self.database.storylines.values()
                          if s.status in (StorylineStatus.ACTIVE, StorylineStatus.PROGRESSING)]),
            'stalled': len(self.database.get_by_status(StorylineStatus.STALLED)),
            'abandoned': len(self.database.get_by_status(StorylineStatus.ABANDONED)),
            'emerging': len(self.database.get_by_status(StorylineStatus.EMERGING)),
        }

    def _get_status_icon(self, status: StorylineStatus) -> str:
        """Get icon for status."""
        icons = {
            StorylineStatus.COMPLETED: "✓",
            StorylineStatus.ACTIVE: "●",
            StorylineStatus.PROGRESSING: "▶",
            StorylineStatus.STALLED: "◌",
            StorylineStatus.ABANDONED: "✗",
            StorylineStatus.EMERGING: "○",
        }
        return icons.get(status, "?")


def generate_storyline_section_for_markdown(
    tracker_or_db,
    include_timeline: bool = True,
    include_themes: bool = True,
) -> str:
    """
    Generate storyline section for inclusion in main markdown output.

    Args:
        tracker_or_db: StorylineTracker or StorylineDatabase
        include_timeline: Include ASCII timeline
        include_themes: Include cross-phase themes

    Returns:
        Markdown section string
    """
    if isinstance(tracker_or_db, StorylineTracker):
        reporter = StorylineReporter(tracker=tracker_or_db)
    else:
        reporter = StorylineReporter(database=tracker_or_db)

    sections = [reporter.generate_storyline_index()]

    if include_timeline:
        sections.append(reporter.generate_timeline_view())

    if include_themes:
        sections.append(reporter.generate_cross_phase_themes())

    return '\n\n'.join(sections)
