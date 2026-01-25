"""Unified storyline extraction from LLM summaries.

This module provides a unified approach to extracting storyline information
from LLM-generated phase summaries, replacing the regex-based approach
in the old summarizer module.
"""

import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ExtractedStoryline:
    """Storyline extracted from LLM summary."""
    title: str
    status: str  # new, continued, completed, stalled
    category: str  # feature, refactor, bugfix, debt, infrastructure, docs
    description: str
    confidence: float = 0.7
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'status': self.status,
            'category': self.category,
            'description': self.description,
            'confidence': self.confidence,
        }


class StorylineExtractor:
    """Extract storylines from LLM-generated phase summaries.

    This class provides multiple extraction strategies:
    1. Structured format: [STATUS:category] title: description
    2. JSON format: {"storylines": [...]}
    3. Natural language heuristics: Pattern matching for narrative elements

    The extractor tries each strategy in order and returns the best results.
    """

    # Pattern for structured format: [STATUS:category] title: description
    STRUCTURED_PATTERN = re.compile(
        r'-\s*\[(\w+):(\w+)\]\s*([^:]+):\s*(.+?)(?=\n-|\Z)',
        re.DOTALL
    )

    # Pattern for section headers
    STORYLINE_SECTION_PATTERN = re.compile(
        r'##\s*(?:Storylines?|Narrative\s+Threads?)\s*\n(.*?)(?=\n##|\Z)',
        re.DOTALL | re.IGNORECASE
    )

    # JSON block pattern
    JSON_PATTERN = re.compile(
        r'```(?:json)?\s*(\{[\s\S]*?"storylines"[\s\S]*?\})\s*```',
        re.IGNORECASE
    )

    # Natural language patterns for implicit storylines
    NARRATIVE_PATTERNS = [
        (r'(?:continued?|continuing|resumed?)\s+(?:work|effort|development)\s+on\s+["\']?([^"\'.\n]+)', 'continued'),
        (r'(?:completed?|finished?|wrapped up)\s+(?:the\s+)?["\']?([^"\'.\n]+)', 'completed'),
        (r'(?:started?|began?|introduced?)\s+(?:work|development|implementation)\s+(?:on\s+)?["\']?([^"\'.\n]+)', 'new'),
        (r'(?:stalled?|paused?|halted?)\s+(?:work\s+on\s+)?["\']?([^"\'.\n]+)', 'stalled'),
    ]

    # Category detection patterns
    CATEGORY_PATTERNS = {
        'feature': [r'feature', r'add(?:ed|ing)?', r'implement', r'new\s+\w+'],
        'refactor': [r'refactor', r'restructur', r'reorganiz', r'clean(?:up|ed)', r'modulariz'],
        'bugfix': [r'fix(?:ed|ing)?', r'bug', r'issue', r'error', r'crash', r'problem'],
        'debt': [r'debt', r'technical\s+debt', r'legacy', r'deprecat'],
        'infrastructure': [r'infra', r'ci/cd', r'build', r'deploy', r'pipeline', r'config'],
        'docs': [r'doc(?:s|umentation)?', r'readme', r'comment', r'wiki'],
        'test': [r'test', r'spec', r'coverage', r'unit\s+test'],
    }

    def __init__(self, confidence_threshold: float = 0.5):
        """Initialize extractor.

        Args:
            confidence_threshold: Minimum confidence for extracted storylines
        """
        self.confidence_threshold = confidence_threshold

    def extract(self, summary: str) -> List[ExtractedStoryline]:
        """Extract storylines from a phase summary.

        Tries multiple extraction strategies in order of reliability:
        1. Structured format (highest confidence)
        2. JSON format
        3. Natural language heuristics (lowest confidence)

        Args:
            summary: LLM-generated phase summary text

        Returns:
            List of extracted storylines
        """
        storylines = []

        # Try structured format first
        structured = self._extract_structured(summary)
        if structured:
            storylines.extend(structured)

        # Try JSON format
        json_storylines = self._extract_json(summary)
        if json_storylines:
            storylines.extend(json_storylines)

        # Try natural language heuristics if no structured storylines found
        if not storylines:
            natural = self._extract_natural_language(summary)
            storylines.extend(natural)

        # Filter by confidence threshold
        return [s for s in storylines if s.confidence >= self.confidence_threshold]

    def _extract_structured(self, summary: str) -> List[ExtractedStoryline]:
        """Extract storylines from structured format.

        Format: - [STATUS:category] title: description
        """
        storylines = []

        # Find storyline section
        section_match = self.STORYLINE_SECTION_PATTERN.search(summary)
        if not section_match:
            return storylines

        section_text = section_match.group(1)

        # Parse individual entries
        matches = self.STRUCTURED_PATTERN.findall(section_text)
        for status, category, title, description in matches:
            storyline = ExtractedStoryline(
                title=title.strip(),
                status=status.lower(),
                category=category.lower(),
                description=description.strip()[:200],
                confidence=0.9,
                raw_text=f"[{status}:{category}] {title}: {description[:50]}..."
            )
            storylines.append(storyline)

        return storylines

    def _extract_json(self, summary: str) -> List[ExtractedStoryline]:
        """Extract storylines from JSON blocks."""
        storylines = []

        json_match = self.JSON_PATTERN.search(summary)
        if not json_match:
            return storylines

        try:
            data = json.loads(json_match.group(1))
            json_storylines = data.get('storylines', [])

            for sl in json_storylines:
                storyline = ExtractedStoryline(
                    title=sl.get('title', 'Unknown'),
                    status=sl.get('status', 'continued').lower(),
                    category=sl.get('category', 'feature').lower(),
                    description=sl.get('description', '')[:200],
                    confidence=0.85,
                    raw_text=json.dumps(sl)[:100]
                )
                storylines.append(storyline)

        except json.JSONDecodeError:
            pass

        return storylines

    def _extract_natural_language(self, summary: str) -> List[ExtractedStoryline]:
        """Extract storylines from natural language patterns."""
        storylines = []
        seen_titles = set()

        for pattern, status in self.NARRATIVE_PATTERNS:
            matches = re.findall(pattern, summary, re.IGNORECASE)
            for match in matches:
                title = match.strip()

                # Skip short or common words
                if len(title) < 5 or title.lower() in seen_titles:
                    continue

                seen_titles.add(title.lower())

                # Detect category
                category = self._detect_category(title, summary)

                storyline = ExtractedStoryline(
                    title=title,
                    status=status,
                    category=category,
                    description=f"Detected from narrative context: {status}",
                    confidence=0.5,
                    raw_text=match[:100]
                )
                storylines.append(storyline)

        return storylines

    def _detect_category(self, title: str, context: str) -> str:
        """Detect storyline category from title and context."""
        text = f"{title} {context}".lower()

        scores = {}
        for category, patterns in self.CATEGORY_PATTERNS.items():
            score = sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))
            if score > 0:
                scores[category] = score

        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        return 'feature'  # Default category

    def extract_from_phases(self, phases) -> Dict[int, List[ExtractedStoryline]]:
        """Extract storylines from multiple phases.

        Args:
            phases: List of Phase objects with summary attribute

        Returns:
            Dict mapping phase_number to list of storylines
        """
        results = {}
        for phase in phases:
            if phase.summary:
                storylines = self.extract(phase.summary)
                if storylines:
                    results[phase.phase_number] = storylines
        return results


# Convenience function for backward compatibility
def parse_storylines(summary: str) -> List[Dict[str, Any]]:
    """Parse storylines from summary text.

    This function provides backward compatibility with the old
    _parse_storylines function in summarizer.py.

    Args:
        summary: Phase summary text

    Returns:
        List of storyline dicts
    """
    extractor = StorylineExtractor()
    storylines = extractor.extract(summary)
    return [sl.to_dict() for sl in storylines]
