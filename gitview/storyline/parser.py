"""Robust parsing of storyline data from LLM responses."""

import re
import json
from typing import List, Dict, Any, Optional, Tuple


class StorylineResponseParser:
    """
    Parse storyline data from LLM responses with multiple fallback strategies.

    Strategies (in order of preference):
    1. JSON code block with structured format
    2. Raw JSON object in response
    3. Regex-based extraction from markdown format
    """

    # JSON schema for validation
    EXPECTED_KEYS = {'title', 'status', 'category'}
    VALID_STATUSES = {'new', 'continued', 'completed', 'stalled'}
    VALID_CATEGORIES = {
        'feature', 'refactor', 'bugfix', 'tech_debt', 'debt',
        'infrastructure', 'documentation', 'docs', 'migration',
        'performance', 'security', 'unknown',
    }

    def parse(self, response: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Parse storylines from LLM response.

        Args:
            response: Full LLM response text

        Returns:
            Tuple of (summary_text, storylines_list)
            - summary_text: The narrative summary without storyline section
            - storylines_list: List of parsed storyline dicts
        """
        # Try JSON extraction first (most reliable if LLM follows format)
        storylines = self._try_json_extraction(response)

        if storylines:
            summary = self._extract_summary_section(response)
            return summary, storylines

        # Fall back to regex parsing
        storylines = self._try_regex_extraction(response)
        summary = self._extract_summary_section(response)

        return summary, storylines

    def _try_json_extraction(self, response: str) -> List[Dict[str, Any]]:
        """Try to extract storylines from JSON format."""
        storylines = []

        # Strategy 1: Look for ```json code block
        json_block_match = re.search(
            r'```json\s*\n?(.*?)\n?```',
            response,
            re.DOTALL | re.IGNORECASE,
        )

        if json_block_match:
            try:
                data = json.loads(json_block_match.group(1))
                storylines = self._extract_from_json_data(data)
                if storylines:
                    return storylines
            except json.JSONDecodeError:
                pass

        # Strategy 2: Look for raw JSON object with "storylines" key
        json_obj_match = re.search(
            r'\{\s*"storylines"\s*:\s*\[.*?\]\s*\}',
            response,
            re.DOTALL,
        )

        if json_obj_match:
            try:
                data = json.loads(json_obj_match.group(0))
                storylines = self._extract_from_json_data(data)
                if storylines:
                    return storylines
            except json.JSONDecodeError:
                pass

        # Strategy 3: Look for JSON array directly
        json_array_match = re.search(
            r'\[\s*\{\s*"title".*?\}\s*\]',
            response,
            re.DOTALL,
        )

        if json_array_match:
            try:
                data = json.loads(json_array_match.group(0))
                if isinstance(data, list):
                    storylines = [self._validate_storyline(s) for s in data]
                    return [s for s in storylines if s]
            except json.JSONDecodeError:
                pass

        return []

    def _extract_from_json_data(self, data: Any) -> List[Dict[str, Any]]:
        """Extract storylines from parsed JSON data."""
        storylines = []

        if isinstance(data, dict):
            # Look for storylines key
            if 'storylines' in data and isinstance(data['storylines'], list):
                for item in data['storylines']:
                    validated = self._validate_storyline(item)
                    if validated:
                        storylines.append(validated)
        elif isinstance(data, list):
            for item in data:
                validated = self._validate_storyline(item)
                if validated:
                    storylines.append(validated)

        return storylines

    def _validate_storyline(self, item: Any) -> Optional[Dict[str, Any]]:
        """Validate and normalize a storyline dict."""
        if not isinstance(item, dict):
            return None

        # Check required keys
        if 'title' not in item:
            return None

        # Normalize and validate
        storyline = {
            'title': str(item.get('title', '')).strip(),
            'status': self._normalize_status(item.get('status', 'continued')),
            'category': self._normalize_category(item.get('category', 'feature')),
            'description': str(item.get('description', item.get('summary', '')))[:300],
        }

        # Add optional fields
        if 'confidence' in item:
            try:
                storyline['confidence'] = float(item['confidence'])
            except (ValueError, TypeError):
                storyline['confidence'] = 0.5

        if 'key_commits' in item and isinstance(item['key_commits'], list):
            storyline['key_commits'] = item['key_commits'][:5]

        if 'key_files' in item and isinstance(item['key_files'], list):
            storyline['key_files'] = item['key_files'][:10]

        return storyline

    def _normalize_status(self, status: Any) -> str:
        """Normalize status value."""
        if not status:
            return 'continued'

        status_str = str(status).lower().strip()

        # Map variations
        mappings = {
            'new': 'new',
            'started': 'new',
            'beginning': 'new',
            'continued': 'continued',
            'ongoing': 'continued',
            'active': 'continued',
            'in_progress': 'continued',
            'in progress': 'continued',
            'completed': 'completed',
            'done': 'completed',
            'finished': 'completed',
            'resolved': 'completed',
            'stalled': 'stalled',
            'blocked': 'stalled',
            'paused': 'stalled',
            'on_hold': 'stalled',
            'on hold': 'stalled',
        }

        return mappings.get(status_str, 'continued')

    def _normalize_category(self, category: Any) -> str:
        """Normalize category value."""
        if not category:
            return 'feature'

        cat_str = str(category).lower().strip().replace('-', '_')

        # Map variations
        mappings = {
            'feature': 'feature',
            'feat': 'feature',
            'enhancement': 'feature',
            'refactor': 'refactor',
            'refactoring': 'refactor',
            'cleanup': 'refactor',
            'bugfix': 'bugfix',
            'bug': 'bugfix',
            'fix': 'bugfix',
            'hotfix': 'bugfix',
            'tech_debt': 'tech_debt',
            'debt': 'tech_debt',
            'technical_debt': 'tech_debt',
            'infrastructure': 'infrastructure',
            'infra': 'infrastructure',
            'ci': 'infrastructure',
            'documentation': 'documentation',
            'docs': 'documentation',
            'doc': 'documentation',
            'migration': 'migration',
            'performance': 'performance',
            'perf': 'performance',
            'security': 'security',
        }

        return mappings.get(cat_str, 'feature')

    def _try_regex_extraction(self, response: str) -> List[Dict[str, Any]]:
        """
        Fall back to regex extraction from markdown format.

        Looks for patterns like:
        - [NEW:feature] Title: Description
        - [CONTINUED:refactor] Title: Description
        """
        storylines = []

        # Find the Storylines section
        section_match = re.search(
            r'##\s*Storylines?\s*\n(.*?)(?=\n##|\Z)',
            response,
            re.DOTALL | re.IGNORECASE,
        )

        if not section_match:
            # Try without ## header
            section_match = re.search(
                r'\*\*Storylines?\*\*:?\s*\n(.*?)(?=\n\*\*|\Z)',
                response,
                re.DOTALL | re.IGNORECASE,
            )

        if not section_match:
            return []

        section = section_match.group(1)

        # Pattern: - [STATUS:category] title: description
        pattern = r'-\s*\[(\w+):(\w+)\]\s*([^:\n]+):\s*([^\n]+)'
        matches = re.findall(pattern, section)

        for status, category, title, description in matches:
            storylines.append({
                'title': title.strip(),
                'status': self._normalize_status(status),
                'category': self._normalize_category(category),
                'description': description.strip()[:300],
            })

        # Also try simpler pattern: - [STATUS] title: description
        if not storylines:
            pattern = r'-\s*\[(\w+)\]\s*([^:\n]+):\s*([^\n]+)'
            matches = re.findall(pattern, section)

            for status, title, description in matches:
                storylines.append({
                    'title': title.strip(),
                    'status': self._normalize_status(status),
                    'category': 'feature',  # Default category
                    'description': description.strip()[:300],
                })

        return storylines

    def _extract_summary_section(self, response: str) -> str:
        """Extract the summary portion without storyline section."""
        # Remove JSON code blocks
        summary = re.sub(r'```json.*?```', '', response, flags=re.DOTALL)

        # Remove storyline section
        summary = re.sub(
            r'##\s*Storylines?\s*\n.*?(?=\n##|\Z)',
            '',
            summary,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Remove ** Storylines section
        summary = re.sub(
            r'\*\*Storylines?\*\*:?\s*\n.*?(?=\n\*\*|\Z)',
            '',
            summary,
            flags=re.DOTALL | re.IGNORECASE,
        )

        return summary.strip()


def build_storyline_prompt_section(active_storylines: List[Dict[str, Any]]) -> str:
    """
    Build the storyline tracking section for prompts.

    Args:
        active_storylines: List of active storylines to track

    Returns:
        Prompt section string
    """
    if not active_storylines:
        return """
After your summary, provide storyline tracking in this JSON format:

```json
{
  "storylines": [
    {
      "title": "Storyline name",
      "status": "new|continued|completed|stalled",
      "category": "feature|refactor|bugfix|tech_debt|infrastructure|documentation",
      "description": "Brief 1-2 sentence description of progress",
      "confidence": 0.8
    }
  ]
}
```

Identify any ongoing initiatives, features being built, refactoring efforts, or other narrative threads.
"""

    # Format active storylines
    lines = ["\n**Active Storylines to Track:**"]

    for sl in active_storylines[:8]:
        status_icon = {
            'active': '[ONGOING]',
            'progressing': '[ONGOING]',
            'stalled': '[STALLED]',
            'emerging': '[EMERGING]',
        }.get(sl.get('status', 'active'), '[ONGOING]')

        phases = sl.get('phases_involved', [sl.get('first_phase', '?')])
        phase_str = f"Phase {phases[0]}" if len(phases) == 1 else f"Phases {phases[0]}-{phases[-1]}"

        lines.append(
            f"- {status_icon} **{sl['title']}** ({sl.get('category', 'feature')}, {phase_str}): "
            f"{sl.get('last_update', sl.get('description', ''))[:100]}"
        )

    lines.append("""
For each storyline above, indicate if it was CONTINUED, COMPLETED, or STALLED in this phase.
Also identify any NEW storylines that started.

Provide your storyline tracking in this JSON format:

```json
{
  "storylines": [
    {
      "title": "Exact storyline title from above (or new title)",
      "status": "new|continued|completed|stalled",
      "category": "feature|refactor|bugfix|tech_debt|infrastructure|documentation",
      "description": "Brief update on progress this phase",
      "confidence": 0.8
    }
  ]
}
```
""")

    return '\n'.join(lines)


# Convenience instance
default_parser = StorylineResponseParser()


def parse_storylines(response: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Convenience function to parse storylines from response."""
    return default_parser.parse(response)
