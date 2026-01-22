"""
AI-powered summarization for file changes.

This module provides intelligent summaries of file changes using LLMs,
with caching to avoid redundant API calls and cost optimization.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import json
import hashlib

from .backends.base import LLMMessage
from .backends.router import LLMRouter


@dataclass
class ChangeSummary:
    """AI-generated summary for a file change"""
    commit_hash: str
    file_path: str
    summary: str
    model: str
    generated_at: str
    cache_key: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChangeSummary':
        """Deserialize from dictionary"""
        return cls(**data)


class ChangeAISummarizer:
    """
    Generates AI summaries for file changes with caching and batch processing
    """

    # Cost estimates per 1000 tokens (approximate)
    COST_PER_1K_TOKENS = {
        'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
        'gpt-4o': {'input': 0.0025, 'output': 0.01},
        'claude-sonnet-4-5-20250929': {'input': 0.003, 'output': 0.015},
        'claude-haiku-3-5-20241022': {'input': 0.0008, 'output': 0.004},
        'ollama': {'input': 0.0, 'output': 0.0},  # Local, free
    }

    def __init__(self, llm_router: LLMRouter, cache_path: Optional[Path] = None):
        """
        Initialize AI summarizer

        Args:
            llm_router: LLM router for making API calls
            cache_path: Path to cache file (optional)
        """
        self.llm = llm_router
        self.cache_path = cache_path or Path('output/file_histories/summaries_cache.json')
        self.cache = self._load_cache()
        self.cache_hits = 0
        self.cache_misses = 0

    def _load_cache(self) -> Dict[str, ChangeSummary]:
        """Load summary cache from disk"""
        if not self.cache_path.exists():
            return {}

        try:
            with open(self.cache_path, 'r') as f:
                data = json.load(f)

            # Convert to ChangeSummary objects
            cache = {}
            for key, value in data.items():
                cache[key] = ChangeSummary.from_dict(value)

            return cache
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _save_cache(self):
        """Save summary cache to disk"""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert ChangeSummary objects to dicts
        data = {}
        for key, summary in self.cache.items():
            data[key] = summary.to_dict()

        with open(self.cache_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _generate_cache_key(self, commit_hash: str, file_path: str, diff_snippet: str) -> str:
        """
        Generate cache key from commit, file, and diff

        Args:
            commit_hash: Git commit hash
            file_path: Path to file
            diff_snippet: Diff content

        Returns:
            Cache key hash
        """
        content = f"{commit_hash}:{file_path}:{diff_snippet}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

    def summarize_change(
        self,
        commit_hash: str,
        file_path: str,
        commit_message: str,
        author: str,
        date: str,
        lines_added: int,
        lines_removed: int,
        diff_snippet: str,
        functions_modified: List[str] = None,
        functions_added: List[str] = None,
        force: bool = False
    ) -> str:
        """
        Generate AI summary for a file change

        Args:
            commit_hash: Git commit hash
            file_path: Path to file
            commit_message: Commit message
            author: Author name
            date: Commit date
            lines_added: Number of lines added
            lines_removed: Number of lines removed
            diff_snippet: Code diff snippet
            functions_modified: List of modified functions
            functions_added: List of added functions
            force: Force regeneration (ignore cache)

        Returns:
            AI-generated summary
        """
        # Generate cache key
        cache_key = self._generate_cache_key(commit_hash, file_path, diff_snippet)

        # Check cache
        if not force and cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key].summary

        self.cache_misses += 1

        # Build prompt
        prompt = self._build_prompt(
            file_path=file_path,
            commit_message=commit_message,
            author=author,
            date=date,
            lines_added=lines_added,
            lines_removed=lines_removed,
            diff_snippet=diff_snippet,
            functions_modified=functions_modified or [],
            functions_added=functions_added or []
        )

        # Generate summary
        messages = [
            LLMMessage(role="system", content="You are a code change analyst. Provide concise, technical summaries of code changes."),
            LLMMessage(role="user", content=prompt)
        ]

        try:
            response = self.llm.generate(messages, max_tokens=200)
            summary = response.content.strip()

            # Cache the result
            cache_entry = ChangeSummary(
                commit_hash=commit_hash,
                file_path=file_path,
                summary=summary,
                model=response.model,
                generated_at=datetime.now().isoformat(),
                cache_key=cache_key
            )

            self.cache[cache_key] = cache_entry
            self._save_cache()

            return summary

        except Exception as e:
            # Return fallback summary on error
            return f"Error generating summary: {str(e)}"

    def _build_prompt(
        self,
        file_path: str,
        commit_message: str,
        author: str,
        date: str,
        lines_added: int,
        lines_removed: int,
        diff_snippet: str,
        functions_modified: List[str],
        functions_added: List[str]
    ) -> str:
        """Build prompt for AI summarization"""

        prompt_parts = [
            f"Analyze this code change and provide a 2-3 sentence technical summary.",
            f"",
            f"File: {file_path}",
            f"Author: {author}",
            f"Date: {date}",
            f"Commit message: {commit_message}",
            f"",
            f"Changes:",
            f"  • Lines: +{lines_added} -{lines_removed}",
        ]

        if functions_modified:
            prompt_parts.append(f"  • Modified functions: {', '.join(functions_modified[:5])}")

        if functions_added:
            prompt_parts.append(f"  • Added functions: {', '.join(functions_added[:5])}")

        if diff_snippet and diff_snippet.strip():
            # Truncate diff if too long
            max_diff_lines = 30
            diff_lines = diff_snippet.split('\n')
            if len(diff_lines) > max_diff_lines:
                truncated_diff = '\n'.join(diff_lines[:max_diff_lines])
                truncated_diff += f"\n... ({len(diff_lines) - max_diff_lines} more lines)"
            else:
                truncated_diff = diff_snippet

            prompt_parts.extend([
                f"",
                f"Diff snippet:",
                f"```",
                truncated_diff,
                f"```",
            ])

        prompt_parts.extend([
            f"",
            f"Provide a concise summary explaining:",
            f"1. What changed (be specific about code elements)",
            f"2. Why (infer from commit message and context)",
            f"3. Impact (breaking changes, new features, bug fixes, refactoring)",
            f"",
            f"Keep it technical and concise (2-3 sentences max).",
        ])

        return '\n'.join(prompt_parts)

    def batch_summarize(
        self,
        changes: List[Dict[str, Any]],
        batch_size: int = 10
    ) -> Dict[str, str]:
        """
        Batch summarize multiple changes for cost efficiency

        Args:
            changes: List of change dictionaries with keys:
                     commit_hash, file_path, commit_message, author, date,
                     lines_added, lines_removed, diff_snippet
            batch_size: Number of changes to process per batch

        Returns:
            Dictionary mapping cache_key to summary
        """
        summaries = {}

        # Process in batches
        for i in range(0, len(changes), batch_size):
            batch = changes[i:i + batch_size]

            for change in batch:
                cache_key = self._generate_cache_key(
                    change['commit_hash'],
                    change['file_path'],
                    change.get('diff_snippet', '')
                )

                # Check cache first
                if cache_key in self.cache:
                    summaries[cache_key] = self.cache[cache_key].summary
                    self.cache_hits += 1
                else:
                    # Generate summary
                    summary = self.summarize_change(
                        commit_hash=change['commit_hash'],
                        file_path=change['file_path'],
                        commit_message=change['commit_message'],
                        author=change['author'],
                        date=change['date'],
                        lines_added=change['lines_added'],
                        lines_removed=change['lines_removed'],
                        diff_snippet=change.get('diff_snippet', ''),
                        functions_modified=change.get('functions_modified', []),
                        functions_added=change.get('functions_added', [])
                    )
                    summaries[cache_key] = summary

        return summaries

    def estimate_cost(
        self,
        num_changes: int,
        avg_tokens_per_change: int = 400
    ) -> Dict[str, Any]:
        """
        Estimate cost for summarizing N changes

        Args:
            num_changes: Number of changes to summarize
            avg_tokens_per_change: Estimated tokens per change (prompt + response)

        Returns:
            Cost estimate dictionary
        """
        if not self.llm:
            # No LLM router, return default estimate
            model = "gpt-4o-mini"
        else:
            model = self.llm.model

        # Get cost per 1K tokens
        if model in self.COST_PER_1K_TOKENS:
            costs = self.COST_PER_1K_TOKENS[model]
        elif 'gpt-4o-mini' in model:
            costs = self.COST_PER_1K_TOKENS['gpt-4o-mini']
        elif 'gpt-4o' in model:
            costs = self.COST_PER_1K_TOKENS['gpt-4o']
        elif 'claude-sonnet' in model:
            costs = self.COST_PER_1K_TOKENS['claude-sonnet-4-5-20250929']
        elif 'claude-haiku' in model:
            costs = self.COST_PER_1K_TOKENS['claude-haiku-3-5-20241022']
        else:
            # Unknown model, use OpenAI mini as fallback
            costs = self.COST_PER_1K_TOKENS['gpt-4o-mini']

        # Estimate tokens (70% input, 30% output)
        input_tokens = int(num_changes * avg_tokens_per_change * 0.7)
        output_tokens = int(num_changes * avg_tokens_per_change * 0.3)

        # Calculate cost
        input_cost = (input_tokens / 1000) * costs['input']
        output_cost = (output_tokens / 1000) * costs['output']
        total_cost = input_cost + output_cost

        return {
            'num_changes': num_changes,
            'model': model,
            'estimated_input_tokens': input_tokens,
            'estimated_output_tokens': output_tokens,
            'estimated_total_tokens': input_tokens + output_tokens,
            'input_cost_usd': input_cost,
            'output_cost_usd': output_cost,
            'total_cost_usd': total_cost,
            'cost_per_change': total_cost / num_changes if num_changes > 0 else 0
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            'cache_size': len(self.cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate_percent': hit_rate,
            'cache_path': str(self.cache_path)
        }
