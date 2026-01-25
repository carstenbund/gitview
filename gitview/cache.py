"""Cache management for GitView analysis artifacts.

This module provides centralized cache management for:
- Commit histories (repo_history.jsonl)
- Phase data (phases/*.json)
- Storyline databases (storylines.json)
- LLM response caches

The CacheManager handles:
- Loading and saving cached data
- Cache invalidation and freshness checks
- Incremental update detection
- Storage optimization (references vs embedded data)
"""

import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from .extractor import GitHistoryExtractor, CommitRecord
from .chunker import HistoryChunker, Phase


@dataclass
class CacheMetadata:
    """Metadata about a cached analysis."""
    last_commit_hash: str
    last_commit_date: str
    total_commits: int
    total_phases: int
    generated_at: str
    repository_path: str
    cache_version: str = "2.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'last_commit_hash': self.last_commit_hash,
            'last_commit_date': self.last_commit_date,
            'total_commits': self.total_commits,
            'total_phases': self.total_phases,
            'generated_at': self.generated_at,
            'repository_path': self.repository_path,
            'cache_version': self.cache_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheMetadata':
        return cls(
            last_commit_hash=data.get('last_commit_hash', ''),
            last_commit_date=data.get('last_commit_date', ''),
            total_commits=data.get('total_commits', 0),
            total_phases=data.get('total_phases', 0),
            generated_at=data.get('generated_at', ''),
            repository_path=data.get('repository_path', ''),
            cache_version=data.get('cache_version', '1.0'),
        )


@dataclass
class CacheStatus:
    """Status of cache relative to current repository state."""
    is_valid: bool
    needs_update: bool
    new_commits_count: int
    reason: str
    metadata: Optional[CacheMetadata] = None


class CacheManager:
    """Manages caching for GitView analysis artifacts.

    This class provides a centralized interface for loading and saving
    cached analysis data, detecting when updates are needed, and
    managing incremental analysis.
    """

    CACHE_VERSION = "2.0"

    def __init__(self, output_dir: str, repo_path: Optional[str] = None):
        """Initialize cache manager.

        Args:
            output_dir: Directory containing cached analysis
            repo_path: Path to git repository (for freshness checks)
        """
        self.output_dir = Path(output_dir)
        self.repo_path = Path(repo_path) if repo_path else None
        self._metadata: Optional[CacheMetadata] = None

    @property
    def history_file(self) -> Path:
        """Path to commit history JSONL file."""
        return self.output_dir / "repo_history.jsonl"

    @property
    def phases_dir(self) -> Path:
        """Path to phases directory."""
        return self.output_dir / "phases"

    @property
    def storylines_file(self) -> Path:
        """Path to storylines database."""
        return self.phases_dir / "storylines.json"

    @property
    def data_file(self) -> Path:
        """Path to main history_data.json file."""
        return self.output_dir / "history_data.json"

    def get_cache_status(self) -> CacheStatus:
        """Check if cache exists and is up-to-date.

        Returns:
            CacheStatus with validity and update needs
        """
        # Check if cache exists
        if not self.data_file.exists():
            return CacheStatus(
                is_valid=False,
                needs_update=True,
                new_commits_count=0,
                reason="No previous analysis found"
            )

        # Load metadata
        metadata = self.load_metadata()
        if not metadata:
            return CacheStatus(
                is_valid=False,
                needs_update=True,
                new_commits_count=0,
                reason="Could not load cache metadata"
            )

        # Check cache version
        if metadata.cache_version != self.CACHE_VERSION:
            return CacheStatus(
                is_valid=False,
                needs_update=True,
                new_commits_count=0,
                reason=f"Cache version mismatch (have {metadata.cache_version}, need {self.CACHE_VERSION})",
                metadata=metadata
            )

        # Check for new commits if repo_path is available
        if self.repo_path and metadata.last_commit_hash:
            new_commits = self._count_new_commits(metadata.last_commit_hash)
            if new_commits > 0:
                return CacheStatus(
                    is_valid=True,
                    needs_update=True,
                    new_commits_count=new_commits,
                    reason=f"Found {new_commits} new commits since last analysis",
                    metadata=metadata
                )

        return CacheStatus(
            is_valid=True,
            needs_update=False,
            new_commits_count=0,
            reason="Cache is up-to-date",
            metadata=metadata
        )

    def _count_new_commits(self, since_hash: str) -> int:
        """Count commits since a given hash."""
        try:
            from git import Repo
            repo = Repo(str(self.repo_path))
            count = 0
            for commit in repo.iter_commits('HEAD'):
                if commit.hexsha == since_hash:
                    break
                count += 1
            return count
        except Exception:
            return 0

    def load_metadata(self) -> Optional[CacheMetadata]:
        """Load cache metadata from history_data.json."""
        if self._metadata:
            return self._metadata

        if not self.data_file.exists():
            return None

        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
            metadata_dict = data.get('metadata', {})
            self._metadata = CacheMetadata.from_dict(metadata_dict)
            return self._metadata
        except (json.JSONDecodeError, IOError):
            return None

    def load_commit_history(self) -> Optional[List[CommitRecord]]:
        """Load cached commit history.

        Returns:
            List of CommitRecords, or None if not cached
        """
        if not self.history_file.exists():
            return None

        try:
            return GitHistoryExtractor.load_from_jsonl(str(self.history_file))
        except Exception:
            return None

    def load_phases(self) -> Optional[List[Phase]]:
        """Load cached phases.

        Returns:
            List of Phase objects, or None if not cached
        """
        if not self.phases_dir.exists():
            return None

        try:
            return HistoryChunker.load_phases(str(self.phases_dir))
        except Exception:
            return None

    def load_storylines(self) -> Optional[Dict[str, Any]]:
        """Load cached storyline database.

        Returns:
            Storyline data dict, or None if not cached
        """
        if not self.storylines_file.exists():
            return None

        try:
            with open(self.storylines_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def save_commit_history(self, records: List[CommitRecord], extractor: GitHistoryExtractor):
        """Save commit history to cache.

        Args:
            records: List of CommitRecords to cache
            extractor: GitHistoryExtractor instance (for save method)
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        extractor.save_to_jsonl(records, str(self.history_file))

    def save_phases(self, phases: List[Phase], chunker: HistoryChunker):
        """Save phases to cache.

        Args:
            phases: List of Phase objects to cache
            chunker: HistoryChunker instance (for save method)
        """
        self.phases_dir.mkdir(parents=True, exist_ok=True)
        chunker.save_phases(phases, str(self.phases_dir))

    def invalidate(self):
        """Invalidate all cached data."""
        import shutil
        if self.output_dir.exists():
            # Remove specific cache files but keep output structure
            for cache_file in [self.history_file, self.data_file]:
                if cache_file.exists():
                    cache_file.unlink()

            if self.phases_dir.exists():
                shutil.rmtree(self.phases_dir)

        self._metadata = None

    def get_incremental_since(self) -> Optional[str]:
        """Get the commit hash to use for incremental analysis.

        Returns:
            Commit hash string, or None if full analysis needed
        """
        status = self.get_cache_status()
        if status.is_valid and status.metadata:
            return status.metadata.last_commit_hash
        return None

    def create_cache_key(self, *args) -> str:
        """Create a cache key from arguments.

        Args:
            *args: Values to hash together

        Returns:
            Hex string cache key
        """
        content = json.dumps(args, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class LLMResponseCache:
    """Cache for LLM responses to avoid redundant API calls.

    This cache stores LLM responses keyed by a hash of the prompt,
    allowing reuse of expensive API calls during incremental analysis.
    """

    def __init__(self, cache_dir: str):
        """Initialize LLM response cache.

        Args:
            cache_dir: Directory to store cached responses
        """
        self.cache_dir = Path(cache_dir) / ".llm_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(self, prompt: str, model: str) -> str:
        """Create cache key from prompt and model."""
        content = f"{model}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, prompt: str, model: str) -> Optional[str]:
        """Get cached response for a prompt.

        Args:
            prompt: The LLM prompt
            model: Model identifier

        Returns:
            Cached response string, or None if not cached
        """
        key = self._make_key(prompt, model)
        cache_file = self.cache_dir / f"{key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)

            # Check if cache is still valid (24 hour expiry)
            cached_at = datetime.fromisoformat(data.get('cached_at', ''))
            if datetime.now() - cached_at > timedelta(hours=24):
                cache_file.unlink()
                return None

            return data.get('response')
        except (json.JSONDecodeError, IOError, ValueError):
            return None

    def set(self, prompt: str, model: str, response: str):
        """Cache a response for a prompt.

        Args:
            prompt: The LLM prompt
            model: Model identifier
            response: The response to cache
        """
        key = self._make_key(prompt, model)
        cache_file = self.cache_dir / f"{key}.json"

        data = {
            'prompt_hash': key,
            'model': model,
            'response': response,
            'cached_at': datetime.now().isoformat(),
        }

        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except IOError:
            pass  # Silently fail on cache write errors

    def clear(self):
        """Clear all cached responses."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
