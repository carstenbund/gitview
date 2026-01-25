"""Multi-signal storyline detection from various data sources."""

import re
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime

from .models import StorylineSignal, StorylineCategory, Storyline


class BaseDetector(ABC):
    """Base class for storyline signal detectors."""

    def __init__(self, weight: float = 0.5):
        """
        Initialize detector with confidence weight.

        Args:
            weight: Base confidence weight for signals from this detector (0.0-1.0)
        """
        self.weight = weight

    @abstractmethod
    def detect(self, phase) -> List[StorylineSignal]:
        """
        Detect storyline signals from a phase.

        Args:
            phase: Phase object to analyze

        Returns:
            List of StorylineSignal objects
        """
        pass

    def _create_signal(
        self,
        title: str,
        category: StorylineCategory,
        description: str,
        phase_number: int,
        commit_hashes: List[str],
        confidence_modifier: float = 1.0,
        data: Dict[str, Any] = None,
        files: List[str] = None,
    ) -> StorylineSignal:
        """Helper to create a signal with proper source attribution."""
        return StorylineSignal(
            source=self.__class__.__name__,
            confidence=min(1.0, self.weight * confidence_modifier),
            phase_number=phase_number,
            commit_hashes=commit_hashes,
            title=title,
            category=category,
            description=description,
            data=data or {},
            files=files or [],
        )


class PRLabelDetector(BaseDetector):
    """Detect storylines from GitHub PR labels - highest confidence source."""

    # Map PR labels to storyline categories
    LABEL_MAPPINGS = {
        # Features
        'feature': StorylineCategory.FEATURE,
        'feat': StorylineCategory.FEATURE,
        'enhancement': StorylineCategory.FEATURE,
        'new feature': StorylineCategory.FEATURE,

        # Bugs
        'bug': StorylineCategory.BUGFIX,
        'bugfix': StorylineCategory.BUGFIX,
        'fix': StorylineCategory.BUGFIX,
        'hotfix': StorylineCategory.BUGFIX,

        # Refactoring
        'refactor': StorylineCategory.REFACTOR,
        'refactoring': StorylineCategory.REFACTOR,
        'cleanup': StorylineCategory.REFACTOR,
        'code-quality': StorylineCategory.REFACTOR,

        # Tech debt
        'tech-debt': StorylineCategory.TECH_DEBT,
        'technical-debt': StorylineCategory.TECH_DEBT,
        'debt': StorylineCategory.TECH_DEBT,

        # Infrastructure
        'infrastructure': StorylineCategory.INFRASTRUCTURE,
        'infra': StorylineCategory.INFRASTRUCTURE,
        'ci': StorylineCategory.INFRASTRUCTURE,
        'ci/cd': StorylineCategory.INFRASTRUCTURE,
        'build': StorylineCategory.INFRASTRUCTURE,
        'devops': StorylineCategory.INFRASTRUCTURE,

        # Documentation
        'documentation': StorylineCategory.DOCUMENTATION,
        'docs': StorylineCategory.DOCUMENTATION,
        'doc': StorylineCategory.DOCUMENTATION,

        # Performance
        'performance': StorylineCategory.PERFORMANCE,
        'perf': StorylineCategory.PERFORMANCE,
        'optimization': StorylineCategory.PERFORMANCE,

        # Security
        'security': StorylineCategory.SECURITY,
        'sec': StorylineCategory.SECURITY,
        'vulnerability': StorylineCategory.SECURITY,

        # Migration
        'migration': StorylineCategory.MIGRATION,
        'migrate': StorylineCategory.MIGRATION,
        'upgrade': StorylineCategory.MIGRATION,
    }

    def __init__(self, weight: float = 0.9):
        """Initialize with high confidence weight (explicit labels are reliable)."""
        super().__init__(weight=weight)

    def detect(self, phase) -> List[StorylineSignal]:
        """Detect signals from PR labels."""
        signals = []
        pr_groups: Dict[str, Dict] = {}  # Group by PR title

        for commit in phase.commits:
            if not commit.has_github_context():
                continue

            labels = commit.get_pr_labels()
            pr_title = commit.get_pr_title()
            pr_body = commit.get_pr_body() or ""

            if not labels or not pr_title:
                continue

            # Determine category from labels
            category = self._get_category_from_labels(labels)
            if category == StorylineCategory.UNKNOWN:
                continue

            # Group commits by PR
            if pr_title not in pr_groups:
                pr_groups[pr_title] = {
                    'title': pr_title,
                    'category': category,
                    'labels': set(labels),
                    'commits': [],
                    'files': set(),
                    'description': pr_body[:300] if pr_body else pr_title,
                    'pr_number': commit.github_context.get('pr_number') if commit.github_context else None,
                }

            pr_groups[pr_title]['commits'].append(commit.short_hash)
            pr_groups[pr_title]['labels'].update(labels)

            # Track files
            if hasattr(commit, 'files') and commit.files:
                pr_groups[pr_title]['files'].update(commit.files)

        # Create signals from grouped PRs
        for pr_title, group in pr_groups.items():
            # Generate storyline title from PR title
            storyline_title = self._generate_storyline_title(pr_title, group['labels'])

            signal = self._create_signal(
                title=storyline_title,
                category=group['category'],
                description=group['description'],
                phase_number=phase.phase_number,
                commit_hashes=group['commits'],
                confidence_modifier=1.0,  # Full confidence for labeled PRs
                data={
                    'pr_title': pr_title,
                    'labels': list(group['labels']),
                    'pr_number': group['pr_number'],
                },
                files=list(group['files']),
            )
            signals.append(signal)

        return signals

    def _get_category_from_labels(self, labels: List[str]) -> StorylineCategory:
        """Determine category from a list of labels."""
        for label in labels:
            label_lower = label.lower().strip()
            if label_lower in self.LABEL_MAPPINGS:
                return self.LABEL_MAPPINGS[label_lower]
        return StorylineCategory.UNKNOWN

    def _generate_storyline_title(self, pr_title: str, labels: Set[str]) -> str:
        """Generate a storyline title from PR title and labels."""
        # Clean up PR title
        title = pr_title.strip()

        # Remove common prefixes like [WIP], feat:, fix:, etc.
        title = re.sub(r'^\[.*?\]\s*', '', title)
        title = re.sub(r'^(feat|fix|docs|refactor|chore|test|ci|perf|build)[\(:].*?[\):]\s*', '', title, flags=re.IGNORECASE)

        # Capitalize first letter
        if title:
            title = title[0].upper() + title[1:]

        return title or pr_title


class PRTitlePatternDetector(BaseDetector):
    """Detect storylines from PR title patterns (for PRs without labels)."""

    # Pattern prefixes that indicate category
    TITLE_PATTERNS = {
        StorylineCategory.FEATURE: [
            r'^feat(?:ure)?[\(:]',
            r'^add\s+',
            r'^implement\s+',
            r'^create\s+',
            r'^new\s+',
        ],
        StorylineCategory.BUGFIX: [
            r'^fix[\(:]',
            r'^bug[\(:]',
            r'^hotfix[\(:]',
            r'^resolve\s+',
            r'^patch\s+',
        ],
        StorylineCategory.REFACTOR: [
            r'^refactor[\(:]',
            r'^cleanup[\(:]',
            r'^clean\s+up\s+',
            r'^reorganize\s+',
            r'^restructure\s+',
        ],
        StorylineCategory.DOCUMENTATION: [
            r'^docs?[\(:]',
            r'^update\s+(readme|docs?|documentation)',
        ],
        StorylineCategory.INFRASTRUCTURE: [
            r'^ci[\(:]',
            r'^build[\(:]',
            r'^chore[\(:]',
            r'^devops[\(:]',
        ],
        StorylineCategory.PERFORMANCE: [
            r'^perf[\(:]',
            r'^optimi[zs]e\s+',
            r'^performance[\(:]',
        ],
        StorylineCategory.SECURITY: [
            r'^security[\(:]',
            r'^sec[\(:]',
            r'^cve[\(:]',
        ],
    }

    def __init__(self, weight: float = 0.8):
        """Initialize with good confidence (patterns are fairly reliable)."""
        super().__init__(weight=weight)
        self._compiled_patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[StorylineCategory, List[re.Pattern]]:
        """Pre-compile regex patterns."""
        compiled = {}
        for category, patterns in self.TITLE_PATTERNS.items():
            compiled[category] = [re.compile(p, re.IGNORECASE) for p in patterns]
        return compiled

    def detect(self, phase) -> List[StorylineSignal]:
        """Detect signals from PR title patterns."""
        signals = []
        pr_groups: Dict[str, Dict] = {}

        for commit in phase.commits:
            if not commit.has_github_context():
                continue

            # Skip if PR has labels (handled by PRLabelDetector)
            labels = commit.get_pr_labels()
            if labels:
                continue

            pr_title = commit.get_pr_title()
            if not pr_title:
                continue

            category = self._match_category(pr_title)
            if category == StorylineCategory.UNKNOWN:
                continue

            # Group by PR title
            if pr_title not in pr_groups:
                pr_groups[pr_title] = {
                    'title': pr_title,
                    'category': category,
                    'commits': [],
                    'files': set(),
                    'description': pr_title,
                }

            pr_groups[pr_title]['commits'].append(commit.short_hash)

        # Create signals
        for pr_title, group in pr_groups.items():
            storyline_title = self._clean_title(pr_title)

            signal = self._create_signal(
                title=storyline_title,
                category=group['category'],
                description=group['description'],
                phase_number=phase.phase_number,
                commit_hashes=group['commits'],
                confidence_modifier=0.9,  # Slightly lower than labeled PRs
                data={'pr_title': pr_title},
                files=list(group['files']),
            )
            signals.append(signal)

        return signals

    def _match_category(self, title: str) -> StorylineCategory:
        """Match title against patterns to determine category."""
        for category, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(title):
                    return category
        return StorylineCategory.UNKNOWN

    def _clean_title(self, title: str) -> str:
        """Clean PR title to create storyline title."""
        # Remove conventional commit prefix
        cleaned = re.sub(r'^[a-z]+[\(:].*?[\):]\s*', '', title, flags=re.IGNORECASE)
        if cleaned:
            return cleaned[0].upper() + cleaned[1:]
        return title


class CommitMessagePatternDetector(BaseDetector):
    """Detect storylines from commit message patterns."""

    # Keywords that indicate different categories
    KEYWORDS = {
        StorylineCategory.FEATURE: [
            'add', 'implement', 'create', 'introduce', 'new',
            'feature', 'support', 'enable',
        ],
        StorylineCategory.BUGFIX: [
            'fix', 'bug', 'issue', 'resolve', 'patch', 'repair',
            'correct', 'handle', 'error',
        ],
        StorylineCategory.REFACTOR: [
            'refactor', 'restructure', 'reorganize', 'cleanup',
            'clean', 'simplify', 'improve', 'optimize code',
        ],
        StorylineCategory.DOCUMENTATION: [
            'doc', 'readme', 'comment', 'document', 'changelog',
        ],
        StorylineCategory.INFRASTRUCTURE: [
            'ci', 'cd', 'build', 'deploy', 'docker', 'kubernetes',
            'config', 'setup', 'workflow', 'action',
        ],
        StorylineCategory.PERFORMANCE: [
            'performance', 'perf', 'optimize', 'speed', 'fast',
            'cache', 'memory', 'efficiency',
        ],
        StorylineCategory.MIGRATION: [
            'migrate', 'migration', 'upgrade', 'update dependency',
            'bump', 'version',
        ],
        StorylineCategory.SECURITY: [
            'security', 'vulnerability', 'cve', 'auth', 'permission',
            'encrypt', 'sanitize',
        ],
    }

    def __init__(self, weight: float = 0.6):
        """Initialize with moderate confidence (commit messages vary in quality)."""
        super().__init__(weight=weight)

    def detect(self, phase) -> List[StorylineSignal]:
        """Detect signals from commit message patterns."""
        signals = []
        theme_groups: Dict[str, Dict] = {}  # Group commits by detected theme

        for commit in phase.commits:
            # Skip commits with GitHub context (handled by PR detectors)
            if commit.has_github_context():
                continue

            message = commit.commit_subject or commit.commit_message or ""
            if not message:
                continue

            category, keywords_found = self._analyze_message(message)
            if category == StorylineCategory.UNKNOWN:
                continue

            # Create a theme key from the category and main action
            theme_key = self._extract_theme_key(message, category)

            if theme_key not in theme_groups:
                theme_groups[theme_key] = {
                    'category': category,
                    'commits': [],
                    'messages': [],
                    'files': set(),
                    'keywords': set(),
                }

            theme_groups[theme_key]['commits'].append(commit.short_hash)
            theme_groups[theme_key]['messages'].append(message)
            theme_groups[theme_key]['keywords'].update(keywords_found)

        # Create signals from groups with multiple commits (more confident)
        for theme_key, group in theme_groups.items():
            # Only create signal if multiple commits share the theme
            if len(group['commits']) < 2:
                continue

            # Generate title from most common message theme
            title = self._generate_title(group['messages'], group['category'])

            confidence_modifier = min(1.0, 0.5 + (len(group['commits']) * 0.1))

            signal = self._create_signal(
                title=title,
                category=group['category'],
                description=f"Detected from {len(group['commits'])} commits with shared theme",
                phase_number=phase.phase_number,
                commit_hashes=group['commits'],
                confidence_modifier=confidence_modifier,
                data={
                    'keywords': list(group['keywords']),
                    'sample_messages': group['messages'][:3],
                },
                files=list(group['files']),
            )
            signals.append(signal)

        return signals

    def _analyze_message(self, message: str) -> Tuple[StorylineCategory, Set[str]]:
        """Analyze commit message to determine category and keywords found."""
        message_lower = message.lower()
        keywords_found = set()
        category_scores = defaultdict(int)

        for category, keywords in self.KEYWORDS.items():
            for keyword in keywords:
                if keyword in message_lower:
                    keywords_found.add(keyword)
                    category_scores[category] += 1

        if not category_scores:
            return StorylineCategory.UNKNOWN, keywords_found

        # Return category with highest score
        best_category = max(category_scores.items(), key=lambda x: x[1])[0]
        return best_category, keywords_found

    def _extract_theme_key(self, message: str, category: StorylineCategory) -> str:
        """Extract a theme key for grouping similar commits."""
        # Combine category with first significant words
        words = message.lower().split()[:5]
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'to', 'for', 'in', 'on', 'and', 'or', 'of'}
        significant = [w for w in words if w not in stop_words and len(w) > 2]
        return f"{category.value}:{'-'.join(significant[:3])}"

    def _generate_title(self, messages: List[str], category: StorylineCategory) -> str:
        """Generate a storyline title from multiple messages."""
        if not messages:
            return f"{category.value.replace('_', ' ').title()} work"

        # Use the first message as base, clean it up
        first = messages[0]
        # Remove common prefixes
        cleaned = re.sub(r'^[a-z]+[\(:].*?[\):]\s*', '', first, flags=re.IGNORECASE)
        cleaned = re.sub(r'^(fix|add|update|remove|refactor)\s+', '', cleaned, flags=re.IGNORECASE)

        if cleaned:
            return cleaned[0].upper() + cleaned[1:].strip()
        return first


class FileClusterDetector(BaseDetector):
    """Detect storylines from files that change together."""

    # File patterns that indicate specific categories
    FILE_CATEGORY_HINTS = {
        StorylineCategory.DOCUMENTATION: [r'\.md$', r'docs?/', r'readme', r'changelog'],
        StorylineCategory.INFRASTRUCTURE: [
            r'\.ya?ml$', r'dockerfile', r'docker-compose', r'\.github/',
            r'ci/', r'\.circleci/', r'jenkinsfile', r'makefile',
        ],
        StorylineCategory.FEATURE: [r'src/', r'lib/', r'app/'],
        StorylineCategory.BUGFIX: [],  # Hard to detect from files alone
        StorylineCategory.REFACTOR: [],
    }

    # Minimum file overlap ratio to consider clusters related
    OVERLAP_THRESHOLD = 0.5

    def __init__(self, weight: float = 0.7):
        """Initialize with good confidence (file patterns are meaningful)."""
        super().__init__(weight=weight)
        self._compiled_hints = self._compile_hints()

    def _compile_hints(self) -> Dict[StorylineCategory, List[re.Pattern]]:
        """Pre-compile file pattern hints."""
        compiled = {}
        for category, patterns in self.FILE_CATEGORY_HINTS.items():
            compiled[category] = [re.compile(p, re.IGNORECASE) for p in patterns]
        return compiled

    def detect(self, phase) -> List[StorylineSignal]:
        """Detect signals from file change clusters."""
        signals = []

        # Build file-to-commits mapping
        file_commits: Dict[str, List[str]] = defaultdict(list)
        commit_files: Dict[str, Set[str]] = {}

        for commit in phase.commits:
            # Get files changed in this commit
            files = self._get_commit_files(commit)
            if not files:
                continue

            commit_files[commit.short_hash] = files
            for f in files:
                file_commits[f].append(commit.short_hash)

        # Find file clusters (files that change together)
        clusters = self._find_clusters(file_commits, commit_files)

        # Create signals from significant clusters
        for cluster_files, cluster_commits in clusters:
            if len(cluster_commits) < 2 or len(cluster_files) < 2:
                continue

            # Determine category from file patterns
            category = self._categorize_from_files(cluster_files)

            # Generate title from file paths
            title = self._generate_title_from_files(cluster_files)

            confidence_modifier = min(1.0, 0.6 + (len(cluster_files) * 0.05))

            signal = self._create_signal(
                title=title,
                category=category,
                description=f"Files changing together: {', '.join(list(cluster_files)[:5])}",
                phase_number=phase.phase_number,
                commit_hashes=list(cluster_commits),
                confidence_modifier=confidence_modifier,
                data={
                    'file_count': len(cluster_files),
                    'sample_files': list(cluster_files)[:10],
                },
                files=list(cluster_files),
            )
            signals.append(signal)

        return signals

    def _get_commit_files(self, commit) -> Set[str]:
        """Extract files changed in a commit."""
        files = set()

        # Try different attributes that might contain file info
        if hasattr(commit, 'files_changed_list') and commit.files_changed_list:
            files.update(commit.files_changed_list)
        elif hasattr(commit, 'files') and commit.files:
            if isinstance(commit.files, dict):
                files.update(commit.files.keys())
            elif isinstance(commit.files, list):
                files.update(commit.files)

        return files

    def _find_clusters(
        self,
        file_commits: Dict[str, List[str]],
        commit_files: Dict[str, Set[str]],
    ) -> List[Tuple[Set[str], Set[str]]]:
        """Find clusters of files that change together."""
        clusters = []
        processed_files = set()

        # Sort files by number of commits (most changed first)
        sorted_files = sorted(file_commits.keys(), key=lambda f: len(file_commits[f]), reverse=True)

        for seed_file in sorted_files:
            if seed_file in processed_files:
                continue

            # Find files that frequently co-change with seed
            cluster_files = {seed_file}
            cluster_commits = set(file_commits[seed_file])

            for other_file, other_commits in file_commits.items():
                if other_file in processed_files or other_file == seed_file:
                    continue

                # Calculate overlap
                overlap = len(set(other_commits) & cluster_commits)
                total = len(set(other_commits) | cluster_commits)

                if total > 0 and overlap / total >= self.OVERLAP_THRESHOLD:
                    cluster_files.add(other_file)
                    cluster_commits.update(other_commits)

            if len(cluster_files) >= 2:
                clusters.append((cluster_files, cluster_commits))
                processed_files.update(cluster_files)

        return clusters

    def _categorize_from_files(self, files: Set[str]) -> StorylineCategory:
        """Determine category from file paths."""
        category_scores = defaultdict(int)

        for file_path in files:
            for category, patterns in self._compiled_hints.items():
                for pattern in patterns:
                    if pattern.search(file_path):
                        category_scores[category] += 1
                        break

        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]

        return StorylineCategory.FEATURE  # Default for code changes

    def _generate_title_from_files(self, files: Set[str]) -> str:
        """Generate a storyline title from file paths."""
        # Find common path prefix
        paths = list(files)
        if not paths:
            return "File changes"

        # Split paths and find common directory
        split_paths = [p.split('/') for p in paths]
        common_parts = []

        for parts in zip(*split_paths):
            if len(set(parts)) == 1:
                common_parts.append(parts[0])
            else:
                break

        if common_parts:
            common_dir = '/'.join(common_parts)
            return f"{common_dir} module changes"

        # Fall back to most common directory
        dirs = [p.split('/')[0] if '/' in p else '' for p in paths]
        if dirs:
            most_common = max(set(dirs), key=dirs.count)
            if most_common:
                return f"{most_common} changes"

        return "Code changes"


class StorylineDetector:
    """Main detector that combines signals from multiple sources."""

    def __init__(
        self,
        pr_label_weight: float = 0.9,
        pr_title_weight: float = 0.8,
        file_cluster_weight: float = 0.7,
        commit_message_weight: float = 0.6,
    ):
        """
        Initialize with configurable detector weights.

        Args:
            pr_label_weight: Confidence weight for PR label signals
            pr_title_weight: Confidence weight for PR title pattern signals
            file_cluster_weight: Confidence weight for file cluster signals
            commit_message_weight: Confidence weight for commit message signals
        """
        self.detectors = [
            PRLabelDetector(weight=pr_label_weight),
            PRTitlePatternDetector(weight=pr_title_weight),
            FileClusterDetector(weight=file_cluster_weight),
            CommitMessagePatternDetector(weight=commit_message_weight),
        ]

    def detect_in_phase(self, phase) -> List[StorylineSignal]:
        """
        Run all detectors on a phase and return signals.

        Args:
            phase: Phase object to analyze

        Returns:
            List of all detected signals
        """
        all_signals = []

        for detector in self.detectors:
            try:
                signals = detector.detect(phase)
                all_signals.extend(signals)
            except Exception as e:
                # Log but don't fail on detector errors
                print(f"Warning: {detector.__class__.__name__} failed: {e}")

        return all_signals

    def merge_signals(
        self,
        signals: List[StorylineSignal],
        title_similarity_threshold: float = 0.6,
    ) -> List[Storyline]:
        """
        Merge related signals into candidate storylines.

        Args:
            signals: List of signals to merge
            title_similarity_threshold: Min similarity ratio for title matching

        Returns:
            List of merged Storyline candidates
        """
        if not signals:
            return []

        # Group signals by normalized title similarity
        groups: Dict[str, List[StorylineSignal]] = {}

        for signal in signals:
            normalized = Storyline.normalize_title(signal.title)

            # Find existing group with similar title
            matched_group = None
            for group_key in groups:
                if self._title_similarity(normalized, group_key) >= title_similarity_threshold:
                    matched_group = group_key
                    break

            if matched_group:
                groups[matched_group].append(signal)
            else:
                groups[normalized] = [signal]

        # Create storylines from groups
        storylines = []

        for group_signals in groups.values():
            storyline = self._create_storyline_from_signals(group_signals)
            storylines.append(storyline)

        return storylines

    def _title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles."""
        # Simple word overlap ratio
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _create_storyline_from_signals(self, signals: List[StorylineSignal]) -> Storyline:
        """Create a storyline from a group of related signals."""
        # Use highest confidence signal for primary attributes
        primary = max(signals, key=lambda s: s.confidence)

        # Aggregate from all signals
        all_commits = set()
        all_files = set()
        categories = defaultdict(int)

        for signal in signals:
            all_commits.update(signal.commit_hashes)
            all_files.update(signal.files)
            categories[signal.category] += signal.confidence

        # Pick category with highest weighted votes
        best_category = max(categories.items(), key=lambda x: x[1])[0]

        # Create storyline
        storyline = Storyline.from_signal(primary)
        storyline.category = best_category
        storyline.key_files = all_files

        # Add additional signals
        for signal in signals:
            if signal != primary:
                storyline.add_signal(signal)

        return storyline
