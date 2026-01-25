"""Cross-phase storyline tracking with persistence."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .models import (
    Storyline,
    StorylineSignal,
    StorylineUpdate,
    StorylineStatus,
    StorylineCategory,
    StorylineDatabase,
)
from .state_machine import StorylineStateMachine
from .detector import StorylineDetector


class StorylineTracker:
    """
    Track storylines across phases with automatic detection and persistence.

    This is the main orchestrator that combines:
    - Multi-signal detection (from commits, PRs, files)
    - State machine for lifecycle management
    - Database for persistence and querying
    """

    def __init__(
        self,
        persist_path: Optional[str] = None,
        stall_threshold: int = 3,
        abandon_threshold: int = 6,
        confidence_threshold: float = 0.6,
        min_signals: int = 2,
        title_similarity_threshold: float = 0.5,
    ):
        """
        Initialize tracker with configuration.

        Args:
            persist_path: Path to storyline database JSON file
            stall_threshold: Phases without updates before STALLED
            abandon_threshold: Phases stalled before ABANDONED
            confidence_threshold: Min confidence for EMERGING -> ACTIVE
            min_signals: Min signals for EMERGING -> ACTIVE
            title_similarity_threshold: Min similarity for signal merging
        """
        self.persist_path = persist_path
        self.title_similarity_threshold = title_similarity_threshold

        # Initialize components
        self.detector = StorylineDetector()
        self.state_machine = StorylineStateMachine(
            stall_threshold=stall_threshold,
            abandon_threshold=abandon_threshold,
            confidence_threshold=confidence_threshold,
            min_signals=min_signals,
        )

        # Load or create database
        self.database = self._load_or_create_database()

    def _load_or_create_database(self) -> StorylineDatabase:
        """Load existing database or create new one."""
        if self.persist_path and Path(self.persist_path).exists():
            try:
                return StorylineDatabase.load(self.persist_path)
            except Exception as e:
                print(f"Warning: Could not load storyline database: {e}")
        return StorylineDatabase()

    def process_phase(self, phase, llm_storylines: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Process a phase and update storyline database.

        Args:
            phase: Phase object to analyze
            llm_storylines: Optional storylines extracted from LLM summary

        Returns:
            Dict with processing results:
            - signals_detected: Number of signals found
            - storylines_updated: Number of existing storylines updated
            - storylines_created: Number of new storylines created
            - transitions: List of state transitions applied
        """
        phase_number = phase.phase_number
        results = {
            'signals_detected': 0,
            'storylines_updated': 0,
            'storylines_created': 0,
            'transitions': [],
        }

        # Step 1: Detect signals from phase data
        signals = self.detector.detect_in_phase(phase)

        # Step 2: Add LLM-extracted storylines as signals (if provided)
        if llm_storylines:
            llm_signals = self._convert_llm_storylines_to_signals(llm_storylines, phase_number)
            signals.extend(llm_signals)

        results['signals_detected'] = len(signals)

        # Step 3: Match signals to existing storylines
        matched, unmatched = self._match_to_existing(signals)

        # Step 4: Update existing storylines with matched signals
        for storyline_id, storyline_signals in matched.items():
            self._update_storyline(storyline_id, phase, storyline_signals)
            results['storylines_updated'] += 1

        # Step 5: Create new storylines from unmatched signals
        new_storylines = self._create_from_signals(unmatched, phase)
        results['storylines_created'] = len(new_storylines)

        # Step 6: Evaluate state transitions for all active storylines
        transitions = self._evaluate_all_transitions(phase_number, matched)
        results['transitions'] = transitions

        # Step 7: Update database metadata
        self.database.last_phase_analyzed = phase_number
        if phase.commits:
            self.database.last_commit_hash = phase.commits[-1].short_hash

        # Step 8: Persist if path configured
        if self.persist_path:
            self._save_database()

        return results

    def _convert_llm_storylines_to_signals(
        self,
        llm_storylines: List[Dict[str, Any]],
        phase_number: int,
    ) -> List[StorylineSignal]:
        """Convert LLM-extracted storylines to signals."""
        signals = []

        for sl in llm_storylines:
            # Map LLM status to signal data
            status = sl.get('status', 'continued').lower()
            is_completion = status == 'completed'

            signal = StorylineSignal(
                source='LLMExtraction',
                confidence=0.5,  # Lower confidence for LLM extraction
                phase_number=phase_number,
                commit_hashes=[],
                title=sl.get('title', 'Unknown'),
                category=StorylineCategory.from_string(sl.get('category', 'feature')),
                description=sl.get('description', ''),
                data={
                    'status': status,
                    'is_new': status == 'new',
                    'is_completion': is_completion,
                    'llm_extracted': True,
                },
            )
            signals.append(signal)

        return signals

    def _match_to_existing(
        self,
        signals: List[StorylineSignal],
    ) -> Tuple[Dict[str, List[StorylineSignal]], List[StorylineSignal]]:
        """
        Match signals to existing storylines.

        Returns:
            Tuple of (matched: {storyline_id: [signals]}, unmatched: [signals])
        """
        matched: Dict[str, List[StorylineSignal]] = {}
        unmatched: List[StorylineSignal] = []

        for signal in signals:
            match_id = self._find_matching_storyline(signal)

            if match_id:
                if match_id not in matched:
                    matched[match_id] = []
                matched[match_id].append(signal)
            else:
                unmatched.append(signal)

        return matched, unmatched

    def _find_matching_storyline(self, signal: StorylineSignal) -> Optional[str]:
        """Find an existing storyline that matches a signal."""
        # Priority 1: Exact title match
        normalized_title = Storyline.normalize_title(signal.title)
        if normalized_title in self.database.title_index:
            return self.database.title_index[normalized_title]

        # Priority 2: PR number match
        pr_number = signal.data.get('pr_number')
        if pr_number and pr_number in self.database.pr_index:
            storyline_ids = self.database.pr_index[pr_number]
            if storyline_ids:
                return storyline_ids[0]

        # Priority 3: High file overlap
        if signal.files:
            best_match = None
            best_overlap = 0.0

            for storyline in self.database.get_active():
                if not storyline.key_files:
                    continue

                overlap = self._calculate_file_overlap(
                    set(signal.files),
                    storyline.key_files,
                )

                if overlap > 0.5 and overlap > best_overlap:
                    best_overlap = overlap
                    best_match = storyline.id

            if best_match:
                return best_match

        # Priority 4: Title similarity
        for storyline in self.database.get_active():
            similarity = self._title_similarity(
                normalized_title,
                Storyline.normalize_title(storyline.title),
            )
            if similarity >= self.title_similarity_threshold:
                return storyline.id

        return None

    def _calculate_file_overlap(self, files1: set, files2: set) -> float:
        """Calculate Jaccard similarity between file sets."""
        if not files1 or not files2:
            return 0.0
        intersection = len(files1 & files2)
        union = len(files1 | files2)
        return intersection / union if union > 0 else 0.0

    def _title_similarity(self, title1: str, title2: str) -> float:
        """Calculate word overlap similarity between titles."""
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    def _update_storyline(
        self,
        storyline_id: str,
        phase,
        signals: List[StorylineSignal],
    ) -> None:
        """Update an existing storyline with new signals."""
        storyline = self.database.storylines.get(storyline_id)
        if not storyline:
            return

        # Add signals
        for signal in signals:
            storyline.add_signal(signal)

        # Calculate update metrics
        phase_commits = [s.commit_hashes for s in signals]
        all_commits = [h for commits in phase_commits for h in commits]

        phase_files = set()
        for signal in signals:
            phase_files.update(signal.files)

        # Create update record
        update = StorylineUpdate(
            phase_number=phase.phase_number,
            timestamp=datetime.now().isoformat(),
            description=signals[0].description if signals else "",
            signals=signals,
            commit_count=len(all_commits),
            insertions=phase.total_insertions,
            deletions=phase.total_deletions,
            key_files=list(phase_files)[:10],
            key_commits=all_commits[:5],
        )

        storyline.add_update(update)

        # Update PR numbers
        for signal in signals:
            pr_num = signal.data.get('pr_number')
            if pr_num and pr_num not in storyline.pr_numbers:
                storyline.pr_numbers.append(pr_num)

        # Refresh indexes
        self.database.update_indexes(storyline)

    def _create_from_signals(
        self,
        signals: List[StorylineSignal],
        phase,
    ) -> List[Storyline]:
        """Create new storylines from unmatched signals."""
        # First, merge related signals
        merged = self.detector.merge_signals(signals, self.title_similarity_threshold)

        new_storylines = []
        for storyline in merged:
            # Only create if confidence is reasonable
            if storyline.confidence < 0.3:
                continue

            self.database.add_storyline(storyline)
            new_storylines.append(storyline)

        return new_storylines

    def _evaluate_all_transitions(
        self,
        current_phase: int,
        matched_signals: Dict[str, List[StorylineSignal]],
    ) -> List[Dict[str, Any]]:
        """Evaluate and apply state transitions for all storylines."""
        transitions = []

        for storyline in list(self.database.storylines.values()):
            # Skip terminal states
            if storyline.status in (StorylineStatus.COMPLETED, StorylineStatus.ABANDONED):
                continue

            signals = matched_signals.get(storyline.id, [])
            has_new_signal = len(signals) > 0

            # Check for explicit completion
            explicit_completion = any(
                s.data.get('is_completion') or s.data.get('status') == 'completed'
                for s in signals
            )

            new_status = self.state_machine.evaluate_transition(
                storyline,
                current_phase,
                has_new_signal=has_new_signal,
                explicit_completion=explicit_completion,
            )

            if new_status is not None:
                old_status = storyline.status
                self.state_machine.apply_transition(storyline, new_status)
                transitions.append({
                    'storyline_id': storyline.id,
                    'title': storyline.title,
                    'from': old_status.value,
                    'to': new_status.value,
                })

        return transitions

    def _save_database(self) -> None:
        """Save database to configured path."""
        if self.persist_path:
            self.database.save(self.persist_path)

    def get_active_storylines(self, limit: int = 10) -> List[Storyline]:
        """Get active storylines for context injection."""
        return self.database.get_active()[:limit]

    def get_storylines_for_prompt(self, limit: int = 8) -> List[Dict[str, Any]]:
        """
        Get storylines formatted for prompt injection.

        Returns simplified dicts compatible with existing summarizer format.
        """
        active = self.get_active_storylines(limit)

        result = []
        for sl in active:
            result.append({
                'title': sl.title,
                'category': sl.category.value,
                'status': sl.status.value,
                'description': sl.description,
                'last_update': sl.current_summary or sl.description,
                'first_phase': sl.first_phase,
                'last_phase': sl.last_phase,
                'phases_involved': sl.phases_involved,
            })

        return result

    def get_all_storylines_for_report(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all storylines grouped by status for reporting.

        Returns dict with keys: completed, active, stalled, abandoned
        """
        groups = {
            'completed': [],
            'active': [],
            'stalled': [],
            'abandoned': [],
            'emerging': [],
        }

        for sl in self.database.storylines.values():
            status_key = sl.status.value
            if status_key == 'progressing':
                status_key = 'active'

            if status_key in groups:
                groups[status_key].append(sl.to_dict())

        # Sort each group by last_phase descending
        for key in groups:
            groups[key].sort(key=lambda x: x.get('last_phase', 0), reverse=True)

        return groups

    def process_incremental(self, new_phases: list) -> Dict[str, Any]:
        """
        Process multiple new phases incrementally.

        Args:
            new_phases: List of Phase objects to process

        Returns:
            Aggregated results from all phases
        """
        total_results = {
            'phases_processed': 0,
            'signals_detected': 0,
            'storylines_updated': 0,
            'storylines_created': 0,
            'transitions': [],
        }

        for phase in new_phases:
            if phase.phase_number <= self.database.last_phase_analyzed:
                continue  # Skip already processed phases

            results = self.process_phase(phase)

            total_results['phases_processed'] += 1
            total_results['signals_detected'] += results['signals_detected']
            total_results['storylines_updated'] += results['storylines_updated']
            total_results['storylines_created'] += results['storylines_created']
            total_results['transitions'].extend(results['transitions'])

        return total_results


def create_tracker_from_legacy(
    legacy_storylines: List[Dict[str, Any]],
    persist_path: Optional[str] = None,
) -> StorylineTracker:
    """
    Create a tracker initialized from legacy storyline format.

    This allows migration from the quick-win format to the new system.
    """
    tracker = StorylineTracker(persist_path=persist_path)

    for sl in legacy_storylines:
        # Convert legacy format to Storyline
        storyline = Storyline(
            id=Storyline.generate_id(sl['title'], sl.get('first_phase', 1)),
            title=sl['title'],
            category=StorylineCategory.from_string(sl.get('category', 'feature')),
            status=_map_legacy_status(sl.get('status', 'active')),
            confidence=0.7,  # Assume reasonable confidence for existing data
            first_phase=sl.get('first_phase', 1),
            last_phase=sl.get('last_phase', 1),
            phases_involved=sl.get('phases_involved', []),
            description=sl.get('description', ''),
            current_summary=sl.get('last_update', sl.get('description', '')),
        )

        tracker.database.add_storyline(storyline)

    return tracker


def _map_legacy_status(status: str) -> StorylineStatus:
    """Map legacy status string to StorylineStatus enum."""
    mapping = {
        'active': StorylineStatus.ACTIVE,
        'new': StorylineStatus.EMERGING,
        'completed': StorylineStatus.COMPLETED,
        'stalled': StorylineStatus.STALLED,
    }
    return mapping.get(status.lower(), StorylineStatus.ACTIVE)
