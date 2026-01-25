"""State machine for storyline lifecycle management."""

from typing import Optional, Dict, Set
from datetime import datetime

from .models import Storyline, StorylineStatus, StorylineSignal


class StorylineStateMachine:
    """Manages storyline state transitions."""

    # Valid transitions from each state
    TRANSITIONS: Dict[StorylineStatus, Set[StorylineStatus]] = {
        StorylineStatus.EMERGING: {
            StorylineStatus.ACTIVE,
            StorylineStatus.ABANDONED,
        },
        StorylineStatus.ACTIVE: {
            StorylineStatus.PROGRESSING,
            StorylineStatus.STALLED,
            StorylineStatus.COMPLETED,
        },
        StorylineStatus.PROGRESSING: {
            StorylineStatus.ACTIVE,
            StorylineStatus.STALLED,
            StorylineStatus.COMPLETED,
        },
        StorylineStatus.STALLED: {
            StorylineStatus.ACTIVE,
            StorylineStatus.PROGRESSING,
            StorylineStatus.ABANDONED,
            StorylineStatus.COMPLETED,
        },
        StorylineStatus.COMPLETED: set(),  # Terminal state
        StorylineStatus.ABANDONED: set(),  # Terminal state
    }

    # Configuration thresholds
    STALL_THRESHOLD_PHASES = 3      # Phases without updates -> STALLED
    ABANDON_THRESHOLD_PHASES = 6    # Phases stalled without recovery -> ABANDONED
    EMERGING_CONFIDENCE_THRESHOLD = 0.6  # Min confidence to become ACTIVE
    EMERGING_SIGNAL_COUNT = 2       # Min signals to become ACTIVE

    def __init__(
        self,
        stall_threshold: int = 3,
        abandon_threshold: int = 6,
        confidence_threshold: float = 0.6,
        min_signals: int = 2,
    ):
        """
        Initialize state machine with custom thresholds.

        Args:
            stall_threshold: Phases without updates before STALLED
            abandon_threshold: Phases stalled before ABANDONED
            confidence_threshold: Min confidence to transition EMERGING -> ACTIVE
            min_signals: Min signals to transition EMERGING -> ACTIVE
        """
        self.stall_threshold = stall_threshold
        self.abandon_threshold = abandon_threshold
        self.confidence_threshold = confidence_threshold
        self.min_signals = min_signals

    def can_transition(self, from_status: StorylineStatus, to_status: StorylineStatus) -> bool:
        """Check if a transition is valid."""
        return to_status in self.TRANSITIONS.get(from_status, set())

    def evaluate_transition(
        self,
        storyline: Storyline,
        current_phase: int,
        has_new_signal: bool = False,
        explicit_completion: bool = False,
    ) -> Optional[StorylineStatus]:
        """
        Determine if storyline should transition states.

        Args:
            storyline: The storyline to evaluate
            current_phase: The current phase number being processed
            has_new_signal: Whether a new signal was detected for this storyline
            explicit_completion: Whether an explicit completion signal was detected

        Returns:
            New status if transition should occur, None otherwise
        """
        current_status = storyline.status
        phases_since_update = current_phase - storyline.last_phase

        # Handle explicit completion (highest priority)
        if explicit_completion and self.can_transition(current_status, StorylineStatus.COMPLETED):
            return StorylineStatus.COMPLETED

        # Terminal states don't transition
        if current_status in (StorylineStatus.COMPLETED, StorylineStatus.ABANDONED):
            return None

        # EMERGING -> ACTIVE: when confidence and signals meet threshold
        if current_status == StorylineStatus.EMERGING:
            if (storyline.confidence >= self.confidence_threshold and
                    len(storyline.signals) >= self.min_signals):
                return StorylineStatus.ACTIVE
            # EMERGING -> ABANDONED: if too long without confirmation
            if phases_since_update >= self.abandon_threshold:
                return StorylineStatus.ABANDONED

        # ACTIVE/PROGRESSING -> STALLED: no updates for threshold phases
        if current_status in (StorylineStatus.ACTIVE, StorylineStatus.PROGRESSING):
            if not has_new_signal and phases_since_update >= self.stall_threshold:
                return StorylineStatus.STALLED
            # ACTIVE -> PROGRESSING: has recent activity
            if has_new_signal and current_status == StorylineStatus.ACTIVE:
                return StorylineStatus.PROGRESSING
            # PROGRESSING -> ACTIVE: no new signal but not yet stalled
            if not has_new_signal and current_status == StorylineStatus.PROGRESSING:
                return StorylineStatus.ACTIVE

        # STALLED -> ACTIVE: resumed activity
        if current_status == StorylineStatus.STALLED:
            if has_new_signal:
                return StorylineStatus.ACTIVE
            # STALLED -> ABANDONED: too long without recovery
            if phases_since_update >= self.abandon_threshold:
                return StorylineStatus.ABANDONED

        return None

    def apply_transition(
        self,
        storyline: Storyline,
        new_status: StorylineStatus,
        reason: str = "",
    ) -> bool:
        """
        Apply a state transition to a storyline.

        Args:
            storyline: The storyline to update
            new_status: The new status to apply
            reason: Optional reason for the transition

        Returns:
            True if transition was applied, False if invalid
        """
        if not self.can_transition(storyline.status, new_status):
            return False

        old_status = storyline.status
        storyline.status = new_status
        storyline.updated_at = datetime.now().isoformat()

        # Add transition to tags for history
        transition_tag = f"transition:{old_status.value}->{new_status.value}"
        if transition_tag not in storyline.tags:
            storyline.tags.append(transition_tag)

        return True

    def evaluate_all(
        self,
        storylines: list,
        current_phase: int,
        signals_by_storyline: Dict[str, list] = None,
    ) -> Dict[str, StorylineStatus]:
        """
        Evaluate transitions for multiple storylines.

        Args:
            storylines: List of storylines to evaluate
            current_phase: Current phase number
            signals_by_storyline: Map of storyline_id -> signals detected this phase

        Returns:
            Dict of storyline_id -> new_status for storylines that should transition
        """
        signals_by_storyline = signals_by_storyline or {}
        transitions = {}

        for storyline in storylines:
            signals = signals_by_storyline.get(storyline.id, [])
            has_new_signal = len(signals) > 0

            # Check for explicit completion signal
            explicit_completion = any(
                s.data.get('status') == 'completed'
                for s in signals
            )

            new_status = self.evaluate_transition(
                storyline,
                current_phase,
                has_new_signal=has_new_signal,
                explicit_completion=explicit_completion,
            )

            if new_status is not None:
                transitions[storyline.id] = new_status

        return transitions


def detect_completion_keywords(text: str) -> bool:
    """Check if text contains completion indicators."""
    completion_patterns = [
        'completed',
        'finished',
        'done',
        'resolved',
        'closed',
        'merged',
        'shipped',
        'released',
        'launched',
    ]
    text_lower = text.lower()
    return any(pattern in text_lower for pattern in completion_patterns)


def detect_stall_keywords(text: str) -> bool:
    """Check if text contains stall/block indicators."""
    stall_patterns = [
        'blocked',
        'waiting',
        'on hold',
        'paused',
        'deferred',
        'delayed',
        'stalled',
        'pending',
        'wip',
        'work in progress',
    ]
    text_lower = text.lower()
    return any(pattern in text_lower for pattern in stall_patterns)
