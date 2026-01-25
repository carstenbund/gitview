"""Tests for the storyline tracking module."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from gitview.storyline.models import (
    StorylineStatus,
    StorylineCategory,
    StorylineSignal,
    StorylineUpdate,
    Storyline,
    StorylineDatabase,
)
from gitview.storyline.state_machine import (
    StorylineStateMachine,
    detect_completion_keywords,
    detect_stall_keywords,
)
from gitview.storyline.detector import (
    PRLabelDetector,
    PRTitlePatternDetector,
    CommitMessagePatternDetector,
    FileClusterDetector,
    StorylineDetector,
)


# =============================================================================
# Model Tests
# =============================================================================

class TestStorylineCategory:
    """Test StorylineCategory enum and helpers."""

    def test_from_string_direct_mapping(self):
        """Test direct string to category mapping."""
        assert StorylineCategory.from_string('feature') == StorylineCategory.FEATURE
        assert StorylineCategory.from_string('bug') == StorylineCategory.BUGFIX
        assert StorylineCategory.from_string('refactor') == StorylineCategory.REFACTOR

    def test_from_string_aliases(self):
        """Test alias mappings."""
        assert StorylineCategory.from_string('feat') == StorylineCategory.FEATURE
        assert StorylineCategory.from_string('fix') == StorylineCategory.BUGFIX
        assert StorylineCategory.from_string('docs') == StorylineCategory.DOCUMENTATION
        assert StorylineCategory.from_string('perf') == StorylineCategory.PERFORMANCE

    def test_from_string_case_insensitive(self):
        """Test case insensitivity."""
        assert StorylineCategory.from_string('FEATURE') == StorylineCategory.FEATURE
        assert StorylineCategory.from_string('BugFix') == StorylineCategory.BUGFIX

    def test_from_string_unknown(self):
        """Test unknown category handling."""
        assert StorylineCategory.from_string('random') == StorylineCategory.UNKNOWN
        assert StorylineCategory.from_string('') == StorylineCategory.UNKNOWN


class TestStorylineSignal:
    """Test StorylineSignal data class."""

    def test_create_signal(self):
        """Test basic signal creation."""
        signal = StorylineSignal(
            source='test',
            confidence=0.8,
            phase_number=1,
            commit_hashes=['abc123'],
            title='Test Signal',
            category=StorylineCategory.FEATURE,
            description='A test signal',
        )

        assert signal.source == 'test'
        assert signal.confidence == 0.8
        assert signal.title == 'Test Signal'

    def test_signal_serialization(self):
        """Test signal to/from dict."""
        signal = StorylineSignal(
            source='test',
            confidence=0.9,
            phase_number=2,
            commit_hashes=['abc', 'def'],
            title='Test',
            category=StorylineCategory.BUGFIX,
            description='Description',
            files=['file1.py', 'file2.py'],
        )

        data = signal.to_dict()
        restored = StorylineSignal.from_dict(data)

        assert restored.source == signal.source
        assert restored.confidence == signal.confidence
        assert restored.category == signal.category
        assert restored.files == signal.files


class TestStoryline:
    """Test Storyline data class."""

    def test_generate_id(self):
        """Test ID generation is deterministic."""
        id1 = Storyline.generate_id('Test Story', 1)
        id2 = Storyline.generate_id('Test Story', 1)
        id3 = Storyline.generate_id('Test Story', 2)

        assert id1 == id2  # Same input = same ID
        assert id1 != id3  # Different phase = different ID
        assert id1.startswith('sl_')

    def test_normalize_title(self):
        """Test title normalization."""
        assert Storyline.normalize_title('Test Story') == 'test story'
        assert Storyline.normalize_title('  Test  Story  ') == 'test story'
        assert Storyline.normalize_title('Test-Story!') == 'test story'

    def test_from_signal(self):
        """Test creating storyline from signal."""
        signal = StorylineSignal(
            source='test',
            confidence=0.7,
            phase_number=3,
            commit_hashes=['abc'],
            title='OAuth Implementation',
            category=StorylineCategory.FEATURE,
            description='Adding OAuth',
            files=['auth.py'],
        )

        storyline = Storyline.from_signal(signal)

        assert storyline.title == 'OAuth Implementation'
        assert storyline.category == StorylineCategory.FEATURE
        assert storyline.status == StorylineStatus.EMERGING
        assert storyline.first_phase == 3
        assert storyline.confidence == 0.7
        assert 'auth.py' in storyline.key_files

    def test_add_signal_updates_confidence(self):
        """Test that adding signals updates confidence."""
        signal1 = StorylineSignal(
            source='test1', confidence=0.6, phase_number=1,
            commit_hashes=['a'], title='Test', category=StorylineCategory.FEATURE,
            description='Test',
        )
        signal2 = StorylineSignal(
            source='test2', confidence=0.8, phase_number=1,
            commit_hashes=['b'], title='Test', category=StorylineCategory.FEATURE,
            description='Test',
        )

        storyline = Storyline.from_signal(signal1)
        initial_confidence = storyline.confidence

        storyline.add_signal(signal2)

        assert storyline.confidence > initial_confidence
        assert len(storyline.signals) == 2

    def test_serialization_roundtrip(self):
        """Test storyline serialization and deserialization."""
        storyline = Storyline(
            id='sl_test123',
            title='Test Storyline',
            category=StorylineCategory.REFACTOR,
            status=StorylineStatus.ACTIVE,
            confidence=0.85,
            first_phase=1,
            last_phase=3,
            phases_involved=[1, 2, 3],
            description='Test description',
            key_files={'file1.py', 'file2.py'},
            key_authors={'author1'},
            pr_numbers=[42, 43],
        )

        data = storyline.to_dict()
        restored = Storyline.from_dict(data)

        assert restored.id == storyline.id
        assert restored.title == storyline.title
        assert restored.category == storyline.category
        assert restored.status == storyline.status
        assert restored.phases_involved == storyline.phases_involved


class TestStorylineDatabase:
    """Test StorylineDatabase container."""

    def test_add_and_retrieve_storyline(self):
        """Test adding and retrieving storylines."""
        db = StorylineDatabase()

        storyline = Storyline(
            id='sl_test',
            title='Test',
            category=StorylineCategory.FEATURE,
            status=StorylineStatus.ACTIVE,
            confidence=0.8,
            first_phase=1,
            last_phase=2,
            phases_involved=[1, 2],
        )

        db.add_storyline(storyline)

        assert db.get_by_title('Test') == storyline
        assert db.get_by_title('test') == storyline  # Case insensitive
        assert storyline in db.get_by_phase(1)
        assert storyline in db.get_by_phase(2)

    def test_get_active_storylines(self):
        """Test filtering active storylines."""
        db = StorylineDatabase()

        active = Storyline(
            id='sl_active', title='Active', category=StorylineCategory.FEATURE,
            status=StorylineStatus.ACTIVE, confidence=0.8, first_phase=1, last_phase=1,
        )
        completed = Storyline(
            id='sl_completed', title='Completed', category=StorylineCategory.FEATURE,
            status=StorylineStatus.COMPLETED, confidence=0.8, first_phase=1, last_phase=1,
        )

        db.add_storyline(active)
        db.add_storyline(completed)

        active_list = db.get_active()
        assert active in active_list
        assert completed not in active_list

    def test_serialization_roundtrip(self):
        """Test database serialization."""
        db = StorylineDatabase()
        db.last_phase_analyzed = 5
        db.last_commit_hash = 'abc123'

        storyline = Storyline(
            id='sl_test', title='Test', category=StorylineCategory.FEATURE,
            status=StorylineStatus.ACTIVE, confidence=0.8, first_phase=1, last_phase=1,
        )
        db.add_storyline(storyline)

        data = db.to_dict()
        restored = StorylineDatabase.from_dict(data)

        assert restored.last_phase_analyzed == 5
        assert restored.last_commit_hash == 'abc123'
        assert 'sl_test' in restored.storylines


# =============================================================================
# State Machine Tests
# =============================================================================

class TestStorylineStateMachine:
    """Test state machine transitions."""

    def test_valid_transitions(self):
        """Test valid transition checking."""
        sm = StorylineStateMachine()

        # Valid transitions
        assert sm.can_transition(StorylineStatus.EMERGING, StorylineStatus.ACTIVE)
        assert sm.can_transition(StorylineStatus.ACTIVE, StorylineStatus.COMPLETED)
        assert sm.can_transition(StorylineStatus.STALLED, StorylineStatus.ACTIVE)

        # Invalid transitions
        assert not sm.can_transition(StorylineStatus.COMPLETED, StorylineStatus.ACTIVE)
        assert not sm.can_transition(StorylineStatus.ABANDONED, StorylineStatus.ACTIVE)
        assert not sm.can_transition(StorylineStatus.EMERGING, StorylineStatus.COMPLETED)

    def test_emerging_to_active_transition(self):
        """Test EMERGING -> ACTIVE when confidence threshold met."""
        sm = StorylineStateMachine(confidence_threshold=0.6, min_signals=2)

        storyline = Storyline(
            id='sl_test', title='Test', category=StorylineCategory.FEATURE,
            status=StorylineStatus.EMERGING, confidence=0.7,
            first_phase=1, last_phase=1, phases_involved=[1],
        )
        # Add signals to meet threshold
        storyline.signals = [MagicMock(), MagicMock()]

        new_status = sm.evaluate_transition(storyline, current_phase=2)
        assert new_status == StorylineStatus.ACTIVE

    def test_stall_transition(self):
        """Test transition to STALLED after inactivity."""
        sm = StorylineStateMachine(stall_threshold=3)

        storyline = Storyline(
            id='sl_test', title='Test', category=StorylineCategory.FEATURE,
            status=StorylineStatus.ACTIVE, confidence=0.8,
            first_phase=1, last_phase=1, phases_involved=[1],
        )

        # No new signal, 4 phases since last update
        new_status = sm.evaluate_transition(storyline, current_phase=5, has_new_signal=False)
        assert new_status == StorylineStatus.STALLED

    def test_stalled_to_active_on_new_signal(self):
        """Test STALLED -> ACTIVE when new signal detected."""
        sm = StorylineStateMachine()

        storyline = Storyline(
            id='sl_test', title='Test', category=StorylineCategory.FEATURE,
            status=StorylineStatus.STALLED, confidence=0.8,
            first_phase=1, last_phase=1, phases_involved=[1],
        )

        new_status = sm.evaluate_transition(storyline, current_phase=5, has_new_signal=True)
        assert new_status == StorylineStatus.ACTIVE

    def test_explicit_completion(self):
        """Test explicit completion overrides other logic."""
        sm = StorylineStateMachine()

        storyline = Storyline(
            id='sl_test', title='Test', category=StorylineCategory.FEATURE,
            status=StorylineStatus.ACTIVE, confidence=0.8,
            first_phase=1, last_phase=1, phases_involved=[1],
        )

        new_status = sm.evaluate_transition(
            storyline, current_phase=2, explicit_completion=True
        )
        assert new_status == StorylineStatus.COMPLETED


class TestKeywordDetection:
    """Test keyword detection helpers."""

    def test_detect_completion_keywords(self):
        """Test completion keyword detection."""
        assert detect_completion_keywords('Feature completed')
        assert detect_completion_keywords('Issue resolved')
        assert detect_completion_keywords('PR merged')
        assert not detect_completion_keywords('Work in progress')

    def test_detect_stall_keywords(self):
        """Test stall keyword detection."""
        assert detect_stall_keywords('Blocked by dependency')
        assert detect_stall_keywords('On hold for review')
        assert detect_stall_keywords('WIP: initial implementation')
        assert not detect_stall_keywords('Feature completed')


# =============================================================================
# Detector Tests
# =============================================================================

class TestPRLabelDetector:
    """Test PR label detection."""

    def test_category_mapping(self):
        """Test label to category mapping."""
        detector = PRLabelDetector()

        assert detector._get_category_from_labels(['feature']) == StorylineCategory.FEATURE
        assert detector._get_category_from_labels(['bug']) == StorylineCategory.BUGFIX
        assert detector._get_category_from_labels(['unknown']) == StorylineCategory.UNKNOWN

    def test_detect_with_labeled_commits(self):
        """Test detection from commits with PR labels."""
        detector = PRLabelDetector()

        # Create mock phase with commits
        mock_commit = MagicMock()
        mock_commit.has_github_context.return_value = True
        mock_commit.get_pr_labels.return_value = ['feature']
        mock_commit.get_pr_title.return_value = 'Add user authentication'
        mock_commit.get_pr_body.return_value = 'Implements OAuth2 flow'
        mock_commit.short_hash = 'abc123'
        mock_commit.github_context = {'pr_number': 42}

        mock_phase = MagicMock()
        mock_phase.phase_number = 1
        mock_phase.commits = [mock_commit]

        signals = detector.detect(mock_phase)

        assert len(signals) == 1
        assert signals[0].category == StorylineCategory.FEATURE
        assert signals[0].confidence == 0.9


class TestPRTitlePatternDetector:
    """Test PR title pattern detection."""

    def test_match_category(self):
        """Test title pattern matching."""
        detector = PRTitlePatternDetector()

        assert detector._match_category('feat: add login') == StorylineCategory.FEATURE
        assert detector._match_category('fix: resolve crash') == StorylineCategory.BUGFIX
        assert detector._match_category('docs: update readme') == StorylineCategory.DOCUMENTATION
        assert detector._match_category('random title') == StorylineCategory.UNKNOWN


class TestCommitMessagePatternDetector:
    """Test commit message pattern detection."""

    def test_analyze_message(self):
        """Test message analysis."""
        detector = CommitMessagePatternDetector()

        category, keywords = detector._analyze_message('Add new feature for users')
        assert category == StorylineCategory.FEATURE
        assert 'add' in keywords or 'new' in keywords or 'feature' in keywords

        category, keywords = detector._analyze_message('Fix bug in authentication')
        assert category == StorylineCategory.BUGFIX


class TestStorylineDetector:
    """Test main storyline detector."""

    def test_merge_signals_by_title(self):
        """Test signal merging by similar titles."""
        detector = StorylineDetector()

        signals = [
            StorylineSignal(
                source='test1', confidence=0.8, phase_number=1,
                commit_hashes=['a'], title='User Authentication',
                category=StorylineCategory.FEATURE, description='Test',
            ),
            StorylineSignal(
                source='test2', confidence=0.7, phase_number=1,
                commit_hashes=['b'], title='User authentication system',
                category=StorylineCategory.FEATURE, description='Test',
            ),
        ]

        storylines = detector.merge_signals(signals, title_similarity_threshold=0.5)

        # Should merge into one storyline due to title similarity
        assert len(storylines) == 1
        assert len(storylines[0].signals) == 2

    def test_title_similarity(self):
        """Test title similarity calculation."""
        detector = StorylineDetector()

        # Identical
        assert detector._title_similarity('test', 'test') == 1.0

        # Similar
        sim = detector._title_similarity('user authentication', 'user auth system')
        assert sim > 0.3

        # Different
        sim = detector._title_similarity('feature one', 'bug fix')
        assert sim < 0.3


# =============================================================================
# Integration Tests
# =============================================================================

class TestStorylineIntegration:
    """Integration tests for storyline workflow."""

    def test_full_workflow(self):
        """Test complete storyline detection workflow."""
        # Create signals
        signal1 = StorylineSignal(
            source='PRLabelDetector', confidence=0.9, phase_number=1,
            commit_hashes=['abc'], title='OAuth Implementation',
            category=StorylineCategory.FEATURE, description='Add OAuth2',
        )
        signal2 = StorylineSignal(
            source='CommitMessagePatternDetector', confidence=0.6, phase_number=2,
            commit_hashes=['def'], title='OAuth Implementation',
            category=StorylineCategory.FEATURE, description='OAuth token refresh',
        )

        # Create storyline from first signal
        storyline = Storyline.from_signal(signal1)
        assert storyline.status == StorylineStatus.EMERGING

        # Add second signal
        storyline.add_signal(signal2)
        storyline.last_phase = 2
        storyline.phases_involved.append(2)

        # Evaluate transition
        sm = StorylineStateMachine(confidence_threshold=0.6, min_signals=2)
        new_status = sm.evaluate_transition(storyline, current_phase=2)

        assert new_status == StorylineStatus.ACTIVE

        # Apply transition
        sm.apply_transition(storyline, new_status)
        assert storyline.status == StorylineStatus.ACTIVE

        # Store in database
        db = StorylineDatabase()
        db.add_storyline(storyline)

        # Retrieve
        found = db.get_by_title('OAuth Implementation')
        assert found is not None
        assert found.status == StorylineStatus.ACTIVE


# =============================================================================
# Parser Tests
# =============================================================================

class TestStorylineResponseParser:
    """Test storyline response parser."""

    def test_parse_json_block(self):
        """Test parsing JSON code block format."""
        from gitview.storyline.parser import StorylineResponseParser

        parser = StorylineResponseParser()

        response = '''Here is my summary of the phase.

```json
{
  "storylines": [
    {
      "title": "OAuth Implementation",
      "status": "continued",
      "category": "feature",
      "description": "Added token refresh"
    }
  ]
}
```
'''

        summary, storylines = parser.parse(response)

        assert len(storylines) == 1
        assert storylines[0]['title'] == 'OAuth Implementation'
        assert storylines[0]['status'] == 'continued'
        assert storylines[0]['category'] == 'feature'

    def test_parse_markdown_format(self):
        """Test parsing markdown format (fallback)."""
        from gitview.storyline.parser import StorylineResponseParser

        parser = StorylineResponseParser()

        response = '''Here is my summary.

## Storylines
- [NEW:feature] User Auth: Started implementing OAuth
- [CONTINUED:refactor] API Cleanup: Migrated more endpoints
- [COMPLETED:bugfix] Token Bug: Fixed refresh issue
'''

        summary, storylines = parser.parse(response)

        assert len(storylines) == 3
        assert storylines[0]['title'] == 'User Auth'
        assert storylines[0]['status'] == 'new'
        assert storylines[1]['status'] == 'continued'
        assert storylines[2]['status'] == 'completed'

    def test_normalize_status(self):
        """Test status normalization."""
        from gitview.storyline.parser import StorylineResponseParser

        parser = StorylineResponseParser()

        assert parser._normalize_status('NEW') == 'new'
        assert parser._normalize_status('started') == 'new'
        assert parser._normalize_status('ongoing') == 'continued'
        assert parser._normalize_status('done') == 'completed'
        assert parser._normalize_status('blocked') == 'stalled'


# =============================================================================
# Tracker Tests
# =============================================================================

class TestStorylineTracker:
    """Test storyline tracker."""

    def test_convert_llm_storylines(self):
        """Test converting LLM storylines to signals."""
        from gitview.storyline.tracker import StorylineTracker

        tracker = StorylineTracker()

        llm_storylines = [
            {'title': 'OAuth', 'status': 'new', 'category': 'feature', 'description': 'Started OAuth'},
            {'title': 'Cleanup', 'status': 'continued', 'category': 'refactor', 'description': 'More cleanup'},
        ]

        signals = tracker._convert_llm_storylines_to_signals(llm_storylines, phase_number=3)

        assert len(signals) == 2
        assert signals[0].title == 'OAuth'
        assert signals[0].phase_number == 3
        assert signals[1].title == 'Cleanup'

    def test_get_storylines_for_prompt(self):
        """Test getting storylines formatted for prompts."""
        from gitview.storyline.tracker import StorylineTracker

        tracker = StorylineTracker()

        # Add a storyline
        signal = StorylineSignal(
            source='test', confidence=0.8, phase_number=1,
            commit_hashes=['abc'], title='Test Feature',
            category=StorylineCategory.FEATURE, description='Test',
        )
        storyline = Storyline.from_signal(signal)
        storyline.status = StorylineStatus.ACTIVE
        tracker.database.add_storyline(storyline)

        prompt_data = tracker.get_storylines_for_prompt()

        assert len(prompt_data) == 1
        assert prompt_data[0]['title'] == 'Test Feature'
        assert 'status' in prompt_data[0]


# =============================================================================
# Reporter Tests
# =============================================================================

class TestStorylineReporter:
    """Test storyline reporter."""

    def test_generate_index(self):
        """Test generating storyline index."""
        from gitview.storyline.reporter import StorylineReporter

        db = StorylineDatabase()

        # Add storylines
        completed = Storyline(
            id='sl_1', title='Completed Feature', category=StorylineCategory.FEATURE,
            status=StorylineStatus.COMPLETED, confidence=0.9,
            first_phase=1, last_phase=3, phases_involved=[1, 2, 3],
        )
        active = Storyline(
            id='sl_2', title='Active Work', category=StorylineCategory.REFACTOR,
            status=StorylineStatus.ACTIVE, confidence=0.8,
            first_phase=2, last_phase=4, phases_involved=[2, 3, 4],
        )

        db.add_storyline(completed)
        db.add_storyline(active)

        reporter = StorylineReporter(database=db)
        index = reporter.generate_storyline_index()

        assert '# Storyline Index' in index
        assert 'Completed Feature' in index
        assert 'Active Work' in index

    def test_generate_timeline_view(self):
        """Test generating ASCII timeline."""
        from gitview.storyline.reporter import StorylineReporter

        db = StorylineDatabase()

        storyline = Storyline(
            id='sl_1', title='Test Feature', category=StorylineCategory.FEATURE,
            status=StorylineStatus.ACTIVE, confidence=0.8,
            first_phase=1, last_phase=3, phases_involved=[1, 2, 3],
        )
        db.add_storyline(storyline)

        reporter = StorylineReporter(database=db)
        timeline = reporter.generate_timeline_view()

        assert '# Storyline Timeline' in timeline
        assert 'Test Feature' in timeline

    def test_get_stats(self):
        """Test statistics gathering."""
        from gitview.storyline.reporter import StorylineReporter

        db = StorylineDatabase()

        completed = Storyline(
            id='sl_1', title='Done', category=StorylineCategory.FEATURE,
            status=StorylineStatus.COMPLETED, confidence=0.9,
            first_phase=1, last_phase=1,
        )
        active = Storyline(
            id='sl_2', title='Active', category=StorylineCategory.FEATURE,
            status=StorylineStatus.ACTIVE, confidence=0.8,
            first_phase=1, last_phase=2,
        )

        db.add_storyline(completed)
        db.add_storyline(active)

        reporter = StorylineReporter(database=db)
        stats = reporter._get_stats()

        assert stats['total'] == 2
        assert stats['completed'] == 1
        assert stats['active'] == 1
