"""The story cache must be invalidated when the prompt inputs change, not only the phases."""

from gitview.chunker import Phase
from gitview.storyteller import StoryTeller


def _phase() -> Phase:
    p = Phase(phase_number=1, start_date='2026-01-01T00:00:00', end_date='2026-01-02T00:00:00',
              commit_count=1, loc_start=0, loc_end=10, loc_delta=10, loc_delta_percent=1.0,
              total_insertions=10, total_deletions=0, languages_start={}, languages_end={},
              has_large_deletion=False, has_large_addition=False, has_refactor=False,
              readme_changed=False, authors=['a'], primary_author='a', commits=[])
    p.summary = 'phase one'
    return p


def test_story_cache_keyed_on_directives_and_critical(tmp_path):
    phases = [_phase()]
    plain = StoryTeller(backend='ollama')
    plain._save_story_cache(phases, {'executive_summary': 'old'}, str(tmp_path))
    assert plain._load_cached_story(phases, str(tmp_path)) == {'executive_summary': 'old'}
    assert StoryTeller(backend='ollama')._load_cached_story(phases, str(tmp_path)) is not None
    assert StoryTeller(backend='ollama', directives='OEBV = Ophtecs Europe BV')._load_cached_story(phases, str(tmp_path)) is None
    assert StoryTeller(backend='ollama', critical_mode=True)._load_cached_story(phases, str(tmp_path)) is None
