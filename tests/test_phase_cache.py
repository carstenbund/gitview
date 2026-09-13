"""The phase cache must ignore non-phase files that share the phase_ prefix."""

import json
from pathlib import Path

from gitview.chunker import HistoryChunker, Phase


def _phase(n: int) -> Phase:
    return Phase(phase_number=n, start_date='2026-01-01T00:00:00', end_date='2026-01-02T00:00:00',
                 commit_count=0, loc_start=0, loc_end=0, loc_delta=0, loc_delta_percent=0.0,
                 total_insertions=0, total_deletions=0, languages_start={}, languages_end={},
                 has_large_deletion=False, has_large_addition=False, has_refactor=False,
                 readme_changed=False, authors=[], primary_author='', commits=[])


def test_load_phases_skips_non_phase_artifacts(tmp_path: Path):
    HistoryChunker(strategy='fixed').save_phases([_phase(1), _phase(2)], str(tmp_path))
    (tmp_path / 'phase_index.json').write_text(json.dumps({'1': 'a summary'}))
    (tmp_path / 'storylines.json').write_text(json.dumps({'version': 1}))
    phases = HistoryChunker.load_phases(str(tmp_path))
    assert [p.phase_number for p in phases] == [1, 2]
