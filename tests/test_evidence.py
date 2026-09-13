"""Tests for deterministic evidence in analyze (gitview.evidence)."""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from gitview.backends.base import LLMResponse
from gitview.backends.router import LLMRouter
from gitview.chunker import HistoryChunker
from gitview.cli import cli
from gitview.evidence import EvidenceLedger, LLM_BUDGETS, cluster_summary, llm_threshold
from gitview.significance_analyzer import SignificanceAnalyzer
from gitview.summarizer import _parse_storylines

from tests.test_graph import _commit, _write_and_commit, synthetic_repo  # noqa: F401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _records(n_routine=6, feature=True):
    """Chronological records: a docs-only routine stretch, then feature work with a refactor."""
    records = [_commit(i, 'README.md', author='Doc', ts=f"2026-01-{i:02d}T00:00:00")
               for i in range(1, n_routine + 1)]
    if feature:
        base = n_routine
        records += [
            _commit(base + 1, 'src/core.py', 'src/api.py', 'tests/test_core.py',
                    ts=f"2026-02-01T00:00:00"),
            _commit(base + 2, 'src/core.py', 'src/api.py', ts="2026-02-02T00:00:00"),
            _commit(base + 3, 'src/core.py', 'src/api.py', ts="2026-02-03T00:00:00"),
            _commit(base + 4, 'src/core.py', 'src/api.py', 'src/util.py', ts="2026-02-04T00:00:00"),
        ]
    if feature:
        records[n_routine + 2].is_refactor = True
        records[n_routine + 2].insertions, records[n_routine + 2].deletions = 160, 150
    for i, r in enumerate(records):
        r.commit_subject = f"{'docs: tweak readme' if r.files_stats.get('README.md') else 'feat: core work'} {i}"
        r.commit_message = r.commit_subject
        r.loc_total = 100 + 10 * i
    return records


def _phases(records, chunk_size):
    return HistoryChunker('fixed').chunk(records, chunk_size=chunk_size)


def _ledger(tmp_path, records, git_repo):
    return EvidenceLedger(git_repo, records, store_path=tmp_path / 'evidence.sqlite')


class _StubBackend:
    """Counts calls and returns a summary with a parseable Storylines section."""

    def __init__(self):
        self.calls = 0
        self.prompts = []

    def generate(self, messages, max_tokens=2000, **kwargs):
        self.calls += 1
        self.prompts.append(messages[-1].content)
        return LLMResponse(content="Model narrative.\n\n## Storylines\n- [NEW:feature] core-work: model said so\n",
                           model='stub')


@pytest.fixture
def stub_llm(monkeypatch):
    stub = _StubBackend()
    monkeypatch.setattr(LLMRouter, '_get_backend', lambda self: stub)
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'x')
    LLMRouter.reset_call_count()
    return stub


# ---------------------------------------------------------------------------
# Budget and scoring
# ---------------------------------------------------------------------------

def test_budget_thresholds():
    assert llm_threshold('full') == float('-inf')
    assert llm_threshold('balanced') < llm_threshold('minimal')
    with pytest.raises(ValueError):
        llm_threshold('lavish')


def test_routine_phase_scores_low_and_feature_phase_high(tmp_path, synthetic_repo):
    records = _records()
    routine, feature = _phases(records, chunk_size=6)
    ledger = _ledger(tmp_path, records, synthetic_repo)

    ev_r, ev_f = ledger.phase_evidence(routine), ledger.phase_evidence(feature)
    assert ev_r.score < LLM_BUDGETS['balanced']
    assert ev_f.score >= LLM_BUDGETS['balanced']
    assert '1 significant commit(s)' in ev_f.reasons
    assert ev_f.active_motifs == []          # repeated co-change is context, not an event
    assert ev_f.top_files[0][0] in {'src/core.py', 'src/api.py'}
    assert ev_f.coupling and ev_f.coupling[0][:2] == ('src/api.py', 'src/core.py')


def test_tiny_refactors_and_timeless_motifs_are_not_signals(tmp_path, synthetic_repo):
    records = _records()
    # A 4-line "refactor" (the extractor flags submodule-pointer bumps this way) must not count.
    records[2].is_refactor = True
    records[2].insertions, records[2].deletions = 2, 2
    routine, _ = _phases(records, chunk_size=6)
    ev = _ledger(tmp_path, records, synthetic_repo).phase_evidence(routine)
    assert ev.significant_commits == 0
    assert ev.active_motifs == []
    assert ev.score < LLM_BUDGETS['balanced']


def test_router_counts_calls(stub_llm):
    router = LLMRouter(backend='anthropic', api_key='x')
    router.generate([SimpleNamespace(content='hi')])
    router.generate([SimpleNamespace(content='hi')])
    assert router.calls == 2
    assert LLMRouter.total_calls == 2
    assert LLMRouter.reset_call_count() == 2
    assert LLMRouter.total_calls == 0


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_render_summary_is_parseable_and_tracks_storyline_status(tmp_path, synthetic_repo):
    records = _records()
    routine, feature = _phases(records, chunk_size=6)
    ledger = _ledger(tmp_path, records, synthetic_repo)

    first = ledger.phase_evidence(routine).render_summary(prior_slugs=set())
    parsed = _parse_storylines(first)
    assert parsed and parsed[0]['status'] == 'new'
    assert 'without an LLM call' in first

    slugs = {s for s, _, _ in ledger.phase_evidence(routine).storylines()}
    again = ledger.phase_evidence(routine).render_summary(prior_slugs=slugs)
    assert _parse_storylines(again)[0]['status'] == 'continued'


def test_prompt_block_mentions_clusters_and_coupling(tmp_path, synthetic_repo):
    records = _records()
    _, feature = _phases(records, chunk_size=6)
    block = _ledger(tmp_path, records, synthetic_repo).phase_evidence(feature).prompt_block()
    assert 'Established Evidence' in block
    assert 'Activity clusters' in block
    assert 'src/api.py + src/core.py' in block


def test_cluster_summary_is_factual():
    records = _records(n_routine=0)
    [cluster] = SignificanceAnalyzer().cluster_commits(records)[:1]
    text = cluster_summary(cluster)
    assert str(len(cluster.commits)) in text
    assert 'src/core.py' in text


def test_repository_sections(tmp_path, synthetic_repo):
    records = _records()
    records[-1].deletions = 500
    records[-1].is_large_deletion = True
    phases = _phases(records, chunk_size=6)
    ledger = _ledger(tmp_path, records, synthetic_repo)

    tech = ledger.technical_evolution(phases)
    assert '### Where change concentrated' in tech and '`src/core.py`' in tech
    assert '**Phase 2**' in tech

    deletions = ledger.deletion_story(phases)
    assert '-500' in deletions and 'Phase 2' in deletions

    arch = ledger.architecture_section()
    assert 'Repeated co-change' in arch
    assert 'structural observation needed' in arch


def test_hard_facts_state_span_people_modules_and_rules(tmp_path, synthetic_repo):
    records = _records()
    phases = _phases(records, chunk_size=6)
    (synthetic_repo / '.gitmodules').write_text(
        '[submodule "api"]\n\tpath = oebv-api\n\turl = x\n[submodule "db"]\n\tpath = oebv-db\n\turl = y\n')
    records[-1].commit_subject = 'chore: bump schema to v6.23'
    facts = _ledger(tmp_path, records, synthetic_repo).hard_facts(phases, 'demo')

    assert 'History covered: 2026-01-01 to 2026-02-04, 10 commits in 2 phase(s)' in facts
    assert 'Nothing after 2026-02-04 has happened' in facts
    assert 'Doc (6 commits)' in facts and 'Dev (4 commits)' in facts
    assert 'Git submodules (complete list): oebv-api, oebv-db' in facts
    assert 'Top-level directories touched: src, tests' in facts
    assert 'Version strings that appear in commit subjects: 6.23' in facts
    assert 'Do not present plans or proposals as implemented work' in facts


# ---------------------------------------------------------------------------
# storyteller: facts in every prompt, batches stitched without a merge call
# ---------------------------------------------------------------------------

def _summary_phases(n):
    from types import SimpleNamespace
    return [SimpleNamespace(
        phase_number=i, start_date=f"2026-01-{i:02d}", end_date=f"2026-01-{i:02d}", commit_count=3,
        loc_delta=10, loc_delta_percent=1.0, total_insertions=12, total_deletions=2, authors=['Dev'],
        primary_author='Dev', has_large_deletion=False, has_large_addition=False, has_refactor=False,
        readme_changed=False, summary=f"phase {i} summary", commits=[]) for i in range(1, n + 1)]


def test_storyteller_puts_facts_before_the_instruction_and_stitches_batches(stub_llm):
    from gitview.storyteller import StoryTeller
    teller = StoryTeller(backend='anthropic', api_key='x', max_phases_per_prompt=2,
                         facts='**Hard facts about demo**\n- History covered: 2026-01-01 to 2026-01-05',
                         precomputed_sections={'technical_evolution': 'T', 'deletion_story': 'D'})
    stories = teller.generate_global_story(_summary_phases(5), 'demo')

    # 5 phases in batches of 2 -> 3 calls per model-written section, no merge call.
    assert stub_llm.calls == 3 * 3
    for prompt in stub_llm.prompts:
        assert 'Hard facts about demo' in prompt
        stripped = prompt.rstrip()
        assert stripped.rpartition('\n')[2].startswith('Write the')   # instruction still closes the prompt
        assert stripped.rfind('Hard facts about demo') > len(stripped) - 700  # facts sit just before it
    assert any('part 2 of 3' in p and 'phases 3-4' in p for p in stub_llm.prompts)
    assert stories['full_narrative'].count('### 2026-01-') == 3     # stitched, chronological
    assert stories['timeline'].count('Model narrative.') == 3
    assert stories['technical_evolution'] == 'T'


def test_storyteller_single_prompt_when_it_fits(stub_llm):
    from gitview.storyteller import StoryTeller
    teller = StoryTeller(backend='anthropic', api_key='x', facts='F',
                         precomputed_sections={'technical_evolution': 'T', 'deletion_story': 'D'})
    teller.generate_global_story(_summary_phases(5), 'demo')
    assert stub_llm.calls == 3
    assert all('part 1 of' not in p for p in stub_llm.prompts)


# ---------------------------------------------------------------------------
# analyze end to end (stubbed model): fewer calls, same outputs
# ---------------------------------------------------------------------------

def _repo_with_history(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    import subprocess
    subprocess.run(['git', 'init', '-q', '-b', 'main', str(repo)], check=True)
    for i in range(1, 7):
        _write_and_commit(repo, f"docs: readme pass {i}", **{'README.md': 'x' * i + '\n'})
    _write_and_commit(repo, "feat: add core", **{'src/core.py': 'def a():\n    return 1\n' * 40,
                                                 'src/api.py': 'import core\n' * 40})
    _write_and_commit(repo, "feat: wire api", **{'src/core.py': 'def a():\n    return 2\n' * 40,
                                                 'src/api.py': 'import core  # v2\n' * 40})
    _write_and_commit(repo, "refactor: restructure core and api", **{
        'src/core.py': 'def b():\n    return 3\n' * 80, 'src/api.py': 'import core  # v3\n' * 80})
    _write_and_commit(repo, "feat: util", **{'src/util.py': 'x = 1\n' * 60,
                                             'src/core.py': 'def b():\n    return 4\n' * 80})
    # A >1000-line addition: the one commit that makes the second phase worth narrating.
    _write_and_commit(repo, "feat: generated client", **{'src/client.py': 'pass\n' * 1200})
    return repo


def _run_analyze(repo, out, *extra):
    return CliRunner().invoke(cli, ['analyze', '--repo', str(repo), '--output', str(out),
                                    '--backend', 'anthropic', '--strategy', 'fixed', '--chunk-size', '6',
                                    *extra])


def test_analyze_balanced_budget_spends_fewer_calls(tmp_path, stub_llm):
    repo = _repo_with_history(tmp_path)
    result = _run_analyze(repo, tmp_path / 'out', '--llm-budget', 'balanced')
    assert result.exit_code == 0, result.output

    # 2 phases (6 docs commits, then 5 feature commits incl. a >1000-line addition);
    # the routine docs phase is written from evidence, the feature phase by the model.
    # Story: 5 sections minus the two pre-rendered ones = 3 calls.
    assert stub_llm.calls == 1 + 3
    assert 'from evidence' in result.output
    assert 'LLM calls this run: 4' in result.output

    story = (tmp_path / 'out' / 'history_story.md').read_text()
    assert '## Architectural Motifs' in story
    assert '### Where change concentrated' in story          # deterministic technical evolution
    data = json.loads((tmp_path / 'out' / 'history_data.json').read_text())
    assert len(data['phases']) == 2
    # The model prompt for the narrated phase carried the evidence block, and
    # every story prompt carried the hard facts.
    assert any('Established Evidence' in p for p in stub_llm.prompts)
    assert sum('Hard facts about' in p for p in stub_llm.prompts) == 3


def test_analyze_full_budget_calls_model_for_every_phase(tmp_path, stub_llm):
    repo = _repo_with_history(tmp_path)
    result = _run_analyze(repo, tmp_path / 'out', '--llm-budget', 'full')
    assert result.exit_code == 0, result.output
    assert stub_llm.calls == 2 + 3


def test_analyze_without_evidence_is_unchanged(tmp_path, stub_llm):
    repo = _repo_with_history(tmp_path)
    result = _run_analyze(repo, tmp_path / 'out', '--no-evidence')
    assert result.exit_code == 0, result.output
    assert stub_llm.calls == 2 + 5
    assert not any('Established Evidence' in p for p in stub_llm.prompts)
    assert '## Architectural Motifs' not in (tmp_path / 'out' / 'history_story.md').read_text()


def test_analyze_regenerate_story_reuses_summaries_and_rebuilds_story(tmp_path, stub_llm):
    repo = _repo_with_history(tmp_path)
    out = tmp_path / 'out'
    first = _run_analyze(repo, out, '--llm-budget', 'balanced')
    assert first.exit_code == 0, first.output
    assert stub_llm.calls == 1 + 3

    # Nothing changed: incremental mode stops before the story.
    again = _run_analyze(repo, out)
    assert again.exit_code == 0, again.output
    assert 'No new commits' in again.output
    assert stub_llm.calls == 1 + 3

    # --regenerate-story keeps the phase summaries (no phase calls) and rebuilds the
    # three model-written sections, so new directives reach the prompts.
    redo = _run_analyze(repo, out, '--regenerate-story', '--directives', 'Mention the moon.')
    assert redo.exit_code == 0, redo.output
    assert 'Regenerating the story' in redo.output
    assert stub_llm.calls == 1 + 3 + 3
    assert any('Mention the moon.' in p and 'Hard facts about' in p for p in stub_llm.prompts[-3:])


def test_story_cache_fingerprint_includes_facts(stub_llm):
    from gitview.storyteller import StoryTeller
    phases = _summary_phases(2)
    a = StoryTeller(backend='anthropic', api_key='x', facts='span A')._hash_phases(phases)
    b = StoryTeller(backend='anthropic', api_key='x', facts='span B')._hash_phases(phases)
    none = StoryTeller(backend='anthropic', api_key='x')._hash_phases(phases)
    assert a != b and a != none


def test_analyze_hierarchical_uses_evidence_for_clusters(tmp_path, stub_llm):
    repo = _repo_with_history(tmp_path)
    result = _run_analyze(repo, tmp_path / 'out', '--hierarchical', '--llm-budget', 'balanced')
    assert result.exit_code == 0, result.output
    # feature phase: 1 narrative call (clusters from evidence); routine phase: 0; timeline: 1
    assert stub_llm.calls == 1 + 1
