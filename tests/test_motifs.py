"""Tests for motif detection (gitview.motifs)."""

import hashlib
import json

from click.testing import CliRunner

from gitview.cli import cli
from gitview.motifs import Evidence, Thresholds, motif_catalog, run_motifs
from gitview.motifs.models import MotifContext
from gitview.structural import StructuralEdge, StructuralNode, StructuralSnapshot

from tests.test_graph import _build, _commit, synthetic_repo  # noqa: F401


def _snap(sha, edges, paths, content=None):
    content = content or f"{sha}:{edges}"
    return StructuralSnapshot(
        provider='test', provider_version='1', observed_sha=sha, observed_at='2026-01-01T00:00:00',
        source='memory', content_hash=hashlib.sha256(content.encode()).hexdigest(),
        nodes=[StructuralNode(p) for p in paths], edges=edges,
    )


def _sha(n):
    return f"{n:040x}"


LOOSE = Thresholds(min_cochanges=2, min_jaccard=0.1, min_touches=2, min_degree_growth=2)


# ---------------------------------------------------------------------------
# Evidence gating
# ---------------------------------------------------------------------------

def test_every_motif_declares_historical_evidence():
    for m in motif_catalog():
        assert Evidence.HISTORICAL in m.requires, m.id


def test_structural_motifs_are_skipped_without_observation():
    store = _build([_commit(1, 'a.py', 'b.py'), _commit(2, 'a.py', 'b.py'), _commit(3, 'a.py', 'b.py')])
    report = run_motifs(store, thresholds=LOOSE)

    assert report.available == [Evidence.HISTORICAL]
    assert {f.motif for f in report.findings} == {'repeated_cochange'}
    assert set(report.skipped) == {'hidden_coupling', 'confirmed_coupling', 'stable_interface',
                                   'emerging_dependency', 'architectural_split', 'centrality_growth'}
    assert 'gitview observe' in report.skipped['hidden_coupling']


def test_series_motifs_need_two_distinct_commits():
    store = _build([_commit(1, 'a.py', 'b.py'), _commit(2, 'a.py', 'b.py')])
    store.insert_structural_snapshot(_snap(_sha(2), [], ['a.py', 'b.py'], content='one'))
    store.insert_structural_snapshot(_snap(_sha(2), [], ['a.py', 'b.py'], content='two'))
    report = run_motifs(store, thresholds=LOOSE)

    assert Evidence.STRUCTURAL in report.available
    assert Evidence.STRUCTURAL_SERIES not in report.available
    assert 'emerging_dependency' in report.skipped
    assert 'hidden_coupling' not in report.skipped


def test_only_filter_runs_selected_motifs():
    store = _build([_commit(1, 'a.py', 'b.py'), _commit(2, 'a.py', 'b.py')])
    report = run_motifs(store, only=['ownership_transition'], thresholds=LOOSE)
    assert report.findings == []
    assert report.skipped == {}


# ---------------------------------------------------------------------------
# Historical motifs
# ---------------------------------------------------------------------------

def test_repeated_cochange():
    store = _build([_commit(1, 'a.py', 'b.py'), _commit(2, 'a.py', 'b.py'),
                    _commit(3, 'a.py', 'b.py'), _commit(4, 'c.py')])
    [f] = run_motifs(store, only=['repeated_cochange'], thresholds=LOOSE).findings
    assert f.files == ['a.py', 'b.py']
    assert f.evidence['cochanges'] == 3
    assert f.evidence['jaccard'] == 1.0
    assert f.level == 'high'


def test_ownership_transition():
    records = [_commit(n, 'core.py', author='Ada') for n in range(1, 5)]
    records += [_commit(n, 'core.py', author='Bob') for n in range(5, 9)]
    records += [_commit(9, 'other.py', author='Ada')]
    store = _build(records)

    [f] = run_motifs(store, only=['ownership_transition'], thresholds=LOOSE).findings
    assert f.files == ['core.py']
    assert (f.evidence['from_author'], f.evidence['to_author']) == ('Ada', 'Bob')
    assert f.evidence['handover_commit'] == _sha(5)


def test_no_ownership_transition_for_stable_owner():
    store = _build([_commit(n, 'core.py', author='Ada') for n in range(1, 9)])
    assert run_motifs(store, only=['ownership_transition'], thresholds=LOOSE).findings == []


# ---------------------------------------------------------------------------
# Structural + historical (single observation)
# ---------------------------------------------------------------------------

def _coupled_store():
    # a↔b co-change 3×, a↔c co-change 3×, d and e change often but never together.
    return _build([
        _commit(1, 'a.py', 'b.py', 'c.py'), _commit(2, 'a.py', 'b.py', 'c.py'),
        _commit(3, 'a.py', 'b.py', 'c.py'),
        _commit(4, 'd.py'), _commit(5, 'e.py'), _commit(6, 'd.py'), _commit(7, 'e.py'),
    ])


def test_hidden_vs_confirmed_coupling():
    store = _coupled_store()
    store.insert_structural_snapshot(_snap(
        _sha(7), [StructuralEdge('a.py', 'c.py', 'imports')], ['a.py', 'b.py', 'c.py', 'd.py', 'e.py']))
    report = run_motifs(store, thresholds=LOOSE)
    by = report.by_motif()

    hidden = {tuple(f.files) for f in by['hidden_coupling']}
    confirmed = {tuple(f.files) for f in by['confirmed_coupling']}
    assert hidden == {('a.py', 'b.py'), ('b.py', 'c.py')}
    assert confirmed == {('a.py', 'c.py')}
    assert by['confirmed_coupling'][0].evidence['structural_edges'][0]['relation'] == 'imports'


def test_edge_index_is_not_shared_between_snapshots():
    """A freed snapshot's address must not hand its edges to the next one.

    The edge index used to live in a module-level dict keyed by id(snapshot);
    CPython reuses addresses, so a later snapshot could inherit the edges of a
    collected one and hidden coupling would silently disappear.
    """
    import gc

    store = _coupled_store()
    with_edge = _snap(_sha(7), [StructuralEdge('a.py', 'c.py', 'imports')],
                      ['a.py', 'b.py', 'c.py'], content='with-edge')
    sid, _ = store.insert_structural_snapshot(with_edge)
    ctx_first = MotifContext(store, LOOSE, store.structural_observations())
    assert ctx_first.edges_between(ctx_first.latest, 'a.py', 'c.py')

    del ctx_first, with_edge
    gc.collect()

    store.delete_structural_snapshot(sid)
    store.insert_structural_snapshot(_snap(_sha(7), [], ['a.py', 'b.py', 'c.py'], content='no-edge'))
    ctx_second = MotifContext(store, LOOSE, store.structural_observations())
    assert ctx_second.edges_between(ctx_second.latest, 'a.py', 'c.py') == []


def test_hidden_coupling_needs_both_files_observed():
    store = _coupled_store()
    # The analyser never saw b.py: silence about a↔b is not evidence of hidden coupling.
    store.insert_structural_snapshot(_snap(_sha(7), [], ['a.py', 'c.py']))
    hidden = run_motifs(store, only=['hidden_coupling'], thresholds=LOOSE).findings
    assert {tuple(f.files) for f in hidden} == {('a.py', 'c.py')}


def test_stable_interface():
    store = _coupled_store()
    store.insert_structural_snapshot(_snap(
        _sha(7),
        [StructuralEdge('d.py', 'e.py', 'imports', 3.0, 3), StructuralEdge('a.py', 'b.py', 'calls', 5.0, 5)],
        ['a.py', 'b.py', 'd.py', 'e.py']))
    findings = run_motifs(store, only=['stable_interface'], thresholds=LOOSE).findings
    # a↔b co-change 3× → not stable; d→e strong dependency with 0 co-changes → stable.
    assert [f.files for f in findings] == [['d.py', 'e.py']]
    assert findings[0].evidence['structural_strength'] == 3


# ---------------------------------------------------------------------------
# Structural + historical (series)
# ---------------------------------------------------------------------------

def _series_store():
    # Observation 1 at commit 2, observation 2 at commit 6.
    # a↔b start co-changing at commit 3, dependency a→b appears by commit 6 (emerging).
    # c→d dependency exists at commit 2, gone by 6, and c/d never co-change after 2 (split).
    # hub.py gains structural neighbours (centrality growth).
    store = _build([
        _commit(1, 'a.py', 'b.py', 'c.py', 'd.py', 'hub.py', 'x.py', 'y.py', 'z.py'),
        _commit(2, 'c.py', 'd.py'),
        _commit(3, 'a.py', 'b.py'),
        _commit(4, 'a.py', 'b.py'),
        _commit(5, 'hub.py', 'x.py'),
        _commit(6, 'a.py', 'b.py', 'hub.py'),
    ])
    paths = ['a.py', 'b.py', 'c.py', 'd.py', 'hub.py', 'x.py', 'y.py', 'z.py']
    store.insert_structural_snapshot(_snap(_sha(2), [
        StructuralEdge('c.py', 'd.py', 'imports'),
        StructuralEdge('hub.py', 'x.py', 'imports'),
    ], paths))
    store.insert_structural_snapshot(_snap(_sha(6), [
        StructuralEdge('a.py', 'b.py', 'calls', 2.0, 2),
        StructuralEdge('hub.py', 'x.py', 'imports'),
        StructuralEdge('hub.py', 'y.py', 'imports'),
        StructuralEdge('hub.py', 'z.py', 'imports'),
        StructuralEdge('a.py', 'hub.py', 'calls'),
    ], paths))
    return store


def test_emerging_dependency_reports_cochange_lead():
    store = _series_store()
    findings = run_motifs(store, only=['emerging_dependency'], thresholds=LOOSE).findings
    # Other pairs (a→hub, hub→y, hub→z) also emerged after a single shared commit;
    # a↔b, with three co-changes, must rank first.
    assert findings[0].files == ['a.py', 'b.py']
    assert {tuple(f.files) for f in findings} >= {('a.py', 'hub.py'), ('hub.py', 'y.py'), ('hub.py', 'z.py')}
    f = findings[0]
    assert f.evidence['first_cochange'] == _sha(1)
    assert f.evidence['lead_commits'] == 5          # first co-change at seq 1, observed at seq 6
    assert (f.evidence['observed_from'], f.evidence['observed_to']) == (_sha(2), _sha(6))


def test_architectural_split_outcome():
    store = _series_store()
    [f] = run_motifs(store, only=['architectural_split'], thresholds=LOOSE).findings
    assert f.files == ['c.py', 'd.py']
    assert f.evidence['outcome'] == 'decoupled'
    assert (f.evidence['cochanges_before'], f.evidence['cochanges_after']) == (2, 0)


def test_centrality_growth():
    store = _series_store()
    [f] = run_motifs(store, only=['centrality_growth'], thresholds=LOOSE).findings
    assert f.files == ['hub.py']
    assert (f.evidence['degree_from'], f.evidence['degree_to']) == (1, 4)
    assert f.evidence['historical_neighbours'] >= 1


def test_series_uses_single_provider():
    store = _series_store()
    other = _snap(_sha(1), [], ['a.py'], content='other-provider')
    other.provider = 'other'
    store.insert_structural_snapshot(other)
    report = run_motifs(store, thresholds=LOOSE)
    assert {o.provider for o in report.observations} == {'test'}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_motifs_cli_json_and_list(synthetic_repo):
    repo_path = synthetic_repo
    runner = CliRunner()
    result = runner.invoke(cli, ['motifs', '--repo', str(repo_path), '--json'])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload['available_evidence'] == ['historical']
    assert 'hidden_coupling' in payload['skipped']

    listing = runner.invoke(cli, ['motifs', '--list'])
    assert listing.exit_code == 0
    assert 'emerging_dependency' in listing.output

    bad = runner.invoke(cli, ['motifs', '--repo', str(repo_path), '--top', '0'])
    assert bad.exit_code != 0
