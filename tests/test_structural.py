"""Tests for optional structural evidence (gitview.structural)."""

import hashlib
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from gitview.cli import cli
from gitview.graph import GraphStore
from gitview.structural import (
    StructuralEdge,
    StructuralNode,
    StructuralProviderError,
    StructuralSnapshot,
    get_provider,
    provider_names,
)
from gitview.structural.graphify import GraphifyProvider, translate
from gitview.structural.models import fold_symbol_edges

from tests.test_graph import _build, _commit, synthetic_repo  # noqa: F401


# ---------------------------------------------------------------------------
# Fixtures: a small Graphify-shaped graph.json
# ---------------------------------------------------------------------------

def _node(nid, source_file, label=None, community=0, community_name='Core', file_type='code'):
    return {'id': nid, 'label': label or nid, 'source_file': source_file, 'community': community,
            'community_name': community_name, 'file_type': file_type, '_origin': 'ast'}


def _link(src, dst, relation='calls', weight=1.0):
    return {'source': src, 'target': dst, 'relation': relation, 'weight': weight}


GRAPHIFY_JSON = {
    'directed': False,
    'built_at_commit': 'a' * 40,
    'nodes': [
        _node('parser_parse', 'parser.py'),
        _node('parser_file', 'parser.py', label='parser.py'),
        _node('models_model', 'models.py'),
        _node('models_other', 'models.py', community=1, community_name='Data'),
        _node('models_third', 'models.py', community=1, community_name='Data'),
        _node('ser_dump', 'serializer.py'),
        _node('doc_heading', 'docs/guide.md', file_type='document'),
    ],
    'links': [
        _link('parser_parse', 'models_model', 'imports'),
        _link('parser_parse', 'models_other', 'calls', 0.5),
        _link('parser_parse', 'models_third', 'calls', 0.5),
        _link('parser_file', 'parser_parse', 'contains'),          # intra-file: dropped
        _link('ser_dump', 'models_model', 'references'),
        _link('doc_heading', 'parser_parse', 'references'),
        _link('ser_dump', 'ghost_node', 'calls'),                   # dangling: dropped
    ],
}


def _write_graph(tmp_path: Path, data=GRAPHIFY_JSON) -> Path:
    out = tmp_path / 'graphify-out'
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'graph.json'
    path.write_text(json.dumps(data))
    return path


def _snapshot(sha='a' * 40, edges=None, nodes=None, content='x') -> StructuralSnapshot:
    return StructuralSnapshot(
        provider='test', provider_version='0', observed_sha=sha, observed_at='2026-01-01T00:00:00',
        source='memory', content_hash=hashlib.sha256(content.encode()).hexdigest(),
        nodes=nodes if nodes is not None else [StructuralNode('parser.py'), StructuralNode('models.py')],
        edges=edges if edges is not None else [StructuralEdge('parser.py', 'models.py', 'imports')],
    )


# ---------------------------------------------------------------------------
# Neutral model
# ---------------------------------------------------------------------------

def test_fold_symbol_edges_aggregates_and_drops_self_edges():
    edges = fold_symbol_edges([
        ('a.py', 'b.py', 'calls', 1.0),
        ('a.py', 'b.py', 'calls', 0.5),
        ('a.py', 'b.py', 'imports', 1.0),
        ('a.py', 'a.py', 'calls', 1.0),
        ('', 'b.py', 'calls', 1.0),
    ])
    assert edges == [
        StructuralEdge('a.py', 'b.py', 'calls', 1.5, 2),
        StructuralEdge('a.py', 'b.py', 'imports', 1.0, 1),
    ]


def test_snapshot_degree_is_undirected_distinct_neighbours():
    snap = _snapshot(edges=[
        StructuralEdge('a.py', 'b.py', 'calls'),
        StructuralEdge('a.py', 'b.py', 'imports'),
        StructuralEdge('c.py', 'a.py', 'calls'),
    ])
    assert snap.degree() == {'a.py': 2, 'b.py': 1, 'c.py': 1}


# ---------------------------------------------------------------------------
# Graphify adapter
# ---------------------------------------------------------------------------

def test_translate_folds_symbols_into_file_nodes_and_edges():
    snap = translate(GRAPHIFY_JSON, sha='b' * 40, source='s', content_hash='h',
                     provider_version='9.9', observed_at='t')

    assert snap.provider == 'graphify'
    assert snap.provider_version == '9.9'
    # Graphify's own commit wins over the caller's guess.
    assert snap.observed_sha == 'a' * 40

    by_path = {n.path: n for n in snap.nodes}
    assert set(by_path) == {'parser.py', 'models.py', 'serializer.py', 'docs/guide.md'}
    assert by_path['parser.py'].symbols == 2
    assert by_path['models.py'].community == '1'            # majority community of its symbols
    assert by_path['models.py'].community_name == 'Data'
    assert by_path['docs/guide.md'].kind == 'document'

    assert snap.edges == [
        StructuralEdge('docs/guide.md', 'parser.py', 'references', 1.0, 1),
        StructuralEdge('parser.py', 'models.py', 'calls', 1.0, 2),
        StructuralEdge('parser.py', 'models.py', 'imports', 1.0, 1),
        StructuralEdge('serializer.py', 'models.py', 'references', 1.0, 1),
    ]


def test_translate_falls_back_to_caller_sha_without_built_at_commit():
    data = {k: v for k, v in GRAPHIFY_JSON.items() if k != 'built_at_commit'}
    snap = translate(data, sha='b' * 40, source='s', content_hash='h', provider_version='v', observed_at='t')
    assert snap.observed_sha == 'b' * 40


def test_provider_reads_existing_output_without_running_graphify(tmp_path):
    graph_file = _write_graph(tmp_path)
    provider = GraphifyProvider(executable='definitely-not-installed-graphify')
    assert provider.available()[0] is False

    snap = provider.snapshot(tmp_path, 'c' * 40)
    assert snap.source == str(graph_file)
    assert snap.content_hash == hashlib.sha256(graph_file.read_bytes()).hexdigest()
    assert len(snap.nodes) == 4


def test_provider_errors_when_tool_missing_and_no_output(tmp_path):
    provider = GraphifyProvider(executable='definitely-not-installed-graphify')
    with pytest.raises(StructuralProviderError, match='not on PATH'):
        provider.snapshot(tmp_path, 'c' * 40)


def test_provider_errors_on_missing_explicit_source(tmp_path):
    provider = GraphifyProvider(executable='definitely-not-installed-graphify')
    with pytest.raises(StructuralProviderError, match='not found'):
        provider.snapshot(tmp_path, 'c' * 40, source=tmp_path / 'nope.json')


def test_registry_knows_graphify_and_rejects_unknown():
    assert 'graphify' in provider_names()
    assert isinstance(get_provider('graphify'), GraphifyProvider)
    with pytest.raises(StructuralProviderError, match='unknown structural provider'):
        get_provider('codeql')


# ---------------------------------------------------------------------------
# Store round-trip and provenance
# ---------------------------------------------------------------------------

def test_store_roundtrip_and_dedupe_by_provenance():
    store = _build([_commit(1, 'parser.py', 'models.py')])
    snap = _snapshot(sha=f"{1:040x}")

    sid, inserted = store.insert_structural_snapshot(snap)
    assert inserted
    assert store.insert_structural_snapshot(snap) == (sid, False)          # same provenance → no-op
    _, inserted_again = store.insert_structural_snapshot(_snapshot(sha=f"{1:040x}", content='y'))
    assert inserted_again                                                   # new content hash → new row

    loaded = store.load_structural_snapshot(sid)
    assert loaded.nodes == sorted(snap.nodes, key=lambda n: n.path)
    assert loaded.edges == snap.edges
    assert loaded.observed_sha == snap.observed_sha
    assert store.counts()['structural_snapshots'] == 2


def test_observations_are_ordered_by_history_position():
    store = _build([_commit(1, 'a.py'), _commit(2, 'a.py'), _commit(3, 'a.py')])
    store.insert_structural_snapshot(_snapshot(sha=f"{3:040x}", content='late'))
    store.insert_structural_snapshot(_snapshot(sha=f"{1:040x}", content='early'))
    store.insert_structural_snapshot(_snapshot(sha='f' * 40, content='unknown'))

    obs = store.structural_observations()
    assert [o.sequence for o in obs] == [1, 3, None]
    assert store.latest_structural_observation().observed_sha == 'f' * 40


def test_reset_drops_structural_tables():
    store = _build([_commit(1, 'a.py')])
    store.insert_structural_snapshot(_snapshot())
    store.reset()
    assert store.counts()['structural_snapshots'] == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_observe_cli_stores_and_reports_drift(synthetic_repo):
    repo_path = synthetic_repo
    _write_graph(repo_path)                      # built_at_commit 'aaaa…' ≠ HEAD
    runner = CliRunner()
    result = runner.invoke(cli, ['observe', '--repo', str(repo_path), '--structural', 'graphify', '--json'])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload['inserted'] is True
    assert payload['drift'] is True
    assert payload['observation']['provider'] == 'graphify'
    assert payload['observation']['node_count'] == 4

    again = runner.invoke(cli, ['observe', '--repo', str(repo_path), '--structural', 'graphify'])
    assert again.exit_code == 0
    assert 'Already stored' in again.output


def test_observe_cli_unknown_provider(synthetic_repo):
    repo_path = synthetic_repo
    result = CliRunner().invoke(cli, ['observe', '--repo', str(repo_path), '--structural', 'nope'])
    assert result.exit_code == 1
    assert 'unknown structural provider' in result.output


def test_graph_cli_structural_option(synthetic_repo):
    repo_path = synthetic_repo
    _write_graph(repo_path)
    result = CliRunner().invoke(cli, ['graph', '--repo', str(repo_path), '--structural', 'graphify', '--json'])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload['stats']['structural_snapshots'] == 1
    assert payload['structural']['observation']['edge_count'] == 4


def test_graph_cli_without_structural_is_unchanged(synthetic_repo):
    repo_path = synthetic_repo
    result = CliRunner().invoke(cli, ['graph', '--repo', str(repo_path), '--json'])
    assert result.exit_code == 0, result.output
    assert 'structural' not in json.loads(result.output)
