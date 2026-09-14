"""Tests for version descriptors, version events and phases (gitview.versioning)."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from gitview.cli import cli
from gitview.versioning import (
    BranchHistory,
    DescriptorError,
    build_timeline,
    detect,
    find_descriptor,
    load_timeline,
    parse_descriptor,
    render_draft,
    write_draft,
)
from gitview.versioning.sources import parse_file

from tests.test_graph import _git


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _commit(repo: Path, message: str, **files) -> str:
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        _git(repo, 'add', name)
    _git(repo, '-c', 'user.name=T', '-c', 'user.email=t@example.com', 'commit', '-q', '--allow-empty', '-m', message)
    return _head(repo)


def _head(repo: Path) -> str:
    import subprocess
    return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo, capture_output=True, text=True).stdout.strip()


def _tag(repo: Path, name: str, message: str = '', ref: str = 'HEAD') -> None:
    args = ['tag', '-a', name, '-m', message, ref] if message else ['tag', name, ref]
    _git(repo, '-c', 'user.name=T', '-c', 'user.email=t@example.com', *args)


def _version_py(major, minor, patch, epoch=None, comment=''):
    lines = ['"""Version contract.', '', 'MINOR - schema revision; every migration bumps it.', '"""']
    if epoch is not None:
        lines.append(f'EPOCH = {epoch}')
    lines += [f'MAJOR = {major}', f'MINOR = {minor}  # {comment}', f'PATCH = {patch}']
    return '\n'.join(lines) + '\n'


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / 'repo'
    path.mkdir()
    _git(path, 'init', '-q', '-b', 'main')
    return path


ASSIGN_SOURCE = {
    'kind': 'file', 'path': 'app/version.py', 'parse': 'python-assign',
    'label': '{EPOCH}.{MAJOR}.{MINOR}.{PATCH}',
    'fields': {'EPOCH': {'role': 'label'}, 'MAJOR': {'role': 'generation'},
               'MINOR': {'role': 'boundary', 'meaning': 'schema revision'}, 'PATCH': {'role': 'step'}},
}
SEMVER_FIELDS = {'major': {'role': 'generation'}, 'minor': {'role': 'boundary'}, 'patch': {'role': 'step'}}


def _timeline(repo, *sources, branch='HEAD'):
    return build_timeline(BranchHistory(repo, branch), parse_descriptor({'source': list(sources)}, origin='test'))


def _errors(tl):
    return [p.message for p in tl.errors]


# ---------------------------------------------------------------------------
# Descriptor lookup and validation
# ---------------------------------------------------------------------------

class TestDescriptor:
    def test_pyproject_without_table_falls_through_to_gitview_toml(self, repo):
        (repo / 'pyproject.toml').write_text('[project]\nname = "x"\n')
        (repo / '.gitview.toml').write_text('[versioning]\n[[versioning.source]]\nkind = "tag"\n'
                                            'fields.minor = { role = "boundary" }\n')
        d = find_descriptor(repo)
        assert d.origin == '.gitview.toml'
        assert d.sources[0].kind == 'tag' and d.scope == ''

    def test_pyproject_table_wins(self, repo):
        (repo / 'pyproject.toml').write_text(
            '[tool.gitview.versioning]\ndescription = "d"\n'
            '[[tool.gitview.versioning.source]]\nkind = "file"\npath = "v.py"\nparse = "python-assign"\n'
            'fields.MINOR = { role = "boundary", meaning = "schema" }\n')
        (repo / '.gitview.toml').write_text('[versioning]\n')
        d = find_descriptor(repo)
        assert (d.origin, d.description) == ('pyproject.toml', 'd')
        assert d.sources[0].field('MINOR').meaning == 'schema'

    def test_package_json_and_cargo(self, repo):
        (repo / 'package.json').write_text(json.dumps({'gitview': {'versioning': {
            'source': [{'kind': 'file', 'path': 'package.json', 'parse': 'json', 'fields': SEMVER_FIELDS}]}}}))
        d = find_descriptor(repo)
        assert d.origin == 'package.json' and d.sources[0].options['key'] == 'version'
        (repo / 'Cargo.toml').write_text('[package.metadata.gitview.versioning]\n'
                                         '[[package.metadata.gitview.versioning.source]]\nkind = "tag"\n'
                                         'fields.minor = { role = "boundary" }\n')
        assert find_descriptor(repo).origin == 'Cargo.toml'

    def test_scope_is_the_descriptor_directory(self, repo):
        sub = repo / 'services' / 'api'
        sub.mkdir(parents=True)
        (sub / '.gitview.toml').write_text('[versioning]\n[[versioning.source]]\nkind = "tag"\n'
                                           'fields.minor = { role = "boundary" }\n')
        assert find_descriptor(sub).scope == 'services/api'
        assert find_descriptor(repo) is None

    @pytest.mark.parametrize('source, message', [
        ({'kind': 'svn'}, 'kind must be one of'),
        ({'kind': 'file', 'parse': 'python-assign', 'fields': {'A': {'role': 'boundary'}}}, 'needs path'),
        ({'kind': 'file', 'path': 'v', 'parse': 'yaml', 'fields': {'A': {'role': 'boundary'}}}, 'parse must be'),
        ({'kind': 'tag', 'fields': {'minor': {'role': 'major'}}}, 'unknown role'),
        ({'kind': 'tag', 'pattern': r'v(\d+)', 'fields': {'minor': {'role': 'boundary'}}}, 'no named group'),
        ({'kind': 'tag', 'label': '{nope}', 'fields': {'minor': {'role': 'boundary'}}}, 'unknown fields'),
        ({'kind': 'file', 'path': 'v', 'parse': 'toml', 'fields': SEMVER_FIELDS}, 'needs key'),
        ({'kind': 'file', 'path': 'v', 'parse': 'plain', 'fields': {'build': {'role': 'step'}}}, 'not in components'),
        ({'kind': 'sequence', 'role': 'boundary'}, 'needs glob'),
    ])
    def test_invalid_sources(self, source, message):
        with pytest.raises(DescriptorError, match=message):
            parse_descriptor({'source': [source]}, origin='t')

    def test_schema_and_empty_sources(self):
        with pytest.raises(DescriptorError, match='unsupported descriptor schema'):
            parse_descriptor({'schema': 2, 'source': [{'kind': 'tag', 'fields': SEMVER_FIELDS}]}, origin='t')
        with pytest.raises(DescriptorError, match='at least one'):
            parse_descriptor({}, origin='t')


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _file_source(parse, **options):
    fields = options.pop('fields', SEMVER_FIELDS)
    return parse_descriptor({'source': [dict(kind='file', path='x', parse=parse, fields=fields, **options)]},
                            origin='t').sources[0]


@pytest.mark.parametrize('parse, content, options, expected', [
    ('python-dunder', '__version__: str = "1.4.2"\n', {}, {'major': 1, 'minor': 4, 'patch': 2}),
    ('plain', '2.3\n', {}, {'major': 2, 'minor': 3, 'patch': 0}),
    ('json', '{"version": "0.9.1"}', {}, {'major': 0, 'minor': 9, 'patch': 1}),
    ('toml', '[project]\nversion = "3.0.7"\n', {'key': 'project.version'}, {'major': 3, 'minor': 0, 'patch': 7}),
    ('regex', 'VERSION_TRIPLE(6, 30, 1)', {'pattern': r'\((?P<major>\d+), (?P<minor>\d+), (?P<patch>\d+)\)'},
     {'major': 6, 'minor': 30, 'patch': 1}),
    ('plain', 'release-6.30', {'pattern': r'(?P<major>\d+)\.(?P<minor>\d+)', 'fields': {
        'major': {'role': 'generation'}, 'minor': {'role': 'boundary'}}}, {'major': 6, 'minor': 30}),
])
def test_parsers(parse, content, options, expected):
    values, _ = parse_file(content, _file_source(parse, **options))
    assert values == expected


def test_python_dunder_rejects_non_numeric_fallback():
    with pytest.raises(ValueError):
        parse_file('__version__ = "unknown"\n', _file_source('python-dunder'))


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

class TestFileSource:
    def test_phases_steps_unversioned_and_open(self, repo):
        c1 = _commit(repo, 'init', **{'README.md': 'x'})
        c2 = _commit(repo, 'version 6.0', **{'app/version.py': _version_py(6, 0, 0)})
        _commit(repo, 'feature')
        c4 = _commit(repo, 'fix', **{'app/version.py': _version_py(6, 0, 1)})
        _commit(repo, 'comment only', **{'app/version.py': _version_py(6, 0, 1, comment='reworded')})
        c6 = _commit(repo, 'schema 6.1 (migration)', **{'app/version.py': _version_py(6, 1, 0, epoch=0)})
        c7 = _commit(repo, 'more')

        tl = _timeline(repo, ASSIGN_SOURCE)
        assert not tl.problems
        assert [e.level for e in tl.events] == ['generation', 'step', 'boundary']
        assert [(p.id, p.level, p.sealed, p.commits) for p in tl.phases] == [
            ('unversioned', 'unversioned', True, 1),
            ('6.0', 'generation', True, 4),
            ('6.1', 'boundary', False, 2),
        ]
        unversioned, first, last = tl.phases
        assert (unversioned.start, unversioned.end) == (c1, c1)
        assert (first.start, first.end) == (c2, _parent(repo, c6))
        assert [s.commit for s in first.steps] == [c4]
        assert (last.start, last.end) == (c6, c7)
        assert first.event.label == '?.6.0.0'           # EPOCH not yet in the file
        assert last.event.label == '0.6.1.0'
        assert last.event.source == 'file:app/version.py'

    def test_parse_error_is_reported_at_its_commit(self, repo):
        _commit(repo, 'v', **{'app/version.py': _version_py(1, 0, 0)})
        bad = _commit(repo, 'broken', **{'app/version.py': 'nothing here\n'})
        tl = _timeline(repo, ASSIGN_SOURCE)
        assert [(p.severity, p.commit) for p in tl.problems] == [('error', bad)]

    def test_backwards_version_is_an_error(self, repo):
        _commit(repo, 'a', **{'app/version.py': _version_py(6, 4, 1)})
        back = _commit(repo, 'b', **{'app/version.py': _version_py(6, 2, 0)})
        _commit(repo, 'c', **{'app/version.py': _version_py(7, 0, 0)})     # generation bump resets: fine
        tl = _timeline(repo, ASSIGN_SOURCE)
        assert [(p.commit, 'backwards' in p.message) for p in tl.errors] == [(back, True)]

    def test_merged_branch_version_lands_on_the_merge(self, repo):
        _commit(repo, 'v', **{'app/version.py': _version_py(1, 0, 0)})
        _git(repo, 'checkout', '-q', '-b', 'feature')
        _commit(repo, 'bump on branch', **{'app/version.py': _version_py(1, 1, 0)})
        _git(repo, 'checkout', '-q', 'main')
        _commit(repo, 'main work', **{'other.txt': 'x'})
        _git(repo, '-c', 'user.name=T', '-c', 'user.email=t@example.com', 'merge', '-q', '--no-ff', 'feature', '-m', 'merge')
        merge = _head(repo)
        tl = _timeline(repo, ASSIGN_SOURCE)
        assert tl.phases[-1].start == merge
        assert tl.phases[-1].commits == 2           # the merge and the branch commit it brings in

    def test_scoped_descriptor_reads_paths_below_its_directory(self, repo):
        _commit(repo, 'v', **{'svc/app/version.py': _version_py(2, 3, 0)})
        (repo / 'svc' / '.gitview.toml').write_text(
            '[versioning]\n[[versioning.source]]\nkind = "file"\npath = "app/version.py"\n'
            'parse = "python-assign"\nfields.MAJOR = { role = "generation" }\nfields.MINOR = { role = "boundary" }\n')
        tl = load_timeline(repo / 'svc')
        assert [p.id for p in tl.phases] == ['2.3']


def _parent(repo, sha):
    import subprocess
    return subprocess.run(['git', 'rev-parse', f'{sha}^'], cwd=repo, capture_output=True, text=True).stdout.strip()


class TestTagSource:
    def test_annotated_and_lightweight_tags(self, repo):
        _commit(repo, 'a')
        _tag(repo, 'v0.1.0', 'first release\n\nnotes')
        _commit(repo, 'b')
        _tag(repo, 'v0.1.1')
        c = _commit(repo, 'c')
        _tag(repo, 'v0.2.0', 'second')
        _tag(repo, 'nightly')                        # not version-shaped: ignored
        tl = _timeline(repo, {'kind': 'tag', 'fields': SEMVER_FIELDS})
        assert [(p.id, p.sealed) for p in tl.phases] == [('0.1', True), ('0.2', False)]
        assert tl.phases[0].event.message == 'first release\n\nnotes'
        assert [s.label for s in tl.phases[0].steps] == ['v0.1.1']
        assert tl.phases[1].start == c and tl.phases[1].event.label == 'v0.2.0'

    def test_tag_off_branch_and_on_merged_branch(self, repo):
        _commit(repo, 'a')
        _tag(repo, 'v1.0.0')
        _git(repo, 'checkout', '-q', '-b', 'side')
        _commit(repo, 'side work')
        _tag(repo, 'v1.1.0')
        _git(repo, 'checkout', '-q', '-b', 'abandoned')
        _commit(repo, 'never merged')
        _tag(repo, 'v9.0.0')
        _git(repo, 'checkout', '-q', 'main')
        _commit(repo, 'main work')
        _git(repo, '-c', 'user.name=T', '-c', 'user.email=t@example.com', 'merge', '-q', '--no-ff', 'side', '-m', 'merge side')
        merge = _head(repo)
        tl = _timeline(repo, {'kind': 'tag', 'fields': SEMVER_FIELDS}, branch='main')
        assert [p.id for p in tl.phases] == ['1.0', '1.1']
        assert tl.phases[1].start == merge
        assert [p.message for p in tl.warnings] == ['tag v9.0.0 is not on main; ignored']


class TestSequenceSource:
    def test_numbered_migrations(self, repo):
        _commit(repo, 'init')
        _commit(repo, 'm1', **{'migrations/001_init.sql': '', 'migrations/001_init_rollback.sql': ''})
        _commit(repo, 'm2', **{'migrations/002_a.sql': '', 'migrations/003_b.sql': '', 'migrations/notes.md': ''})
        _commit(repo, 'hotfix rollback only', **{'migrations/004_c_rollback.sql': ''})
        (repo / 'migrations' / 'history').mkdir()
        _git(repo, 'mv', 'migrations/001_init.sql', 'migrations/history/001_init.sql')
        _commit(repo, 'archive old migrations')
        last = _commit(repo, 'm5', **{'migrations/005_d.sql': ''})
        tl = _timeline(repo, {'kind': 'sequence', 'glob': 'migrations/**/[0-9][0-9][0-9]_*.sql',
                              'exclude': '*rollback*', 'role': 'boundary', 'meaning': 'one DDL revision'})
        assert not tl.problems
        assert [(p.id, p.commits) for p in tl.phases] == [('unversioned', 1), ('1', 1), ('3', 3), ('5', 1)]
        assert tl.phases[-1].start == last and tl.phases[-1].event.raw == '005_d.sql'


class TestMultipleSources:
    def test_file_covers_history_before_first_tag_and_tags_must_agree(self, repo):
        _commit(repo, 'a', **{'pkg/__init__.py': '__version__ = "0.1.0"\n'})
        _commit(repo, 'b', **{'pkg/__init__.py': '__version__ = "0.2.0"\n'})
        _tag(repo, 'v0.2.0')
        wrong = _commit(repo, 'c', **{'pkg/__init__.py': '__version__ = "0.3.0"\n'})
        _tag(repo, 'v0.4.0')
        tag = {'kind': 'tag', 'fields': SEMVER_FIELDS}
        file = {'kind': 'file', 'path': 'pkg/__init__.py', 'parse': 'python-dunder', 'fields': SEMVER_FIELDS}
        tl = _timeline(repo, tag, file)
        tag_source = f"tag:{tl.descriptor.sources[0].options['pattern']}"
        # Before the first tag the file decides; from then on the first-listed source (tags) does.
        assert [(p.id, p.event.source) for p in tl.phases] == [('0.1', 'file:pkg/__init__.py'),
                                                               ('0.2', tag_source), ('0.4', tag_source)]
        assert [(p.commit, 'says v0.4.0 but' in p.message) for p in tl.errors] == [(wrong, True)]


class TestRoles:
    def test_unresolved_roles_fail_but_preview_with_suggestions(self, repo):
        _commit(repo, 'a', **{'app/version.py': _version_py(1, 0, 0)})
        _commit(repo, 'b', **{'app/version.py': _version_py(1, 1, 0)})
        source = {'kind': 'file', 'path': 'app/version.py', 'parse': 'python-assign', 'fields': {
            'MAJOR': {'role': '?', 'suggested': 'generation'}, 'MINOR': {'role': '?', 'suggested': 'boundary'},
            'PATCH': {'role': '?'}}}
        tl = _timeline(repo, source)
        assert [p.id for p in tl.phases] == ['1.0', '1.1']
        assert _errors(tl) == ['field MAJOR has no role yet (suggested: generation)',
                               'field MINOR has no role yet (suggested: boundary)',
                               'field PATCH has no role yet']

    def test_nothing_seals_a_phase(self, repo):
        _commit(repo, 'a', **{'app/version.py': _version_py(1, 0, 0)})
        tl = _timeline(repo, {'kind': 'file', 'path': 'app/version.py', 'parse': 'python-assign',
                              'fields': {'PATCH': {'role': 'step'}}})
        assert tl.phases == []
        assert 'nothing seals a phase' in _errors(tl)[0]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class TestDetect:
    def test_version_file_with_docstring_and_ddl_evidence(self, repo):
        (repo / 'pyproject.toml').write_text('[tool.pytest.ini_options]\naddopts = "-q"\n')
        _git(repo, 'add', 'pyproject.toml')
        _commit(repo, 'v', **{'app/core/version.py': _version_py(6, 0, 0, epoch=0)})
        _commit(repo, 'fix', **{'app/core/version.py': _version_py(6, 0, 1, epoch=0)})
        _commit(repo, 'schema 6.1 (migration 12)', **{'app/core/version.py': _version_py(6, 1, 0, epoch=0)})

        detection = detect(repo)
        assert (detection.target, detection.table_prefix) == ('pyproject.toml', 'tool.gitview.versioning')
        draft = render_draft(detection)
        assert 'fields.MINOR = { role = "?", suggested = "boundary" }' in draft
        assert '# MINOR changed in 1 commit(s); 1 of their messages mention schema/migration/DDL' in draft
        assert '# Docstring: "MINOR - schema revision; every migration bumps it."' in draft

        path = write_draft(detection)
        text = path.read_text()
        assert text.startswith('[tool.pytest.ini_options]\naddopts = "-q"\n\n# --- GitView')
        tl = load_timeline(repo)
        assert len(tl.errors) == 4 and [p.id for p in tl.phases] == ['6.0', '6.1']
        path.write_text(text.replace('role = "?", suggested = "', 'role = "').replace('" }', '" }'))
        assert not load_timeline(repo).errors
        with pytest.raises(FileExistsError):
            write_draft(detect(repo))

    def test_tags_and_project_version(self, repo):
        (repo / 'pyproject.toml').write_text('[project]\nname = "x"\nversion = "0.2.0"\n')
        _git(repo, 'add', 'pyproject.toml')
        _commit(repo, 'a', **{'x/__init__.py': 'try:\n    __version__ = v()\nexcept E:\n    __version__ = "unknown"\n'})
        _tag(repo, 'v0.2.0')
        detection = detect(repo)
        kinds = [(c.source.kind, c.source.options.get('path'), c.active) for c in detection.candidates]
        assert ('file', 'pyproject.toml', True) in kinds
        assert ('tag', None, True) in kinds
        tag = next(c for c in detection.candidates if c.source.kind == 'tag')
        assert tag.source.options['pattern'] == r'v(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)'
        assert "pattern = 'v(?P<major>\\d+)" in render_draft(detection)

    def test_migrations_create_gitview_toml(self, repo):
        _commit(repo, 'm', **{'db/migrations/0001_a.sql': '', 'db/migrations/0001_a_rollback.sql': '',
                             'db/migrations/0002_b.sql': ''})
        detection = detect(repo)
        assert detection.target == '.gitview.toml'
        source = detection.candidates[0].source
        assert source.options == {'glob': 'db/migrations/**/[0-9][0-9][0-9][0-9]_*.sql', 'exclude': '*rollback*'}
        write_draft(detection)
        d = find_descriptor(repo)
        assert d.sources[0].fields[0].suggested == 'boundary'

    def test_nothing_to_detect(self, repo):
        _commit(repo, 'a', **{'README.md': 'x'})
        assert detect(repo).candidates == []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCli:
    def test_without_descriptor(self, repo):
        _commit(repo, 'a')
        result = CliRunner().invoke(cli, ['versions', '--repo', str(repo)])
        assert result.exit_code == 1
        assert '[tool.gitview.versioning]' in result.output

    def test_detect_write_check_list(self, repo):
        _commit(repo, 'a', **{'app/version.py': _version_py(1, 0, 0)})
        _commit(repo, 'b', **{'app/version.py': _version_py(1, 1, 0)})
        runner = CliRunner()
        printed = runner.invoke(cli, ['versions', 'detect', '--repo', str(repo)])
        assert printed.exit_code == 0 and '[[versioning.source]]' in printed.output
        assert not (repo / '.gitview.toml').exists()

        assert runner.invoke(cli, ['versions', 'detect', '--repo', str(repo), '--write']).exit_code == 0
        failed = runner.invoke(cli, ['versions', 'check', '--repo', str(repo)])
        assert failed.exit_code == 1 and 'has no role yet' in failed.output

        toml = repo / '.gitview.toml'
        toml.write_text(toml.read_text().replace('role = "?", suggested = "', 'role = "'))
        passed = runner.invoke(cli, ['versions', 'check', '--repo', str(repo), '--json'])
        assert passed.exit_code == 0
        assert json.loads(passed.output)['ok'] is True

        listed = runner.invoke(cli, ['versions', '--repo', str(repo), '--json'])
        payload = json.loads(listed.output)
        assert [(p['id'], p['sealed']) for p in payload['phases']] == [('1.0', True), ('1.1', False)]
        assert payload['phases'][1]['version']['fields'] == {'MAJOR': 1, 'MINOR': 1, 'PATCH': 0}

        again = runner.invoke(cli, ['versions', 'detect', '--repo', str(repo), '--write'])
        assert again.exit_code == 1 and 'already has a descriptor' in again.output
