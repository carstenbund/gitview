"""Draft a version descriptor from what a repository already shows.

Detection proposes *where* versions live and *how* to parse them. It never
decides what a component means: every role is written as ``"?"`` with a
suggestion, for the repository owner to confirm.
"""

import ast
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .descriptor import LOCATIONS, find_descriptor, git_root, read_table
from .models import UNRESOLVED, FieldSpec, SourceSpec
from .sources import BranchHistory, Observation, observe_source, parse_file

KNOWN_ASSIGN_NAMES = ('EPOCH', 'RELEASE', 'GENERATION', 'MAJOR', 'MINOR', 'REVISION', 'PATCH', 'MICRO', 'BUILD')
SUGGESTED_ROLES = {
    'epoch': 'label', 'release': 'label',
    'generation': 'generation', 'major': 'generation',
    'minor': 'boundary', 'revision': 'boundary',
    'patch': 'step', 'micro': 'step', 'build': 'step',
    'sequence': 'boundary',
}
COMPONENT_NAMES = {1: ['major'], 2: ['major', 'minor'], 3: ['major', 'minor', 'patch'],
                   4: ['epoch', 'major', 'minor', 'patch']}
DDL_WORDS = re.compile(r'\b(schema|migration|migrations|ddl)\b', re.I)
SKIP_DIRS = {'venv', '.venv', 'node_modules', 'site-packages', 'vendor', 'third_party', 'dist', 'build', '.tox'}
MIGRATION_FILE = re.compile(r'(?:^|/)(?P<dir>(?:[^/]+/)*?(?:migrations?|migrate|alembic/versions))/'
                            r'(?:[^/]+/)*(?P<num>\d{3,})[_-][^/]*\.(?P<ext>sql|py)$')


@dataclass
class Candidate:
    source: SourceSpec
    evidence: List[str] = field(default_factory=list)
    field_evidence: Dict[str, List[str]] = field(default_factory=dict)
    observations: int = 0
    active: bool = True


@dataclass
class Detection:
    project_dir: Path
    target: str                      # file a --write would append to or create
    table_prefix: str                # e.g. "tool.gitview.versioning"
    notes: List[str]
    candidates: List[Candidate]

    @property
    def active(self) -> List[Candidate]:
        return [c for c in self.candidates if c.active]


def detect(project_dir: Union[str, Path], branch: str = 'HEAD') -> Detection:
    project_dir = Path(project_dir).resolve()
    root = git_root(project_dir)
    scope = project_dir.relative_to(root).as_posix()
    scope = '' if scope == '.' else scope
    history = BranchHistory(root, branch)
    subjects = dict(line.split('\x00', 1) for line in
                    history.run('log', '--first-parent', '--format=%H%x00%s', branch).splitlines() if '\x00' in line)
    tracked = [p for p in history.run('-C', str(project_dir), 'ls-files').splitlines()
               if not (set(p.split('/')[:-1]) & SKIP_DIRS)]

    notes: List[str] = []
    candidates: List[Candidate] = []
    tags_are_versions = _project_file_evidence(project_dir, notes, candidates)
    candidates.extend(_tag_candidates(history, notes))
    candidates.extend(_version_file_candidates(history, scope, tracked, candidates))
    candidates.extend(_migration_candidates(tracked))

    kept = []
    for c in candidates:
        observations, problems = observe_source(history, c.source, scope)
        if c.source.kind == 'file' and not _parses_at_tip(history, c.source, scope):
            notes.append(f"{c.source.options['path']} was considered but does not parse at the branch tip")
            continue
        if not observations:
            continue
        c.observations = len(observations)
        c.evidence.append(f"{len(observations)} version change(s) on {branch}")
        if problems:
            c.evidence.append(f"{len(problems)} commit(s) could not be read with this source")
        _field_evidence(c, observations, history, subjects)
        kept.append(c)

    kept.sort(key=lambda c: (-c.observations, c.source.kind != 'tag'))
    _choose_active(kept, tags_are_versions)
    target, prefix = _target(project_dir)
    return Detection(project_dir, target, prefix, notes, kept)


# ---------------------------------------------------------------------------
# Evidence gatherers
# ---------------------------------------------------------------------------

def _fields(names: List[str]) -> List[FieldSpec]:
    return [FieldSpec(n, UNRESOLVED, suggested=SUGGESTED_ROLES.get(n.lower(), '')) for n in names]


def _project_file_evidence(project_dir: Path, notes: List[str], out: List[Candidate]) -> bool:
    tags_are_versions = False
    pyproject = project_dir / 'pyproject.toml'
    if pyproject.is_file():
        data = read_table(pyproject, ()) or {}
        tool = data.get('tool', {})
        requires = ' '.join(data.get('build-system', {}).get('requires', []))
        if 'setuptools_scm' in tool or 'setuptools_scm' in requires or 'setuptools-scm' in requires \
                or tool.get('hatch', {}).get('version', {}).get('source') == 'vcs' or 'hatch-vcs' in requires:
            tags_are_versions = True
            notes.append("pyproject.toml derives the version from git tags (setuptools-scm / hatch-vcs)")
        attr = tool.get('setuptools', {}).get('dynamic', {}).get('version', {})
        if isinstance(attr, dict) and isinstance(attr.get('attr'), str):
            module = attr['attr'].rsplit('.', 1)[0].replace('.', '/')
            for path in (f'{module}.py', f'{module}/__init__.py'):
                if (project_dir / path).is_file():
                    out.append(Candidate(SourceSpec('file', _fields(COMPONENT_NAMES[3]), {
                        'path': path, 'parse': 'python-dunder', 'components': COMPONENT_NAMES[3]}),
                        [f"pyproject.toml reads the version from {attr['attr']}"]))
                    break
        for keys, label in ((('project', 'version'), '[project] version'),
                            (('tool', 'poetry', 'version'), '[tool.poetry] version')):
            value = _dig(data, keys)
            if isinstance(value, str):
                out.append(_string_candidate('pyproject.toml', 'toml', '.'.join(keys), value,
                                             f"pyproject.toml declares {label} = \"{value}\""))
        bump = tool.get('bumpversion')
        if isinstance(bump, dict):
            files = [f.get('filename') for f in bump.get('files', []) if isinstance(f, dict)]
            notes.append(f"[tool.bumpversion] manages {', '.join(filter(None, files)) or 'the version'}"
                         + (f" with parse = {bump['parse']!r}" if bump.get('parse') else ''))
        cz = tool.get('commitizen')
        if isinstance(cz, dict):
            notes.append(f"[tool.commitizen] tag_format = {cz.get('tag_format', '$version')!r}, "
                         f"version_files = {cz.get('version_files', [])}")
    cargo = project_dir / 'Cargo.toml'
    if cargo.is_file():
        value = _dig(read_table(cargo, ()) or {}, ('package', 'version'))
        if isinstance(value, str):
            out.append(_string_candidate('Cargo.toml', 'toml', 'package.version', value,
                                         f"Cargo.toml declares version = \"{value}\""))
    package = project_dir / 'package.json'
    if package.is_file():
        try:
            value = json.loads(package.read_text(encoding='utf-8')).get('version')
        except (ValueError, OSError, AttributeError):
            value = None
        if isinstance(value, str):
            out.append(_string_candidate('package.json', 'json', 'version', value,
                                         f"package.json declares \"version\": \"{value}\""))
    return tags_are_versions


def _string_candidate(path: str, parse: str, key: str, value: str, evidence: str) -> Candidate:
    names = COMPONENT_NAMES.get(len(re.findall(r'\d+', value)), COMPONENT_NAMES[3])
    return Candidate(SourceSpec('file', _fields(names), {'path': path, 'parse': parse, 'key': key,
                                                         'components': names}), [evidence])


def _tag_candidates(history: BranchHistory, notes: List[str]) -> List[Candidate]:
    tags = history.run('tag', '--list').split()
    if not tags:
        return []
    shapes = Counter(re.sub(r'\d+', '#', t) for t in tags)
    shape, count = shapes.most_common(1)[0]
    parts = shape.split('#')
    names = COMPONENT_NAMES.get(len(parts) - 1)
    if not names:
        notes.append(f"{len(tags)} tag(s) found, none shaped like a version")
        return []
    pattern = ''.join(re.escape(p) + (f'(?P<{names[i]}>\\d+)' if i < len(names) else '')
                      for i, p in enumerate(parts))
    others = len(tags) - count
    evidence = [f"{count} tag(s) shaped like {shape.replace('#', 'N')}"
                + (f"; {others} other tag(s) ignored" if others else '')]
    return [Candidate(SourceSpec('tag', _fields(names), {'pattern': pattern}), evidence)]


def _version_file_candidates(history: BranchHistory, scope: str, tracked: List[str],
                             existing: List[Candidate]) -> List[Candidate]:
    seen = {c.source.options.get('path') for c in existing}
    out = []
    for path in tracked:
        base = path.rsplit('/', 1)[-1]
        if path in seen:
            continue
        if base in ('VERSION', 'VERSION.txt', 'version.txt'):
            content = _tip(history, scope, path) or ''
            names = COMPONENT_NAMES.get(len(re.findall(r'\d+', content.strip().split('\n')[0])))
            if names:
                out.append(Candidate(SourceSpec('file', _fields(names), {
                    'path': path, 'parse': 'plain', 'components': names}), [f"{path} holds a plain version"]))
            continue
        if not base.endswith('.py') or not ('version' in base or base in ('__init__.py', '__about__.py')):
            continue
        content = _tip(history, scope, path) or ''
        names = [m.group(1) for m in re.finditer(r'^([A-Z][A-Z_]*)\s*(?::[^=\n]+)?=\s*\d+\b', content, re.M)
                 if m.group(1) in KNOWN_ASSIGN_NAMES]
        names = list(dict.fromkeys(names))
        if len(names) >= 2:
            label = '.'.join('{%s}' % n for n in names)
            c = Candidate(SourceSpec('file', _fields(names), {'path': path, 'parse': 'python-assign', 'label': label}),
                          [f"{path} assigns {', '.join(names)}"])
            c.field_evidence = {n: _docstring_lines(content, n, names) for n in names}
            out.append(c)
        elif re.search(r'''^\s*__version__\s*=\s*['"]\d''', content, re.M):
            out.append(Candidate(SourceSpec('file', _fields(COMPONENT_NAMES[3]), {
                'path': path, 'parse': 'python-dunder', 'components': COMPONENT_NAMES[3]}),
                [f"{path} assigns __version__"]))
    return out


def _migration_candidates(tracked: List[str]) -> List[Candidate]:
    groups: Dict[Tuple[str, str], List[str]] = {}
    for path in tracked:
        m = MIGRATION_FILE.search(path)
        if m:
            groups.setdefault((m.group('dir'), m.group('ext')), []).append(path)
    out = []
    for (directory, ext), paths in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        width = Counter(len(MIGRATION_FILE.search(p).group('num')) for p in paths).most_common(1)[0][0]
        options: Dict[str, Any] = {'glob': f"{directory}/**/{'[0-9]' * width}_*.{ext}"}
        rollbacks = [p for p in paths if re.search(r'rollback|_down\b|\.down\.', p)]
        if rollbacks:
            options['exclude'] = '*rollback*' if any('rollback' in p for p in rollbacks) else '*down*'
        out.append(Candidate(SourceSpec('sequence', [FieldSpec('sequence', UNRESOLVED, suggested='boundary')], options),
                             [f"{len(paths)} numbered migration file(s) under {directory}/"]))
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dig(data: Any, keys: Tuple[str, ...]) -> Any:
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data


def _tip(history: BranchHistory, scope: str, path: str) -> Optional[str]:
    if not history.chain:
        return None
    return history.show(history.chain[-1], f'{scope}/{path}' if scope else path)


def _parses_at_tip(history: BranchHistory, source: SourceSpec, scope: str) -> bool:
    content = _tip(history, scope, source.options['path'])
    if content is None:
        return False
    try:
        parse_file(content, source)
        return True
    except Exception:
        return False


def _docstring_lines(content: str, name: str, names: List[str]) -> List[str]:
    """The docstring passage that explains ``name``, if the module has one."""
    try:
        doc = ast.get_docstring(ast.parse(content)) or ''
    except (SyntaxError, ValueError):
        return []
    lines = doc.splitlines()
    for i, line in enumerate(lines):
        mentioned = [n for n in names if re.search(rf'\b{n}\b', line)]
        if name not in mentioned or len(mentioned) > 2:     # skip lines that list every field
            continue
        passage = [line]
        for nxt in lines[i + 1:]:
            stripped = nxt.strip()
            if not stripped or stripped[0] in '*-' or any(re.match(rf'{n}\b', stripped) for n in names):
                break
            passage.append(stripped)
        text = re.sub(r'\s+', ' ', re.sub(r'[*`]', '', ' '.join(passage))).strip()
        if len(text) > 160:
            cut = text[:160]
            text = cut[:cut.rfind('. ') + 1] if '. ' in cut else cut.rsplit(' ', 1)[0] + ' …'
        return [f'Docstring: "{text}"']
    return []


def _field_evidence(c: Candidate, observations: List[Observation], history: BranchHistory,
                    subjects: Dict[str, str]) -> None:
    for f in c.source.fields:
        changed = [o for prev, o in zip(observations, observations[1:]) if prev.values.get(f.name) != o.values.get(f.name)]
        if not changed:
            continue
        ddl = sum(1 for o in changed if DDL_WORDS.search(subjects.get(history.chain[o.position], '')))
        line = f"{f.name} changed in {len(changed)} commit(s)"
        if ddl:
            line += f"; {ddl} of their messages mention schema/migration/DDL"
        c.field_evidence.setdefault(f.name, []).insert(0, line)


def _choose_active(candidates: List[Candidate], tags_are_versions: bool) -> None:
    """One primary source, listed first, plus tags as a cross-check; the rest stay commented out.

    A version file is primary unless the project derives its version from tags:
    it records every bump, while tags usually mark only some of them.
    """
    by_kind = {kind: next((c for c in candidates if c.source.kind == kind), None)
               for kind in ('file', 'tag', 'sequence')}
    order = ('tag', 'file', 'sequence') if tags_are_versions else ('file', 'tag', 'sequence')
    primary = next((by_kind[k] for k in order if by_kind[k] is not None), None)
    for c in candidates:
        c.active = c is primary or (c is by_kind['tag'] and primary is not None
                                    and primary.source.kind == 'file')
    candidates.sort(key=lambda c: (c is not primary, not c.active))


def _target(project_dir: Path) -> Tuple[str, str]:
    for name, keys, _ in LOCATIONS:
        if name.endswith('.toml') and (project_dir / name).is_file():
            return name, '.'.join(keys)
    return '.gitview.toml', 'versioning'


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _toml_str(value: str) -> str:
    return f"'{value}'" if "'" not in value and '\n' not in value and '\\' in value else json.dumps(value)


def _toml_value(value: Any) -> str:
    if isinstance(value, str):
        return _toml_str(value)
    if isinstance(value, list):
        return '[' + ', '.join(_toml_value(v) for v in value) + ']'
    return json.dumps(value)


def render_draft(detection: Detection) -> str:
    prefix = detection.table_prefix
    lines = [
        "# --- GitView version descriptor, drafted by `gitview versions detect` ---",
        "# Replace every role = \"?\" with generation, boundary, step or label, then run",
        "# `gitview versions check`. A boundary seals a phase (by convention: any DDL",
        "# change); a step is a code-only change inside a phase; label is display only.",
    ]
    lines += [f"# Note: {n}" for n in detection.notes]
    lines += [f"[{prefix}]", "schema = 1", 'description = ""']
    for c in detection.candidates:
        block = _render_source(c, prefix)
        lines.append("")
        if not c.active:
            lines.append("# Alternative source (uncomment to use):")
            block = [line if line.startswith('#') else f"# {line}" for line in block]
        lines += block
    return '\n'.join(lines) + '\n'


def _render_source(c: Candidate, prefix: str) -> List[str]:
    s = c.source
    lines = [f"# {e}" for e in c.evidence]
    lines.append(f"[[{prefix}.source]]")
    lines.append(f"kind = {_toml_str(s.kind)}")
    for key in ('path', 'parse', 'key', 'pattern', 'glob', 'exclude', 'components', 'label'):
        if key in s.options:
            lines.append(f"{key} = {_toml_value(s.options[key])}")
    if s.kind == 'sequence':
        f = s.fields[0]
        lines.append(f'role = "{f.role}"')
        if f.suggested:
            lines.append(f'suggested = "{f.suggested}"')
        return lines
    for f in s.fields:
        for e in c.field_evidence.get(f.name, []):
            lines.append(f"# {e}")
        suggestion = f', suggested = "{f.suggested}"' if f.suggested else ''
        lines.append(f'fields.{f.name} = {{ role = "{f.role}"{suggestion} }}')
    return lines


def write_draft(detection: Detection) -> Path:
    """Append the draft to the project file (or create ``.gitview.toml``). Refuses to overwrite."""
    if find_descriptor(detection.project_dir) is not None:
        raise FileExistsError(f"{detection.project_dir} already has a version descriptor")
    path = detection.project_dir / detection.target
    text = render_draft(detection)
    if path.is_file():
        existing = path.read_text(encoding='utf-8')
        separator = '' if existing.endswith('\n\n') else ('\n' if existing.endswith('\n') else '\n\n')
        path.write_text(existing + separator + text, encoding='utf-8')
    else:
        path.write_text(text, encoding='utf-8')
    return path
