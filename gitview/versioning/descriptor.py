"""Find and load a repository's version descriptor.

Lookup order in the project directory, first match wins:

1. ``pyproject.toml``  → ``[tool.gitview.versioning]``
2. ``Cargo.toml``      → ``[package.metadata.gitview.versioning]``
3. ``package.json``    → ``"gitview": {"versioning": ...}``
4. ``.gitview.toml``   → ``[versioning]``

A descriptor applies to the directory of the file that holds it; paths inside
it are relative to that directory.
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:  # Python 3.11+
    import tomllib
except ImportError:  # pragma: no cover - exercised on 3.8-3.10 CI
    import tomli as tomllib

from .models import (
    DESCRIPTOR_SCHEMA,
    FILE_PARSERS,
    ROLES,
    SOURCE_KINDS,
    UNRESOLVED,
    Descriptor,
    DescriptorError,
    FieldSpec,
    SourceSpec,
)

DEFAULT_COMPONENTS = ('major', 'minor', 'patch')
DEFAULT_TAG_PATTERN = r'v?(?P<major>\d+)\.(?P<minor>\d+)(?:\.(?P<patch>\d+))?'
DEFAULT_SEQUENCE_NUMBER = r'^(\d+)'

# (file name, path of keys to the versioning table, human-readable location)
LOCATIONS: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    ('pyproject.toml', ('tool', 'gitview', 'versioning'), '[tool.gitview.versioning]'),
    ('Cargo.toml', ('package', 'metadata', 'gitview', 'versioning'), '[package.metadata.gitview.versioning]'),
    ('package.json', ('gitview', 'versioning'), '"gitview": {"versioning": ...}'),
    ('.gitview.toml', ('versioning',), '[versioning]'),
)


def git_root(path: Union[str, Path]) -> Path:
    out = subprocess.run(['git', '-C', str(path), 'rev-parse', '--show-toplevel'],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise DescriptorError(f"{path} is not inside a git repository")
    return Path(out.stdout.strip()).resolve()


def read_table(file_path: Path, keys: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
    """Return the nested table at ``keys`` in a TOML/JSON file, or None if absent."""
    try:
        text = file_path.read_text(encoding='utf-8')
    except OSError:
        return None
    try:
        data = json.loads(text) if file_path.suffix == '.json' else tomllib.loads(text)
    except (ValueError, tomllib.TOMLDecodeError) as exc:
        raise DescriptorError(f"{file_path.name}: cannot parse ({exc})") from exc
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    if not isinstance(data, dict):
        raise DescriptorError(f"{file_path.name}: {'.'.join(keys)} must be a table")
    return data


def find_descriptor(project_dir: Union[str, Path]) -> Optional[Descriptor]:
    """Load the first descriptor found in ``project_dir``; None when there is none."""
    project_dir = Path(project_dir).resolve()
    root = git_root(project_dir)
    scope = project_dir.relative_to(root).as_posix()
    scope = '' if scope == '.' else scope
    for name, keys, _ in LOCATIONS:
        path = project_dir / name
        if not path.is_file():
            continue
        table = read_table(path, keys)
        if table is not None:
            return parse_descriptor(table, origin=name, scope=scope)
    return None


# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------

def parse_descriptor(table: Dict[str, Any], *, origin: str, scope: str = '') -> Descriptor:
    schema = table.get('schema', DESCRIPTOR_SCHEMA)
    if schema != DESCRIPTOR_SCHEMA:
        raise DescriptorError(f"{origin}: unsupported descriptor schema {schema!r} (expected {DESCRIPTOR_SCHEMA})")
    raw_sources = table.get('source')
    if not isinstance(raw_sources, list) or not raw_sources:
        raise DescriptorError(f"{origin}: at least one [[...source]] entry is required")
    sources = [_parse_source(raw, f"{origin}: source {i + 1}") for i, raw in enumerate(raw_sources)]
    return Descriptor(origin=origin, scope=scope, sources=sources, schema=schema,
                      description=str(table.get('description', '')))


def _parse_field(name: str, spec: Any, where: str) -> FieldSpec:
    if not isinstance(spec, dict):
        raise DescriptorError(f"{where}: field {name} must be a table like {{ role = \"boundary\" }}")
    role = spec.get('role', UNRESOLVED)
    if role not in ROLES and role != UNRESOLVED:
        raise DescriptorError(f"{where}: field {name} has unknown role {role!r} "
                              f"(expected one of {', '.join(ROLES)})")
    return FieldSpec(name=name, role=role, meaning=str(spec.get('meaning', '')),
                     suggested=str(spec.get('suggested', '')))


def _compile(pattern: str, where: str) -> 're.Pattern':
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise DescriptorError(f"{where}: invalid pattern {pattern!r} ({exc})") from exc


def _parse_source(raw: Any, where: str) -> SourceSpec:
    if not isinstance(raw, dict):
        raise DescriptorError(f"{where}: must be a table")
    kind = raw.get('kind')
    if kind not in SOURCE_KINDS:
        raise DescriptorError(f"{where}: kind must be one of {', '.join(SOURCE_KINDS)} (got {kind!r})")
    options = {k: v for k, v in raw.items() if k not in ('kind', 'fields', 'role', 'meaning', 'suggested')}

    if kind == 'sequence':
        if not options.get('glob'):
            raise DescriptorError(f"{where}: sequence source needs glob")
        name = str(options.get('field', 'sequence'))
        options.setdefault('number', DEFAULT_SEQUENCE_NUMBER)
        _compile(options['number'], where)
        return SourceSpec(kind, [_parse_field(name, raw, where)], options)

    fields_raw = raw.get('fields')
    if not isinstance(fields_raw, dict) or not fields_raw:
        raise DescriptorError(f"{where}: {kind} source needs fields, e.g. fields.minor = {{ role = \"boundary\" }}")
    fields = [_parse_field(str(n), s, where) for n, s in fields_raw.items()]
    names = [f.name for f in fields]

    if kind == 'tag':
        options.setdefault('pattern', DEFAULT_TAG_PATTERN)
        _require_groups(_compile(options['pattern'], where), names, where)
    else:
        if not options.get('path'):
            raise DescriptorError(f"{where}: file source needs path")
        parse = options.get('parse')
        if parse not in FILE_PARSERS:
            raise DescriptorError(f"{where}: parse must be one of {', '.join(FILE_PARSERS)} (got {parse!r})")
        if parse == 'regex':
            if not options.get('pattern'):
                raise DescriptorError(f"{where}: parse = \"regex\" needs pattern")
            _require_groups(_compile(options['pattern'], where), names, where)
        elif parse in ('json', 'toml') and not options.get('key'):
            if parse == 'json':
                options['key'] = 'version'
            else:
                raise DescriptorError(f"{where}: parse = \"toml\" needs key, e.g. key = \"project.version\"")
        if parse in ('python-dunder', 'plain', 'json', 'toml'):
            if 'pattern' in options:
                _require_groups(_compile(options['pattern'], where), names, where)
            else:
                components = options.setdefault('components', list(DEFAULT_COMPONENTS))
                missing = [n for n in names if n not in components]
                if missing:
                    raise DescriptorError(f"{where}: fields {', '.join(missing)} are not in components "
                                          f"{components}; set components = [...] or a pattern")

    label = options.get('label')
    if label:
        unknown = [n for n in re.findall(r'\{(\w+)\}', label) if n not in names]
        if unknown:
            raise DescriptorError(f"{where}: label uses unknown fields {', '.join(unknown)}")
    return SourceSpec(kind, fields, options)


def _require_groups(pattern: 're.Pattern', names: List[str], where: str) -> None:
    missing = [n for n in names if n not in pattern.groupindex]
    if missing:
        raise DescriptorError(f"{where}: pattern has no named group for {', '.join(missing)}")
