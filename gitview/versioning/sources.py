"""Translate descriptor sources into observations on a branch's history.

All positions are indexes into the branch's first-parent chain, oldest first:
a version enters the branch at the commit where the chain first contains it.
"""

import fnmatch
import json
import posixpath
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

try:  # Python 3.11+
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib

from .descriptor import DEFAULT_COMPONENTS, DEFAULT_SEQUENCE_NUMBER, DEFAULT_TAG_PATTERN
from .models import Problem, SourceSpec

Values = Dict[str, Optional[int]]


class GitError(RuntimeError):
    pass


class BranchHistory:
    """Thin wrapper over ``git`` for one branch of one repository."""

    def __init__(self, root: Union[str, Path], branch: str = 'HEAD') -> None:
        self.root = str(root)
        self.branch = branch
        lines = self.run('log', '--first-parent', '--reverse', '--format=%H %cI', branch).splitlines()
        self.chain: List[str] = []
        self.timestamps: List[str] = []
        for line in lines:
            sha, ts = line.split(' ', 1)
            self.chain.append(sha)
            self.timestamps.append(ts)
        self.index = {sha: i for i, sha in enumerate(self.chain)}

    def run(self, *args: str, check: bool = True) -> str:
        out = subprocess.run(['git', '-C', self.root, *args], capture_output=True, text=True,
                             encoding='utf-8', errors='replace')
        if check and out.returncode != 0:
            raise GitError(f"git {' '.join(args)}: {out.stderr.strip()}")
        return out.stdout if out.returncode == 0 else ''

    def show(self, sha: str, path: str) -> Optional[str]:
        out = subprocess.run(['git', '-C', self.root, 'show', f'{sha}:{path}'], capture_output=True,
                             text=True, encoding='utf-8', errors='replace')
        return out.stdout if out.returncode == 0 else None

    def is_ancestor(self, older: str, newer: str) -> bool:
        return subprocess.run(['git', '-C', self.root, 'merge-base', '--is-ancestor', older, newer],
                              capture_output=True).returncode == 0

    def position_of(self, sha: str) -> Optional[int]:
        """Index of the first chain commit that contains ``sha``; None if the branch never does."""
        if sha in self.index:
            return self.index[sha]
        if not self.chain or not self.is_ancestor(sha, self.chain[-1]):
            return None
        # Containment is monotonic along the first-parent chain: binary search.
        lo, hi = 0, len(self.chain) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.is_ancestor(sha, self.chain[mid]):
                hi = mid
            else:
                lo = mid + 1
        return lo

    def count_range(self, start_exclusive: Optional[str], end: str) -> int:
        spec = f'{start_exclusive}..{end}' if start_exclusive else end
        return int(self.run('rev-list', '--count', spec).strip() or 0)


@dataclass
class Observation:
    position: int
    values: Values
    raw: str
    message: str = ''


# ---------------------------------------------------------------------------
# Value parsing
# ---------------------------------------------------------------------------

def values_from_string(text: str, source: SourceSpec) -> Values:
    names = [f.name for f in source.fields]
    pattern = source.options.get('pattern')
    if pattern:
        m = re.search(pattern, text)
        if not m:
            raise ValueError(f"{text!r} does not match pattern {pattern!r}")
        return {n: _int_or_none(m.group(n)) for n in names}
    numbers = re.findall(r'\d+', text)
    if not numbers:
        raise ValueError(f"no version numbers in {text!r}")
    components = source.options.get('components', DEFAULT_COMPONENTS)
    by_component = {c: (int(numbers[i]) if i < len(numbers) else 0) for i, c in enumerate(components)}
    return {n: by_component[n] for n in names}


def parse_file(content: str, source: SourceSpec) -> Tuple[Values, str]:
    """Return ``(values, raw)`` for one version of a file; raises ValueError."""
    parse = source.options['parse']
    if parse == 'python-assign':
        values: Values = {}
        for f in source.fields:
            m = re.search(rf'^\s*{re.escape(f.name)}\s*(?::[^=\n]+)?=\s*(\d+)\b', content, re.M)
            values[f.name] = int(m.group(1)) if m else None
        missing = [f.name for f in source.fields if values[f.name] is None and f.role not in ('label',)]
        if len(missing) == len([f for f in source.fields if f.role != 'label']):
            raise ValueError(f"none of {', '.join(missing)} assigned an integer")
        return values, '.'.join(str(v) for v in values.values() if v is not None)
    if parse == 'regex':
        m = re.search(source.options['pattern'], content, re.M)
        if not m:
            raise ValueError(f"pattern {source.options['pattern']!r} not found")
        return {f.name: _int_or_none(m.group(f.name)) for f in source.fields}, m.group(0)

    if parse == 'python-dunder':
        m = re.search(r'''^\s*__version__\s*(?::[^=\n]+)?=\s*['"]([^'"]+)['"]''', content, re.M)
        if not m:
            raise ValueError("no __version__ = \"...\" assignment")
        text = m.group(1)
    elif parse == 'plain':
        text = content.strip().splitlines()[0].strip() if content.strip() else ''
    else:
        data = json.loads(content) if parse == 'json' else tomllib.loads(content)
        for key in str(source.options['key']).split('.'):
            if not isinstance(data, dict) or key not in data:
                raise ValueError(f"key {source.options['key']!r} not found")
            data = data[key]
        text = str(data)
    return values_from_string(text, source), text


def _int_or_none(value: Optional[str]) -> Optional[int]:
    return int(value) if value not in (None, '') else None


def render_label(source: SourceSpec, values: Values, raw: str) -> str:
    template = source.options.get('label')
    if not template:
        return raw
    return re.sub(r'\{(\w+)\}', lambda m: '?' if values.get(m.group(1)) is None else str(values[m.group(1)]),
                  template)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def observe_source(history: BranchHistory, source: SourceSpec, scope: str) -> Tuple[List[Observation], List[Problem]]:
    if source.kind == 'tag':
        return _observe_tags(history, source)
    if source.kind == 'file':
        return _observe_file(history, source, scope)
    return _observe_sequence(history, source, scope)


def _observe_tags(history: BranchHistory, source: SourceSpec):
    pattern = re.compile(source.options.get('pattern', DEFAULT_TAG_PATTERN))
    names = [f.name for f in source.fields]
    fmt = '%(refname:strip=2)%00%(objecttype)%00%(objectname)%00%(*objectname)%00%(contents)%1e'
    records = history.run('for-each-ref', 'refs/tags', f'--format={fmt}').split('\x1e')
    observations, problems = [], []
    for record in records:
        parts = record.strip('\n').split('\x00')
        if len(parts) < 5:
            continue
        name, objtype, obj, peeled, contents = parts[0], parts[1], parts[2], parts[3], parts[4]
        m = pattern.fullmatch(name)
        if not m:
            continue
        commit = peeled if objtype == 'tag' else obj
        position = history.position_of(commit)
        if position is None:
            problems.append(Problem('warning', f"tag {name} is not on {history.branch}; ignored",
                                    commit=commit, source=source.describe()))
            continue
        message = contents.strip() if objtype == 'tag' else ''
        observations.append(Observation(position, {n: _int_or_none(m.group(n)) for n in names}, name, message))
    observations.sort(key=lambda o: (o.position, _sort_key(o.values)))
    return observations, problems


def _observe_file(history: BranchHistory, source: SourceSpec, scope: str):
    path = posixpath.join(scope, source.options['path']) if scope else source.options['path']
    shas = history.run('log', '--first-parent', '--format=%H', history.branch, '--', path).split()
    positions = sorted(history.index[s] for s in shas if s in history.index)
    observations, problems = [], []
    previous: Optional[Values] = None
    for position in positions:
        sha = history.chain[position]
        content = history.show(sha, path)
        if content is None:          # deleted at this commit
            continue
        try:
            values, raw = parse_file(content, source)
        except (ValueError, tomllib.TOMLDecodeError) as exc:
            problems.append(Problem('error', f"{path}: {exc}", commit=sha, source=source.describe()))
            continue
        if values == previous:       # comments or unrelated lines changed
            continue
        previous = values
        observations.append(Observation(position, values, raw))
    return observations, problems


def _observe_sequence(history: BranchHistory, source: SourceSpec, scope: str):
    glob = posixpath.join(scope, source.options['glob']) if scope else source.options['glob']
    excludes = source.options.get('exclude') or []
    excludes = [excludes] if isinstance(excludes, str) else list(excludes)
    number = re.compile(source.options.get('number', DEFAULT_SEQUENCE_NUMBER))
    name = source.fields[0].name
    out = history.run('log', '--first-parent', '--no-renames', '--diff-filter=A', '--format=%x1e%H',
                      '--name-only', history.branch, '--', f':(glob){glob}')
    added: List[Tuple[int, List[str]]] = []
    for block in out.split('\x1e'):
        lines = [line for line in block.splitlines() if line.strip()]
        if not lines or lines[0] not in history.index:
            continue
        added.append((history.index[lines[0]], lines[1:]))
    added.sort()

    observations, highest = [], None
    for position, paths in added:
        best: Optional[Tuple[int, str]] = None
        for p in paths:
            base = posixpath.basename(p)
            if any(fnmatch.fnmatch(base, ex) or fnmatch.fnmatch(p, ex) for ex in excludes):
                continue
            m = number.search(base)
            if m and (best is None or int(m.group(1)) > best[0]):
                best = (int(m.group(1)), base)
        if best and (highest is None or best[0] > highest):
            highest = best[0]
            observations.append(Observation(position, {name: best[0]}, best[1]))
    return observations, []


def _sort_key(values: Values):
    return tuple(-1 if v is None else v for v in values.values())
