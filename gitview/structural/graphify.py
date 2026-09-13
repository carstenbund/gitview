"""Graphify adapter: translate ``graphify-out/graph.json`` into the neutral model.

Graphify emits a symbol-level graph (functions, classes, headings) where every
node carries ``source_file``. GitView only needs file granularity, so nodes are
grouped by file and cross-file symbol edges are folded into file → file edges.

Nothing outside this module refers to Graphify's representation.
"""

import hashlib
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .models import StructuralNode, StructuralSnapshot, fold_symbol_edges
from .provider import StructuralProvider, StructuralProviderError

DEFAULT_OUTPUT_DIR = 'graphify-out'
DEFAULT_GRAPH_FILE = 'graph.json'
#: Written by Graphify next to graph.json: the absolute directory it was built from.
ROOT_MARKER_FILE = '.graphify_root'

#: Graphify relations that describe containment rather than dependency.
_NON_DEPENDENCY_RELATIONS = {'contains', 'method', 'defines'}

#: Graphify ``file_type`` → neutral node kind.
_KIND = {'code': 'code', 'document': 'document', 'rationale': 'document'}


def graphify_version() -> str:
    """Installed Graphify version, or ``'unknown'``."""
    from importlib import metadata
    for dist in ('graphifyy', 'graphify'):
        try:
            return metadata.version(dist)
        except metadata.PackageNotFoundError:
            continue
    return 'unknown'


class GraphifyProvider(StructuralProvider):
    name = 'graphify'

    def __init__(self, executable: str = 'graphify') -> None:
        self.executable = executable

    # ----------------------------------------------------------- protocol

    def available(self) -> Tuple[bool, str]:
        if shutil.which(self.executable):
            return True, ''
        return False, (f"'{self.executable}' is not on PATH; install Graphify or pass "
                       f"--source pointing at an existing graph.json")

    def snapshot(
        self,
        repo_path: Union[str, Path],
        sha: str,
        *,
        source: Optional[Union[str, Path]] = None,
        refresh: bool = False,
    ) -> StructuralSnapshot:
        repo = Path(repo_path).resolve()
        graph_file, graph_root = self._locate(repo, source)

        if refresh or not graph_file.exists():
            if source and not refresh:
                raise StructuralProviderError(f"structural source not found: {graph_file}")
            self._run_update(graph_root or repo)
            if not graph_file.exists():
                raise StructuralProviderError(
                    f"Graphify ran but produced no {graph_file}")

        raw = graph_file.read_bytes()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise StructuralProviderError(f"{graph_file} is not valid JSON: {exc}") from exc

        prefix = _path_prefix(repo, graph_root)
        return translate(
            data,
            sha=sha,
            source=str(graph_file) + (f" [{prefix}/]" if prefix else ''),
            content_hash=hashlib.sha256(raw).hexdigest(),
            provider_version=graphify_version(),
            observed_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
            path_prefix=prefix,
        )

    # ----------------------------------------------------------- internals

    def _locate(self, repo: Path, source) -> Tuple[Path, Optional[Path]]:
        """Return ``(graph_file, graph_root)``.

        ``graph_root`` is the directory the graph describes; paths in
        ``graph.json`` are relative to it. It is ``None`` when unknown (an
        explicit ``--source`` with no ``.graphify_root`` marker), in which case
        the graph is taken to describe ``repo`` itself.

        Resolution order without ``--source``: the repository's own
        ``graphify-out/graph.json``; otherwise, when the repository is a git
        submodule, the enclosing superproject's graph (walking up through
        nested superprojects). One graph built at the superproject root then
        serves every module, re-based under the module's path.
        """
        if source:
            graph_file = Path(source).resolve()
            return graph_file, _root_marker(graph_file)
        graph_file = repo / DEFAULT_OUTPUT_DIR / DEFAULT_GRAPH_FILE
        if graph_file.exists():
            return graph_file, repo
        parent = _superproject_of(repo)
        while parent is not None:
            candidate = parent / DEFAULT_OUTPUT_DIR / DEFAULT_GRAPH_FILE
            if candidate.exists():
                return candidate, _root_marker(candidate) or parent
            parent = _superproject_of(parent)
        return graph_file, repo

    def _run_update(self, repo: Path) -> None:
        ok, reason = self.available()
        if not ok:
            raise StructuralProviderError(reason)
        # ``graphify update`` re-extracts code without an LLM; ``--no-cluster``
        # is deliberately not passed so community ids stay available.
        proc = subprocess.run(
            [self.executable, 'update', str(repo)],
            cwd=str(repo), capture_output=True, text=True,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
            raise StructuralProviderError(
                "graphify update failed:\n" + "\n".join(tail))


def translate(
    data: Dict[str, Any],
    *,
    sha: str,
    source: str,
    content_hash: str,
    provider_version: str,
    observed_at: str,
    path_prefix: str = '',
) -> StructuralSnapshot:
    """Pure translation of a parsed ``graph.json`` into a :class:`StructuralSnapshot`.

    The commit recorded by Graphify (``built_at_commit``) wins over ``sha``
    when both are present: the observation belongs to the tree it was taken
    from, and the caller reports the drift.

    ``path_prefix`` re-bases a graph built one or more directories above the
    repository (a git superproject): only nodes under ``<prefix>/`` are kept,
    with the prefix stripped, and edges are folded only between kept files.
    Edges into files outside the prefix are dropped — they are dependencies on
    another repository and have no counterpart in this one's history. A
    re-based observation records ``sha`` rather than ``built_at_commit``,
    which is the superproject's commit, not this repository's.
    """
    nodes_raw: List[Dict[str, Any]] = data.get('nodes') or []
    links_raw: List[Dict[str, Any]] = data.get('links') or data.get('edges') or []

    file_of: Dict[str, str] = {}
    symbols: Counter = Counter()
    kinds: Dict[str, Counter] = defaultdict(Counter)
    communities: Dict[str, Counter] = defaultdict(Counter)
    community_names: Dict[str, str] = {}

    prefix = path_prefix.strip('/')
    prefix_slash = prefix + '/' if prefix else ''

    for n in nodes_raw:
        path = _norm_path(n.get('source_file'))
        if not path:
            continue
        if prefix_slash:
            if not path.startswith(prefix_slash):
                continue
            path = path[len(prefix_slash):]
        nid = n.get('id')
        if nid is not None:
            file_of[str(nid)] = path
        symbols[path] += 1
        kinds[path][_KIND.get(str(n.get('file_type', 'code')), 'other')] += 1
        community = n.get('community')
        if community is not None:
            communities[path][str(community)] += 1
            name = n.get('community_name')
            if name:
                community_names[str(community)] = str(name)

    nodes: List[StructuralNode] = []
    for path in sorted(symbols):
        community = communities[path].most_common(1)[0][0] if communities[path] else None
        nodes.append(StructuralNode(
            path=path,
            kind=kinds[path].most_common(1)[0][0],
            community=community,
            community_name=community_names.get(community) if community else None,
            symbols=symbols[path],
        ))

    pairs = []
    for e in links_raw:
        relation = str(e.get('relation', 'references'))
        if relation in _NON_DEPENDENCY_RELATIONS:
            continue
        src = file_of.get(str(e.get('source')))
        dst = file_of.get(str(e.get('target')))
        if not src or not dst or src == dst:
            continue
        pairs.append((src, dst, relation, float(e.get('weight', 1.0) or 1.0)))

    observed = sha if prefix else str(data.get('built_at_commit') or sha)
    return StructuralSnapshot(
        provider='graphify',
        provider_version=provider_version,
        observed_sha=observed,
        observed_at=observed_at,
        source=source,
        content_hash=content_hash,
        nodes=nodes,
        edges=fold_symbol_edges(pairs),
    )


def _norm_path(value: Any) -> str:
    if not value:
        return ''
    return str(value).replace('\\', '/').lstrip('./')


def _root_marker(graph_file: Path) -> Optional[Path]:
    """The directory recorded in ``.graphify_root`` next to ``graph_file``, if any."""
    marker = graph_file.parent / ROOT_MARKER_FILE
    try:
        text = marker.read_text(encoding='utf-8').strip()
    except OSError:
        return None
    if not text:
        return None
    root = Path(text)
    return root.resolve() if root.is_dir() else None


def _path_prefix(repo: Path, graph_root: Optional[Path]) -> str:
    """``repo`` relative to ``graph_root`` as a POSIX path, or ``''`` when the graph is the repo's own."""
    if graph_root is None:
        return ''
    try:
        rel = repo.resolve().relative_to(graph_root.resolve())
    except ValueError:
        return ''
    return rel.as_posix() if str(rel) != '.' else ''


def _superproject_of(repo: Path) -> Optional[Path]:
    """The working tree of the git superproject ``repo`` is a submodule of, or ``None``."""
    try:
        proc = subprocess.run(
            ['git', '-C', str(repo), 'rev-parse', '--show-superproject-working-tree'],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    out = proc.stdout.strip()
    if proc.returncode != 0 or not out:
        return None
    return Path(out).resolve()

