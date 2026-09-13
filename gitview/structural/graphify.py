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
        graph_file = Path(source).resolve() if source else repo / DEFAULT_OUTPUT_DIR / DEFAULT_GRAPH_FILE

        if refresh or not graph_file.exists():
            if source and not refresh:
                raise StructuralProviderError(f"structural source not found: {graph_file}")
            self._run_update(repo)
            if not graph_file.exists():
                raise StructuralProviderError(
                    f"Graphify ran but produced no {graph_file}")

        raw = graph_file.read_bytes()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise StructuralProviderError(f"{graph_file} is not valid JSON: {exc}") from exc

        return translate(
            data,
            sha=sha,
            source=str(graph_file),
            content_hash=hashlib.sha256(raw).hexdigest(),
            provider_version=graphify_version(),
            observed_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        )

    # ----------------------------------------------------------- internals

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
) -> StructuralSnapshot:
    """Pure translation of a parsed ``graph.json`` into a :class:`StructuralSnapshot`.

    The commit recorded by Graphify (``built_at_commit``) wins over ``sha``
    when both are present: the observation belongs to the tree it was taken
    from, and the caller reports the drift.
    """
    nodes_raw: List[Dict[str, Any]] = data.get('nodes') or []
    links_raw: List[Dict[str, Any]] = data.get('links') or data.get('edges') or []

    file_of: Dict[str, str] = {}
    symbols: Counter = Counter()
    kinds: Dict[str, Counter] = defaultdict(Counter)
    communities: Dict[str, Counter] = defaultdict(Counter)
    community_names: Dict[str, str] = {}

    for n in nodes_raw:
        path = _norm_path(n.get('source_file'))
        if not path:
            continue
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

    observed = str(data.get('built_at_commit') or sha)
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
