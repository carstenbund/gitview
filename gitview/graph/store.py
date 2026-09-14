"""SQLite-backed persistent graph store.

Explicit tables for primary evidence (commits, files, authors, PRs and the
edges between them) plus one derived table (``file_edges``) that is always
regenerable from the primary tables. Nothing here depends on an LLM.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

from ..extractor import CommitRecord
from .models import (
    DEFAULT_MAX_PROJECTION_FILES,
    FileCouplingEdge,
    FileDegree,
    FileTouchStats,
    GraphMetadata,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS graph_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS authors (
    id    INTEGER PRIMARY KEY,
    name  TEXT NOT NULL,
    email TEXT NOT NULL,
    UNIQUE (name, email)
);

CREATE TABLE IF NOT EXISTS commits (
    id                    INTEGER PRIMARY KEY,
    hash                  TEXT NOT NULL UNIQUE,
    short_hash            TEXT NOT NULL,
    sequence              INTEGER NOT NULL,
    timestamp             TEXT NOT NULL,
    author_id             INTEGER NOT NULL REFERENCES authors(id),
    subject               TEXT NOT NULL,
    insertions            INTEGER NOT NULL,
    deletions             INTEGER NOT NULL,
    files_changed         INTEGER NOT NULL,
    is_merge              INTEGER NOT NULL,
    is_refactor           INTEGER NOT NULL,
    is_large_addition     INTEGER NOT NULL,
    is_large_deletion     INTEGER NOT NULL,
    projection_suppressed INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_commits_sequence ON commits(sequence);

CREATE TABLE IF NOT EXISTS commit_parents (
    commit_id   INTEGER NOT NULL REFERENCES commits(id),
    parent_hash TEXT NOT NULL,
    position    INTEGER NOT NULL,
    PRIMARY KEY (commit_id, parent_hash)
);
CREATE INDEX IF NOT EXISTS idx_commit_parents_parent ON commit_parents(parent_hash);

CREATE TABLE IF NOT EXISTS files (
    id                    INTEGER PRIMARY KEY,
    path                  TEXT NOT NULL UNIQUE,
    first_seen            TEXT NOT NULL,
    last_seen             TEXT NOT NULL,
    touch_count           INTEGER NOT NULL DEFAULT 0,
    projected_touch_count INTEGER NOT NULL DEFAULT 0,
    total_insertions      INTEGER NOT NULL DEFAULT 0,
    total_deletions       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS commit_files (
    commit_id  INTEGER NOT NULL REFERENCES commits(id),
    file_id    INTEGER NOT NULL REFERENCES files(id),
    insertions INTEGER NOT NULL,
    deletions  INTEGER NOT NULL,
    PRIMARY KEY (commit_id, file_id)
);
CREATE INDEX IF NOT EXISTS idx_commit_files_file ON commit_files(file_id);

CREATE TABLE IF NOT EXISTS pull_requests (
    id            INTEGER PRIMARY KEY,
    number        INTEGER NOT NULL UNIQUE,
    title         TEXT,
    body          TEXT,
    state         TEXT,
    merged        INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS commit_prs (
    commit_id INTEGER NOT NULL REFERENCES commits(id),
    pr_id     INTEGER NOT NULL REFERENCES pull_requests(id),
    PRIMARY KEY (commit_id, pr_id)
);

CREATE TABLE IF NOT EXISTS file_edges (
    file_a         INTEGER NOT NULL REFERENCES files(id),
    file_b         INTEGER NOT NULL REFERENCES files(id),
    cochange_count INTEGER NOT NULL,
    first_seen     TEXT NOT NULL,
    last_seen      TEXT NOT NULL,
    PRIMARY KEY (file_a, file_b),
    CHECK (file_a < file_b)
);
CREATE INDEX IF NOT EXISTS idx_file_edges_b ON file_edges(file_b);

-- Structural evidence (optional; written by ``gitview observe``). Keyed by path
-- rather than files(id) so an observation never mutates the historical tables.
CREATE TABLE IF NOT EXISTS structural_snapshots (
    id               INTEGER PRIMARY KEY,
    provider         TEXT NOT NULL,
    provider_version TEXT NOT NULL,
    observed_sha     TEXT NOT NULL,
    observed_at      TEXT NOT NULL,
    source           TEXT NOT NULL,
    content_hash     TEXT NOT NULL,
    node_count       INTEGER NOT NULL,
    edge_count       INTEGER NOT NULL,
    UNIQUE (provider, observed_sha, content_hash)
);

CREATE TABLE IF NOT EXISTS structural_nodes (
    snapshot_id    INTEGER NOT NULL REFERENCES structural_snapshots(id) ON DELETE CASCADE,
    path           TEXT NOT NULL,
    kind           TEXT NOT NULL,
    community      TEXT,
    community_name TEXT,
    symbols        INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (snapshot_id, path)
);

CREATE TABLE IF NOT EXISTS structural_edges (
    snapshot_id INTEGER NOT NULL REFERENCES structural_snapshots(id) ON DELETE CASCADE,
    source      TEXT NOT NULL,
    target      TEXT NOT NULL,
    relation    TEXT NOT NULL,
    weight      REAL NOT NULL,
    count       INTEGER NOT NULL,
    PRIMARY KEY (snapshot_id, source, target, relation)
);
CREATE INDEX IF NOT EXISTS idx_structural_edges_target ON structural_edges(snapshot_id, target);
"""

_HISTORY_TABLES = (
    'file_edges', 'commit_prs', 'pull_requests', 'commit_files', 'files',
    'commit_parents', 'commits', 'authors', 'graph_metadata',
)
# Observations of trees, not derived from git history: a history rebuild cannot
# recreate them, so ``reset`` keeps them unless told otherwise. They reference
# commits only by SHA, never by row id, so they survive the history tables.
_STRUCTURAL_TABLES = ('structural_edges', 'structural_nodes', 'structural_snapshots')

# Keys of github_context worth persisting as PR metadata (review comment
# text is deliberately left out; it belongs to the enrichment cache).
_PR_METADATA_KEYS = ('pr_labels', 'pr_author', 'pr_reviewers', 'pr_review_states',
                     'source_branch', 'target_branch')


class GraphStore:
    """Thin, explicit SQLite wrapper. Use as a context manager or call ``close``."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL") if self.path != ':memory:' else None

    # ------------------------------------------------------------------ lifecycle

    def __enter__(self) -> 'GraphStore':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    # ------------------------------------------------------------------ schema

    def initialize(self) -> None:
        """Create tables if missing."""
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def reset(self, *, keep_structural: bool = True) -> None:
        """Drop the history data and recreate the schema.

        Structural observations are kept by default: they cannot be rebuilt
        from git. Observations whose commit is absent from the rebuilt history
        stay stored with an unknown position (``sequence is None``).
        """
        tables = _HISTORY_TABLES if keep_structural else _STRUCTURAL_TABLES + _HISTORY_TABLES
        for table in tables:
            self.conn.execute(f"DROP TABLE IF EXISTS {table}")
        self.conn.commit()
        self.initialize()

    # ------------------------------------------------------------------ metadata

    def get_metadata(self) -> Optional[GraphMetadata]:
        try:
            rows = self.conn.execute("SELECT key, value FROM graph_metadata").fetchall()
        except sqlite3.OperationalError:
            return None
        if not rows:
            return None
        data = {row['key']: json.loads(row['value']) for row in rows}
        if 'schema_version' not in data:
            return None
        return GraphMetadata.from_dict(data)

    def set_metadata(self, metadata: GraphMetadata) -> None:
        self.conn.executemany(
            "INSERT INTO graph_metadata (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            [(k, json.dumps(v)) for k, v in metadata.to_dict().items()],
        )
        self.conn.commit()

    # ------------------------------------------------------------------ primary evidence

    def has_commit(self, commit_hash: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM commits WHERE hash = ?", (commit_hash,)).fetchone()
        return row is not None

    def max_sequence(self) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(sequence), 0) AS s FROM commits").fetchone()
        return int(row['s'])

    def insert_commits(
        self,
        records: Iterable[CommitRecord],
        max_projection_files: int = DEFAULT_MAX_PROJECTION_FILES,
    ) -> int:
        """Insert commits (with parents, files, authors, PRs) in the given order.

        Records whose hash is already stored are skipped. Returns the number
        of commits inserted. Does not touch ``file_edges``; run the projection
        afterwards.
        """
        conn = self.conn
        sequence = self.max_sequence()
        inserted = 0

        for record in records:
            if self.has_commit(record.commit_hash):
                continue
            sequence += 1

            author_id = self._author_id(record.author or '', record.author_email or '')
            files = record.get_changed_files()
            suppressed = 1 if len(files) > max_projection_files else 0

            cur = conn.execute(
                """INSERT INTO commits (hash, short_hash, sequence, timestamp, author_id, subject,
                       insertions, deletions, files_changed, is_merge, is_refactor,
                       is_large_addition, is_large_deletion, projection_suppressed)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.commit_hash, record.short_hash, sequence, record.timestamp, author_id,
                    record.commit_subject or '', int(record.insertions), int(record.deletions),
                    len(files), int(record.is_merge), int(bool(record.is_refactor)),
                    int(bool(record.is_large_addition)), int(bool(record.is_large_deletion)),
                    suppressed,
                ),
            )
            commit_id = cur.lastrowid

            conn.executemany(
                "INSERT OR IGNORE INTO commit_parents (commit_id, parent_hash, position) VALUES (?, ?, ?)",
                [(commit_id, parent, position) for position, parent in enumerate(record.parent_hashes)],
            )

            for path in files:
                stats = record.files_stats.get(path) or {}
                ins = int(stats.get('insertions', 0) or 0)
                dels = int(stats.get('deletions', 0) or 0)
                file_id = self._upsert_file(path, record.timestamp, ins, dels)
                conn.execute(
                    "INSERT OR IGNORE INTO commit_files (commit_id, file_id, insertions, deletions) "
                    "VALUES (?, ?, ?, ?)",
                    (commit_id, file_id, ins, dels),
                )

            pr_number = record.get_pr_number()
            if pr_number is not None:
                pr_id = self._upsert_pr(pr_number, record.github_context or {})
                conn.execute(
                    "INSERT OR IGNORE INTO commit_prs (commit_id, pr_id) VALUES (?, ?)",
                    (commit_id, pr_id),
                )

            inserted += 1

        conn.commit()
        return inserted

    def _author_id(self, name: str, email: str) -> int:
        self.conn.execute(
            "INSERT OR IGNORE INTO authors (name, email) VALUES (?, ?)", (name, email)
        )
        row = self.conn.execute(
            "SELECT id FROM authors WHERE name = ? AND email = ?", (name, email)
        ).fetchone()
        return int(row['id'])

    def _upsert_file(self, path: str, timestamp: str, insertions: int, deletions: int) -> int:
        self.conn.execute(
            """INSERT INTO files (path, first_seen, last_seen, touch_count,
                                  total_insertions, total_deletions)
               VALUES (?, ?, ?, 1, ?, ?)
               ON CONFLICT(path) DO UPDATE SET
                   first_seen       = MIN(first_seen, excluded.first_seen),
                   last_seen        = MAX(last_seen, excluded.last_seen),
                   touch_count      = touch_count + 1,
                   total_insertions = total_insertions + excluded.total_insertions,
                   total_deletions  = total_deletions + excluded.total_deletions""",
            (path, timestamp, timestamp, insertions, deletions),
        )
        row = self.conn.execute("SELECT id FROM files WHERE path = ?", (path,)).fetchone()
        return int(row['id'])

    def _upsert_pr(self, number: int, context: Dict) -> int:
        metadata = {k: context.get(k) for k in _PR_METADATA_KEYS if context.get(k)}
        self.conn.execute(
            """INSERT INTO pull_requests (number, title, body, state, merged, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(number) DO UPDATE SET
                   title         = COALESCE(excluded.title, title),
                   body          = COALESCE(excluded.body, body),
                   state         = COALESCE(excluded.state, state),
                   merged        = MAX(merged, excluded.merged),
                   metadata_json = COALESCE(excluded.metadata_json, metadata_json)""",
            (
                int(number), context.get('pr_title'), context.get('pr_body'),
                context.get('pr_state'), int(bool(context.get('pr_merged'))),
                json.dumps(metadata, sort_keys=True) if metadata else None,
            ),
        )
        row = self.conn.execute("SELECT id FROM pull_requests WHERE number = ?", (int(number),)).fetchone()
        return int(row['id'])

    # ------------------------------------------------------------------ derived

    def clear_projection(self) -> None:
        self.conn.execute("DELETE FROM file_edges")
        self.conn.execute("UPDATE files SET projected_touch_count = 0")
        self.conn.commit()

    def apply_cochange_projection(self, since_sequence: int = 0) -> None:
        from .projections.file_cochange import project_cochange
        project_cochange(self.conn, since_sequence)
        self.conn.commit()

    # ------------------------------------------------------------------ read side

    def counts(self) -> Dict[str, int]:
        q = self.conn.execute
        return {
            'commits': q("SELECT COUNT(*) AS n FROM commits").fetchone()['n'],
            'merge_commits': q("SELECT COUNT(*) AS n FROM commits WHERE is_merge = 1").fetchone()['n'],
            'suppressed_commits': q(
                "SELECT COUNT(*) AS n FROM commits WHERE projection_suppressed = 1").fetchone()['n'],
            'files': q("SELECT COUNT(*) AS n FROM files").fetchone()['n'],
            'authors': q("SELECT COUNT(*) AS n FROM authors").fetchone()['n'],
            'pull_requests': q("SELECT COUNT(*) AS n FROM pull_requests").fetchone()['n'],
            'commit_file_edges': q("SELECT COUNT(*) AS n FROM commit_files").fetchone()['n'],
            'file_edges': q("SELECT COUNT(*) AS n FROM file_edges").fetchone()['n'],
            'structural_snapshots': q(
                "SELECT COUNT(*) AS n FROM structural_snapshots").fetchone()['n'],
        }

    def most_changed(self, limit: int = 10) -> List[FileTouchStats]:
        rows = self.conn.execute(
            """SELECT path, touch_count, total_insertions, total_deletions FROM files
               ORDER BY touch_count DESC, total_insertions + total_deletions DESC, path ASC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [FileTouchStats(r['path'], r['touch_count'], r['total_insertions'], r['total_deletions'])
                for r in rows]

    def most_coupled(self, limit: int = 10, min_cochanges: int = 2) -> List[FileCouplingEdge]:
        rows = self.conn.execute(
            """SELECT MIN(fa.path, fb.path) AS path_a, MAX(fa.path, fb.path) AS path_b,
                      e.cochange_count,
                      CAST(e.cochange_count AS REAL)
                        / (fa.projected_touch_count + fb.projected_touch_count - e.cochange_count) AS jaccard,
                      e.first_seen, e.last_seen
               FROM file_edges e
               JOIN files fa ON fa.id = e.file_a
               JOIN files fb ON fb.id = e.file_b
               WHERE e.cochange_count >= ?
               ORDER BY e.cochange_count DESC, jaccard DESC, path_a ASC, path_b ASC
               LIMIT ?""",
            (min_cochanges, limit),
        ).fetchall()
        return [FileCouplingEdge(r['path_a'], r['path_b'], r['cochange_count'],
                                 round(float(r['jaccard']), 4), r['first_seen'], r['last_seen'])
                for r in rows]

    def most_connected(self, limit: int = 10) -> List[FileDegree]:
        rows = self.conn.execute(
            """SELECT f.path, COUNT(*) AS degree, SUM(x.w) AS weighted
               FROM (SELECT file_a AS fid, cochange_count AS w FROM file_edges
                     UNION ALL
                     SELECT file_b AS fid, cochange_count AS w FROM file_edges) x
               JOIN files f ON f.id = x.fid
               GROUP BY f.id
               ORDER BY degree DESC, weighted DESC, f.path ASC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [FileDegree(r['path'], int(r['degree']), int(r['weighted'])) for r in rows]

    def parents_of(self, commit_hash: str) -> List[str]:
        rows = self.conn.execute(
            """SELECT p.parent_hash FROM commit_parents p
               JOIN commits c ON c.id = p.commit_id
               WHERE c.hash = ? ORDER BY p.position""",
            (commit_hash,),
        ).fetchall()
        return [r['parent_hash'] for r in rows]

    def files_of(self, commit_hash: str) -> List[str]:
        rows = self.conn.execute(
            """SELECT f.path FROM commit_files cf
               JOIN commits c ON c.id = cf.commit_id
               JOIN files f ON f.id = cf.file_id
               WHERE c.hash = ? ORDER BY f.path""",
            (commit_hash,),
        ).fetchall()
        return [r['path'] for r in rows]

    def edge(self, path_a: str, path_b: str) -> Optional[FileCouplingEdge]:
        """Look up one coupling edge regardless of argument order."""
        rows = self.conn.execute(
            """SELECT MIN(fa.path, fb.path) AS path_a, MAX(fa.path, fb.path) AS path_b,
                      e.cochange_count,
                      CAST(e.cochange_count AS REAL)
                        / (fa.projected_touch_count + fb.projected_touch_count - e.cochange_count) AS jaccard,
                      e.first_seen, e.last_seen
               FROM file_edges e
               JOIN files fa ON fa.id = e.file_a
               JOIN files fb ON fb.id = e.file_b
               WHERE (fa.path = ? AND fb.path = ?) OR (fa.path = ? AND fb.path = ?)""",
            (path_a, path_b, path_b, path_a),
        ).fetchall()
        if not rows:
            return None
        r = rows[0]
        return FileCouplingEdge(r['path_a'], r['path_b'], r['cochange_count'],
                                round(float(r['jaccard']), 4), r['first_seen'], r['last_seen'])

    def all_edges(self) -> List[tuple]:
        """Every coupling edge as ``(path_a, path_b, cochange_count, first_seen, last_seen)``, sorted."""
        rows = self.conn.execute(
            """SELECT fa.path AS a, fb.path AS b, e.cochange_count, e.first_seen, e.last_seen
               FROM file_edges e JOIN files fa ON fa.id = e.file_a JOIN files fb ON fb.id = e.file_b
               ORDER BY fa.path, fb.path"""
        ).fetchall()
        edges = [(*sorted((r['a'], r['b'])), r['cochange_count'], r['first_seen'], r['last_seen'])
                 for r in rows]
        return sorted(edges)

    # ------------------------------------------------------------------ structural evidence

    def insert_structural_snapshot(self, snapshot) -> Tuple[int, bool]:
        """Persist a :class:`~gitview.structural.StructuralSnapshot`.

        Returns ``(snapshot_id, inserted)``. An observation with the same
        provider, commit and content hash is stored once; re-observing an
        unchanged tree is a no-op.
        """
        row = self.conn.execute(
            "SELECT id FROM structural_snapshots WHERE provider = ? AND observed_sha = ? AND content_hash = ?",
            (snapshot.provider, snapshot.observed_sha, snapshot.content_hash),
        ).fetchone()
        if row is not None:
            return int(row['id']), False

        cur = self.conn.execute(
            """INSERT INTO structural_snapshots
               (provider, provider_version, observed_sha, observed_at, source, content_hash,
                node_count, edge_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (snapshot.provider, snapshot.provider_version, snapshot.observed_sha,
             snapshot.observed_at, snapshot.source, snapshot.content_hash,
             len(snapshot.nodes), len(snapshot.edges)),
        )
        sid = int(cur.lastrowid)
        self.conn.executemany(
            "INSERT INTO structural_nodes (snapshot_id, path, kind, community, community_name, symbols) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [(sid, n.path, n.kind, n.community, n.community_name, n.symbols) for n in snapshot.nodes],
        )
        self.conn.executemany(
            "INSERT INTO structural_edges (snapshot_id, source, target, relation, weight, count) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [(sid, e.source, e.target, e.relation, e.weight, e.count) for e in snapshot.edges],
        )
        self.conn.commit()
        return sid, True

    def structural_observations(self, provider: Optional[str] = None) -> List['StructuralObservation']:
        """All stored observations, oldest first by history position, then by time."""
        from ..structural.models import StructuralObservation
        where = "WHERE s.provider = ?" if provider else ""
        rows = self.conn.execute(
            f"""SELECT s.*, c.sequence AS sequence
                FROM structural_snapshots s
                LEFT JOIN commits c ON c.hash = s.observed_sha
                {where}
                ORDER BY c.sequence IS NULL, c.sequence ASC, s.observed_at ASC, s.id ASC""",
            (provider,) if provider else (),
        ).fetchall()
        return [StructuralObservation(
            id=int(r['id']), provider=r['provider'], provider_version=r['provider_version'],
            observed_sha=r['observed_sha'], observed_at=r['observed_at'], source=r['source'],
            content_hash=r['content_hash'], node_count=int(r['node_count']),
            edge_count=int(r['edge_count']),
            sequence=int(r['sequence']) if r['sequence'] is not None else None,
        ) for r in rows]

    def latest_structural_observation(self, provider: Optional[str] = None):
        observations = self.structural_observations(provider)
        return observations[-1] if observations else None

    def structural_nodes(self, snapshot_id: int) -> List['StructuralNode']:
        from ..structural.models import StructuralNode
        rows = self.conn.execute(
            "SELECT path, kind, community, community_name, symbols FROM structural_nodes "
            "WHERE snapshot_id = ? ORDER BY path", (snapshot_id,)).fetchall()
        return [StructuralNode(r['path'], r['kind'], r['community'], r['community_name'], int(r['symbols']))
                for r in rows]

    def structural_edges(self, snapshot_id: int) -> List['StructuralEdge']:
        from ..structural.models import StructuralEdge
        rows = self.conn.execute(
            "SELECT source, target, relation, weight, count FROM structural_edges "
            "WHERE snapshot_id = ? ORDER BY source, target, relation", (snapshot_id,)).fetchall()
        return [StructuralEdge(r['source'], r['target'], r['relation'], float(r['weight']), int(r['count']))
                for r in rows]

    def load_structural_snapshot(self, snapshot_id: int) -> Optional['StructuralSnapshot']:
        """Rebuild a full :class:`StructuralSnapshot` from the store."""
        from ..structural.models import StructuralSnapshot
        r = self.conn.execute(
            "SELECT * FROM structural_snapshots WHERE id = ?", (snapshot_id,)).fetchone()
        if r is None:
            return None
        return StructuralSnapshot(
            provider=r['provider'], provider_version=r['provider_version'],
            observed_sha=r['observed_sha'], observed_at=r['observed_at'], source=r['source'],
            content_hash=r['content_hash'],
            nodes=self.structural_nodes(snapshot_id), edges=self.structural_edges(snapshot_id),
        )

    def delete_structural_snapshot(self, snapshot_id: int) -> None:
        self.conn.execute("DELETE FROM structural_snapshots WHERE id = ?", (snapshot_id,))
        self.conn.commit()

    # ------------------------------------------------------------------ history queries used by motifs

    def sequence_of(self, commit_hash: str) -> Optional[int]:
        row = self.conn.execute("SELECT sequence FROM commits WHERE hash = ?", (commit_hash,)).fetchone()
        return int(row['sequence']) if row else None

    def cochange_commits(self, path_a: str, path_b: str) -> List[Tuple[int, str, str]]:
        """Non-merge commits touching both files as ``(sequence, hash, timestamp)``, oldest first."""
        rows = self.conn.execute(
            """SELECT c.sequence, c.hash, c.timestamp
               FROM commits c
               JOIN commit_files x ON x.commit_id = c.id
               JOIN commit_files y ON y.commit_id = c.id
               JOIN files fa ON fa.id = x.file_id
               JOIN files fb ON fb.id = y.file_id
               WHERE fa.path = ? AND fb.path = ? AND c.is_merge = 0 AND c.projection_suppressed = 0
               ORDER BY c.sequence""",
            (path_a, path_b),
        ).fetchall()
        return [(int(r['sequence']), r['hash'], r['timestamp']) for r in rows]

    def file_touches(self, path: str) -> List[Tuple[int, str, str, str]]:
        """Non-merge commits touching ``path`` as ``(sequence, hash, timestamp, author)``, oldest first."""
        rows = self.conn.execute(
            """SELECT c.sequence, c.hash, c.timestamp, a.name AS author
               FROM commits c
               JOIN commit_files cf ON cf.commit_id = c.id
               JOIN files f ON f.id = cf.file_id
               JOIN authors a ON a.id = c.author_id
               WHERE f.path = ? AND c.is_merge = 0
               ORDER BY c.sequence""",
            (path,),
        ).fetchall()
        return [(int(r['sequence']), r['hash'], r['timestamp'], r['author']) for r in rows]

    def cochange_neighbours(self, path: str) -> List[Tuple[str, int]]:
        """Files co-changed with ``path`` as ``(other_path, cochange_count)``, strongest first."""
        rows = self.conn.execute(
            """SELECT CASE WHEN fa.path = ? THEN fb.path ELSE fa.path END AS other, e.cochange_count
               FROM file_edges e JOIN files fa ON fa.id = e.file_a JOIN files fb ON fb.id = e.file_b
               WHERE fa.path = ? OR fb.path = ?
               ORDER BY e.cochange_count DESC, other""",
            (path, path, path),
        ).fetchall()
        return [(r['other'], int(r['cochange_count'])) for r in rows]

    def touch_counts(self) -> Dict[str, int]:
        rows = self.conn.execute("SELECT path, touch_count FROM files").fetchall()
        return {r['path']: int(r['touch_count']) for r in rows}

    def commit_span(self) -> Tuple[int, int]:
        row = self.conn.execute(
            "SELECT COALESCE(MIN(sequence), 0) AS lo, COALESCE(MAX(sequence), 0) AS hi FROM commits").fetchone()
        return int(row['lo']), int(row['hi'])

    def commit_info(self, sequence: int) -> Optional[Tuple[str, str, str]]:
        row = self.conn.execute(
            "SELECT hash, timestamp, subject FROM commits WHERE sequence = ?", (sequence,)).fetchone()
        return (row['hash'], row['timestamp'], row['subject']) if row else None
