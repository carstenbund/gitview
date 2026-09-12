"""File co-change projection: ``File ↔ File`` edges from shared commits.

Only additive primaries are stored (``cochange_count`` per pair,
``projected_touch_count`` per file). Jaccard is derived on read as::

    cochange / (touch_a + touch_b - cochange)

because adding a single commit changes the Jaccard denominator of every
pair involving a touched file; storing counts keeps incremental updates a
pure append.

Merge commits are excluded: GitPython's ``commit.stats`` diffs a merge
against its first parent, so a PR merge re-reports every file the branch
already touched. Commits flagged ``projection_suppressed`` (more files than
the configured cap) are excluded too.
"""

import sqlite3

_ELIGIBLE = """
    SELECT id, timestamp FROM commits
    WHERE sequence > :since AND is_merge = 0 AND projection_suppressed = 0
"""

_UPSERT_EDGES = f"""
    INSERT INTO file_edges (file_a, file_b, cochange_count, first_seen, last_seen)
    SELECT file_a, file_b, cochange_count, first_seen, last_seen FROM (
        SELECT a.file_id AS file_a,
               b.file_id AS file_b,
               COUNT(*)      AS cochange_count,
               MIN(e.timestamp) AS first_seen,
               MAX(e.timestamp) AS last_seen
        FROM ({_ELIGIBLE}) e
        JOIN commit_files a ON a.commit_id = e.id
        JOIN commit_files b ON b.commit_id = e.id AND b.file_id > a.file_id
        GROUP BY a.file_id, b.file_id
    ) WHERE true
    ON CONFLICT(file_a, file_b) DO UPDATE SET
        cochange_count = cochange_count + excluded.cochange_count,
        first_seen     = MIN(first_seen, excluded.first_seen),
        last_seen      = MAX(last_seen, excluded.last_seen)
"""

_BUMP_TOUCHES = f"""
    UPDATE files SET projected_touch_count = projected_touch_count + (
        SELECT COUNT(*) FROM commit_files cf
        JOIN ({_ELIGIBLE}) e ON e.id = cf.commit_id
        WHERE cf.file_id = files.id
    )
"""


def project_cochange(conn: sqlite3.Connection, since_sequence: int = 0) -> None:
    """Fold commits with ``sequence > since_sequence`` into ``file_edges``.

    Set-based and deterministic: one upsert statement, no Python pair loops.
    Call with ``since_sequence=0`` after ``clear_projection`` for a full
    rebuild, or with the previous max sequence for an incremental append.
    """
    params = {'since': since_sequence}
    conn.execute(_UPSERT_EDGES, params)
    conn.execute(_BUMP_TOUCHES, params)
