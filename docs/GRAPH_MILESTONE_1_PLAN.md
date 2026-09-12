# Graph Analysis — Milestone 1 Coding Plan

Status: **in progress** (this document is the concrete plan for the first
implementation branch; the design rationale lives in the graph-analysis
proposal that motivated it).

Milestone 1 establishes the persistent evidence substrate and nothing else:

```
CommitRecord normalization
+ SQLite graph store under <repo>/.gitview/graph.sqlite
+ Commit → File graph construction (plus parents, authors, PRs)
+ repository-wide file co-change projection
+ basic graph statistics
+ tests
```

Exposed as:

```bash
gitview graph            # build or update, print summary
gitview graph --stats    # summary plus top lists
gitview graph --rebuild  # drop and rebuild
gitview graph --json     # machine-readable stats
```

No LLM. No storyline rewrite. No phase rewrite. No communities. No new
runtime dependencies (SQLite + standard library only).

---

## Decisions locked for this milestone

| Question | Decision | Why |
|---|---|---|
| Where does `graph.sqlite` live? | `<repo>/.gitview/graph.sqlite`, next to the extraction cache `brief` already keeps | The graph is a property of the repository, not of one `output/` run; `brief` is the first consumer; `CacheManager` isn't wired into `analyze` today |
| Shared history extraction | `brief`'s cached incremental extraction moves to `gitview/history_cache.py` and both `brief` and `graph` call it | One ancestry check, one cache file (`.gitview/history.jsonl`) |
| Merge commits in the projection | Stored as commit→file edges, **excluded** from co-change pairs | GitPython stats diff a merge against its first parent, so a PR merge re-reports every file the branch touched |
| Giant commits | `files_changed > max_projection_files` (default 100) → `projection_suppressed = 1`, no pairs | A 6,000-file mechanical commit must not create 18M relationships |
| What is stored for coupling | Additive primaries only: `cochange_count` per pair, `projected_touch_count` per file | Jaccard denominators change for every neighbour of a touched file; deriving on read keeps incremental update a pure append |
| History rewrite | If the stored `last_commit_hash` is not an ancestor of HEAD → full rebuild | Correctness over seconds; merge-base invalidation is a later optimization |
| Renames | Not tracked; a rename is delete + add (matches current numstat extraction) | Documented limitation; rename detection is a later extractor change |
| Branch nodes | Not in milestone 1 | Extraction is single-branch; only `analyze --branches` would populate them |
| Determinism | Every top-list is ordered by score desc, then path asc; every projection is a set-based SQL statement | Identical input → byte-identical output is a test, not a hope |

---

## Stage 1 — normalize evidence (bugfix)

### `gitview/extractor.py`

```python
class CommitRecord:
    def get_changed_files(self) -> List[str]:
        """Paths touched by this commit, in extraction order (git numstat order)."""
        return list(self.files_stats.keys())

    def get_pr_number(self) -> Optional[int]:
        """PR number from GitHub context, or None."""

    @property
    def is_merge(self) -> bool:
        return len(self.parent_hashes) > 1
```

### `gitview/storyline/detector.py`

- `FileClusterDetector._get_commit_files` → `set(commit.get_changed_files())`
- `PRLabelDetector.detect` → `group['files'].update(commit.get_changed_files())`
- `PRTitlePatternDetector.detect`, `CommitMessagePatternDetector.detect` →
  populate the already-declared `files` sets the same way.

Effect: `signal.files` is non-empty for every deterministic detector, so
`StorylineTracker._find_matching_storyline` Priority 3 (file overlap) and
`Storyline.key_files` finally receive data; `SignificanceAnalyzer.cluster_commits`
stops crashing on consecutive GitHub-enriched commits.

### Tests — `tests/test_commit_record.py`

- `get_changed_files` returns `files_stats` keys; empty when none
- `get_pr_number` → `None` without context, `int` with
- `is_merge` false for 0/1 parents, true for 2
- `FileClusterDetector` on real `CommitRecord`s emits a cluster signal with
  `files` populated (regression for the dead detector)
- `PRLabelDetector` signal carries the PR's changed files
- `SignificanceAnalyzer.cluster_commits` splits on PR number without raising

---

## Stage 2 — history cache + graph store

### `gitview/history_cache.py` (moved out of `commands/brief.py`)

```python
HISTORY_CACHE_FILE = ".gitview/history.jsonl"

def head_descends_from(repo: Repo, cached_head: str) -> bool:
    """True when cached_head is an ancestor of (or equal to) HEAD."""

def load_or_extract_history(
    extractor: GitHistoryExtractor,
    repo: Repo,
    repo_path: Path,
    branch: str = "HEAD",
    max_commits: Optional[int] = None,
    head_sha: Optional[str] = None,
) -> List[CommitRecord]:
    """Cached, incremental extraction; falls back to full extraction on any
    failure. Bounded runs (max_commits) bypass the cache entirely."""
```

`BriefCommand._extract_records` becomes a thin delegate.

### `gitview/graph/models.py`

```python
GRAPH_SCHEMA_VERSION = 1
PROJECTION_VERSION = 1

@dataclass(frozen=True)
class GraphMetadata:
    schema_version: int
    repository_path: str
    branch: str
    built_at: str
    last_commit_hash: str
    last_commit_timestamp: str
    projection_version: int
    max_projection_files: int

@dataclass(frozen=True)
class FileTouchStats:      # "most changed"
    path: str
    touch_count: int
    total_insertions: int
    total_deletions: int

@dataclass(frozen=True)
class FileCouplingEdge:    # "most coupled"
    path_a: str
    path_b: str
    cochange_count: int
    jaccard: float
    first_seen: str
    last_seen: str

@dataclass(frozen=True)
class FileDegree:          # "most connected"
    path: str
    degree: int
    weighted_degree: int

@dataclass
class GraphStats:
    commits: int
    merge_commits: int
    suppressed_commits: int
    files: int
    authors: int
    pull_requests: int
    commit_file_edges: int
    file_edges: int
    most_changed: List[FileTouchStats]
    most_coupled: List[FileCouplingEdge]
    most_connected: List[FileDegree]
    def to_dict(self) -> Dict[str, Any]: ...
```

### `gitview/graph/store.py` — SQLite DDL (schema version 1)

```sql
CREATE TABLE graph_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE authors (
    id    INTEGER PRIMARY KEY,
    name  TEXT NOT NULL,
    email TEXT NOT NULL,
    UNIQUE (name, email)
);

CREATE TABLE commits (
    id                    INTEGER PRIMARY KEY,
    hash                  TEXT NOT NULL UNIQUE,
    short_hash            TEXT NOT NULL,
    sequence              INTEGER NOT NULL,          -- chronological position
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
CREATE INDEX idx_commits_sequence ON commits(sequence);

CREATE TABLE commit_parents (
    commit_id   INTEGER NOT NULL REFERENCES commits(id),
    parent_hash TEXT NOT NULL,
    position    INTEGER NOT NULL,
    PRIMARY KEY (commit_id, parent_hash)
);
CREATE INDEX idx_commit_parents_parent ON commit_parents(parent_hash);

CREATE TABLE files (
    id                    INTEGER PRIMARY KEY,
    path                  TEXT NOT NULL UNIQUE,
    first_seen            TEXT NOT NULL,
    last_seen             TEXT NOT NULL,
    touch_count           INTEGER NOT NULL DEFAULT 0,  -- all commits
    projected_touch_count INTEGER NOT NULL DEFAULT 0,  -- non-merge, non-suppressed
    total_insertions      INTEGER NOT NULL DEFAULT 0,
    total_deletions       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE commit_files (
    commit_id  INTEGER NOT NULL REFERENCES commits(id),
    file_id    INTEGER NOT NULL REFERENCES files(id),
    insertions INTEGER NOT NULL,
    deletions  INTEGER NOT NULL,
    PRIMARY KEY (commit_id, file_id)
);
CREATE INDEX idx_commit_files_file ON commit_files(file_id);

CREATE TABLE pull_requests (
    id            INTEGER PRIMARY KEY,
    number        INTEGER NOT NULL UNIQUE,
    title         TEXT,
    body          TEXT,
    state         TEXT,
    merged        INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT
);

CREATE TABLE commit_prs (
    commit_id INTEGER NOT NULL REFERENCES commits(id),
    pr_id     INTEGER NOT NULL REFERENCES pull_requests(id),
    PRIMARY KEY (commit_id, pr_id)
);

CREATE TABLE file_edges (
    file_a         INTEGER NOT NULL REFERENCES files(id),
    file_b         INTEGER NOT NULL REFERENCES files(id),
    cochange_count INTEGER NOT NULL,
    first_seen     TEXT NOT NULL,
    last_seen      TEXT NOT NULL,
    PRIMARY KEY (file_a, file_b),
    CHECK (file_a < file_b)
);
CREATE INDEX idx_file_edges_b ON file_edges(file_b);
```

`jaccard` is derived on read:

```
jaccard(a, b) = cochange_count / (projected_touch_a + projected_touch_b - cochange_count)
```

### `GraphStore` interface

```python
class GraphStore:
    def __init__(self, path: Union[str, Path]) -> None
    def __enter__(self) -> "GraphStore"; def __exit__(...) -> None
    def close(self) -> None

    # schema / metadata
    def initialize(self) -> None                 # create tables if missing
    def reset(self) -> None                      # drop all data, recreate schema
    def get_metadata(self) -> Optional[GraphMetadata]
    def set_metadata(self, metadata: GraphMetadata) -> None

    # primary evidence
    def insert_commits(self, records: Iterable[CommitRecord],
                       max_projection_files: int) -> int   # returns inserted count, skips known hashes
    def has_commit(self, commit_hash: str) -> bool
    def max_sequence(self) -> int

    # derived
    def apply_cochange_projection(self, since_sequence: int) -> None
    def clear_projection(self) -> None

    # read side
    def counts(self) -> Dict[str, int]
    def most_changed(self, limit: int) -> List[FileTouchStats]
    def most_coupled(self, limit: int, min_cochanges: int = 2) -> List[FileCouplingEdge]
    def most_connected(self, limit: int) -> List[FileDegree]
    def parents_of(self, commit_hash: str) -> List[str]
    def files_of(self, commit_hash: str) -> List[str]
```

### `gitview/graph/builder.py`

```python
class GraphBuilder:
    def __init__(self, store: GraphStore, *, repository_path: str, branch: str = "HEAD",
                 max_projection_files: int = 100) -> None
    def build(self, records: List[CommitRecord]) -> GraphMetadata   # reset + insert + project
    def update(self, records: List[CommitRecord]) -> GraphMetadata  # append + project(since)
```

### `gitview/graph/projections/file_cochange.py`

```python
def project_cochange(conn: sqlite3.Connection, since_sequence: int) -> None:
    """Upsert file_edges from commit_files for commits with
    sequence > since_sequence, is_merge = 0 and projection_suppressed = 0;
    bump files.projected_touch_count for the same commits."""
```

One set-based `INSERT ... ON CONFLICT DO UPDATE` over a self-join of
`commit_files`; no Python-side pair loops.

### `gitview/graph/updater.py`

```python
class GraphUpdater:
    def __init__(self, repo_path: Path, *, branch: str = "HEAD",
                 max_projection_files: int = 100, store_path: Optional[Path] = None) -> None
    def sync(self, *, rebuild: bool = False,
             records: Optional[List[CommitRecord]] = None) -> SyncResult
```

`SyncResult(action: "built"|"updated"|"unchanged", new_commits: int,
metadata: GraphMetadata)`. Logic:

```
open store
meta = store.get_metadata()
if rebuild or meta is None or meta.schema_version != GRAPH_SCHEMA_VERSION
   or not head_descends_from(repo, meta.last_commit_hash):
       records = records or load_or_extract_history(...)
       builder.build(records)  → "built"
elif meta.last_commit_hash == HEAD:
       → "unchanged"
else:
       new = records filtered to unknown hashes, or extractor.extract_incremental(since=meta.last_commit_hash)
       builder.update(new)      → "updated"
```

### `gitview/graph/analysis/stats.py`

```python
def compute_graph_stats(store: GraphStore, *, top: int = 10) -> GraphStats
```

### `gitview/commands/graph.py` + `gitview/cli.py`

```python
class GraphCommand(BaseCommand):
    options: repo, branch, rebuild, stats, top, json_output, max_projection_files
```

Console output for `--stats` matches the milestone spec (counts block, then
Most changed / Most coupled / Most connected tables).

### Tests — `tests/test_graph.py`

Unit (in-memory records, `:memory:` or tmp SQLite):

- schema created; metadata round-trips
- build inserts commits, parents, files, commit_files, authors, PRs with the expected counts
- one commit touching A,B,C → exactly 3 edges, each `cochange_count = 1`
- merge commit contributes commit_files but **no** pairs
- commit with more files than the cap → `projection_suppressed = 1`, no pairs, files still recorded
- Jaccard example from the proposal (parser/models/test_parser; parser/test_parser; models/serializer):
  `jaccard(parser, test_parser) = 1.0`, `jaccard(parser, models) = 1/3`
- `update()` after `build()` on a prefix yields the same `file_edges` rows as `build()` on the whole list
- `insert_commits` skips hashes already present
- top lists are deterministic: ties ordered by path
- `compute_graph_stats().to_dict()` is JSON-serializable

Synthetic repository (real `git` via subprocess, as `tests/test_brief.py` does):

- `GraphUpdater.sync()` on a fresh repo → `built`; second call → `unchanged`;
  after a new commit → `updated` with `new_commits = 1`
- after `git reset --hard HEAD~1` + a different commit → `built` (rewrite detected)
- `gitview graph --json` via `click.testing.CliRunner` returns valid JSON with the counts

### CI — `.github/workflows/test.yml`

Add a pytest step (`pip install pytest`, run the suite, deselect the live
GitHub GraphQL integration class which needs network access).

### Docs

- `AGENTS.md` architecture map: `graph/` package and `history_cache.py`
- `README.md`: `gitview graph` in the feature list and command examples

---

## Explicitly out of scope (milestone 2+)

communities · centrality beyond degree · hotspots · bridges · temporal
activity · lifecycle · transitions · anomalies · evidence packets ·
`--strategy graph` · branch nodes · rename tracking · GraphML/DOT export ·
YAML configuration.
