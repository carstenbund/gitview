# Phased History — Design

Status: **milestone 1 (versions) implemented; milestones 2–5 are design.**
Captures the decisions agreed on 2026-09-14. Milestones at the end; each
milestone gets its own plan when it starts.

---

## 1. Why

An agent that used GitView across six repositories reported:

- **Useful:** the most-changed-files list and the co-change pairs from
  `gitview motifs` — "touch X, check Y" hints you can't get from reading code.
- **Not useful:** storylines ("app changes", "docs changes") and contributor
  tables.
- **Fragile:** a graph rebuild silently dropped the Graphify observations
  (fixed in #88).
- **Costly:** committed briefs that are stale by construction.

Reading `history_story.md` and the storyline report confirms the larger
problem: they are a long repetition of single facts. That is useful as an
inventory or for a final project review, and very hard to remember.

The pipeline also works against how history behaves. History is append-only:
the first run over a repository is the expensive one, and every later run only
has a new stretch of commits to add. Today an incremental `analyze` still
regenerates the executive summary, timeline and full narrative over the whole
history (the story cache key covers every phase), and phase boundaries move as
commits arrive, so nothing is ever finished.

## 2. Goals

The output answers four questions:

| Question | Who asks | Structure facts (Graphify) | History facts (GitView) |
|---|---|---|---|
| Where are we now? | Agent starting a session | Modules, communities, hub files | Active initiatives, recent churn |
| How did we get here? | Newcomer, reviewer | Structural change per phase | Arcs per phase |
| What changed between A and B? | Step report | Structural diff A → B | Commits, PRs, finished/started work |
| Before touching X, what should I know? | Agent editing code | X's dependencies and dependents | Co-change partners, past fixes, churn |

**Non-goals:** retelling every commit as prose; explaining *why* beyond what
commits, PRs and tag messages state.

## 3. Principles

1. **Append-only.** A finished phase is sealed and never recomputed
   automatically. A run does work proportional to what is new.
2. **Graphify is current by design; GitView is historical by design.** A
   Graphify observation describes one tree and is gone once the tree moves on.
   Structural history exists only if it is captured when a phase is sealed and
   kept as a digest.
3. **LLM output is data, not prose.** Summaries are stored as machine-readable
   JSON with evidence pointers. Documents never contain stored LLM text
   verbatim as their source of truth; they are rendered on command.
4. **Summaries first, detail on demand.** Every document leads with a short
   list of claims; each claim links down to the evidence.
5. **Versions are declared, not guessed.** Phase boundaries come from the
   repository's own versioning, described once by its owner.

## 4. Layers

```
            ┌──────────────────────────────────────────────┐
  views     │  rendered docs (markdown)   — no LLM, on command
            └───────────────▲──────────────────────────────┘
                            │ pure function of ↓
            ┌───────────────┴──────────────────────────────┐
  records   │  phase records (JSON)      — sealed per phase
            │    history · structure · summary
            └───────────────▲──────────────────────────────┘
                            │ built from ↓
            ┌───────────────┴──────────────────────────────┐
  ledger    │  .gitview/graph.sqlite      — commits, files, PRs,
            │  co-change, structural snapshots
            └───────────────▲──────────────────────────────┘
                            │
                     git  +  version descriptor  +  Graphify
```

| Layer | Contents | Cost | Recreated when |
|---|---|---|---|
| Ledger | Primary evidence, co-change projection, full structural snapshots | Git reads | Automatically when invalid (cheap); structural snapshots are always kept |
| Records | One JSON file per phase | LLM, once per phase; Graphify, once per phase | Never automatically; explicitly, per phase |
| Views | Markdown documents | None | Every time they are asked for |

## 5. Version descriptor

GitView cannot know what a version number *means* in a repository; the owner
can. A **version descriptor** says where and how a repository is versioned and
maps its version components onto GitView's roles. `versions detect` drafts it;
the owner confirms it.

### 5.1 Where it lives

Lookup order, first match wins:

1. The project file that usually already exists:
   - `pyproject.toml` → `[tool.gitview.versioning]`
   - `Cargo.toml` → `[package.metadata.gitview.versioning]`
   - `package.json` → `"gitview": { "versioning": … }`
2. `.gitview.toml` at the repository root → `[versioning]`
3. Neither: the repository has no phased history. `graph`, `motifs` and
   `observe` still work; phased commands explain what to add.

A descriptor is scoped to the directory of the file that holds it, so each
project in a monorepo (e.g. the three `pyproject.toml` files in `ADO-api`)
describes itself, and paths are relative to that file.

### 5.2 Roles

| Role | Meaning | Effect |
|---|---|---|
| `generation` | A new baseline | Seals a phase and a chapter |
| `boundary` | Any change that alters persistent structure — by convention, any DDL | Seals a phase |
| `step` | Code-only change | Marker inside a phase; appears in the detail index |
| `label` | Display only (e.g. release epoch) | None |

The working rule for repositories that follow it: *a major version is any DDL,
a minor version is code only, there is no build number* — i.e. the DDL counter
takes the `boundary` role and the code counter takes `step`. Components are
named by the repository; roles are GitView's. They are kept apart because the
same words (major/minor) mean different things in different repositories.

### 5.3 Sources

| `kind` | Reads | Keys |
|---|---|---|
| `tag` | Tags matching a pattern | `pattern` (regex with named groups) |
| `file` | A file's content at each commit that changes it | `path`, `parse` = `python-assign` \| `python-dunder` \| `json` \| `toml` \| `regex` \| `plain` |
| `sequence` | Numbered files appearing over time; the highest number so far is the version | `glob`, `exclude`, `number` (regex, default `^(\d+)`), `field` (default `sequence`) |
| `changelog` | Version headings (later) | `path`, `pattern` |

Sources are listed in priority order. For each commit the first source that
yields a version wins; `versions check` reports where sources disagree. A
`pyproject.toml` with `version = {attr = "pkg.__version__"}` is followed to
the file it names.

A `file` source produces an event only when a parsed value changes — an edit
to comments in the version file is not a version. A field missing from an
older version of the file (e.g. an `EPOCH` added later) is read as unknown,
not as an error, unless every non-label field is missing.

Versions are placed on the branch's first-parent chain: a version bumped or
tagged on a merged branch takes effect at the merge commit, and a tag the
branch never contains is reported and ignored.

### 5.4 Examples

`oebv-api/pyproject.toml` (version fields in `app/core/version.py`, no tags):

```toml
[tool.gitview.versioning]
schema = 1
description = "Schema-compatibility contract shared with shop-api and the live DB"

[[tool.gitview.versioning.source]]
kind  = "file"
path  = "app/core/version.py"
parse = "python-assign"
label = "{EPOCH}.{MAJOR}.{MINOR}.{PATCH}"
fields.EPOCH = { role = "label",      meaning = "release marker, display only" }
fields.MAJOR = { role = "generation", meaning = "schema generation (v6 baseline)" }
fields.MINOR = { role = "boundary",   meaning = "schema revision; every oebv-db migration bumps it" }
fields.PATCH = { role = "step",       meaning = "code-only change in this service" }
```

`drm_screen/pyproject.toml` (annotated tags; the version file covers history
before the first tag):

```toml
[tool.gitview.versioning]
schema = 1

[[tool.gitview.versioning.source]]
kind    = "tag"
pattern = 'v(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)'
fields.major = { role = "generation" }
fields.minor = { role = "boundary" }
fields.patch = { role = "step" }

[[tool.gitview.versioning.source]]
kind  = "file"
path  = "drm_screen/__init__.py"
parse = "python-dunder"
```

`oebv-db/.gitview.toml` (no version, numbered migrations):

```toml
[versioning]
schema = 1

[[versioning.source]]
kind    = "sequence"
glob    = "migrations/[0-9][0-9][0-9]_*.sql"
exclude = "*_rollback.sql"
role    = "boundary"
meaning = "each migration file is one DDL revision"
```

### 5.5 Normalized events

Every source is translated into the same event:

```python
@dataclass(frozen=True)
class VersionEvent:
    commit: str            # full SHA
    timestamp: str         # commit time, ISO 8601
    source: str            # "tag:v0.2.2", "file:app/core/version.py", …
    raw: str               # the version as the repository writes it
    label: str             # rendered label
    fields: Dict[str, int] # component values
    level: str             # highest role that changed: generation | boundary | step | label
    message: str = ""      # tag annotation, if any
```

Everything downstream — phases, sealing, records, rendering — reads events
only, never repository-specific formats.

### 5.6 Detection

`gitview versions detect` gathers evidence in this order and prints a draft
descriptor with the evidence as comments:

1. **Existing tool configuration** in the project file:
   `setuptools_scm` / `hatch-vcs` (versions are tags),
   `[tool.setuptools.dynamic] version.attr` (a file source),
   `[project] version`, `[tool.bumpversion]` (parse regex, serialize format,
   files), `[tool.commitizen]` (`tag_format`, `version_files`).
2. **Tags**, with a pattern inferred from their names.
3. **Conventional version files** (`__version__`, `VERSION`, `version.py`,
   `package.json`, `Cargo.toml`).
4. **Frequently changing version-like fields** in any file.
5. **Numbered migration directories.**
6. **Changelog headings.**

Tools describe *where* a version lives and *how* to parse it; none describe
which component is structural. Roles are therefore suggested, never confirmed:

```toml
# app/core/version.py changed in 32 commits; values changed in 26.
# MINOR changed in 14 commits, 12 of which also touch *.sql → looks like a boundary.
# Docstring: "MINOR — the versioned schema's revision. Every migration bumps it."
fields.MINOR = { role = "?", suggested = "boundary" }
```

Detection lists one primary source first and tags second as a cross-check;
other candidates are written as commented-out alternatives. A version file is
primary (it records every bump, tags usually only some), unless the project
derives its version from tags (setuptools-scm, hatch-vcs). A candidate file
that does not parse at the branch tip (e.g. a `__version__ = "unknown"`
fallback) is dropped with a note.

`--write` appends the table as text to the existing project file (formatting
and comments untouched) or creates `.gitview.toml`. It refuses when a
descriptor already exists.

*Not in milestone 1:* changelog headings (step 6) and scanning arbitrary files
for frequently changing version-like fields (step 4); `detect` covers project
files, tags, `version`-named Python files, `__init__.py`/`__about__.py`,
`VERSION` files and numbered migration directories.

### 5.7 Check

`gitview versions check` replays the descriptor over the full history and
fails on:

- an unresolved `role = "?"`;
- a value that cannot be parsed;
- two fields sharing the `generation`, `boundary` or `step` role;
- no field with a sealing role;
- a component that decreases without a higher-level component increasing;
- sources that disagree at a commit where one of them changes.

Tags off the branch and sources that yield nothing are warnings. While roles
are unresolved, phases are still previewed using the `suggested` roles.

It prints the resulting phases. Phased commands refuse to run until `check`
passes.

## 6. Phases

- A phase starts at a `generation` or `boundary` event and ends at the commit
  before the next one. Ranges are git ranges: `end(P-1)..end(P)`, where
  `end(P)` is the first parent of the next boundary commit.
- **Sealed:** a later boundary exists. A sealed phase's range never changes.
- **Open:** the phase after the last boundary, ending at the branch tip. Its
  record is provisional and replaced on each run until it is sealed.
- **Unversioned:** commits before the first event form one sealed phase, so
  old history is not dropped.
- **Moved or deleted tags:** each record stores the commit its boundary
  pointed at. If the descriptor now yields a different commit, GitView warns
  and keeps the stored record.
- Phase id = the boundary label (`6.30`, `0.2`). Step events inside a phase
  are listed in its detail index.

This replaces the heuristic phase chunking for phased history. The existing
chunkers remain for `analyze` until milestone 5.

## 7. Phase records

One JSON file per phase, `.gitview/phases/<phase-id>.json`, committed with the
repository (see open question 1). Values below are illustrative:

```json
{
  "schema": "gitview.phase/1",
  "phase": {
    "id": "6.30", "level": "boundary", "sealed": true,
    "start": "<sha>", "end": "<sha>",
    "from": "2026-09-02", "to": "2026-09-13",
    "version": {"raw": "0.6.30.0", "source": "file:app/core/version.py", "message": ""},
    "steps": [{"label": "0.6.30.1", "commit": "…"}]
  },

  "history": {
    "generator": {"gitview": "0.8.0", "ledger_schema": 2},
    "commits": 14, "merges": 2, "prs": [91, 92],
    "authors": [["Carsten Bund", 12]],
    "hot_files": [["app/services/pricing.py", 6]],
    "cochange": [["app/services/pricing.py", "tests/test_pricing.py", 5]],
    "storylines": {"started": [], "continued": [], "completed": [], "stalled": []}
  },

  "structure": {
    "provider": "graphify", "provider_version": "0.4.2", "observed_sha": "<end sha>",
    "files": 212,
    "communities": [{"name": "Pricing", "files": 18}],
    "hubs": [["app/core/db.py", 41]],
    "diff_vs_previous": {"new_files": 6, "removed_files": 1,
                         "new_communities": [], "merged": [], "split": []},
    "hidden_coupling": [["app/services/pricing.py", "app/routes/cart.py"]]
  },

  "summary": {
    "generator": {"gitview": "0.8.0", "backend": "anthropic",
                  "model": "claude-sonnet-5", "prompt_version": 1},
    "headline": "Legacy individual rebates projected into customer prices",
    "claims": [
      {"id": "rebates", "kind": "change", "area": "app/services/pricing",
       "text": "klantprijs/klantkorting from ado.klantrebate now populate product_price customer rows",
       "evidence": {"commits": ["<sha>"], "prs": [], "files": ["app/services/pricing.py"]}}
    ],
    "open_threads": []
  }
}
```

### 7.1 The three parts

| Part | Source | Recomputable | Written |
|---|---|---|---|
| `history` | Ledger | Always, from git | At seal; refreshed while open |
| `structure` | Structural snapshot at `end` | Only by re-running the analyser on that commit | At seal; `null` if no analyser ran |
| `summary` | LLM over `history` + `structure` + commit/PR text | Only by paying again | At seal; provisional while open |

`history` is stored although it is recomputable, so a record is readable
without the ledger and diffs meaningfully in review. `structure` must be
stored: the full snapshot stays in `graph.sqlite`, but the digest in the
record is what survives a deleted `.gitview/` cache.

Hidden coupling — files that co-change in `history` without a structural edge
in `structure` — is computed per phase, because only there do both parts
describe the same range.

### 7.2 Evidence validation

When a summary is written, every commit, PR and file it cites must exist in
the phase's range in the ledger. A claim citing unknown evidence is rejected
and reported, never stored.

### 7.3 Structure capture and backfill

- **Forward:** sealing a phase runs the structural provider at `end` (or uses
  an existing observation of that commit) and stores the digest.
- **Backfill (optional, explicit):** `gitview summarize --backfill-structure`
  checks out each sealed phase's `end` in a temporary worktree, runs the
  provider once, and fills `structure` in records where it is `null`.

### 7.4 Roll-ups

- **Chapter record** per `generation` (`.gitview/phases/generation-6.json`):
  summary over its phase records only. Regenerated when a phase in it is
  sealed.
- **Repository record** (`.gitview/phases/_summary.json`): summary over
  chapter and phase headlines. Regenerated when a phase is sealed.

Roll-ups read records, never commits, so their cost does not grow with
history length.

## 8. Views

Rendered by `gitview render`, no LLM, into `docs/history/` by default:

```
docs/history/
  SUMMARY.md            where we are, major arcs, open threads → links
  phases/INDEX.md       one line per phase: id · dates · headline
  phases/6.30.md        claims → detail index (commits, PRs, files, steps)
  storylines/<slug>.md  one page per initiative → phases and commits
  reference/hotspots.md hot files and co-change hints
```

- Claims come from `summary`; numbers, lists and the detail index come from
  `history`, `structure` and the ledger. A claim's evidence ids become links
  (commit/PR URLs when a remote is known, anchors otherwise).
- Descriptor `meaning` texts are rendered so a reader learns what "6.30" means.
- The version file(s) named by the descriptor are bookkeeping: excluded from
  hot files and co-change pairs.
- **Step report:** `gitview render --from 6.24 --to 6.30` (phase ids, tags or
  dates). Whole phases are composed from their records at no cost; a partial
  phase at either edge is summarized once and cached under its range.
- Without records, `render` produces the mechanical view — this is what
  `brief` becomes.
- `history_story.md` remains as the inventory/final-review export of `analyze`.

## 9. Rebuild and version rules

GitView applies the same rule to its own data as it asks of repositories:

| Change in GitView | Effect |
|---|---|
| Code only | Nothing is recomputed. Records keep the generator version that wrote them. |
| Ledger schema or projection (DDL) | Ledger rebuilt from git automatically; structural snapshots kept (#88). Records untouched. |
| Record schema (DDL) | The affected part is migrated. Where migration is impossible, records are flagged and regenerated only on request, phase by phase. |
| New prompt version | Nothing automatic. `summarize --phase 6.29 --force` or `--older-than-prompt 2` regenerate selected phases. |

Historical phases are never redone as a side effect of upgrading GitView.

## 10. Commands

| Command | LLM | Purpose |
|---|---|---|
| `gitview versions detect [--write]` | no | Draft a descriptor with evidence |
| `gitview versions check` | no | Validate the descriptor against history |
| `gitview versions [--json]` | no | List phases: id, range, dates, commits, sealed |
| `gitview summarize [--phase ID] [--force] [--backfill-structure]` | yes | Write records for phases without one; refresh the open phase |
| `gitview render [--from A --to B] [-o DIR]` | no | Write the views |

`graph`, `observe` and `motifs` are unchanged.

## 11. Existing defects to fix along the way

Found while reviewing the incremental path; each is fixed in the milestone
that touches the code.

| Defect | Where | Milestone |
|---|---|---|
| Incremental `analyze` overwrites `repo_history.jsonl` with only the new commits | `commands/analyze.py:577` | 2 |
| Storyline state re-applies every old phase on each run (`process_incremental` unused) | `commands/analyze.py:1066` | 4 |
| `analyze` keeps its own history copy instead of the ledger | `commands/analyze.py` | 2 |
| Storylines section listed in the table of contents but never written | `writer.py:65,102` | 5 |
| Version bookkeeping files reported as hotspots | `commands/brief.py`, `graph/analysis/stats.py` | 2 |
| Heuristic storyline titles ("app changes") | `storyline/detector.py:717` | 4 |

## 12. Milestones

1. **Versions** *(implemented)*. Descriptor lookup and loading, source translators (`tag`,
   `file`, `sequence`), `VersionEvent`, phase derivation, `versions detect`,
   `versions check`, `versions`. Verified on `drm_screen` (tags), `oebv-api`
   (fields) and `oebv-db` (sequence).
2. **History records and rendering.** `history` part from the ledger, sealing,
   open phase, `render` without summaries (replaces `brief`). Ledger becomes
   the single history source for `analyze`.
3. **Structure capture.** Digest at seal, diff against the previous phase,
   per-phase hidden coupling, optional backfill.
4. **Summaries.** `summarize`, evidence validation, generator versions,
   storyline lifecycle from records and commit trailers.
5. **Roll-ups and step reports.** Chapter and repository records,
   `render --from/--to`, partial-phase cache; `history_story.md` becomes an
   export.

## 13. Open questions

1. **Committing records.** Proposed: commit `.gitview/phases/` (the expensive,
   non-recomputable data) and keep `graph.sqlite` and `history.jsonl` ignored.
   Rendered docs: committed or generated on demand?
2. **Cross-repository alignment.** `oebv-db` migration numbers do not map
   one-to-one to schema revisions. Give `oebv-db` its own version, or let a
   descriptor reference another repository's component ("my boundary is
   oebv-api's MINOR")? The second is the path to a system-level history of
   `oebv-system`.
3. **Very small phases.** A boundary with two commits: its own record with a
   one-line summary, or summarized together with the next phase while keeping
   separate ids?
4. **Unit of analysis for superprojects.** Per submodule with links, or one
   system history aligned on shared boundaries?
