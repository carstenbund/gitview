This project is about analyzing git projects. The gitview/cli.py orchestrates a series of actions: 

 - 1. retrieval of the git in question,
 - 2. read and chunk it up in different phases (based on preference)
 - 3. send each phase to a LLM for evaluation
 - 4. summarize the phases
 - 5. create a report

For commit stats, a phase timeline, and detected storylines, read
**[AGENT_BRIEF.md](AGENT_BRIEF.md)** first — it's regenerated mechanically
(no LLM) via `gitview brief` and is usually cheaper to read than re-deriving
the same facts from `git log`. Regenerate it after a batch of commits with
`gitview brief` (auto-skips if already fresh; `--force` to always rebuild).
The architecture map and proposal-status notes below are hand-maintained and
won't be touched by that command — update them yourself when they go stale.

---

## Architecture Map

**Pipeline commands** (`gitview/commands/`, wired up in `gitview/cli.py`):
- `analyze` — the full pipeline: extract → (optional GitHub enrich) → chunk →
  LLM-summarize → track storylines → generate narrative → write
  `history_story.md`/`history_data.json`/`timeline.md`. Supports
  `--critical` (gap-focused mode), `--hierarchical` (richer multi-level LLM
  summaries), `--adaptive` (discovery-driven agent mode), `--skip-llm`.
- `brief` — deterministic, no-LLM agent digest (this file's companion,
  `AGENT_BRIEF.md`). See `gitview/commands/brief.py`.
- `extract` / `chunk` — standalone, no-LLM steps of the analyze pipeline.
- `track-files` / `file-history` / `inject-history` / `remove-history` —
  per-file change history, optionally AI-summarized and cached, injectable
  as source header comments.
- `compare-branches` — file-history divergence analysis between two branches.
- `worklog` — GitHub-based work log (billing/reporting) across all branches,
  via the GraphQL client.
- `storyline list/show/report/timeline/export` — read `output/phases/storylines.json`
  (written by `analyze`) and render it in different ways.

**Core analysis (no LLM):**
- `extractor.py` — `GitHistoryExtractor`/`CommitRecord`: walks git log,
  extracts LOC/language/README/comment metadata per commit.
- `chunker.py` — `HistoryChunker`/`Phase`: splits commits into phases
  (`adaptive`/`fixed`/`time` strategies).
- `significance_analyzer.py` — `SignificanceAnalyzer`/`CommitCluster`: groups
  commits by type (feature/bugfix/refactor/docs/infra) and picks the most
  representative commit per group. Feeds the hierarchical summarizer and
  `gitview brief`'s phase highlights.
- `cache.py` — `CacheManager`: commit-SHA-keyed freshness checks for
  `output/`. **Gap:** built but not actually wired into `analyze.py`, which
  still reimplements similar logic ad hoc (see Proposal Docs Status below).

**LLM narrative generation** (requires `--backend`/API key or Ollama):
- `backends/{anthropic,openai,ollama}_backend.py` + `router.py` — pluggable
  LLM backends.
- `summarizer.py` — `PhaseSummarizer`, the simple (non-hierarchical) per-phase
  summary strategy.
- `hierarchical_summarizer.py` / `hierarchical_storyteller.py` — richer
  multi-level summarization (`--hierarchical`).
- `storyteller.py` — combines phase summaries into the global narrative
  (executive summary, timeline, technical retrospective, deletion story).
- `writer.py` / `index_writer.py` — writes the markdown/JSON reports and
  multi-branch indexes.

**Storyline tracking** (`storyline/`, mostly non-LLM):
- `models.py` — `Storyline`/`StorylineSignal`/`StorylineDatabase` dataclasses,
  `StorylineCategory`/`StorylineStatus` enums.
- `detector.py` — multi-signal detectors, highest to lowest confidence: this
  project's own `Storyline: [status:category] Title` commit trailer (0.95),
  PR labels (0.9), PR title patterns (0.8), file-change clusters (0.7),
  commit-message keywords (0.6). All deterministic/non-LLM.
- `state_machine.py` — lifecycle transitions (emerging → active/progressing →
  stalled/completed/abandoned).
- `tracker.py` — `StorylineTracker`: orchestrates detection + state machine +
  persistence to `output/phases/storylines.json`.
- `reporter.py` — `StorylineReporter`: renders the tracked database as
  markdown (index, ASCII timeline, cross-phase themes) — reused directly by
  `gitview brief`.
- `parser.py` / `extractor.py` (the storyline one) — parses storylines OUT
  of LLM phase-summary text (the lowest-confidence, 0.5, "LLM extraction"
  signal source).

**GitHub integration:**
- `github_graphql.py` — GraphQL client (PRs, reviews, commits), 24h local
  cache at `~/.gitview/cache/github`.
- `github_enricher.py` — attaches PR/review context to `CommitRecord.github_context`.
- `remote.py` — resolves/clones `org/repo` or full-URL repo specs.
- `branches.py` / `branch_comparator.py` — multi-branch listing and
  file-history divergence comparison.

**Adaptive agent** (`adaptive/`): `AdaptiveReviewAgent` — discovery-driven
mode (`analyze --adaptive`) that reacts to findings (security concerns,
breaking changes, large diffs) instead of following the fixed 7-step pipeline.

## Proposal Docs Status

Root-level `*_PROPOSAL.md`/`*_INVESTIGATION.md`/`SOLUTION_SUMMARY.md` and
`docs/*.md` are design docs, not necessarily current state. As of this
writing:

| Doc | Status |
|---|---|
| `GRAPHQL_INVESTIGATION.md` | Implemented — `github_graphql.py`/`github_enricher.py` power `analyze --github-token` and `worklog` |
| `FILE_HISTORY_PROPOSAL.md` | Implemented — `track-files`/`file-history`/`inject-history`/`compare-branches` |
| `docs/STORYLINE_IMPLEMENTATION_PLAN.md`, `docs/HIERARCHICAL_STRATEGY.md` | Implemented — `storyline/` package, `hierarchical_summarizer.py`/`hierarchical_storyteller.py` |
| `OPTIMIZATION_PROPOSAL.md` | Partially implemented — `CacheManager` (`cache.py`) was built but is not wired into `analyze.py`; `history_data.json` still embeds full phase objects rather than references |
| `JSON_TRACKER_PROPOSAL.md` / `docs/json_tracker_architecture.md` | Not implemented — proposes tracking JSON *config file* diffs specifically; no `json_tracker.py` exists |

---

# use GitView Commit Message Format
```
<type>(<scope>): <subject>

<body>

<footer>

Storyline: [status:category] Initiative Title
```

## Storyline Categories

GitView recognizes these development categories:

| Category | Description | Trigger Keywords |
|----------|-------------|------------------|
| `feature` | New functionality | add, implement, new, feature, create |
| `refactor` | Code restructuring | refactor, restructure, reorganize, cleanup, modularize |
| `bugfix` | Bug fixes | fix, bug, issue, error, crash, problem |
| `debt` | Technical debt | debt, technical debt, legacy, deprecate |
| `infrastructure` | Build/deploy systems | infra, ci/cd, build, deploy, pipeline, config |
| `docs` | Documentation | doc, documentation, readme, comment, wiki |
| `test` | Testing | test, spec, coverage, unit test |

---

## Storyline Statuses

Track initiative lifecycle with these statuses:

| Status | Meaning | When to Use |
|--------|---------|-------------|
| `new` | Starting a new initiative | First commit of a feature/effort |
| `continued` | Work in progress | Ongoing development |
| `completed` | Initiative finished | Final commit completing the work |
| `stalled` | Work paused/blocked | When stopping temporarily |

---

Quick Reference:
Status: new, continued, completed, stalled
Category: feature, refactor, bugfix, debt, infrastructure, docs, test
Key Rules:
Keep storyline titles consistent across commits
One initiative per commit
Mark completions explicitly
Group related changes together
