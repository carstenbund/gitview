# Changelog

All notable changes to GitView. The last published PyPI release is `0.1.3`
(2025-12-06); every version listed above it here was developed since then and
reaches PyPI together as **0.7.1**.

Versions follow [semantic versioning](https://semver.org/) loosely: the minor
number moves when a command or a pipeline stage is added.

## Unreleased

### Fixed

- **Rebuilding the repository graph no longer deletes structural
  observations.** Every rebuild (`graph --rebuild`, a history rewrite, a branch
  change, or the version bump that makes the graph rebuild once after an
  upgrade) dropped the `graphify` snapshots along with the history tables,
  without a message. They cannot be recreated from git, so the history-plus-
  structure motifs quietly disappeared. The history tables are now rebuilt on
  their own; `graph` reports how many observations were kept and warns about
  any whose commit is no longer in the history.

## 0.7.1

### Fixed

- **Line counts now match git's own arithmetic.** Merge commits carry no churn
  of their own (GitPython's `commit.stats` diffs a merge against its first
  parent and re-reported everything the merged branch had already committed,
  counting branch work twice), and renames are detected with
  `git diff-tree -M`, so a moved file is no longer a full deletion plus a full
  insertion. On a 182-commit repository the totals moved from 577k/418k lines
  to 442k/355k, which is what `git log --numstat -M` reports.
- **Caches carry an extraction version.** `EXTRACTION_VERSION` is written as a
  header line in every JSONL history; `analyze` re-extracts a history written
  by an older GitView instead of reusing its numbers, `.gitview/history.jsonl`
  is ignored when stale, and the repository graph rebuilds once.
- **Anthropic backend works with current models.** Text is collected from the
  response's text blocks (Sonnet 5 returns a thinking block first, which raised
  `AttributeError`), `temperature` is no longer sent to models that reject it
  (Sonnet 5, Opus 5, Opus 4.7/4.8, Fable, Mythos), and thinking is switched off
  for summarization on the models that allow it so the small `max_tokens` caps
  are spent on the summary.
- **The timeline is no longer truncated.** Its output grows with each phase and
  used to hit the output-token cap part-way through; it is now generated in
  batches of at most ten phases, and every section warns when the model stops
  at `max_tokens`. Storyline report bullets clip at a word boundary.
- **Motif results were non-deterministic.** The structural edge index was
  cached in a module-level dict keyed by `id(snapshot)`; once a snapshot was
  garbage collected its address could be reused, and a later snapshot inherited
  the freed one's edges, which made hidden coupling disappear. The index is
  cached per context now, on objects that context keeps alive.
- The phase cache was ignored because `load_phases` picked up `phase_index.json`;
  the story cache ignored `--directives`, `--critical` and `--todo`.
- Help text is ASCII, so `--help` renders on Windows cp1252 consoles.
- A missing API key is reported before extraction rather than after it.

### Added

- `analyze --regenerate-story` rebuilds the global narrative from the cached
  history and phase summaries, for applying new `--directives` or prompt
  changes without re-summarizing phases.

## 0.7.0

### Added

- **Evidence-first analysis.** `analyze` builds the repository graph and runs
  the motif catalogue before calling a model, then spends model calls only
  where the evidence justifies one. `--llm-budget full|balanced|minimal`
  (default `balanced`) writes routine phases from evidence in the same shape a
  model would produce; the technical-evolution and deletion sections are
  rendered from the graph; an *Architectural Motifs* section is added to the
  report. `--no-evidence` restores the previous behaviour. Every prompt that is
  still sent carries an evidence block and a block of hard facts from git
  (exact span, contributors, submodules, most changed files, version strings)
  with rules against inventing technologies, periods or motives.
- **Repository graph** (`gitview graph`): a persistent, incremental SQLite
  graph of commits, files, authors, PRs, commit→file edges and a repository-wide
  file co-change projection in `<repo>/.gitview/graph.sqlite`. Deterministic,
  no LLM, and rebuilt automatically when history is rewritten.
- **Structural evidence** (`gitview observe`): an optional, neutral model of
  what an external code analyser sees at one commit — `StructuralSnapshot`,
  `StructuralNode`, `StructuralEdge` — stored with provenance (provider,
  version, observed commit, content hash). Graphify is the first provider;
  nothing in the core names it, and every command works with no analyser
  installed. A git submodule is served from the superproject's graph.
- **Motifs** (`gitview motifs`): recurring historical and architectural
  patterns that declare the evidence they need and are skipped, with a reason,
  when it is missing. Historical: repeated co-change, ownership transition.
  With one structural observation: hidden coupling, confirmed coupling, stable
  interface. With observations at two or more commits: emerging dependency,
  architectural split, centrality growth.
- **Claude Code CLI backend** (`--backend claude-cli`): runs generation through
  a logged-in local `claude` CLI, billed to the Claude plan, with no separate
  `ANTHROPIC_API_KEY`.

## 0.6.x

### Added

- **Agent brief** (`gitview brief`): a compact, no-LLM project history digest
  meant to be committed and read once per session, as a token-efficient
  substitute for an agent re-deriving project history.
- **Work log** (`gitview worklog`): a GitHub-based work log across all branches
  for a date range, as Markdown or CSV, built on the existing GraphQL client.
- **Adaptive review agent** (`analyze --adaptive`): discovery-driven analysis
  that reacts to what it finds instead of following a fixed pipeline.
- Extraction caches its records per repository and extracts incrementally;
  diff stats are computed from numstat instead of full patch text, which keeps
  peak memory flat on large repositories.

## 0.5.x

### Added

- **Storyline tracking**: narrative threads (features, refactors, bug
  campaigns) detected from multiple signals — commit trailers, PR labels, file
  clusters — and carried across phases by a lifecycle state machine, with a
  `gitview storyline` command group and storyline sections in the report.
- The CLI was split into a `commands/` package, and analysis artifacts are
  cached between runs.

## 0.4.x

### Added

- **File history tracking** (`gitview track-files`, `gitview file-history`):
  detailed per-file change history with optional AI summaries and caching.
- **Header injection** (`gitview inject-history`, `gitview remove-history`):
  writes a file's change history into its header as comments, with
  multi-language comment syntax, and removes it again.
- **Branch comparison** (`gitview compare-branches`): divergence analysis
  between two branches.

## 0.2.x

### Added

- **Hierarchical summarization** (`--hierarchical`): commits are clustered by
  significance and summarized per cluster before the phase narrative, so detail
  survives on large repositories.
- Batched storytelling, context-window handling and input truncation for
  histories too large for one prompt.
- Incremental analysis that reuses cached artifacts, and cached remote clones.
- The version is defined once, in `gitview/__init__.py`.

## 0.1.x

Initial releases: history extraction to JSONL, phase chunking (adaptive, fixed,
time), LLM phase summaries, global story generation, markdown/JSON/timeline
output, GitHub PR and review enrichment, critical examination mode, and the
Anthropic, OpenAI and Ollama backends.
