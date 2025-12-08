# GitView Configuration Examples

These example command lines show how to combine GitView CLI options for different review goals. Adjust repository paths and output directories as needed.

## 1) Initial examination for an executive review
Use adaptive chunking with the hierarchical summarization defaults for balanced coverage, keep costs predictable, and write polished summaries.

```bash
gitview analyze \
  --repo /path/to/repo \
  --output ./reports/exec-initial \
  --strategy adaptive \
  --backend openai \
  --model gpt-4o-mini \
  --max-commits 400 \
  --repo-name "Project Exec Brief" \
  --github-token "$GITHUB_TOKEN"
```

**Why this works**
- Adaptive chunking surfaces major shifts automatically while keeping phases readable.
- GitHub enrichment adds PR titles/descriptions for better “why” context.
- Limiting commits to a few hundred keeps turnaround and API spend predictable for a first pass.

## 2) “Changes since last month” snapshot
Combine incremental mode with a date boundary to only process new work and keep prior summaries.

```bash
gitview analyze \
  --repo /path/to/repo \
  --output ./reports/monthly \
  --incremental \
  --since-date 2025-01-01 \
  --strategy time --period month \
  --backend openai \
  --model gpt-4o-mini
```

**Why this works**
- `--since-date` restricts extraction to recent commits, while `--incremental` reuses cached summaries from earlier runs.
- Time-based chunking with `--period month` keeps the summary aligned to the requested window.
- Leaving `--github-token` optional avoids failing when you only need commit-level changes.

## 3) Significant changes in the last year
Favor significance-aware chunking and critical examination to spotlight high-impact shifts and gaps.

```bash
gitview analyze \
  --repo /path/to/repo \
  --output ./reports/yearly-critical \
  --strategy adaptive \
  --since-date 2024-01-01 \
  --critical \
  --todo ROADMAP.md \
  --directives "Highlight architectural shifts and large deletions" \
  --backend anthropic \
  --model claude-haiku-3-5-20241022
```

**Why this works**
- Adaptive chunking plus the hierarchical summarizer emphasizes large deltas and rewrites.
- Critical mode, paired with a goals file and directives, frames the narrative around impact and gaps.
- Using a budget Anthropic model balances tone quality with cost for long-range reviews.

## 4) Most active contributor by significance
Use per-branch analysis and GitHub enrichment to attribute large changes to authors and reviewers.

```bash
gitview analyze \
  --repo /path/to/repo \
  --output ./reports/contributors \
  --strategy adaptive \
  --github-token "$GITHUB_TOKEN" \
  --branch main \
  --max-commits 800
```

Follow with a filtered timeline or JSON post-processing:

- Inspect `output/repo_history.jsonl` to aggregate `author_name` and `loc_added/loc_deleted`.
- Use the enriched PR metadata (reviewers, labels) to weight significance by review volume or label types.

## Known limitations and workarounds
- There is no built-in `--until-date` or `--author` filter; narrowing to specific periods or people requires combining `--since-date` with manual post-processing of `repo_history.jsonl` or a temporary branch that pre-filters commits.
- Contributor significance scores are not computed automatically; you must aggregate LOC deltas or PR metadata yourself after extraction.
- Time-based chunking (`--strategy time --period ...`) sets period boundaries but does not stop extraction outside that range unless paired with `--since-date`.
- Hierarchical summaries focus on semantic clusters, not quantitative rankings; for strict metrics, rely on the JSON outputs alongside the narrative reports.
