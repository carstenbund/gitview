# GitView Overview

GitView is a git history analyzer that turns repository activity into a narrative report, using LLMs to explain how a codebase evolved. It extracts commit data, chunks history into phases, summarizes each phase, and then assembles executive summaries, timelines, and deletion stories for stakeholders. Critical mode shifts the tone to focus on gaps and technical debt.

## Core Capabilities
- Extract commit metadata, file stats, language mix, README snapshots, and comment signals.
- Chunk history with adaptive, fixed-size, or time-based strategies to identify meaningful phases.
- Summarize phases and generate combined reports such as executive summaries, timelines, technical retrospectives, and deletion analyses.
- Enrich narratives with GitHub PR context (titles, descriptions, review comments, labels, reviewers) when provided with a token.
- Support multiple LLM backends: Anthropic Claude (default), OpenAI, or local Ollama.

## Typical Workflows
- **Full pipeline:** `gitview analyze [--repo PATH] [--strategy adaptive|fixed|time] [--backend anthropic|openai|ollama]`
- **Extraction only:** `gitview extract --repo /path/to/repo --output history.jsonl`
- **Chunking only:** `gitview chunk history.jsonl --strategy adaptive --output ./phases`
- **Critical assessment:** add `--critical` with optional `--todo GOALS.md` and `--directives "Focus area"` to emphasize risks and alignment gaps.

## Installation Options
- PyPI install for the `gitview` CLI: `pip3 install gitview`.
- Editable install from source: `pip install -e .` after cloning.
- Run directly from the repo via `./bin/gitview` or `python -m gitview.cli` after installing dependencies.

## Output Snapshot
A typical run writes to `output/` with `repo_history.jsonl`, phased JSON files, and reports such as `history_story.md` and `timeline.md`, capturing the full narrative of repository evolution.
