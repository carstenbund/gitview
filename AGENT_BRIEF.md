<!-- gitview:brief head=9b48dae0842a2227fd49a6e2d39201287452b439 generated=2026-07-16T06:04:26+00:00 commits=91 gitview_version=0.6.2 -->
# Agent Brief — gitview

> Auto-generated, token-efficient history digest for AI coding agents.
> Read this once at the start of a session instead of re-deriving project
> history from `git log` and file exploration.
>
> Regenerate after new commits: `gitview brief` (skips automatically if
> already up to date; add `--force` to always regenerate). Check freshness
> only, no write: `gitview brief --check`.

**Generated:** 2026-07-16T06:04:26+00:00 at commit `9b48dae0` (91 commits analyzed)

For a hand-maintained architecture map and known-gaps list, see [AGENTS.md](AGENTS.md).

---

## Project

- **Name:** gitview
- **Description:** Git history analyzer with LLM-powered narrative generation

## At a Glance

| Metric | Value |
|---|---|
| Commits analyzed | 91 (⚠ shallow clone — full history is longer) |
| Contributors | 2 |
| First commit | 2025-12-08 |
| Last commit | 2026-03-25 |
| Phases | 12 |

**Top contributors:** Carsten Bund (60), Claude (31)

**Language mix (at HEAD):** Python 67%, Markdown 21%, Other 6%, Shell 4%, YAML 1%

## Timeline

| Phase | Period | Commits | LOC Δ | Highlight |
|---|---|---|---|---|
| 1 | 2025-12-08 → 2025-12-08 | 5 | +221 | Bump version to 0.2.2 |
| 2 | 2025-12-08 → 2025-12-08 | 5 | +354 | Merge pull request #32 from carstenbund/claude/centralize-version-mana… |
| 3 | 2025-12-08 → 2025-12-08 | 5 | +262 | Merge branch 'main' into codex/create-example-configurations-for-repo-… |
| 4 | 2025-12-08 → 2025-12-08 | 6 | +25 | Merge pull request #37 from carstenbund/codex/fix-cached-run-error-on-… |
| 5 | 2026-01-18 → 2026-01-22 | 5 | +4,910 | Implement Phase 1: File history tracker core functionality |
| 6 | 2026-01-22 → 2026-01-24 | 9 | +1,738 | Merge pull request #48 from carstenbund/claude/track-json-file-changes… |
| 7 | 2026-01-24 → 2026-01-25 | 16 | +2,436 | Bump version to 0.4.1 |
| 8 | 2026-01-25 → 2026-01-25 | 5 | +7,761 | Add detailed storyline architecture implementation plan |
| 9 | 2026-01-25 → 2026-01-25 | 5 | +2,390 | Merge pull request #58 from carstenbund/claude/review-narrative-strate… |
| 10 | 2026-01-25 → 2026-01-25 | 20 | +1,931 | docs: Add commit strategy guide for GitView-compatible tagging |
| 11 | 2026-02-05 → 2026-03-24 | 5 | +4,219 | feat(adaptive): add discovery-driven adaptive review agent |
| 12 | 2026-03-25 → 2026-03-25 | 5 | +71 | refactor(worklog): replace REST client with existing GraphQL infrastru… |

## Storylines

*Detected from PR labels/titles, file-change clusters, commit-message patterns, and this project's `Storyline:` commit trailer — no LLM used.*

### Storyline Index

#### Summary

- **Total Storylines:** 7
- **Completed:** 0
- **Active:** 5
- **Stalled:** 0
- **Abandoned:** 1

#### Active Storylines

| Title | Category | Started | Last Update | Confidence |
|-------|----------|---------|-------------|------------|
| GitHub worklog analyze mode | feature | Phase 11 | Phase 12 | 95% |
| Adaptive Review Agent | feature | Phase 11 | Phase 11 | 95% |
| GitView Documentation | documentation | Phase 10 | Phase 10 | 95% |
| Commit Strategy Alignment | documentation | Phase 10 | Phase 10 | 95% |
| Merge pull request #61 from carstenbund/claude/implement-optimization-proposal-XSbFH | feature | Phase 10 | Phase 10 | 60% |

#### Emerging Storylines (Unconfirmed)

- **Bump version from 0.1.3 to 0.2.2** (migration): Detected from 2 commits with shared theme


### Storyline Timeline

```
Storyline                                 1  2  3  4  5  6  7  8  9 10 11 12 
----------------------------------------------------------------------------
Bump version from 0.1.3 to 0.2.2          ┌─ ·  ·  ·  ·  · ─→                
Merge pull request #37 from carstenbun..           ┌─                        
Commit Strategy Alignment                                            ┌─      
GitView Documentation                                                ┌─      
Merge pull request #61 from carstenbun..                             ┌─      
Adaptive Review Agent                                                   ┌─   
GitHub worklog analyze mode                                             ┌──→ 
```

Legend: ┌─ start, ─┘ completed, ─→ ongoing, ─╳ stalled, · gap

## Most-Changed Files

| File | Commits touching it |
|---|---|
| `gitview/cli.py` | 36 |
| `gitview/__init__.py` | 14 |
| `gitview/commands/analyze.py` | 6 |
| `CLAUDE.md` | 6 |
| `gitview/commands/worklog.py` | 6 |
| `docs/json_tracker_architecture.md` | 5 |
| `gitview/storyline/__init__.py` | 5 |
| `tests/test_storyline.py` | 5 |
| `gitview/storyteller.py` | 4 |
| `gitview/chunker.py` | 4 |
| `AGENTS.md` | 4 |
| `gitview/file_tracker.py` | 4 |
| `README.md` | 4 |
| `gitview/writer.py` | 4 |
| `gitview/commands/__init__.py` | 4 |

## Limitations

This digest is fully mechanical — no LLM calls. Dates, counts, and storyline signals come directly from git metadata and commit/PR patterns; it does not explain *why* decisions were made. Run `gitview analyze` for full LLM-generated narrative prose.
