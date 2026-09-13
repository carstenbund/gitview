"""Deterministic evidence for ``analyze``: what GitView knows *before* asking an LLM.

The repository graph, the significance clusters and the motif catalogue are
all computed without a model. This module turns them into three things the
analyze pipeline can use to spend fewer LLM calls:

* a **signal score** per phase, so routine phases (docs churn, config bumps,
  a handful of small commits with no motifs or PR narrative) get a
  deterministic summary instead of a model call;
* an **evidence block** injected into the prompts that *are* sent, so the
  model writes from established facts (clusters, hot files, coupling,
  motifs) rather than re-deriving them from a commit dump;
* **pre-rendered story sections** (technical evolution, deletions,
  architectural motifs) that are fact listings by nature and need no prose
  model at all.

Everything here is safe to run on every ``analyze``; it only reads git
history that has already been extracted.
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from .chunker import Phase
from .extractor import CommitRecord
from .graph import GraphStore, GraphUpdater, SyncResult, compute_graph_stats
from .graph.models import GraphStats
from .motifs import MotifFinding, MotifReport, Thresholds, run_motifs
from .significance_analyzer import CommitCluster, SignificanceAnalyzer

# --------------------------------------------------------------------------- budget

#: Phases scoring below the budget's threshold are summarized deterministically.
LLM_BUDGETS: Dict[str, float] = {
    'full': float('-inf'),  # never skip a phase (pre-evidence behaviour), sections still pre-rendered
    'balanced': 0.35,       # skip routine phases
    'minimal': 0.6,         # LLM only for phases with strong signals
}

_CATEGORY = {
    'feature': 'feature', 'bugfix': 'bugfix', 'refactor': 'refactor', 'docs': 'docs',
    'infrastructure': 'infrastructure', 'general': 'feature',
}


def llm_threshold(budget: str) -> float:
    try:
        return LLM_BUDGETS[budget]
    except KeyError:
        raise ValueError(f"unknown LLM budget '{budget}' (known: {', '.join(LLM_BUDGETS)})")


# --------------------------------------------------------------------------- models

@dataclass
class ClusterEvidence:
    type: str
    commits: int
    insertions: int
    deletions: int
    key_hash: str
    key_subject: str
    pr_title: Optional[str]
    labels: List[str]
    top_files: List[str]
    subjects: List[str]                     # a few commit subjects, for small clusters

    @property
    def headline(self) -> str:
        return self.pr_title or self.key_subject

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type, 'commits': self.commits, 'insertions': self.insertions,
            'deletions': self.deletions, 'key_commit': self.key_hash, 'headline': self.headline,
            'labels': self.labels, 'top_files': self.top_files,
        }


@dataclass
class PhaseEvidence:
    phase_number: int
    start_date: str
    end_date: str
    commit_count: int
    authors: List[str]
    primary_author: str
    loc_delta: int
    insertions: int
    deletions: int
    top_files: List[Tuple[str, int]]                       # (path, commits in phase)
    clusters: List[ClusterEvidence]
    coupling: List[Tuple[str, str, int, int]]              # (a, b, in-phase co-changes, all-time)
    motifs: List[MotifFinding]
    significant_commits: int
    pr_narratives: int
    score: float
    reasons: List[str] = field(default_factory=list)

    # ------------------------------------------------------------ prompt block

    def prompt_block(self) -> str:
        """Compact, factual context for an LLM prompt (a few hundred tokens)."""
        lines = ["**Established Evidence (deterministic, from the repository graph):**"]
        if self.clusters:
            lines.append("Activity clusters:")
            for c in self.clusters[:6]:
                files = ', '.join(c.top_files[:3])
                lines.append(f"- {c.type}: {c.commits} commit(s), +{c.insertions}/-{c.deletions}; "
                             f"\"{_short(c.headline, 90)}\"" + (f" [{files}]" if files else ''))
        if self.top_files:
            lines.append("Most touched files: " + ', '.join(f"{p} ({n})" for p, n in self.top_files[:6]))
        if self.coupling:
            lines.append("Files changed together in this phase: " + '; '.join(
                f"{a} + {b} ({n}x here, {t}x overall)" for a, b, n, t in self.coupling[:5]))
        if self.motifs:
            lines.append("Motifs involving these files:")
            for m in self.motifs[:6]:
                lines.append(f"- {m.title}: {', '.join(m.files)} — {_short(m.summary, 140)}")
        lines.append("Use these facts as given; do not restate them as discoveries.")
        return '\n'.join(lines)

    # ------------------------------------------------------------ deterministic summary

    def storylines(self) -> List[Tuple[str, str, str]]:
        """``(slug, category, description)`` per cluster, stable across phases."""
        out, seen = [], set()
        for c in self.clusters:
            category = _CATEGORY.get(c.type, 'feature')
            slug = f"{category}-{_area(c.top_files) or 'core'}"
            if slug in seen:
                continue
            seen.add(slug)
            out.append((slug, category, _short(c.headline, 120)))
        return out

    def render_summary(self, prior_slugs: Set[str]) -> str:
        """A factual phase summary in the shape the LLM would have produced.

        Ends with a ``## Storylines`` section in the exact format the
        storyline parser expects, so tracking works unchanged.
        """
        who = self.primary_author
        if len(self.authors) > 1:
            who += f" and {len(self.authors) - 1} other(s)"
        paragraphs = [
            f"Between {self.start_date} and {self.end_date}, {self.commit_count} commit(s) by {who} "
            f"changed the code base by {self.loc_delta:+,d} lines (+{self.insertions:,}/-{self.deletions:,})."
        ]
        if self.top_files:
            paragraphs[0] += " Work concentrated in " + ', '.join(
                f"{p}" for p, _ in self.top_files[:3]) + "."

        if self.clusters:
            parts = []
            for c in self.clusters[:5]:
                parts.append(f"{c.type} work ({c.commits} commit(s), +{c.insertions}/-{c.deletions}) "
                             f"centred on \"{_short(c.headline, 100)}\"")
            paragraphs.append("Activity grouped as " + '; '.join(parts) + ".")

        facts = []
        for a, b, n, t in self.coupling[:3]:
            facts.append(f"{a} and {b} changed together {n} time(s) in this phase ({t} overall)")
        for m in self.motifs[:3]:
            facts.append(f"{m.title.lower()}: {', '.join(m.files)} ({_short(m.summary, 120)})")
        if facts:
            paragraphs.append("Established patterns: " + '; '.join(facts) + ".")

        paragraphs.append(f"*Summary derived from repository evidence without an LLM call "
                          f"(signal score {self.score:.2f}: {', '.join(self.reasons) or 'routine activity'}).*")

        lines = ["## Storylines"]
        for slug, category, description in self.storylines():
            status = 'CONTINUED' if slug in prior_slugs else 'NEW'
            lines.append(f"- [{status}:{category}] {slug}: {description}")
        if len(lines) == 1:
            lines.append(f"- [CONTINUED:feature] general-maintenance: {self.commit_count} routine commit(s)")

        return '\n\n'.join(paragraphs) + '\n\n' + '\n'.join(lines) + '\n'

    def hierarchical_result(self, summary: str, loc_delta_percent: float) -> Dict[str, Any]:
        """The record ``HierarchicalStoryTeller`` expects, built from evidence alone."""
        by_type: Dict[str, List[ClusterEvidence]] = {}
        for c in self.clusters:
            by_type.setdefault(c.type, []).append(c)
        highlights = []
        for kind, group in by_type.items():
            main = max(group, key=lambda c: c.insertions + c.deletions)
            h = {'type': kind, 'count': sum(c.commits for c in group),
                 'summary': f"{main.commits} commit(s): {_short(main.headline, 120)}"}
            if main.pr_title:
                h['pr_title'] = main.pr_title
            highlights.append(h)
        return {
            'full_summary': summary,
            'clusters': [{'type': c.type, 'summary': _short(c.headline, 160), 'key_commit': c.key_subject,
                          'commit_count': c.commits, 'insertions': c.insertions, 'deletions': c.deletions}
                         for c in self.clusters],
            'timeline_context': {
                'highlights': highlights, 'cluster_count': len(self.clusters),
                'total_commits': self.commit_count, 'loc_delta': self.loc_delta,
                'loc_delta_percent': loc_delta_percent,
            },
            'cluster_details': [],
            'evidence': self.to_dict(),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            'phase_number': self.phase_number, 'score': round(self.score, 3), 'reasons': self.reasons,
            'top_files': self.top_files, 'clusters': [c.to_dict() for c in self.clusters],
            'coupling': self.coupling, 'motifs': [m.to_dict() for m in self.motifs],
            'significant_commits': self.significant_commits, 'pr_narratives': self.pr_narratives,
        }


# --------------------------------------------------------------------------- ledger

class EvidenceLedger:
    """Everything deterministic that ``analyze`` can consult for one run.

    Builds or updates the repository graph from the records already
    extracted (no second ``git log``), runs the motif catalogue once and
    answers per-phase and repository-wide questions from memory.
    """

    def __init__(
        self,
        repo_path: Union[str, Path],
        records: Sequence[CommitRecord],
        *,
        branch: str = 'HEAD',
        store_path: Optional[Union[str, Path]] = None,
        thresholds: Optional[Thresholds] = None,
        top: int = 15,
    ) -> None:
        self.repo_path = Path(repo_path)
        self.records = list(records)
        updater = GraphUpdater(self.repo_path, branch=branch, store_path=store_path)
        self.sync: SyncResult = updater.sync(records_loader=lambda: self.records)
        self.store_path = updater.store_path
        with GraphStore(self.store_path) as store:
            store.initialize()
            self.report: MotifReport = run_motifs(store, thresholds=thresholds)
            self.stats: GraphStats = compute_graph_stats(store, top=top)
            self._all_time: Dict[Tuple[str, str], int] = {
                (a, b): count for a, b, count, _, _ in store.all_edges()}
        self._by_file: Dict[str, List[MotifFinding]] = defaultdict(list)
        for f in self.report.findings:
            for path in f.files:
                self._by_file[path].append(f)
        self._analyzer = SignificanceAnalyzer()
        self._cache: Dict[int, PhaseEvidence] = {}

    # ------------------------------------------------------------ per phase

    def phase_evidence(self, phase: Phase) -> PhaseEvidence:
        key = id(phase)
        if key in self._cache:
            return self._cache[key]

        commits = [c for c in phase.commits if not c.is_merge]
        touched: Counter = Counter()
        for c in commits:
            if len(c.files_stats) <= 100:
                touched.update(c.files_stats.keys())
        top_files = touched.most_common(8)
        top_set = {p for p, _ in touched.most_common(12)}

        pair_counts: Counter = Counter()
        for c in commits:
            paths = sorted(p for p in c.files_stats if p in top_set)
            if 1 < len(paths) <= 100:
                for i, a in enumerate(paths):
                    for b in paths[i + 1:]:
                        pair_counts[(a, b)] += 1
        coupling = [(a, b, n, self._all_time.get((a, b), n))
                    for (a, b), n in pair_counts.most_common(6) if n >= 2]

        clusters = [self._cluster_evidence(c) for c in self._analyzer.cluster_commits(commits)] \
            if commits else []

        motifs = self._motifs_for(touched, top_set, phase.start_date[:10], phase.end_date[:10])
        significant = sum(1 for c in commits if c.is_large_deletion or c.is_large_addition or c.is_refactor)
        pr_narratives = sum(1 for c in commits if c.has_github_context() and c.get_pr_title())

        score, reasons = _score(phase, clusters, motifs, significant, pr_narratives)
        evidence = PhaseEvidence(
            phase_number=phase.phase_number, start_date=phase.start_date[:10], end_date=phase.end_date[:10],
            commit_count=phase.commit_count, authors=list(phase.authors), primary_author=phase.primary_author,
            loc_delta=phase.loc_delta, insertions=phase.total_insertions, deletions=phase.total_deletions,
            top_files=top_files, clusters=clusters, coupling=coupling, motifs=motifs,
            significant_commits=significant, pr_narratives=pr_narratives, score=score, reasons=reasons,
        )
        self._cache[key] = evidence
        return evidence

    def _cluster_evidence(self, cluster: CommitCluster) -> ClusterEvidence:
        key = cluster.key_commit
        top = sorted(cluster.file_changes.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        return ClusterEvidence(
            type=cluster.cluster_type, commits=len(cluster.commits),
            insertions=cluster.total_insertions, deletions=cluster.total_deletions,
            key_hash=key.short_hash, key_subject=key.commit_subject,
            pr_title=key.get_pr_title() if key.has_github_context() else None,
            labels=list(key.get_pr_labels()) if key.has_github_context() else [],
            top_files=[p for p, _ in top],
            subjects=[c.commit_subject for c in cluster.commits[:4]],
        )

    def _motifs_for(self, touched: Counter, top_set: Set[str], start: str, end: str) -> List[MotifFinding]:
        seen: Set[str] = set()
        out: List[MotifFinding] = []
        for path in top_set:
            for f in self._by_file.get(path, []):
                if f.key in seen:
                    continue
                if not _overlaps(f, start, end):
                    continue
                seen.add(f.key)
                out.append(f)
        out.sort(key=lambda f: (-f.confidence, f.key))
        return out[:6]

    # ------------------------------------------------------------ repository-wide sections

    def technical_evolution(self, phases: Sequence[Phase]) -> str:
        """Markdown: how the code base's shape moved, from evidence alone."""
        lines: List[str] = []
        if phases:
            first, last = phases[0], phases[-1]
            lines.append(f"The history spans {len(phases)} phases, {first.start_date[:10]} to "
                         f"{last.end_date[:10]}, moving from {first.loc_start:,} to {last.loc_end:,} lines.")
            lang = _language_shift(first.languages_start, last.languages_end)
            if lang:
                lines.append(lang)
            lines.append('')

        lines.append("### Where change concentrated")
        lines.append('')
        for f in self.stats.most_changed[:10]:
            lines.append(f"- `{f.path}` — {f.touch_count} commits (+{f.total_insertions:,}/-{f.total_deletions:,})")
        lines.append('')

        if self.stats.most_coupled:
            lines.append("### Files that move together")
            lines.append('')
            for e in self.stats.most_coupled[:8]:
                lines.append(f"- `{e.path_a}` + `{e.path_b}` — {e.cochange_count} co-changes "
                             f"(Jaccard {e.jaccard:.2f}), {e.first_seen[:10]} to {e.last_seen[:10]}")
            lines.append('')

        lines.append("### Phase by phase")
        lines.append('')
        for phase in phases:
            ev = self.phase_evidence(phase)
            hot = ', '.join(f"`{p}`" for p, _ in ev.top_files[:3]) or '—'
            kinds = ', '.join(f"{c.type} ({c.commits})" for c in ev.clusters[:4]) or 'no clusters'
            lines.append(f"- **Phase {phase.phase_number}** ({ev.start_date} to {ev.end_date}, "
                         f"{ev.loc_delta:+,d} lines): {kinds}; hot files {hot}")
        lines.append('')
        return '\n'.join(lines).rstrip() + '\n'

    def deletion_story(self, phases: Sequence[Phase]) -> str:
        """Markdown: what was removed, when, and how large it was."""
        events = []
        for phase in phases:
            for c in phase.commits:
                if c.is_large_deletion or (c.deletions >= 200 and c.deletions > 2 * max(c.insertions, 1)):
                    events.append((phase.phase_number, c))
        if not events:
            total = sum(p.total_deletions for p in phases)
            return (f"No single commit removed a large block of code. Deletions were incremental: "
                    f"{total:,} lines removed across {len(phases)} phases.\n")
        lines = [f"{len(events)} commit(s) removed code in bulk:", '']
        for phase_number, c in events[:25]:
            headline = c.get_pr_title() if c.has_github_context() else None
            lines.append(f"- Phase {phase_number}, {c.timestamp[:10]}, `{c.short_hash}` by {c.author}: "
                         f"-{c.deletions:,}/+{c.insertions:,} — {_short(headline or c.commit_subject, 100)}")
        removed = sum(c.deletions for _, c in events)
        lines.append('')
        lines.append(f"Together these removed {removed:,} lines.")
        return '\n'.join(lines) + '\n'

    def architecture_section(self) -> str:
        """Markdown: motif findings grouped by evidence, for the report."""
        report = self.report
        grouped = report.by_motif()
        if not report.findings:
            return ''
        lines: List[str] = []
        obs = report.observations
        if obs:
            latest = obs[-1]
            lines.append(f"*Structural evidence: {latest.provider} {latest.provider_version} "
                         f"at {latest.observed_sha[:8]} ({len(obs)} observation(s)).*")
        else:
            lines.append("*Historical evidence only; run `gitview observe` to add structural motifs.*")
        lines.append('')
        from .motifs import motif_catalog
        for motif in motif_catalog():
            findings = grouped.get(motif.id)
            if not findings:
                continue
            lines.append(f"### {motif.title}")
            lines.append('')
            lines.append(f"*{motif.interpretation}*")
            lines.append('')
            for f in findings[:8]:
                lines.append(f"- {', '.join(f'`{p}`' for p in f.files)} — {f.summary} "
                             f"(confidence {f.level})")
            lines.append('')
        if report.skipped:
            lines.append(f"*Not evaluated: {', '.join(sorted(report.skipped))} — structural observation needed.*")
        return '\n'.join(lines).rstrip() + '\n'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'graph': {'path': str(self.store_path), 'action': self.sync.action, 'reason': self.sync.reason},
            'stats': self.stats.to_dict(),
            'motifs': self.report.to_dict(),
        }


# --------------------------------------------------------------------------- deterministic cluster summary

def cluster_summary(cluster: CommitCluster) -> str:
    """One or two factual sentences for a commit cluster; replaces a per-cluster LLM call."""
    key = cluster.key_commit
    headline = (key.get_pr_title() if key.has_github_context() else None) or key.commit_subject
    top = sorted(cluster.file_changes.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    files = ', '.join(p for p, _ in top)
    n = len(cluster.commits)
    text = (f"{cluster.cluster_type.capitalize()} work in {n} commit(s) "
            f"(+{cluster.total_insertions}/-{cluster.total_deletions})")
    text += f" centred on \"{_short(headline, 100)}\""
    if files:
        text += f", mainly in {files}"
    text += '.'
    body = (key.get_pr_body() if key.has_github_context() else None) or key.commit_body
    first = _first_sentence(body)
    if first:
        text += f" {first}"
    return text


# --------------------------------------------------------------------------- helpers

def _score(phase: Phase, clusters: List[ClusterEvidence], motifs: List[MotifFinding],
           significant: int, pr_narratives: int) -> Tuple[float, List[str]]:
    score, reasons = 0.0, []
    if significant:
        score += 0.35
        reasons.append(f"{significant} significant commit(s)")
    if motifs:
        score += 0.25
        reasons.append(f"{len(motifs)} motif(s)")
    if pr_narratives:
        score += 0.15
        reasons.append(f"{pr_narratives} PR narrative(s)")
    rich = [c for c in clusters if c.commits >= 2]
    if len({c.type for c in rich}) >= 2:
        score += 0.15
        reasons.append("mixed activity")
    if phase.commit_count >= 10 or abs(phase.loc_delta_percent) >= 10:
        score += 0.10
        reasons.append("large phase")
    if phase.readme_changed:
        score += 0.05
    return min(score, 1.0), reasons


def _overlaps(finding: MotifFinding, start: str, end: str) -> bool:
    """Time-bounded motifs must touch the phase window; structural ones always may."""
    ev = finding.evidence
    first, last = ev.get('first_seen'), ev.get('last_seen')
    if first and last:
        return first[:10] <= end and last[:10] >= start
    handover = ev.get('handover_date')
    if handover:
        return start <= handover[:10] <= end
    return True


def _area(paths: Sequence[str]) -> str:
    parts: Counter = Counter()
    for p in paths:
        segs = p.split('/')
        parts[segs[0] if len(segs) > 1 else Path(p).stem] += 1
    if not parts:
        return ''
    return re.sub(r'[^a-z0-9]+', '-', parts.most_common(1)[0][0].lower()).strip('-')


def _language_shift(start: Dict[str, int], end: Dict[str, int]) -> str:
    def mix(d: Dict[str, int]) -> str:
        total = sum(d.values()) or 1
        top = sorted(d.items(), key=lambda kv: -kv[1])[:3]
        return ', '.join(f"{k} {v / total:.0%}" for k, v in top)
    if not start and not end:
        return ''
    return f"Language mix went from {mix(start) or 'n/a'} to {mix(end) or 'n/a'}."


def _short(text: Optional[str], n: int) -> str:
    text = (text or '').strip().splitlines()[0] if text else ''
    return text if len(text) <= n else text[:n - 1] + '…'


def _first_sentence(text: Optional[str]) -> str:
    if not text:
        return ''
    text = text.strip().splitlines()[0].strip()
    if not text or text.startswith(('#', '-', '*', 'Co-Authored', 'Signed-off')):
        return ''
    m = re.match(r'(.+?[.!?])(\s|$)', text)
    sentence = m.group(1) if m else text
    return _short(sentence, 160)
