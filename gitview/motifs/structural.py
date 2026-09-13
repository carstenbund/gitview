"""Motifs that combine git history with structural observations."""

from typing import List

from .base import Motif
from .models import Evidence, MotifContext, MotifFinding
from .historical import _coupling_confidence


def _relations(edges) -> str:
    return ', '.join(sorted({f"{e.source.split('/')[-1]} {e.relation} {e.target.split('/')[-1]}" for e in edges}))


class HiddenCoupling(Motif):
    id = 'hidden_coupling'
    title = 'Hidden coupling'
    interpretation = 'files co-change repeatedly with no source dependency: an implicit architectural or business link'
    requires = frozenset({Evidence.HISTORICAL, Evidence.STRUCTURAL})

    def detect(self, ctx: MotifContext) -> List[MotifFinding]:
        snap = ctx.latest
        known = snap.paths()
        out = []
        for e in ctx.coupling:
            if e.cochange_count < ctx.t.min_cochanges:
                break
            if e.path_a not in known or e.path_b not in known:
                continue          # the analyser never saw one side; absence is not evidence
            if ctx.edges_between(snap, e.path_a, e.path_b):
                continue
            out.append(self._finding(
                [e.path_a, e.path_b],
                f"{e.cochange_count} co-changes (Jaccard {e.jaccard:.2f}); structural edge: none",
                _coupling_confidence(e.cochange_count, e.jaccard),
                cochanges=e.cochange_count, jaccard=e.jaccard, structural_edge=None,
                observed_sha=snap.observed_sha,
            ))
            if len(out) >= ctx.t.top:
                break
        return out


class ConfirmedCoupling(Motif):
    id = 'confirmed_coupling'
    title = 'Confirmed coupling'
    interpretation = 'files co-change and one depends on the other: a strong maintenance dependency'
    requires = frozenset({Evidence.HISTORICAL, Evidence.STRUCTURAL})

    def detect(self, ctx: MotifContext) -> List[MotifFinding]:
        snap = ctx.latest
        out = []
        for e in ctx.coupling:
            if e.cochange_count < ctx.t.min_cochanges:
                break
            edges = ctx.edges_between(snap, e.path_a, e.path_b)
            if not edges:
                continue
            out.append(self._finding(
                [e.path_a, e.path_b],
                f"{e.cochange_count} co-changes (Jaccard {e.jaccard:.2f}); structural: {_relations(edges)}",
                min(1.0, _coupling_confidence(e.cochange_count, e.jaccard) + 0.15),
                cochanges=e.cochange_count, jaccard=e.jaccard,
                structural_edges=[x.to_dict() for x in edges], observed_sha=snap.observed_sha,
            ))
            if len(out) >= ctx.t.top:
                break
        return out


class StableInterface(Motif):
    id = 'stable_interface'
    title = 'Stable interface'
    interpretation = 'a strong source dependency whose sides rarely change together: a mature boundary'
    requires = frozenset({Evidence.HISTORICAL, Evidence.STRUCTURAL})

    def detect(self, ctx: MotifContext) -> List[MotifFinding]:
        snap = ctx.latest
        strength = {}
        for e in snap.edges:
            key = tuple(sorted((e.source, e.target)))
            strength[key] = strength.get(key, 0) + e.count
        out = []
        for (a, b), count in sorted(strength.items(), key=lambda kv: (-kv[1], kv[0])):
            if count < 2:
                break
            ta, tb = ctx.touches.get(a, 0), ctx.touches.get(b, 0)
            if ta < ctx.t.min_touches or tb < ctx.t.min_touches:
                continue
            co = ctx.cochange(a, b)
            cochanges = co.cochange_count if co else 0
            if cochanges > 1:
                continue
            out.append(self._finding(
                [a, b],
                f"{count} symbol-level dependencies, {cochanges} co-change(s) across {ta}+{tb} commits",
                0.4 + min(count, 10) / 25 + min(ta + tb, 40) / 100 - cochanges * 0.1,
                structural_strength=count, cochanges=cochanges, touches=[ta, tb],
                observed_sha=snap.observed_sha,
            ))
        out.sort(key=lambda f: (-f.confidence, f.files))
        return out[:ctx.t.top]


class EmergingDependency(Motif):
    id = 'emerging_dependency'
    title = 'Emerging dependency'
    interpretation = 'a source dependency appeared between observations, often after a period of co-change'
    requires = frozenset({Evidence.HISTORICAL, Evidence.STRUCTURAL_SERIES})

    def detect(self, ctx: MotifContext) -> List[MotifFinding]:
        early, late = ctx.earliest, ctx.latest
        early_pairs = ctx.undirected_pairs(early)
        early_known = early.paths()
        candidates = []
        for pair in sorted(ctx.undirected_pairs(late) - early_pairs):
            a, b = pair
            if a not in early_known or b not in early_known:
                continue      # a brand-new file, not a dependency forming between existing ones
            co = ctx.cochange(a, b)
            if co is None:
                continue
            history = ctx.store.cochange_commits(a, b)
            lead = None
            if history and ctx.latest_observation.sequence is not None:
                lead = ctx.latest_observation.sequence - history[0][0]
            edges = ctx.edges_between(late, a, b)
            lead_text = (f"; co-change began {lead} commit(s) before it was observed ({history[0][1][:8]})"
                         if lead is not None and lead > 0 else "")
            candidates.append(self._finding(
                [a, b],
                f"dependency first observed at {late.observed_sha[:8]} ({_relations(edges)}); "
                f"{co.cochange_count} co-changes{lead_text}",
                0.45 + min(co.cochange_count, 10) / 25 + (0.15 if lead and lead > 0 else 0),
                cochanges=co.cochange_count, first_cochange=history[0][1] if history else None,
                lead_commits=lead, observed_from=early.observed_sha, observed_to=late.observed_sha,
                structural_edges=[x.to_dict() for x in edges],
            ))
        candidates.sort(key=lambda f: (-f.confidence, f.files))
        return candidates[:ctx.t.top]


class ArchitecturalSplit(Motif):
    id = 'architectural_split'
    title = 'Architectural split'
    interpretation = 'a source dependency disappeared; if co-change stopped too the decoupling succeeded'
    requires = frozenset({Evidence.HISTORICAL, Evidence.STRUCTURAL_SERIES})

    def detect(self, ctx: MotifContext) -> List[MotifFinding]:
        early, late = ctx.earliest, ctx.latest
        late_known = late.paths()
        cut = ctx.earliest_observation.sequence
        candidates = []
        for a, b in sorted(ctx.undirected_pairs(early) - ctx.undirected_pairs(late)):
            if a not in late_known or b not in late_known:
                continue      # a file was deleted or moved, not decoupled
            history = ctx.store.cochange_commits(a, b)
            before = sum(1 for h in history if cut is None or h[0] <= cut)
            after = len(history) - before
            outcome = 'decoupled' if after == 0 else 'dependency removed but co-change persists'
            candidates.append(self._finding(
                [a, b],
                f"dependency present at {early.observed_sha[:8]}, absent at {late.observed_sha[:8]}; "
                f"co-changes before/after: {before}/{after} — {outcome}",
                0.5 + (0.25 if after == 0 else -0.1) + min(before, 10) / 50,
                outcome=outcome, cochanges_before=before, cochanges_after=after,
                observed_from=early.observed_sha, observed_to=late.observed_sha,
            ))
        candidates.sort(key=lambda f: (-f.confidence, f.files))
        return candidates[:ctx.t.top]


class CentralityGrowth(Motif):
    id = 'centrality_growth'
    title = 'Centrality growth'
    interpretation = 'a file is acquiring structural neighbours and historical concerns: a future god module?'
    requires = frozenset({Evidence.HISTORICAL, Evidence.STRUCTURAL_SERIES})

    def detect(self, ctx: MotifContext) -> List[MotifFinding]:
        early, late = ctx.earliest, ctx.latest
        d0, d1 = early.degree(), late.degree()
        candidates = []
        for path, degree in d1.items():
            base = d0.get(path)
            if base is None:
                continue
            growth = degree - base
            if growth < ctx.t.min_degree_growth or degree < base * 1.5:
                continue
            neighbours = ctx.store.cochange_neighbours(path)
            candidates.append(self._finding(
                [path],
                f"structural degree {base} → {degree} between {early.observed_sha[:8]} and "
                f"{late.observed_sha[:8]}; {len(neighbours)} historical co-change neighbours, "
                f"{ctx.touches.get(path, 0)} commits",
                0.4 + min(growth, 15) / 30 + min(len(neighbours), 20) / 100,
                degree_from=base, degree_to=degree, historical_neighbours=len(neighbours),
                touches=ctx.touches.get(path, 0), observed_from=early.observed_sha, observed_to=late.observed_sha,
            ))
        candidates.sort(key=lambda f: (-f.confidence, f.files))
        return candidates[:ctx.t.top]


STRUCTURAL_MOTIFS = [HiddenCoupling, ConfirmedCoupling, StableInterface,
                     EmergingDependency, ArchitecturalSplit, CentralityGrowth]
