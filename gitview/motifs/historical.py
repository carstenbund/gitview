"""Motifs that need nothing but git history."""

from collections import Counter
from typing import List

from .base import Motif
from .models import Evidence, MotifContext, MotifFinding


def _coupling_confidence(cochanges: int, jaccard: float) -> float:
    return 0.4 + min(cochanges, 20) / 50 + jaccard / 2


class RepeatedCochange(Motif):
    id = 'repeated_cochange'
    title = 'Repeated co-change'
    interpretation = 'two files keep changing together; something ties them'
    requires = frozenset({Evidence.HISTORICAL})

    def detect(self, ctx: MotifContext) -> List[MotifFinding]:
        out = []
        for e in ctx.coupling:
            if e.cochange_count < ctx.t.min_cochanges or e.jaccard < ctx.t.min_jaccard:
                continue
            out.append(self._finding(
                [e.path_a, e.path_b],
                f"{e.cochange_count} co-changes (Jaccard {e.jaccard:.2f}), "
                f"{e.first_seen[:10]} to {e.last_seen[:10]}",
                _coupling_confidence(e.cochange_count, e.jaccard),
                cochanges=e.cochange_count, jaccard=e.jaccard,
                first_seen=e.first_seen, last_seen=e.last_seen,
            ))
            if len(out) >= ctx.t.top:
                break
        return out


class OwnershipTransition(Motif):
    id = 'ownership_transition'
    title = 'Ownership transition'
    interpretation = 'the person who mostly changes a file is no longer the one who used to'
    requires = frozenset({Evidence.HISTORICAL})

    def detect(self, ctx: MotifContext) -> List[MotifFinding]:
        candidates = []
        need = max(4, 2 * ctx.t.min_touches)
        for path, touches in ctx.touches.items():
            if touches < need:
                continue
            history = ctx.store.file_touches(path)
            if len(history) < need:
                continue
            half = len(history) // 2
            before, after = Counter(t[3] for t in history[:half]), Counter(t[3] for t in history[half:])
            (old, old_n), (new, new_n) = before.most_common(1)[0], after.most_common(1)[0]
            old_share, new_share = old_n / half, new_n / (len(history) - half)
            if old == new or old_share < 0.5 or new_share < 0.5:
                continue
            handover = next((t for t in history[half:] if t[3] == new), history[half])
            confidence = 0.3 + (old_share + new_share) / 4 + min(len(history), 30) / 100
            candidates.append(self._finding(
                [path],
                f"{old} ({old_share:.0%} of the first {half} commits) → {new} "
                f"({new_share:.0%} of the last {len(history) - half}), from {handover[1][:8]} on {handover[2][:10]}",
                confidence,
                from_author=old, to_author=new, from_share=round(old_share, 2), to_share=round(new_share, 2),
                commits=len(history), handover_commit=handover[1], handover_date=handover[2],
            ))
        candidates.sort(key=lambda f: (-f.confidence, f.files[0]))
        return candidates[:ctx.t.top]


HISTORICAL_MOTIFS = [RepeatedCochange, OwnershipTransition]
