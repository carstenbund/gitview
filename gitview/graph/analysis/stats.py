"""Basic graph statistics: counts plus most-changed / most-coupled / most-connected."""

from ..models import GraphStats
from ..store import GraphStore


def compute_graph_stats(store: GraphStore, *, top: int = 10, min_cochanges: int = 2) -> GraphStats:
    counts = store.counts()
    return GraphStats(
        commits=counts['commits'],
        merge_commits=counts['merge_commits'],
        suppressed_commits=counts['suppressed_commits'],
        files=counts['files'],
        authors=counts['authors'],
        pull_requests=counts['pull_requests'],
        commit_file_edges=counts['commit_file_edges'],
        file_edges=counts['file_edges'],
        structural_snapshots=counts.get('structural_snapshots', 0),
        most_changed=store.most_changed(top),
        most_coupled=store.most_coupled(top, min_cochanges=min_cochanges),
        most_connected=store.most_connected(top),
    )
