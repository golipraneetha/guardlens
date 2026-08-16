import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guardlens.registry import TrackedCluster
from guardlens.queue import ReviewQueue


def tc(uid, emergence):
    return TrackedCluster(uid=uid, local_id=0, indices=np.array([]),
                          centroid=np.zeros(3), density=0.1, size=5,
                          age=1, prev_size=None, emergence=emergence)


def test_top_k_ranking_order():
    clusters = [tc(1, 0.3), tc(2, 0.9), tc(3, 0.1), tc(4, 0.5)]
    q = ReviewQueue(top_k=2)
    ranked = q.rank(clusters)
    assert [e.cluster.uid for e in ranked] == [2, 4]
    assert [e.rank for e in ranked] == [1, 2]


def test_top_k_larger_than_available_returns_all():
    clusters = [tc(1, 0.3), tc(2, 0.9)]
    q = ReviewQueue(top_k=5)
    ranked = q.rank(clusters)
    assert len(ranked) == 2
