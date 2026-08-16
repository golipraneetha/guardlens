import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guardlens.clusterer import WindowCluster
from guardlens.registry import ClusterRegistry
from guardlens.emergence import score_clusters


def _normalize(v):
    return v / np.linalg.norm(v)


def wc(centroid, size, density=0.1):
    return WindowCluster(local_id=0, indices=np.arange(size),
                         centroid=_normalize(centroid), density=density)


def test_new_cluster_gets_neutral_growth_and_full_novelty():
    reg = ClusterRegistry()
    c = wc(np.array([1.0, 0.0, 0.0]), size=10)
    tracked = reg.register([c], cycle=1)
    scored = score_clusters(tracked, reg)
    assert scored[0].growth == 1.0
    assert scored[0].novelty == 1.0
    assert scored[0].emergence == scored[0].density * 1.0 * 1.0


def test_growing_cluster_has_higher_growth_than_shrinking():
    # Same recurring centroid in both cases (novelty collapses to ~0 for
    # both, so isolate the growth term itself rather than full emergence).
    reg = ClusterRegistry()
    reg.register([wc(np.array([1.0, 0.0, 0.0]), size=10)], cycle=1)
    grown = reg.register([wc(np.array([1.0, 0.0, 0.0]), size=40)], cycle=2)
    growth_up = score_clusters(grown, reg)[0].growth

    reg2 = ClusterRegistry()
    reg2.register([wc(np.array([1.0, 0.0, 0.0]), size=40)], cycle=1)
    shrunk = reg2.register([wc(np.array([1.0, 0.0, 0.0]), size=10)], cycle=2)
    growth_down = score_clusters(shrunk, reg2)[0].growth

    assert growth_up > 1.0 > growth_down


def test_emergence_equals_product_of_its_three_components():
    # A cluster that persists unchanged across cycles necessarily loses
    # novelty (it was already seen last cycle) -- growth and novelty are
    # therefore *not* independent to vary freely in a same-centroid toy
    # scenario. Rather than fight that confound, assert the formula itself:
    # emergence is exactly density * max(growth, 0.1) * novelty, always.
    reg = ClusterRegistry()
    reg.register([wc(np.array([0.0, 0.0, 1.0]), size=5)], cycle=1)
    tracked = reg.register([wc(np.array([1.0, 0.0, 0.0]), size=40)], cycle=2)
    scored = score_clusters(tracked, reg)[0]
    expected = scored.density * max(scored.growth, 0.1) * scored.novelty
    assert scored.emergence == expected


def test_novel_cluster_scores_higher_than_recurring_cluster():
    reg = ClusterRegistry()
    # cycle 1: an established benign cluster
    reg.register([wc(np.array([1.0, 0.0, 0.0]), size=20)], cycle=1)
    # cycle 2: same cluster recurs (low novelty) + a brand new orthogonal cluster
    recurring = wc(np.array([1.0, 0.0, 0.0]), size=20)
    novel = wc(np.array([0.0, 1.0, 0.0]), size=20)
    tracked = reg.register([recurring, novel], cycle=2)
    scored = score_clusters(tracked, reg)
    by_centroid = {tuple(np.round(c.centroid, 2)): c for c in scored}
    recurring_score = by_centroid[(1.0, 0.0, 0.0)]
    novel_score = by_centroid[(0.0, 1.0, 0.0)]
    assert novel_score.novelty > recurring_score.novelty
    assert novel_score.emergence > recurring_score.emergence


def test_emergence_zero_when_any_component_effectively_zero():
    reg = ClusterRegistry()
    # cluster identical to something already seen many times -> novelty ~ 0
    reg.register([wc(np.array([1.0, 0.0, 0.0]), size=10)], cycle=1)
    tracked = reg.register([wc(np.array([1.0, 0.0, 0.0]), size=10)], cycle=2)
    scored = score_clusters(tracked, reg)
    assert scored[0].novelty < 1e-6
    assert scored[0].emergence < 1e-6
