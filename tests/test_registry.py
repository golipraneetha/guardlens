import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guardlens.clusterer import WindowCluster
from guardlens.registry import ClusterRegistry


def _normalize(v):
    return v / np.linalg.norm(v)


def wc(local_id, centroid, size):
    return WindowCluster(local_id=local_id, indices=np.arange(size),
                         centroid=_normalize(centroid), density=0.1)


def test_matching_cluster_inherits_uid_and_ages():
    reg = ClusterRegistry(match_threshold=0.85)
    c1 = wc(0, np.array([1.0, 0.0, 0.0]), size=10)
    t1 = reg.register([c1], cycle=1)
    assert t1[0].age == 1
    assert t1[0].prev_size is None

    # nearly identical centroid, larger size -> same identity
    c2 = wc(0, np.array([0.99, 0.01, 0.0]), size=20)
    t2 = reg.register([c2], cycle=2)
    assert t2[0].uid == t1[0].uid
    assert t2[0].age == 2
    assert t2[0].prev_size == 10


def test_dissimilar_cluster_gets_new_uid():
    reg = ClusterRegistry(match_threshold=0.85)
    c1 = wc(0, np.array([1.0, 0.0, 0.0]), size=10)
    t1 = reg.register([c1], cycle=1)

    c2 = wc(0, np.array([0.0, 1.0, 0.0]), size=8)   # orthogonal, sim=0
    t2 = reg.register([c2], cycle=2)
    assert t2[0].uid != t1[0].uid
    assert t2[0].age == 1
    assert t2[0].prev_size is None


def test_history_bounded_by_history_len():
    reg = ClusterRegistry(match_threshold=0.85, history_len=3)
    c = wc(0, np.array([1.0, 0.0, 0.0]), size=5)
    for cycle in range(1, 6):
        reg.register([wc(0, np.array([1.0, 0.0, 0.0]), size=5 + cycle)], cycle=cycle)
    uid = 1
    assert len(reg._history[uid]) == 3


def test_historical_centroids_excludes_self_current_entry():
    reg = ClusterRegistry(match_threshold=0.85)
    c1 = wc(0, np.array([1.0, 0.0, 0.0]), size=5)
    t1 = reg.register([c1], cycle=1)
    uid = t1[0].uid
    # after one registration, historical centroids excluding this uid's
    # current (only) entry should be empty
    hist = reg.historical_centroids(exclude_uid=uid)
    assert hist == []


def test_two_new_clusters_never_collide_on_the_same_prior_uid():
    # Simulates HDBSCAN splitting one previously-tracked cluster into two in
    # the same cycle: both new clusters are closest to the same prior uid,
    # but only one may actually claim it.
    reg = ClusterRegistry(match_threshold=0.85)
    c1 = wc(0, np.array([1.0, 0.0, 0.0]), size=20)
    t1 = reg.register([c1], cycle=1)
    prior_uid = t1[0].uid

    split_a = wc(0, np.array([1.0, 0.0, 0.0]), size=10)
    split_b = wc(1, np.array([0.99, 0.01, 0.0]), size=10)
    t2 = reg.register([split_a, split_b], cycle=2)

    assert t2[0].uid != t2[1].uid
    assert prior_uid in (t2[0].uid, t2[1].uid)


def test_stale_uid_expires_after_history_len_cycles():
    reg = ClusterRegistry(match_threshold=0.85, history_len=3)
    c1 = wc(0, np.array([1.0, 0.0, 0.0]), size=10)
    t1 = reg.register([c1], cycle=1)
    uid = t1[0].uid

    # An unrelated cluster keeps the registry moving forward without ever
    # matching uid again.
    for cycle in range(2, 8):
        reg.register([wc(0, np.array([0.0, 1.0, 0.0]), size=5)], cycle=cycle)

    assert uid not in reg._history
    assert all(uid != h_uid for h_uid in reg._history)
    hist = reg.historical_centroids()
    assert not any(np.allclose(h, c1.centroid) for h in hist)


def test_age_keeps_growing_past_history_len():
    reg = ClusterRegistry(match_threshold=0.85, history_len=2)
    centroid = np.array([1.0, 0.0, 0.0])
    ages = []
    for cycle in range(1, 6):
        tracked = reg.register([wc(0, centroid, size=5)], cycle=cycle)
        ages.append(tracked[0].age)
    assert ages == [1, 2, 3, 4, 5]
