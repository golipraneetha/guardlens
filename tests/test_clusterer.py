import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guardlens.clusterer import cluster_window


def _normalize(v):
    return v / np.linalg.norm(v)


def make_blob(center, n, spread=0.02, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    pts = center + rng.normal(scale=spread, size=(n, dim))
    return np.array([_normalize(p) for p in pts])


def test_recovers_two_separated_blobs():
    dim = 8
    rng = np.random.default_rng(1)
    c1 = _normalize(rng.normal(size=dim))
    c2 = _normalize(-c1)  # maximally separated
    blob1 = make_blob(c1, 20, seed=1)
    blob2 = make_blob(c2, 15, seed=2)
    embs = np.vstack([blob1, blob2])

    clusters = cluster_window(embs, min_cluster_size=5, min_samples=3)
    sizes = sorted(c.size for c in clusters)
    assert len(clusters) == 2
    assert sizes == [15, 20]


def test_too_few_points_returns_empty():
    embs = np.random.default_rng(0).normal(size=(3, 8))
    clusters = cluster_window(embs, min_cluster_size=5)
    assert clusters == []


def test_pure_noise_may_return_no_clusters():
    dim = 8
    rng = np.random.default_rng(2)
    # uniformly scattered points on the sphere -- no real structure
    pts = rng.normal(size=(30, dim))
    embs = np.array([_normalize(p) for p in pts])
    clusters = cluster_window(embs, min_cluster_size=8, min_samples=3)
    # Should not crash; may or may not find spurious clusters, but any
    # found cluster's indices must be valid and disjoint.
    seen = set()
    for c in clusters:
        assert len(c.indices) == c.size
        assert not (seen & set(c.indices.tolist()))
        seen |= set(c.indices.tolist())


def test_density_is_floored():
    dim = 8
    rng = np.random.default_rng(3)
    c1 = _normalize(rng.normal(size=dim))
    blob = make_blob(c1, 10, seed=3)
    clusters = cluster_window(blob, min_cluster_size=5, min_samples=3, min_density=0.05)
    for c in clusters:
        assert c.density >= 0.05
