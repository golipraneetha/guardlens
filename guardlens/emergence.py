"""The Emergence Score: the one metric GuardLens ranks clusters by.

    emergence = density * growth * novelty

density  - HDBSCAN cluster_persistence_, i.e. how tight/stable the cluster is.
growth   - log(size_t / size_{t-1}) + 1, i.e. how fast a tracked cluster is
           expanding cycle-over-cycle. A brand-new cluster (no previous size)
           gets a neutral growth of 1.0 rather than an undefined ratio -- it
           gets one cycle of benefit of the doubt before growth becomes
           informative.
novelty  - 1 - max cosine_sim(centroid, historical centroids). A cluster that
           closely resembles something already seen recently scores near 0;
           a cluster unlike anything in the bounded history scores near 1.

All three terms are deliberately simple and independently inspectable --
there is exactly one formula here, not five separate heuristics.
"""
from __future__ import annotations

import math

import numpy as np

from .registry import TrackedCluster, ClusterRegistry
from ._math import cosine


def _growth(cluster: TrackedCluster) -> float:
    if cluster.prev_size is None or cluster.prev_size <= 0:
        return 1.0
    return math.log(cluster.size / cluster.prev_size) + 1.0


def _novelty(cluster: TrackedCluster, historical_centroids: list[np.ndarray]) -> float:
    if not historical_centroids:
        return 1.0
    max_sim = max(cosine(cluster.centroid, h) for h in historical_centroids)
    return max(1.0 - max_sim, 0.0)


ABLATION_MODES = ("full", "no_density", "no_growth", "no_novelty")


def score_clusters(clusters: list[TrackedCluster], registry: ClusterRegistry,
                   ablation: str = "full", growth_floor: float = 0.1,
                   weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
                   ) -> list[TrackedCluster]:
    """Mutates and returns clusters with growth/novelty/emergence populated.

    ablation controls leave-one-out experiments:
      full        density * max(growth, floor) * novelty   (default)
      no_density  1       * max(growth, floor) * novelty
      no_growth   density * 1                  * novelty
      no_novelty  density * max(growth, floor) * 1

    growth_floor and weights (alpha, beta, gamma exponents on density,
    growth, novelty respectively) generalize the fixed-form default
    (floor=0.1, weights=(1,1,1)) for the joint hyperparameter sweep
    (Section V-H) -- E = D^alpha * max(G, floor)^beta * N^gamma.
    """
    alpha, beta, gamma = weights
    for c in clusters:
        c.growth = _growth(c)
        hist = registry.historical_centroids(exclude_uid=c.uid)
        c.novelty = _novelty(c, hist)

        d = 1.0 if ablation == "no_density" else c.density ** alpha
        g = 1.0 if ablation == "no_growth" else max(c.growth, growth_floor) ** beta
        n = 1.0 if ablation == "no_novelty" else c.novelty ** gamma
        c.emergence = d * g * n
    return clusters
