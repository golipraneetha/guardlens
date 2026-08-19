"""Baseline 6: Local Outlier Factor anomaly detection.

Per-cycle LOF fit on approved traffic embeddings. Top-K most-anomalous
points (lowest local density relative to their neighbors) are surfaced
under the same budget as GuardLens. Detection = any of the top-K flagged
points is a ground-truth attack.

Where this differs from Isolation Forest: IF partitions the embedding
space with random trees and scores by how quickly a point isolates: it
is a global, tree-ensemble notion of "how weird is this point overall."
LOF instead compares each point's local density to its k nearest
neighbors' local densities: a point is anomalous if it sits in a
sparser neighborhood than its neighbors' neighborhoods, which is a
fundamentally local, density-ratio notion rather than a global one.
This is the same reason a coordinated burst (R2) is hard for IF (a
tight, self-consistent cluster looks "normal" to a global isolation
score) but the comparison isolates a different failure mode: LOF can
also rate a genuinely novel but internally dense cluster as normal,
since local density around its own members is what LOF measures, not
novelty relative to the rest of the window.

Like Isolation Forest, this is point-based (no cluster aggregation) and
per-cycle (no cross-cycle temporal state).
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import LocalOutlierFactor


class LOFBaseline:
    def __init__(self, top_k: int = 3, n_neighbors: int = 20):
        self.top_k = top_k
        self.n_neighbors = n_neighbors
        self.first_detection_cycle: int | None = None

    def process_cycle(self, cycle: int, embeddings: np.ndarray,
                      labels: list[int]) -> bool:
        n = len(embeddings)
        if n < max(2, self.top_k):
            return False

        k = min(self.n_neighbors, n - 1)
        lof = LocalOutlierFactor(n_neighbors=k, novelty=False)
        lof.fit_predict(embeddings)
        # negative_outlier_factor_: higher (closer to 0) = more normal,
        # so most anomalous = lowest (most negative).
        scores = lof.negative_outlier_factor_
        top_indices = np.argsort(scores)[: self.top_k]

        hit = any(labels[i] == 1 for i in top_indices)
        if hit and self.first_detection_cycle is None:
            self.first_detection_cycle = cycle
        return hit
