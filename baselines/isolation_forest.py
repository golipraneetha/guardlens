"""Baseline 5: Isolation Forest anomaly detection.

Per-cycle Isolation Forest fit on approved traffic embeddings. Top-K
most-anomalous points are surfaced under the same budget as GuardLens.
Detection = any of the top-K flagged points is a ground-truth attack.

This is a standard unsupervised anomaly-detection baseline. It differs
from GuardLens architecturally:
  - Point-based (per-example anomaly score), not cluster-based -- no
    aggregation across semantically similar items.
  - Per-cycle fitting, no cross-cycle temporal state -- comparable to
    OneShotClusterBaseline in that respect, but flags points rather than
    clusters.
  - No emergence signal (density x growth x novelty); ranks purely by
    IF's anomaly score.

Comparing to this baseline isolates the value of GuardLens's clustering
+ temporal-emergence design versus a strong single-point anomaly
detector operating on the same embeddings.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestBaseline:
    def __init__(self, top_k: int = 3, n_estimators: int = 100,
                 random_state: int = 0):
        self.top_k = top_k
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.first_detection_cycle: int | None = None

    def process_cycle(self, cycle: int, embeddings: np.ndarray,
                      labels: list[int]) -> bool:
        if len(embeddings) < max(2, self.top_k):
            return False

        iforest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination="auto",
            random_state=self.random_state,
        )
        iforest.fit(embeddings)
        # decision_function: higher = more normal, so most anomalous = lowest
        scores = iforest.decision_function(embeddings)
        top_indices = np.argsort(scores)[: self.top_k]

        hit = any(labels[i] == 1 for i in top_indices)
        if hit and self.first_detection_cycle is None:
            self.first_detection_cycle = cycle
        return hit
