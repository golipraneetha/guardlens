"""Baseline 7: One-Class SVM anomaly detection.

Per-cycle One-Class SVM fit on approved traffic embeddings. Top-K points
furthest on the outlier side of the learned decision boundary are
surfaced under the same budget as GuardLens. Detection = any of the
top-K flagged points is a ground-truth attack.

Where this differs from Isolation Forest and LOF: both of those are
density/isolation notions of anomaly. One-Class SVM instead learns a
boundary (in this case via an RBF kernel) that encloses the bulk of the
data as tightly as the nu parameter allows, then scores points by
signed distance to that boundary. It is the standard margin-based
anomaly-detection baseline, distinct in mechanism from the two
density-based methods above, rounding out the three unsupervised
novelty-detection approaches most commonly compared in the anomaly-
detection literature (Isolation Forest, LOF, One-Class SVM).

Point-based and per-cycle, like the other two -- no cluster aggregation,
no cross-cycle temporal state.
"""
from __future__ import annotations

import numpy as np
from sklearn.svm import OneClassSVM


class OneClassSVMBaseline:
    def __init__(self, top_k: int = 3, nu: float = 0.1, gamma: str = "scale"):
        self.top_k = top_k
        self.nu = nu
        self.gamma = gamma
        self.first_detection_cycle: int | None = None

    def process_cycle(self, cycle: int, embeddings: np.ndarray,
                      labels: list[int]) -> bool:
        if len(embeddings) < max(2, self.top_k):
            return False

        ocsvm = OneClassSVM(kernel="rbf", nu=self.nu, gamma=self.gamma)
        ocsvm.fit(embeddings)
        # decision_function: higher = more normal (further inside the
        # boundary), so most anomalous = lowest.
        scores = ocsvm.decision_function(embeddings)
        top_indices = np.argsort(scores)[: self.top_k]

        hit = any(labels[i] == 1 for i in top_indices)
        if hit and self.first_detection_cycle is None:
            self.first_detection_cycle = cycle
        return hit
