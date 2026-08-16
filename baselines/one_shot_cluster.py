"""Baseline 4: one-shot clustering. Runs the same HDBSCAN clusterer
GuardLens uses, but independently on each cycle's approved traffic --
no sliding window, no cluster registry, no growth or novelty (neither is
defined without cross-cycle history). Clusters are ranked by density alone.

This isolates GuardLens's actual contribution: if one-shot clustering
matches GuardLens on detection/latency, the temporal tracking (registry +
Emergence Score) isn't doing real work and HDBSCAN alone would suffice. If
GuardLens wins, temporal tracking is load-bearing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from guardlens.clusterer import cluster_window, WindowCluster


@dataclass
class OneShotEntry:
    rank: int
    cluster: WindowCluster


class OneShotClusterBaseline:
    def __init__(self, top_k: int = 5, min_cluster_size: int = 5, min_samples: int = 3):
        self.top_k = top_k
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.first_detection_cycle: int | None = None

    def process_cycle(self, cycle: int, embeddings: np.ndarray) -> list[OneShotEntry]:
        clusters = cluster_window(embeddings, min_cluster_size=self.min_cluster_size,
                                  min_samples=self.min_samples)
        ranked = sorted(clusters, key=lambda c: -c.density)[:self.top_k]
        return [OneShotEntry(rank=i + 1, cluster=c) for i, c in enumerate(ranked)]
