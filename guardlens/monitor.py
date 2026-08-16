"""GuardLensMonitor: the end-to-end pipeline, one cycle at a time.

    approved traffic (sliding window)
        -> embed
        -> cluster_window (HDBSCAN)
        -> ClusterRegistry.register (persistence across cycles)
        -> score_clusters (Emergence Score)
        -> ReviewQueue.rank (Top-K)

Callers drive it one cycle at a time via `process_cycle`, passing the
approved (guardrail-passed) texts for that cycle. The monitor keeps its own
sliding window of raw texts internally so windowing is not the caller's
concern.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embedder import Embedder
from .clusterer import cluster_window
from .registry import ClusterRegistry, TrackedCluster
from .emergence import score_clusters
from .queue import ReviewQueue, QueueEntry


@dataclass
class CycleResult:
    cycle: int
    n_items: int
    clusters: list[TrackedCluster]
    queue: list[QueueEntry]


class GuardLensMonitor:
    def __init__(self, embedder: Embedder | None = None, window_size: int = 3,
                 top_k: int = 5, min_cluster_size: int = 5, min_samples: int = 3,
                 match_threshold: float = 0.85, history_len: int = 7,
                 ablation: str = "full"):
        self.embedder = embedder or Embedder()
        self.window_size = window_size
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.ablation = ablation
        self.registry = ClusterRegistry(match_threshold=match_threshold,
                                        history_len=history_len)
        self.queue = ReviewQueue(top_k=top_k)
        self._window_texts: list[list[str]] = []   # list of per-cycle text lists

    def process_cycle(self, cycle: int, approved_texts: list[str]) -> CycleResult:
        self._window_texts.append(approved_texts)
        self._window_texts = self._window_texts[-self.window_size:]

        window_flat = [t for cycle_texts in self._window_texts for t in cycle_texts]
        if len(window_flat) < self.min_cluster_size:
            return CycleResult(cycle=cycle, n_items=len(approved_texts),
                              clusters=[], queue=[])

        embs = self.embedder.encode(window_flat)
        raw_clusters = cluster_window(embs, min_cluster_size=self.min_cluster_size,
                                      min_samples=self.min_samples)
        tracked = self.registry.register(raw_clusters, cycle)
        scored = score_clusters(tracked, self.registry, ablation=self.ablation)
        ranked = self.queue.rank(scored)
        return CycleResult(cycle=cycle, n_items=len(approved_texts),
                          clusters=scored, queue=ranked)
