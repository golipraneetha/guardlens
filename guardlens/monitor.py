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

import time
from dataclasses import dataclass, field

import numpy as np

from .embedder import Embedder
from .clusterer import cluster_window
from .registry import ClusterRegistry, TrackedCluster
from .emergence import score_clusters
from .queue import ReviewQueue, QueueEntry


@dataclass
class CycleTimingProfile:
    """Wall-clock seconds for each pipeline stage in one process_cycle call.
    All fields are 0.0 when the cycle short-circuits below min_cluster_size."""
    embed_seconds: float = 0.0
    cluster_seconds: float = 0.0
    registry_seconds: float = 0.0
    score_seconds: float = 0.0
    queue_seconds: float = 0.0

    @property
    def total_seconds(self) -> float:
        return (self.embed_seconds + self.cluster_seconds + self.registry_seconds
                + self.score_seconds + self.queue_seconds)


@dataclass
class CycleResult:
    cycle: int
    n_items: int
    clusters: list[TrackedCluster]
    queue: list[QueueEntry]
    timing: CycleTimingProfile = field(default_factory=CycleTimingProfile)


class GuardLensMonitor:
    def __init__(self, embedder: Embedder | None = None, window_size: int = 3,
                 top_k: int = 5, min_cluster_size: int = 5, min_samples: int = 3,
                 match_threshold: float = 0.85, history_len: int = 7,
                 ablation: str = "full", growth_floor: float = 0.1,
                 score_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)):
        self.embedder = embedder or Embedder()
        self.window_size = window_size
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.ablation = ablation
        self.growth_floor = growth_floor
        self.score_weights = score_weights
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

        timing = CycleTimingProfile()

        t0 = time.perf_counter()
        embs = self.embedder.encode(window_flat)
        timing.embed_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        raw_clusters = cluster_window(embs, min_cluster_size=self.min_cluster_size,
                                      min_samples=self.min_samples)
        timing.cluster_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        tracked = self.registry.register(raw_clusters, cycle)
        timing.registry_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        scored = score_clusters(tracked, self.registry, ablation=self.ablation,
                                growth_floor=self.growth_floor,
                                weights=self.score_weights)
        timing.score_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        ranked = self.queue.rank(scored)
        timing.queue_seconds = time.perf_counter() - t0

        return CycleResult(cycle=cycle, n_items=len(approved_texts),
                          clusters=scored, queue=ranked, timing=timing)
