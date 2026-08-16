"""Turns scored clusters into a bounded, ranked analyst review queue.

This is the operational contribution: instead of an unbounded stream of
"suspicious" flags, the analyst gets exactly K clusters per cycle, ranked
by Emergence Score, regardless of how much traffic came through.
"""
from __future__ import annotations

from dataclasses import dataclass

from .registry import TrackedCluster


@dataclass
class QueueEntry:
    rank: int
    cluster: TrackedCluster


class ReviewQueue:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def rank(self, clusters: list[TrackedCluster]) -> list[QueueEntry]:
        ranked = sorted(clusters, key=lambda c: -c.emergence)[:self.top_k]
        return [QueueEntry(rank=i + 1, cluster=c) for i, c in enumerate(ranked)]
