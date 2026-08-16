"""Cluster persistence across cycles.

A WindowCluster only knows about the current window. ClusterRegistry gives
clusters a stable identity over time by matching each new cluster to the
nearest cluster from the previous cycle (cosine similarity on centroids).
A match above `match_threshold` means "this is the same cluster, it moved
slightly, and it now has this new size" — below it, the cluster is treated
as newly born.

This is what makes growth and novelty in emergence.py meaningful: growth
compares a tracked cluster's size to its own size last cycle, and novelty
compares a cluster's centroid to the bounded history of centroids seen in
prior cycles, not just the current one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._math import cosine


@dataclass
class TrackedCluster:
    """A WindowCluster annotated with identity + history, after registration."""
    uid: int
    local_id: int
    indices: np.ndarray
    centroid: np.ndarray
    density: float
    size: int
    age: int                     # consecutive cycles this uid has existed
    prev_size: int | None        # size last cycle this uid was seen, or None if new
    growth: float = 0.0
    novelty: float = 0.0
    emergence: float = 0.0


class ClusterRegistry:
    """Tracks cluster identity across cycles within a bounded history window.

    `history_len` bounds both memory and the "recent past" that novelty is
    computed against -- a cluster stops being penalized for resembling
    something seen 30 cycles ago once that cycle falls out of the window.
    """

    def __init__(self, match_threshold: float = 0.85, history_len: int = 7):
        self.match_threshold = match_threshold
        self.history_len = history_len
        self._next_uid = 1
        # uid -> list of (cycle, size, centroid), most recent last
        self._history: dict[int, list[tuple[int, int, np.ndarray]]] = {}
        self._active: dict[int, np.ndarray] = {}   # uid -> centroid, previous cycle only
        self._age: dict[int, int] = {}             # uid -> consecutive cycles seen, unbounded
        self._cycles_seen: list[int] = []

    def register(self, clusters, cycle: int) -> list[TrackedCluster]:
        # Match new clusters to previous-cycle uids as a one-to-one assignment:
        # rank all (cluster, uid) pairs above threshold by similarity and assign
        # greedily, so two new clusters can never both claim the same prior uid
        # (e.g. when HDBSCAN splits one cluster into two in the same cycle).
        candidates = []
        for i, c in enumerate(clusters):
            for uid, prev_centroid in self._active.items():
                sim = cosine(c.centroid, prev_centroid)
                if sim >= self.match_threshold:
                    candidates.append((sim, i, uid))
        candidates.sort(key=lambda x: -x[0])

        assigned_uid: dict[int, int] = {}
        claimed_uids: set[int] = set()
        for sim, i, uid in candidates:
            if i in assigned_uid or uid in claimed_uids:
                continue
            assigned_uid[i] = uid
            claimed_uids.add(uid)

        tracked = []
        for i, c in enumerate(clusters):
            if i in assigned_uid:
                uid = assigned_uid[i]
                prior_hist = self._history[uid]
                prev_size = prior_hist[-1][1]
                age = self._age[uid] + 1
            else:
                uid = self._next_uid
                self._next_uid += 1
                prev_size = None
                age = 1
                self._history[uid] = []

            self._age[uid] = age
            self._history[uid].append((cycle, c.size, c.centroid))
            self._history[uid] = self._history[uid][-self.history_len:]

            tracked.append(TrackedCluster(
                uid=uid, local_id=c.local_id, indices=c.indices,
                centroid=c.centroid, density=c.density, size=c.size,
                age=age, prev_size=prev_size,
            ))

        self._active = {t.uid: t.centroid for t in tracked}
        self._cycles_seen.append(cycle)
        self._cycles_seen = self._cycles_seen[-self.history_len:]

        # Uids that haven't recurred within the history window fall out of
        # tracking entirely, so novelty/history stay bounded to recent cycles
        # instead of accumulating every cluster identity ever seen.
        stale_uids = [uid for uid, hist in self._history.items()
                     if hist and cycle - hist[-1][0] > self.history_len]
        for uid in stale_uids:
            del self._history[uid]
            self._age.pop(uid, None)

        return tracked

    def historical_centroids(self, exclude_uid: int | None = None) -> list[np.ndarray]:
        """All centroids from prior cycles (not including the just-registered
        cycle's own entries used as 'current'), within the bounded history."""
        out = []
        for uid, hist in self._history.items():
            if uid == exclude_uid:
                entries = hist[:-1]
            else:
                entries = hist
            out.extend(centroid for _, _, centroid in entries)
        return out
