"""Metric computation shared by all regimes.

Detection latency is the headline metric: cycles elapsed between
onset_cycle (ground-truth first appearance of the attack family) and the
cycle where a method first surfaces it, under its review budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MethodResult:
    name: str
    detection_cycle: int | None
    onset_cycle: int
    n_cycles: int

    @property
    def latency(self) -> int | None:
        if self.detection_cycle is None:
            return None
        return self.detection_cycle - self.onset_cycle

    @property
    def detected(self) -> bool:
        return self.detection_cycle is not None


@dataclass
class GuardLensCycleMetrics:
    cycle: int
    top1_purity: float | None       # purity of highest-Emergence cluster, if any
    attack_items_in_topk: int
    total_attack_items_in_window: int
    n_queue_entries: int


def cluster_purity(cluster_indices: np.ndarray, is_attack: np.ndarray) -> float:
    if len(cluster_indices) == 0:
        return 0.0
    return float(is_attack[cluster_indices].mean())


def coverage(topk_clusters, is_attack: np.ndarray, purity_threshold: float = 0.5) -> float:
    """Fraction of all ground-truth attacks in the window that fall inside
    a Top-K cluster whose purity clears `purity_threshold`."""
    total_attacks = int(is_attack.sum())
    if total_attacks == 0:
        return 0.0
    captured = set()
    for entry in topk_clusters:
        idx = entry.cluster.indices
        if cluster_purity(idx, is_attack) >= purity_threshold:
            captured.update(idx[is_attack[idx] == 1].tolist())
    return len(captured) / total_attacks


def attack_cluster_fragmentation(topk_clusters, is_attack: np.ndarray,
                                 purity_threshold: float = 0.5) -> int:
    """Number of distinct top-K clusters that qualify as attack clusters
    (purity >= threshold). For evasion analysis: a coordinated homogeneous
    burst produces fragmentation=1; a diversified attack that survives
    clustering at all should produce fragmentation>=1; a diversified attack
    that defeats clustering produces fragmentation=0."""
    return sum(1 for e in topk_clusters
               if cluster_purity(e.cluster.indices, is_attack) >= purity_threshold)


def precision_at_k(topk_clusters, is_attack: np.ndarray,
                   purity_threshold: float = 0.5) -> float:
    if not topk_clusters:
        return 0.0
    attack_clusters = sum(
        1 for e in topk_clusters
        if cluster_purity(e.cluster.indices, is_attack) >= purity_threshold
    )
    return attack_clusters / len(topk_clusters)


def false_positive_reduction(unfiltered_queue, filtered_queue,
                             is_attack: np.ndarray,
                             purity_threshold: float = 0.5) -> float:
    benign_uids = {
        e.cluster.uid for e in unfiltered_queue
        if cluster_purity(e.cluster.indices, is_attack) < purity_threshold
    }
    if not benign_uids:
        return 0.0
    surviving = {e.cluster.uid for e in filtered_queue}
    return len(benign_uids - surviving) / len(benign_uids)


def recall_preservation(unfiltered_queue, filtered_queue,
                        is_attack: np.ndarray,
                        purity_threshold: float = 0.5) -> float:
    attack_uids = {
        e.cluster.uid for e in unfiltered_queue
        if cluster_purity(e.cluster.indices, is_attack) >= purity_threshold
    }
    if not attack_uids:
        return 1.0
    surviving = {e.cluster.uid for e in filtered_queue}
    return len(attack_uids & surviving) / len(attack_uids)
