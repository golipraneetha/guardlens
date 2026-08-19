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


# --- R4: benign demand shift / queue competition (Section V-E) ---------
#
# These use ground-truth origin labels (traffic/streams.py OriginBatch),
# not a purity threshold: a small clean attack cluster and a mixed
# benign-trend cluster are both misclassified by purity<0.5 heuristics,
# which is exactly the distinction queue-competition analysis needs to
# get right.

@dataclass
class LedgerEntry:
    cycle: int
    rank: int              # 1-indexed position in the Top-K queue
    cluster_uid: int
    label: str              # "attack" | "benign_trend" | "benign_stable"
    emergence_score: float


def cluster_origin_label(cluster_indices: np.ndarray, window_origins: list[str]) -> str:
    """Ground-truth label for a cluster: the plurality origin bucket among
    its members. window_origins[i] is one of "attack", "benign_stable", or
    "benign_trend:<name>" (bucketed here by stripping the ":<name>" suffix)
    aligned index-for-index with the monitor's current sliding window, the
    same convention run_experiment.py already uses for window_is_attack."""
    if len(cluster_indices) == 0:
        return "unknown"
    from collections import Counter
    buckets = Counter(window_origins[i].split(":")[0] for i in cluster_indices)
    return buckets.most_common(1)[0][0]


def build_cycle_ledger(cycle: int, topk_clusters, window_origins: list[str]
                       ) -> list[LedgerEntry]:
    """One LedgerEntry per Top-K slot for this cycle. Accumulate across
    cycles (ledger.extend(...)) and pass the combined list to
    benign_cluster_rate / queue_pollution / analyst_burden below."""
    entries = []
    for rank, entry in enumerate(topk_clusters, start=1):
        entries.append(LedgerEntry(
            cycle=cycle, rank=rank, cluster_uid=entry.cluster.uid,
            label=cluster_origin_label(entry.cluster.indices, window_origins),
            emergence_score=entry.cluster.emergence,
        ))
    return entries


def benign_cluster_rate(ledger: list[LedgerEntry]) -> float:
    """Fraction of all Top-K slots, across the whole run, occupied by a
    benign-trend cluster rather than an attack or benign-stable one."""
    if not ledger:
        return 0.0
    return sum(1 for e in ledger if e.label == "benign_trend") / len(ledger)


def queue_pollution(ledger: list[LedgerEntry]) -> float:
    """Fraction of cycles where at least one Top-K slot went to a
    benign-trend cluster -- the queue-competition rate reviewers asked
    about: how often does emergent-but-benign traffic cost a review slot
    that could have gone to an attack cluster."""
    by_cycle: dict[int, list[LedgerEntry]] = {}
    for e in ledger:
        by_cycle.setdefault(e.cycle, []).append(e)
    if not by_cycle:
        return 0.0
    polluted = sum(1 for entries in by_cycle.values()
                   if any(e.label == "benign_trend" for e in entries))
    return polluted / len(by_cycle)


def analyst_burden(ledger: list[LedgerEntry]) -> float:
    """Mean number of benign-trend clusters reviewed per monitoring
    cycle -- the false-positive workload implied by Top-K, independent of
    whether an attack cluster was also present that cycle."""
    by_cycle: dict[int, list[LedgerEntry]] = {}
    for e in ledger:
        by_cycle.setdefault(e.cycle, []).append(e)
    if not by_cycle:
        return 0.0
    total_benign_trend = sum(1 for e in ledger if e.label == "benign_trend")
    return total_benign_trend / len(by_cycle)
