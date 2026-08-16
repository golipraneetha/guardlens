"""HDBSCAN-based clustering of a single window's approved-traffic embeddings.

Clustering is infrastructure, not the contribution: this module's only job
is to turn a batch of embeddings into a list of WindowCluster objects with a
centroid and a density (stability) score. Everything about tracking clusters
over time and ranking them lives in registry.py / emergence.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import hdbscan


@dataclass
class WindowCluster:
    local_id: int
    indices: np.ndarray          # positions into the window's item array
    centroid: np.ndarray         # L2-normalized
    density: float               # HDBSCAN cluster_persistence_, floored
    size: int = field(init=False)

    def __post_init__(self):
        self.size = int(len(self.indices))


def cluster_window(embeddings: np.ndarray, min_cluster_size: int = 5,
                   min_samples: int = 3, min_density: float = 0.05
                   ) -> list[WindowCluster]:
    """Cluster one window's (already L2-normalized) embeddings.

    Returns an empty list if there's too little data to cluster or HDBSCAN
    finds only noise — both are valid, common states (e.g. cold start).
    """
    if len(embeddings) < min_cluster_size:
        return []

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(embeddings)
    persistence = getattr(clusterer, "cluster_persistence_", None)

    clusters = []
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        idx = np.where(labels == cid)[0]
        centroid = embeddings[idx].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        else:
            # Degenerate case: member embeddings cancel out exactly (e.g. two
            # antipodal duplicates). Fall back to the first member's own
            # (already unit) embedding rather than leaving a zero vector,
            # since every centroid must be normalized for cosine() downstream.
            centroid = embeddings[idx[0]]
        density = float(persistence[cid]) if persistence is not None and cid < len(persistence) else min_density
        clusters.append(WindowCluster(
            local_id=int(cid), indices=idx, centroid=centroid,
            density=max(density, min_density),
        ))
    return clusters
