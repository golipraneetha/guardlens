"""Baseline 2: Maximum Mean Discrepancy two-sample test between a reference
embedding distribution and the current cycle's approved-traffic embeddings.
Detects THAT the distribution has shifted; unlike GuardLens, it cannot say
WHICH items are responsible or localize a reviewable cluster.
"""
from __future__ import annotations

import numpy as np


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    sq_x = (X ** 2).sum(axis=1)[:, None]
    sq_y = (Y ** 2).sum(axis=1)[None, :]
    sq_dists = sq_x + sq_y - 2 * X @ Y.T
    return np.exp(-gamma * np.maximum(sq_dists, 0))


def _median_heuristic_gamma(X: np.ndarray, Y: np.ndarray) -> float:
    Z = np.vstack([X, Y])
    n = min(len(Z), 200)
    idx = np.random.default_rng(0).choice(len(Z), n, replace=False)
    Zs = Z[idx]
    sq = ((Zs[:, None, :] - Zs[None, :, :]) ** 2).sum(-1)
    med = np.median(sq[sq > 0]) if np.any(sq > 0) else 1.0
    return 1.0 / (2 * med) if med > 0 else 1.0


def mmd_squared(X: np.ndarray, Y: np.ndarray, gamma: float | None = None) -> float:
    if gamma is None:
        gamma = _median_heuristic_gamma(X, Y)
    Kxx = _rbf_kernel(X, X, gamma)
    Kyy = _rbf_kernel(Y, Y, gamma)
    Kxy = _rbf_kernel(X, Y, gamma)
    m, n = len(X), len(Y)
    term_x = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1)) if m > 1 else 0.0
    term_y = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1)) if n > 1 else 0.0
    term_xy = Kxy.sum() / (m * n)
    return term_x + term_y - 2 * term_xy


def permutation_pvalue(X: np.ndarray, Y: np.ndarray, n_perm: int = 200,
                       seed: int = 0) -> tuple[float, float]:
    """Returns (observed_mmd2, p_value)."""
    gamma = _median_heuristic_gamma(X, Y)
    observed = mmd_squared(X, Y, gamma)
    Z = np.vstack([X, Y])
    m = len(X)
    rng = np.random.default_rng(seed)
    count_ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(Z))
        Xp, Yp = Z[perm[:m]], Z[perm[m:]]
        stat = mmd_squared(Xp, Yp, gamma)
        if stat >= observed:
            count_ge += 1
    p_value = (count_ge + 1) / (n_perm + 1)
    return observed, p_value


class MMDDriftBaseline:
    """Reference distribution = the embeddings from the first `ref_cycles`
    cycles (assumed attack-free at deployment time). Each subsequent cycle
    is tested against that fixed reference."""

    def __init__(self, alpha: float = 0.05, n_perm: int = 200, seed: int = 0):
        self.alpha = alpha
        self.n_perm = n_perm
        self.seed = seed
        self.reference: np.ndarray | None = None
        self.first_detection_cycle: int | None = None

    def set_reference(self, embeddings: np.ndarray) -> None:
        self.reference = embeddings

    def process_cycle(self, cycle: int, embeddings: np.ndarray) -> tuple[bool, float]:
        if self.reference is None or len(embeddings) < 5:
            return False, 1.0
        _, p_value = permutation_pvalue(self.reference, embeddings,
                                        n_perm=self.n_perm, seed=self.seed + cycle)
        detected = p_value < self.alpha
        if detected and self.first_detection_cycle is None:
            self.first_detection_cycle = cycle
        return detected, p_value
