"""Baseline 1b: stratified random sampling from the bottom-scoring approved
items (closest to the deployment threshold) rather than uniformly across
all approved traffic. This is a materially harder baseline than plain
random audit -- an analyst who knows to "look at the borderline calls
first" already captures some of what clustering buys you. If GuardLens
can't beat this, the clustering machinery isn't adding much.
"""
from __future__ import annotations

import random


def sample_cycle(texts: list[str], labels: list[int], scores: list[float],
                 budget: int, rng: random.Random, fraction: float = 0.3
                 ) -> tuple[list[int], bool]:
    """Samples `budget` items uniformly from the bottom `fraction` of items
    by classifier score (i.e. the ones that most narrowly avoided being
    blocked)."""
    if not texts:
        return [], False
    order = sorted(range(len(texts)), key=lambda i: -scores[i])  # highest score first
    pool_size = max(budget, round(len(texts) * fraction))
    pool = order[:pool_size]
    idx = rng.sample(pool, min(budget, len(pool)))
    hit = any(labels[i] == 1 for i in idx)
    return idx, hit


class StratifiedRandomBaseline:
    def __init__(self, budget: int, seed: int = 0, fraction: float = 0.3):
        self.budget = budget
        self.fraction = fraction
        self._rng = random.Random(seed)
        self.first_detection_cycle: int | None = None

    def process_cycle(self, cycle: int, texts: list[str], labels: list[int],
                      scores: list[float]) -> bool:
        _, hit = sample_cycle(texts, labels, scores, self.budget, self._rng,
                              self.fraction)
        if hit and self.first_detection_cycle is None:
            self.first_detection_cycle = cycle
        return hit
