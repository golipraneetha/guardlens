"""Baseline 1: uniform random sampling of approved traffic under the same
per-cycle review budget as GuardLens's Top-K queue. No intelligence, no
localization -- just "does the analyst happen to see an attack this cycle."
"""
from __future__ import annotations

import random


def sample_cycle(texts: list[str], labels: list[int], budget: int,
                 rng: random.Random) -> tuple[list[int], bool]:
    """Returns (sampled_indices, hit) where hit = at least one sampled item
    is a ground-truth attack."""
    if not texts:
        return [], False
    idx = rng.sample(range(len(texts)), min(budget, len(texts)))
    hit = any(labels[i] == 1 for i in idx)
    return idx, hit


class RandomAuditBaseline:
    def __init__(self, budget: int, seed: int = 0):
        self.budget = budget
        self._rng = random.Random(seed)
        self.first_detection_cycle: int | None = None

    def process_cycle(self, cycle: int, texts: list[str], labels: list[int]) -> bool:
        _, hit = sample_cycle(texts, labels, self.budget, self._rng)
        if hit and self.first_detection_cycle is None:
            self.first_detection_cycle = cycle
        return hit
