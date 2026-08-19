"""Cycle-by-cycle traffic generation.

A CycleBatch is what a deployed guardrail sees in one cycle: raw texts and
their ground-truth labels (1=attack, 0=benign). The classifier decides what
gets approved; GuardLens only ever sees the approved subset.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class CycleBatch:
    cycle: int
    texts: list[str]
    labels: list[int]   # 1 = ground-truth attack, 0 = benign


@dataclass
class OriginBatch(CycleBatch):
    """A CycleBatch with a finer-grained ground-truth origin per item,
    for regimes (R4, benign_demand_shift) where 'benign' isn't one
    category. labels stays a plain 0/1 attack indicator so every existing
    baseline and metric (cluster_purity, coverage, ...) keeps working
    unmodified on OriginBatch instances; origins is additional, read only
    by R4-specific analysis (experiments/run_benign_study.py)."""
    origins: list[str]   # "benign_stable" | "benign_trend:<name>" | "attack"


class BenignPool:
    """Draws without replacement from a shuffled benign pool, cycling back
    to the start (with a re-shuffle) if exhausted -- traffic simulations
    can run longer than the pool's raw size without ever repeating within
    a short window."""

    def __init__(self, pool: list[str], seed: int = 0):
        self._rng = random.Random(seed)
        self._pool = list(pool)
        self._rng.shuffle(self._pool)
        self._ptr = 0

    def draw(self, n: int) -> list[str]:
        out = []
        while len(out) < n:
            remaining = len(self._pool) - self._ptr
            take = min(n - len(out), remaining)
            out.extend(self._pool[self._ptr:self._ptr + take])
            self._ptr += take
            if self._ptr >= len(self._pool):
                self._rng.shuffle(self._pool)
                self._ptr = 0
        return out
