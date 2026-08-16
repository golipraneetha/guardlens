"""Blind-spot regimes, each a different mechanism by which an attack
family can slip past a deployed guardrail undetected:

  1. novel_family_regime   -- a previously-unseen attack style appears and
     ramps up over a handful of cycles (step onset, then growth).
  2. coordinated_attack_regime -- one attacker finds one working template
     and spams near-duplicate variants of it in a sudden burst.
  3. slow_drift_regime -- an attack family bleeds in very gradually, a
     percentage point or two per cycle, with no sharp onset at all.
  4. diverse_attack_regime -- an attacker aware of clustering-based
     detection splits the same burst volume across N semantically distinct
     template families to fragment the attack cluster below HDBSCAN's
     min_cluster_size. Direct evasion of coordinated_attack_regime.

Each regime returns (list[CycleBatch], onset_cycle). onset_cycle is the
first cycle where ground-truth attack traffic appears at all -- the
reference point detection latency is measured from.
"""
from __future__ import annotations

import random

from .streams import BenignPool, CycleBatch


def novel_family_regime(benign_pool: BenignPool, attack_pool: list[str],
                        n_cycles: int, benign_per_cycle: int = 200,
                        onset_cycle: int = 5, seed: int = 0,
                        schedule: dict[int, int] | None = None
                        ) -> tuple[list[CycleBatch], int]:
    rng = random.Random(seed)
    attacks = list(attack_pool)
    rng.shuffle(attacks)
    ptr = 0

    if schedule is None:
        # ramps 5% -> 20% of benign_per_cycle across the post-onset cycles
        schedule = {}
        for i, c in enumerate(range(onset_cycle, n_cycles + 1)):
            frac = 0.05 + 0.03 * i
            schedule[c] = max(1, round(benign_per_cycle * min(frac, 0.25)))

    batches = []
    for c in range(1, n_cycles + 1):
        texts = list(benign_pool.draw(benign_per_cycle))
        labels = [0] * len(texts)
        n_atk = schedule.get(c, 0)
        if n_atk > 0:
            take = attacks[ptr:ptr + n_atk]
            ptr += n_atk
            if ptr > len(attacks):
                rng.shuffle(attacks)
                ptr = 0
            texts.extend(take)
            labels.extend([1] * len(take))
        combined = list(zip(texts, labels))
        rng.shuffle(combined)
        batches.append(CycleBatch(cycle=c, texts=[t for t, _ in combined],
                                  labels=[l for _, l in combined]))
    return batches, onset_cycle


def coordinated_attack_regime(benign_pool: BenignPool, template_variants: list[str],
                              n_cycles: int, benign_per_cycle: int = 200,
                              onset_cycle: int = 5, burst_size: int = 30,
                              seed: int = 0) -> tuple[list[CycleBatch], int]:
    """A single burst of near-duplicate template variants at onset_cycle,
    then a smaller trickle for a couple of cycles after (an attacker
    doesn't usually stop after the first burst gets through)."""
    rng = random.Random(seed)
    variants = list(template_variants)
    rng.shuffle(variants)

    schedule = {onset_cycle: min(burst_size, len(variants))}
    trickle = max(1, burst_size // 6)
    for c in (onset_cycle + 1, onset_cycle + 2):
        if c <= n_cycles:
            schedule[c] = trickle

    batches = []
    ptr = 0
    for c in range(1, n_cycles + 1):
        texts = list(benign_pool.draw(benign_per_cycle))
        labels = [0] * len(texts)
        n_atk = schedule.get(c, 0)
        if n_atk > 0:
            take = variants[ptr:ptr + n_atk]
            ptr += n_atk
            if ptr > len(variants):
                rng.shuffle(variants)
                ptr = 0
            texts.extend(take)
            labels.extend([1] * len(take))
        combined = list(zip(texts, labels))
        rng.shuffle(combined)
        batches.append(CycleBatch(cycle=c, texts=[t for t, _ in combined],
                                  labels=[l for _, l in combined]))
    return batches, onset_cycle


def diverse_attack_regime(benign_pool: BenignPool,
                          communities_by_name: dict[str, list[str]],
                          community_names: list[str],
                          n_cycles: int, benign_per_cycle: int = 200,
                          onset_cycle: int = 5, burst_size: int = 30,
                          seed: int = 0) -> tuple[list[CycleBatch], int]:
    """Same burst timing and total volume as coordinated_attack_regime, but
    the burst is split evenly across `community_names` -- each community
    contributes burst_size // N templates. This is the direct evasion move
    against clustering-based detection: an attacker who knows we cluster
    can defeat cluster formation by keeping each sub-family small enough
    to fall below HDBSCAN's min_cluster_size threshold."""
    rng = random.Random(seed)
    n = len(community_names)
    if n == 0:
        raise ValueError("community_names must be non-empty")
    per_community = burst_size // n
    trickle_per_community = max(1, per_community // 6)

    # Independent shuffle + pointer per community so each contributes its
    # own sample without repeating across cycles until exhausted.
    variants_by_name = {}
    ptr_by_name = {}
    for name in community_names:
        if name not in communities_by_name:
            raise KeyError(f"community '{name}' not in provided pool")
        v = list(communities_by_name[name])
        rng.shuffle(v)
        variants_by_name[name] = v
        ptr_by_name[name] = 0

    def take_from(name: str, k: int) -> list[str]:
        v = variants_by_name[name]
        ptr = ptr_by_name[name]
        out = v[ptr:ptr + k]
        ptr_by_name[name] = ptr + k
        if ptr_by_name[name] >= len(v):
            rng.shuffle(v)
            ptr_by_name[name] = 0
        return out

    schedule = {onset_cycle: per_community}
    for c in (onset_cycle + 1, onset_cycle + 2):
        if c <= n_cycles:
            schedule[c] = trickle_per_community

    batches = []
    for c in range(1, n_cycles + 1):
        texts = list(benign_pool.draw(benign_per_cycle))
        labels = [0] * len(texts)
        k_per = schedule.get(c, 0)
        if k_per > 0:
            take: list[str] = []
            for name in community_names:
                take.extend(take_from(name, k_per))
            texts.extend(take)
            labels.extend([1] * len(take))
        combined = list(zip(texts, labels))
        rng.shuffle(combined)
        batches.append(CycleBatch(cycle=c, texts=[t for t, _ in combined],
                                  labels=[l for _, l in combined]))
    return batches, onset_cycle


def slow_drift_regime(benign_pool: BenignPool, attack_pool: list[str],
                      n_cycles: int, benign_per_cycle: int = 200,
                      onset_cycle: int = 3, ramp_per_cycle_pct: float = 0.015,
                      seed: int = 0) -> tuple[list[CycleBatch], int]:
    """Attack fraction increases by a small fixed percentage each cycle
    starting at onset_cycle -- no sharp step, no accelerating burst. This
    is the hardest regime: density-based clustering needs enough absolute
    attack volume in a window before a coherent cluster can form."""
    rng = random.Random(seed)
    attacks = list(attack_pool)
    rng.shuffle(attacks)
    ptr = 0

    batches = []
    for c in range(1, n_cycles + 1):
        texts = list(benign_pool.draw(benign_per_cycle))
        labels = [0] * len(texts)
        if c >= onset_cycle:
            frac = ramp_per_cycle_pct * (c - onset_cycle + 1)
            n_atk = max(1, round(benign_per_cycle * frac))
            take = attacks[ptr:ptr + n_atk]
            ptr += n_atk
            if ptr > len(attacks):
                rng.shuffle(attacks)
                ptr = 0
            texts.extend(take)
            labels.extend([1] * len(take))
        combined = list(zip(texts, labels))
        rng.shuffle(combined)
        batches.append(CycleBatch(cycle=c, texts=[t for t, _ in combined],
                                  labels=[l for _, l in combined]))
    return batches, onset_cycle
