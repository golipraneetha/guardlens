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
from dataclasses import dataclass

from .streams import BenignPool, CycleBatch, OriginBatch


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


def embedding_evasion_regime(benign_pool: BenignPool, attack_pool: list[str],
                             n_cycles: int, benign_per_cycle: int = 200,
                             onset_cycle: int = 5, seed: int = 0,
                             schedule: dict[int, int] | None = None
                             ) -> tuple[list[CycleBatch], int]:
    """R1 traffic mechanics (novel_family_regime's ramp-onset schedule),
    applied to an attack_pool already generated by
    traffic.attack_variants.generate_diverse_embeddings rather than raw
    benchmark text or Tier B/C paraphrases. The adversarial condition
    lives entirely in what attack_pool contains -- this is a thin,
    explicitly-named wrapper so the embedding-aware condition shows up in
    experiment configs and output filenames instead of being implicit in
    which pool happened to get passed to novel_family_regime."""
    return novel_family_regime(benign_pool, attack_pool, n_cycles=n_cycles,
                               benign_per_cycle=benign_per_cycle,
                               onset_cycle=onset_cycle, seed=seed, schedule=schedule)


@dataclass
class TrendSpec:
    """One benign topic that appears, grows, and decays -- a stand-in for
    a product launch or seasonal spike in production traffic. `pool`
    should be semantically coherent (e.g. one of the three individual
    realistic-traffic sources from realistic_traffic.load_realistic_benign_sources,
    not the pre-mixed blend) so it can actually form a density-based
    cluster; a topic drawn from an already-homogenized mix wouldn't."""
    pool: BenignPool
    name: str
    onset_cycle: int
    peak_cycle: int
    end_cycle: int
    peak_frac: float = 0.35   # fraction of benign_per_cycle at peak


def _trend_frac(spec: TrendSpec, cycle: int) -> float:
    if cycle < spec.onset_cycle or cycle > spec.end_cycle:
        return 0.0
    if cycle <= spec.peak_cycle:
        span = max(spec.peak_cycle - spec.onset_cycle, 1)
        return spec.peak_frac * (cycle - spec.onset_cycle) / span
    span = max(spec.end_cycle - spec.peak_cycle, 1)
    return spec.peak_frac * (1 - (cycle - spec.peak_cycle) / span)


def benign_demand_shift_regime(baseline_pool: BenignPool, trends: list[TrendSpec],
                               n_cycles: int, benign_per_cycle: int = 200,
                               attack_pool: list[str] | None = None,
                               attack_onset_cycle: int = 8, seed: int = 0
                               ) -> tuple[list[OriginBatch], dict]:
    """R4 -- benign demand shift. One or more benign topics (`trends`)
    appear, grow, and decay against a baseline of stable benign traffic,
    optionally with an attack family injected concurrently (same ramping
    schedule as novel_family_regime, starting at attack_onset_cycle) to
    test whether attack clusters keep ranking above benign-trend clusters
    in the Top-K queue under realistic competition (Section V-E).

    Unlike the other regimes, there's no single onset_cycle to report --
    returns a metadata dict instead: {trend_windows: {name: (onset, peak,
    end)}, attack_onset_cycle: int | None}. GuardLens ground truth for
    benign vs. attack still lives in CycleBatch.labels (0/1, unchanged
    contract); OriginBatch.origins additionally distinguishes
    benign_stable from benign_trend:<name> for R4-specific queue-
    competition analysis (experiments/metrics.py cluster ledger)."""
    rng = random.Random(seed)
    attacks = list(attack_pool) if attack_pool else []
    rng.shuffle(attacks)
    ptr = 0

    attack_schedule = {}
    if attacks:
        for i, c in enumerate(range(attack_onset_cycle, n_cycles + 1)):
            frac = 0.05 + 0.03 * i
            attack_schedule[c] = max(1, round(benign_per_cycle * min(frac, 0.25)))

    batches = []
    for c in range(1, n_cycles + 1):
        trend_fracs = {t.name: _trend_frac(t, c) for t in trends}
        total_trend_frac = min(sum(trend_fracs.values()), 0.9)
        n_baseline = benign_per_cycle - round(benign_per_cycle * total_trend_frac)

        texts: list[str] = []
        labels: list[int] = []
        origins: list[str] = []

        base_texts = baseline_pool.draw(max(n_baseline, 0))
        texts.extend(base_texts)
        labels.extend([0] * len(base_texts))
        origins.extend(["benign_stable"] * len(base_texts))

        for t in trends:
            frac = trend_fracs[t.name]
            if frac <= 0:
                continue
            n = round(benign_per_cycle * frac)
            if n <= 0:
                continue
            trend_texts = t.pool.draw(n)
            texts.extend(trend_texts)
            labels.extend([0] * len(trend_texts))
            origins.extend([f"benign_trend:{t.name}"] * len(trend_texts))

        n_atk = attack_schedule.get(c, 0)
        if n_atk > 0 and attacks:
            take = attacks[ptr:ptr + n_atk]
            ptr += n_atk
            if ptr > len(attacks):
                rng.shuffle(attacks)
                ptr = 0
            texts.extend(take)
            labels.extend([1] * len(take))
            origins.extend(["attack"] * len(take))

        combined = list(zip(texts, labels, origins))
        rng.shuffle(combined)
        batches.append(OriginBatch(
            cycle=c,
            texts=[t for t, _, _ in combined],
            labels=[l for _, l, _ in combined],
            origins=[o for _, _, o in combined],
        ))

    meta = dict(
        trend_windows={t.name: (t.onset_cycle, t.peak_cycle, t.end_cycle) for t in trends},
        attack_onset_cycle=attack_onset_cycle if attacks else None,
    )
    return batches, meta
