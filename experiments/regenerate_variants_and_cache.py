"""One-time prep step for the N=50 unseen-variant re-run.

Reviewer concern: 30 Tier B + 30 Tier C (60 unseen attacks/regime) is a
small sample. This regenerates Tier B/C at n_per_tier=50 (100 unseen
attacks/regime) for all 3 regimes, then builds ONE master DeBERTa score
cache covering:
  - the full 15,000-text realistic benign pool (covers any
    benign_per_cycle/seed/cycle-count draw from it, not just the specific
    draws needed today -- a permanent investment, not a one-off)
  - all attack texts (Tier A/B/C) for all 3 regimes at the new n_per_tier

The master cache is then written to every score-cache path that
run_realistic_eval.py, run_hyperparam_sweep.py, and run_scale_test.py look
for, so none of those scripts re-trigger a full DeBERTa load (their
cache-or-nothing logic re-scores EVERYTHING if even one text is missing
from their specific cache file).

Usage:
    python3 experiments/regenerate_variants_and_cache.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/Users/chaituprani/Downloads/tenant-calibration")

N_PER_TIER = 50
VARIANT_MODEL = "qwen3:8b"
REGIMES = ["novel_family", "coordinated_attack", "slow_drift"]

REALISTIC_EVAL_CACHE_DIR = Path(__file__).parent / "realistic_eval" / "score_cache"
SCALE_TEST_CACHE_DIR = Path(__file__).parent / "scale_test" / "score_cache"
MASTER_CACHE = Path(__file__).parent / "master_score_cache.json"

EXISTING_CACHES = [
    REALISTIC_EVAL_CACHE_DIR / "novel_family_realistic_a.json",
    REALISTIC_EVAL_CACHE_DIR / "novel_family_realistic_bc.json",
    REALISTIC_EVAL_CACHE_DIR / "r2_r3_merged.json",
    SCALE_TEST_CACHE_DIR / "combined.json",
    MASTER_CACHE,
]

# Every path run_realistic_eval.py / run_hyperparam_sweep.py / run_scale_test.py
# might look up, so we write the master cache to all of them.
DEST_PATHS = [
    REALISTIC_EVAL_CACHE_DIR / "novel_family_realistic_tiera.json",
    REALISTIC_EVAL_CACHE_DIR / "novel_family_realistic_tierbc.json",
    REALISTIC_EVAL_CACHE_DIR / "coordinated_attack_realistic_tiera.json",
    REALISTIC_EVAL_CACHE_DIR / "coordinated_attack_realistic_tierbc.json",
    REALISTIC_EVAL_CACHE_DIR / "slow_drift_realistic_tiera.json",
    REALISTIC_EVAL_CACHE_DIR / "slow_drift_realistic_tierbc.json",
    REALISTIC_EVAL_CACHE_DIR / "novel_family_realistic_bc.json",  # legacy path (run_hyperparam_sweep.py)
    SCALE_TEST_CACHE_DIR / "combined.json",
]


def merge_existing_caches() -> dict[str, float]:
    merged: dict[str, float] = {}
    for p in EXISTING_CACHES:
        if p.exists():
            data = json.loads(p.read_text())
            merged.update(data)
            print(f"  merged {len(data)} entries from {p}")
    print(f"  total merged: {len(merged)} unique texts")
    return merged


def main():
    from traffic.realistic_traffic import load_realistic_benign_pool
    from traffic.attack_variants import build_tiered_attack_pool
    from traffic.datasets import (load_advbench, load_harmbench,
                                  load_jailbreak_community_templates)

    print("=" * 70)
    print(f"STEP 1: Regenerate Tier B/C variants at n_per_tier={N_PER_TIER}")
    print("=" * 70)

    variant_cache_dir = Path(__file__).parent / "variant_cache"
    attack_pools: dict[str, dict[str, list[str]]] = {}

    raw_advbench = load_advbench()
    print(f"\nnovel_family: {len(raw_advbench)} raw advbench items")
    attack_pools["novel_family"] = build_tiered_attack_pool(
        raw_advbench, category="harmful_behavior",
        n_per_tier=N_PER_TIER, model=VARIANT_MODEL, cache_dir=variant_cache_dir)
    print(f"  tier_b={len(attack_pools['novel_family']['tier_b'])} "
          f"tier_c={len(attack_pools['novel_family']['tier_c'])}")

    communities = load_jailbreak_community_templates()
    raw_community = communities["Advanced"]
    print(f"\ncoordinated_attack: {len(raw_community)} raw community items")
    attack_pools["coordinated_attack"] = build_tiered_attack_pool(
        raw_community, category="jailbreak_template",
        n_per_tier=N_PER_TIER, model=VARIANT_MODEL, cache_dir=variant_cache_dir)
    print(f"  tier_b={len(attack_pools['coordinated_attack']['tier_b'])} "
          f"tier_c={len(attack_pools['coordinated_attack']['tier_c'])}")

    raw_harmbench = load_harmbench(semantic_category="cybercrime_intrusion")
    print(f"\nslow_drift: {len(raw_harmbench)} raw harmbench items")
    attack_pools["slow_drift"] = build_tiered_attack_pool(
        raw_harmbench, category="cybercrime_intrusion",
        n_per_tier=N_PER_TIER, model=VARIANT_MODEL, cache_dir=variant_cache_dir)
    print(f"  tier_b={len(attack_pools['slow_drift']['tier_b'])} "
          f"tier_c={len(attack_pools['slow_drift']['tier_c'])}")

    print("\n" + "=" * 70)
    print("STEP 2: Build master text universe")
    print("=" * 70)

    print("\nLoading full realistic benign pool (covers any future draw)...")
    benign_pool_texts = load_realistic_benign_pool()
    print(f"  {len(benign_pool_texts)} texts")

    all_texts: set[str] = set(benign_pool_texts)
    for regime, pools in attack_pools.items():
        all_texts.update(pools["tier_a"])
        all_texts.update(pools["tier_b"])
        all_texts.update(pools["tier_c"])
    # Tier A for novel_family always uses the FULL raw advbench pool (520
    # items), not just the n_per_tier slice -- run_experiment.py's "a"
    # branch uses raw_attacks directly, not build_tiered_attack_pool's
    # tier_a. Cover it fully so Table IX's Tier A rows are never a cache miss.
    all_texts.update(raw_advbench)
    all_texts.update(raw_community)
    all_texts.update(raw_harmbench)
    for community_texts in communities.values():
        all_texts.update(community_texts)

    print(f"  total unique texts needed: {len(all_texts)}")

    print("\n" + "=" * 70)
    print("STEP 3: Merge existing score caches")
    print("=" * 70)
    merged = merge_existing_caches()

    missing = [t for t in all_texts if t not in merged]
    print(f"\nalready cached: {len(all_texts) - len(missing)}")
    print(f"missing (need DeBERTa): {len(missing)}")

    if missing:
        print("\n" + "=" * 70)
        print(f"STEP 4: Score {len(missing)} missing texts with DeBERTa")
        print("=" * 70)
        from real_classifiers import DebertaInjectionClassifier
        classifier = DebertaInjectionClassifier()
        t0 = time.time()
        classifier.warm_cache(missing)
        elapsed = time.time() - t0
        print(f"  scored {len(missing)} texts in {elapsed:.1f}s")
        for t in missing:
            merged[t] = classifier.confidence(t)
    else:
        print("\nAll texts already cached -- no DeBERTa scoring needed.")

    master = {t: merged[t] for t in all_texts if t in merged}
    print(f"\nMaster cache: {len(master)} entries")
    MASTER_CACHE.write_text(json.dumps(master))
    print(f"Saved master cache to {MASTER_CACHE}")

    print("\n" + "=" * 70)
    print("STEP 5: Write master cache to all downstream paths")
    print("=" * 70)
    for dest in DEST_PATHS:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(master))
        print(f"  wrote {len(master)} entries to {dest}")

    print("\nDone. All downstream experiment scripts should now hit 100% cache.")


if __name__ == "__main__":
    main()
