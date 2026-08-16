"""Scale test: GuardLens detection latency at 500 benign/cycle.

Re-runs the Table VI scale experiment with realistic traffic (Alpaca +
OASST1 + UltraChat) and unseen attack variants (Tier B+C), then compares
against the existing 200/cycle realistic-eval results.

Key insight from the regime code:
  - R1 (novel_family) and R3 (slow_drift) scale attack count proportionally
    with benign_per_cycle, so the attack ratio stays constant -> detection
    should hold at 500/cycle.
  - R2 (coordinated_attack) uses a fixed burst of 30 items regardless of
    benign_per_cycle -> attack ratio drops from ~15% (at 200) to ~6% (at
    500), which caused the original detection degradation.

Part 2 sweeps min_cluster_size {3, 5, 8} on R2 at 500/cycle to test whether
hyperparameter tuning recovers detection at the diluted attack ratio.

Usage:
    python3 experiments/run_scale_test.py
    python3 experiments/run_scale_test.py --skip-pre-score
    python3 experiments/run_scale_test.py --only novel_family coordinated_attack
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REGIMES = ["novel_family", "coordinated_attack", "slow_drift"]
SEEDS = 5
CYCLES = 10
BENIGN_PER_CYCLE_SCALE = 500
TRAFFIC_SOURCE = "realistic"
ATTACK_TIER = "bc"
N_PER_TIER = 50
VARIANT_MODEL = "qwen3:8b"

OUT_DIR = Path(__file__).parent / "scale_test"
SCORE_CACHE_DIR = OUT_DIR / "score_cache"
COMBINED_CACHE = SCORE_CACHE_DIR / "combined.json"

EXISTING_200_DIR = Path(__file__).parent / "realistic_eval"
EXISTING_200_FILES = {
    "novel_family": EXISTING_200_DIR / "novel_family_tierBC_realistic.json",
    "coordinated_attack": EXISTING_200_DIR / "coordinated_attack_tierBC_realistic.json",
    "slow_drift": EXISTING_200_DIR / "slow_drift_tierBC_realistic.json",
}

EXISTING_CACHES = [
    EXISTING_200_DIR / "score_cache" / "novel_family_realistic_bc.json",
    EXISTING_200_DIR / "score_cache" / "novel_family_realistic_a.json",
    EXISTING_200_DIR / "score_cache" / "r2_r3_merged.json",
]

R2_MCS_SWEEP = [3, 5, 8]


def merge_existing_caches() -> dict[str, float]:
    merged: dict[str, float] = {}
    for p in EXISTING_CACHES:
        if p.exists():
            data = json.loads(p.read_text())
            merged.update(data)
            print(f"  merged {len(data)} entries from {p.name}")
    print(f"  total merged: {len(merged)} unique texts")
    return merged


def pre_score_phase():
    """Build a comprehensive DeBERTa score cache covering all texts needed
    for the 500/cycle experiments. Merges existing caches and scores only
    the missing texts."""
    print("=" * 70)
    print("PRE-SCORING PHASE")
    print("=" * 70)

    SCORE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if COMBINED_CACHE.exists():
        existing = json.loads(COMBINED_CACHE.read_text())
        print(f"Combined cache already exists with {len(existing)} entries.")
        print("Checking if it covers all needed texts...")
    else:
        existing = {}

    print("\nMerging existing score caches...")
    merged = merge_existing_caches()
    merged.update(existing)

    print("\nCollecting all texts needed for 500/cycle runs...")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, "/Users/chaituprani/Downloads/tenant-calibration")

    from traffic.realistic_traffic import load_realistic_benign_pool
    from traffic.attack_variants import build_tiered_attack_pool
    from traffic.datasets import (load_advbench, load_harmbench,
                                  load_jailbreak_community_templates)
    from traffic.streams import BenignPool
    from traffic.regimes import (novel_family_regime, coordinated_attack_regime,
                                 slow_drift_regime)

    benign_pool_texts = load_realistic_benign_pool()
    print(f"  realistic benign pool: {len(benign_pool_texts)} texts")

    variant_cache_dir = Path(__file__).parent / "variant_cache"

    attack_pools = {}
    raw_advbench = load_advbench()
    tiers_r1 = build_tiered_attack_pool(
        raw_advbench, category="harmful_behavior",
        n_per_tier=N_PER_TIER, model=VARIANT_MODEL, cache_dir=variant_cache_dir)
    attack_pools["novel_family"] = tiers_r1["tier_b"] + tiers_r1["tier_c"]

    communities = load_jailbreak_community_templates()
    raw_community = communities["Advanced"]
    tiers_r2 = build_tiered_attack_pool(
        raw_community, category="jailbreak_template",
        n_per_tier=N_PER_TIER, model=VARIANT_MODEL, cache_dir=variant_cache_dir)
    attack_pools["coordinated_attack"] = tiers_r2["tier_b"] + tiers_r2["tier_c"]

    raw_harmbench = load_harmbench(semantic_category="cybercrime_intrusion")
    tiers_r3 = build_tiered_attack_pool(
        raw_harmbench, category="cybercrime_intrusion",
        n_per_tier=N_PER_TIER, model=VARIANT_MODEL, cache_dir=variant_cache_dir)
    attack_pools["slow_drift"] = tiers_r3["tier_b"] + tiers_r3["tier_c"]

    all_texts = set()
    for regime in REGIMES:
        if regime == "novel_family":
            atk = attack_pools["novel_family"]
        elif regime == "coordinated_attack":
            atk = attack_pools["coordinated_attack"]
        else:
            atk = attack_pools["slow_drift"]

        for seed in range(SEEDS):
            bp = BenignPool(benign_pool_texts, seed=seed)
            if regime == "novel_family":
                batches, _ = novel_family_regime(
                    bp, atk, n_cycles=CYCLES,
                    benign_per_cycle=BENIGN_PER_CYCLE_SCALE, seed=seed)
            elif regime == "coordinated_attack":
                batches, _ = coordinated_attack_regime(
                    bp, atk, n_cycles=CYCLES,
                    benign_per_cycle=BENIGN_PER_CYCLE_SCALE, seed=seed)
            else:
                batches, _ = slow_drift_regime(
                    bp, atk, n_cycles=CYCLES,
                    benign_per_cycle=BENIGN_PER_CYCLE_SCALE, seed=seed)
            for b in batches:
                all_texts.update(b.texts)

    print(f"  total unique texts needed: {len(all_texts)}")
    missing = [t for t in all_texts if t not in merged]
    print(f"  already cached: {len(all_texts) - len(missing)}")
    print(f"  missing (need DeBERTa): {len(missing)}")

    if not missing:
        print("All texts already cached! Saving combined cache...")
        filtered = {t: merged[t] for t in all_texts if t in merged}
        COMBINED_CACHE.write_text(json.dumps(filtered))
        print(f"Saved {len(filtered)} entries to {COMBINED_CACHE}")
        return

    print(f"\nLoading DeBERTa to score {len(missing)} missing texts...")
    from real_classifiers import DebertaInjectionClassifier
    classifier = DebertaInjectionClassifier()

    t0 = time.time()
    classifier.warm_cache(missing)
    elapsed = time.time() - t0
    print(f"  scored {len(missing)} texts in {elapsed:.1f}s")

    for t in missing:
        merged[t] = classifier.confidence(t)

    combined = {t: merged[t] for t in all_texts if t in merged}
    COMBINED_CACHE.write_text(json.dumps(combined))
    print(f"Saved combined cache ({len(combined)} entries) to {COMBINED_CACHE}")


def run_one(regime: str, benign_per_cycle: int,
            min_cluster_size: int = 5) -> Path:
    suffix = f"{regime}_500"
    if min_cluster_size != 5:
        suffix += f"_mcs{min_cluster_size}"
    out_file = OUT_DIR / f"{suffix}.json"

    cmd = [
        sys.executable, str(Path(__file__).parent / "run_experiment.py"),
        "--regime", regime,
        "--seeds", str(SEEDS),
        "--cycles", str(CYCLES),
        "--benign-per-cycle", str(benign_per_cycle),
        "--traffic-source", TRAFFIC_SOURCE,
        "--attack-tier", ATTACK_TIER,
        "--n-per-tier", str(N_PER_TIER),
        "--variant-model", VARIANT_MODEL,
        "--min-cluster-size", str(min_cluster_size),
        "--score-cache", str(COMBINED_CACHE),
        "--out", str(out_file),
    ]

    print(f"\n{'=' * 70}")
    print(f"  {regime} | benign/cycle={benign_per_cycle} | "
          f"min_cluster_size={min_cluster_size}")
    print(f"{'=' * 70}\n")

    subprocess.run(cmd, check=True)
    return out_file


def load_gl_summary(path: Path) -> dict | None:
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return None
    data = json.loads(path.read_text())
    return data.get("summary", {}).get("guardlens", {})


def print_scale_table(regime_results_500: dict[str, Path]):
    print(f"\n{'=' * 70}")
    print("TABLE — Scale: 200 vs 500 benign/cycle")
    print("(realistic traffic, unseen variants, 5 seeds)")
    print(f"{'=' * 70}\n")

    header = (f"{'Regime':<25} {'benign/cycle':<14} {'Det.Rate':<10} "
              f"{'Med.Latency':<13} {'Purity':<10} {'Coverage':<10}")
    print(header)
    print("-" * len(header))

    rows = []
    for regime in REGIMES:
        for bpc, source in [(200, "existing"), (500, "new")]:
            if bpc == 200:
                path = EXISTING_200_FILES[regime]
            else:
                path = regime_results_500[regime]

            gl = load_gl_summary(path)
            if gl is None:
                continue

            det = gl.get("detection_rate", 0)
            lat = gl.get("median_latency")
            pur = gl.get("mean_purity_at_detection")
            cov = gl.get("mean_coverage_at_detection")

            det_s = f"{det:.0%}"
            lat_s = f"{lat}" if lat is not None else "—"
            pur_s = f"{pur:.2f}" if isinstance(pur, (int, float)) else "—"
            cov_s = f"{cov:.2f}" if isinstance(cov, (int, float)) else "—"

            label = {"novel_family": "R1: Novel Family",
                     "coordinated_attack": "R2: Coordinated",
                     "slow_drift": "R3: Slow Drift"}[regime]
            print(f"{label:<25} {bpc:<14} {det_s:<10} {lat_s:<13} "
                  f"{pur_s:<10} {cov_s:<10}")
            rows.append(dict(regime=regime, benign_per_cycle=bpc,
                             detection_rate=det, median_latency=lat,
                             purity=pur, coverage=cov))
    return rows


def print_hyperparam_table(mcs_results: dict[int, Path]):
    print(f"\n{'=' * 70}")
    print("TABLE — R2 min_cluster_size Tuning at 500/cycle")
    print("(realistic traffic, unseen variants, 5 seeds)")
    print(f"{'=' * 70}\n")

    header = (f"{'min_cluster_size':<18} {'Det.Rate':<10} {'Med.Latency':<13} "
              f"{'Purity':<10} {'Coverage':<10}")
    print(header)
    print("-" * len(header))

    rows = []
    for mcs in R2_MCS_SWEEP:
        path = mcs_results[mcs]
        gl = load_gl_summary(path)
        if gl is None:
            continue

        det = gl.get("detection_rate", 0)
        lat = gl.get("median_latency")
        pur = gl.get("mean_purity_at_detection")
        cov = gl.get("mean_coverage_at_detection")

        det_s = f"{det:.0%}"
        lat_s = f"{lat}" if lat is not None else "—"
        pur_s = f"{pur:.2f}" if isinstance(pur, (int, float)) else "—"
        cov_s = f"{cov:.2f}" if isinstance(cov, (int, float)) else "—"

        default = " (default)" if mcs == 5 else ""
        print(f"{mcs}{default:<18} {det_s:<10} {lat_s:<13} "
              f"{pur_s:<10} {cov_s:<10}")
        rows.append(dict(min_cluster_size=mcs, detection_rate=det,
                         median_latency=lat, purity=pur, coverage=cov))
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-pre-score", action="store_true",
                    help="Skip DeBERTa pre-scoring (use existing combined cache)")
    ap.add_argument("--only", nargs="+", default=None,
                    choices=REGIMES,
                    help="Run only these regimes (default: all 3)")
    ap.add_argument("--skip-hyperparam", action="store_true",
                    help="Skip min_cluster_size sweep on R2")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SCORE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    regimes_to_run = args.only if args.only else REGIMES

    # --- Phase 1: Pre-score ---
    if not args.skip_pre_score:
        pre_score_phase()
    elif not COMBINED_CACHE.exists():
        print("ERROR: --skip-pre-score but combined cache does not exist.")
        print(f"  expected: {COMBINED_CACHE}")
        sys.exit(1)
    else:
        cached = json.loads(COMBINED_CACHE.read_text())
        print(f"Using existing combined cache ({len(cached)} entries)")

    # --- Phase 2: Run 500/cycle experiments ---
    regime_results_500: dict[str, Path] = {}
    for regime in regimes_to_run:
        out = run_one(regime, BENIGN_PER_CYCLE_SCALE)
        regime_results_500[regime] = out

    # --- Phase 3: R2 min_cluster_size sweep at 500/cycle ---
    mcs_results: dict[int, Path] = {}
    if not args.skip_hyperparam and "coordinated_attack" in regimes_to_run:
        for mcs in R2_MCS_SWEEP:
            if mcs == 5:
                mcs_results[mcs] = regime_results_500["coordinated_attack"]
            else:
                out = run_one("coordinated_attack", BENIGN_PER_CYCLE_SCALE,
                              min_cluster_size=mcs)
                mcs_results[mcs] = out
    elif "coordinated_attack" in regime_results_500:
        mcs_results[5] = regime_results_500["coordinated_attack"]

    # --- Phase 4: Print tables ---
    scale_rows = []
    if len(regime_results_500) == len(REGIMES):
        scale_rows = print_scale_table(regime_results_500)

    hp_rows = []
    if len(mcs_results) == len(R2_MCS_SWEEP):
        hp_rows = print_hyperparam_table(mcs_results)

    # --- Phase 5: Save summary ---
    summary = {
        "scale_comparison": scale_rows,
        "r2_hyperparam_sweep": hp_rows,
        "config": {
            "benign_per_cycle_scale": BENIGN_PER_CYCLE_SCALE,
            "seeds": SEEDS,
            "cycles": CYCLES,
            "traffic_source": TRAFFIC_SOURCE,
            "attack_tier": ATTACK_TIER,
            "r2_mcs_sweep": R2_MCS_SWEEP,
        },
    }
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary to {summary_path}")

    print(f"\nAll results saved to: {OUT_DIR}")
    for f in sorted(OUT_DIR.glob("*.json")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
