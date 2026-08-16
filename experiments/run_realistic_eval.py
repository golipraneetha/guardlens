"""Realistic evaluation sweep: runs each regime with realistic benign
traffic and tiered attack variants, comparing detection across tiers.

This directly addresses the "synthetic evaluation" reviewer concern by
showing GuardLens detects attack variants it has never seen before (Tier B/C)
with similar performance to raw benchmark replay (Tier A).

Usage:
    python3 experiments/run_realistic_eval.py
    python3 experiments/run_realistic_eval.py --regimes novel_family --tiers a bc
    python3 experiments/run_realistic_eval.py --variant-model qwen3:8b --seeds 5
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REGIMES = ["novel_family", "coordinated_attack", "slow_drift"]
TIERS = ["a", "bc"]
SEEDS = 5
CYCLES = 10

OUT_DIR = Path(__file__).parent / "realistic_eval"


def run_one(regime: str, tier: str, seeds: int, traffic_source: str,
            variant_model: str, score_cache_dir: Path) -> Path:
    out_file = OUT_DIR / f"{regime}_tier{tier}_{traffic_source}.json"
    score_cache = score_cache_dir / f"{regime}_{traffic_source}_tier{tier}.json"

    cmd = [
        sys.executable, str(Path(__file__).parent / "run_experiment.py"),
        "--regime", regime,
        "--seeds", str(seeds),
        "--cycles", str(CYCLES),
        "--traffic-source", traffic_source,
        "--attack-tier", tier,
        "--variant-model", variant_model,
        "--score-cache", str(score_cache),
        "--out", str(out_file),
    ]

    print(f"\n{'='*70}")
    print(f"  {regime} | tier={tier} | traffic={traffic_source}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    subprocess.run(cmd, check=True)
    return out_file


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", nargs="+", default=REGIMES,
                    choices=REGIMES)
    ap.add_argument("--tiers", nargs="+", default=TIERS,
                    choices=["a", "b", "c", "bc"])
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--traffic-source", default="realistic",
                    choices=["benchmark", "realistic"])
    ap.add_argument("--variant-model", default="qwen3:8b")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    score_cache_dir = OUT_DIR / "score_cache"
    score_cache_dir.mkdir(exist_ok=True)

    results_files = []
    for regime in args.regimes:
        for tier in args.tiers:
            out = run_one(regime, tier, args.seeds, args.traffic_source,
                          args.variant_model, score_cache_dir)
            results_files.append(out)

    print(f"\n\n{'='*70}")
    print("COMPARATIVE SUMMARY")
    print(f"{'='*70}\n")

    print(f"{'Regime':<25} {'Tier':<6} {'Traffic':<12} {'Det.Rate':<10} "
          f"{'Latency':<10} {'Purity':<10} {'Coverage':<10}")
    print("-" * 83)

    for f in results_files:
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        cfg = data.get("config", {})
        s = data.get("summary", {}).get("guardlens", {})

        regime = cfg.get("regime", "?")
        tier = cfg.get("attack_tier", "?")
        traffic = cfg.get("traffic_source", "?")
        det_rate = s.get("detection_rate", 0)
        latency = s.get("median_latency", "—")
        purity = s.get("mean_purity_at_detection", "—")
        coverage = s.get("mean_coverage_at_detection", "—")

        det_str = f"{det_rate:.0%}"
        lat_str = f"{latency}" if latency is not None else "—"
        pur_str = f"{purity:.3f}" if isinstance(purity, (int, float)) else "—"
        cov_str = f"{coverage:.3f}" if isinstance(coverage, (int, float)) else "—"

        print(f"{regime:<25} {tier:<6} {traffic:<12} {det_str:<10} "
              f"{lat_str:<10} {pur_str:<10} {cov_str:<10}")

    print(f"\nAll results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
