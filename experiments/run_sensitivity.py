"""Sensitivity analysis: vary key hyperparameters one at a time.

Sweeps:
  window_size:      [2, 3, 5]       (default 3)
  min_cluster_size: [3, 5, 8]       (default 5)
  top_k:            [3, 5, 10]      (default 5)

Runs all 3 regimes for each setting. Uses --score-cache to skip
repeated DeBERTa scoring.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REGIME_DEFAULTS = {
    "novel_family": dict(
        cycles=10, benign_per_cycle=200, top_k=3, window_size=3,
        onset_cycle=5, min_cluster_size=5, extra=[],
    ),
    "coordinated_attack": dict(
        cycles=10, benign_per_cycle=200, top_k=3, window_size=3,
        onset_cycle=5, min_cluster_size=5,
        extra=["--burst-size", "30", "--community", "Advanced"],
    ),
    "slow_drift": dict(
        cycles=10, benign_per_cycle=150, top_k=5, window_size=3,
        onset_cycle=3, min_cluster_size=5,
        extra=["--ramp-pct", "0.015", "--harmbench-category", "cybercrime_intrusion"],
    ),
}

SWEEPS = {
    "window_size": [2, 3, 5],
    "min_cluster_size": [3, 5, 8],
    "top_k": [3, 5, 10],
}

RESULTS_DIR = Path(__file__).parent / "sensitivity"
CACHE_DIR = Path(__file__).parent / "score_cache"


def run_one(regime: str, overrides: dict, tag: str) -> dict | None:
    defaults = REGIME_DEFAULTS[regime]
    out_path = RESULTS_DIR / f"{regime}_{tag}.json"
    cache_path = CACHE_DIR / f"{regime}.json"

    merged = {**defaults, **overrides}
    cmd = [
        sys.executable, str(Path(__file__).parent / "run_experiment.py"),
        "--regime", regime,
        "--seeds", "3",
        "--cycles", str(merged["cycles"]),
        "--benign-per-cycle", str(merged["benign_per_cycle"]),
        "--top-k", str(merged["top_k"]),
        "--window-size", str(merged["window_size"]),
        "--min-cluster-size", str(merged["min_cluster_size"]),
        "--onset-cycle", str(merged["onset_cycle"]),
        "--score-cache", str(cache_path),
        "--out", str(out_path),
        *merged.get("extra", []),
    ]
    print(f"\n{'='*60}")
    print(f"  {regime} / {tag}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    print(f"  finished in {elapsed:.1f}s (exit code {result.returncode})")
    if result.returncode != 0:
        return None
    return json.loads(out_path.read_text())


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    all_results = {}

    for param_name, values in SWEEPS.items():
        for value in values:
            tag = f"{param_name}_{value}"
            for regime in REGIME_DEFAULTS:
                key = f"{regime}/{tag}"
                data = run_one(regime, {param_name: value}, tag)
                if data:
                    all_results[key] = data.get("summary", {})

    print(f"\n\n{'='*70}")
    print("SENSITIVITY SUMMARY")
    print(f"{'='*70}")
    print(f"{'Regime':<22} {'Parameter':<22} {'Det%':>6} {'MedLat':>7} {'Purity':>7}")
    print("-" * 70)
    for key, summary in all_results.items():
        regime, tag = key.split("/")
        gl = summary.get("guardlens", {})
        det = f"{gl.get('detection_rate', 0):.0%}"
        lat = gl.get("median_latency")
        lat_s = f"{lat:.1f}" if lat is not None else "N/A"
        pur = gl.get("mean_purity_at_detection")
        pur_s = f"{pur:.2f}" if pur is not None else "N/A"
        print(f"{regime:<22} {tag:<22} {det:>6} {lat_s:>7} {pur_s:>7}")

    combined_path = RESULTS_DIR / "sensitivity_summary.json"
    combined_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {combined_path}")


if __name__ == "__main__":
    main()
