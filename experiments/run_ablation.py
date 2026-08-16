"""Run ablation study: leave-one-out on each emergence score component.

Produces results for: full, no_density, no_growth, no_novelty
across all 3 regimes. Uses --score-cache so DeBERTa scoring only
happens once per regime (~20 min), then ablation variants are instant.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

### top_k is forced to 1 here (overriding the main-experiment defaults of
### 3/3/5). At the main-experiment budgets, HDBSCAN rarely produces more raw
### clusters than the review queue can hold, so truncation never happens and
### leave-one-out ablation of density/growth/novelty can't change which
### clusters make the queue -- every ablation variant scores identically.
### Forcing top_k=1 makes rank-1 selection the whole game, so removing a
### scoring component can actually change the outcome.
REGIMES = {
    "novel_family": [
        "--cycles", "10", "--benign-per-cycle", "200", "--top-k", "1",
        "--window-size", "3", "--onset-cycle", "5",
    ],
    "coordinated_attack": [
        "--cycles", "10", "--benign-per-cycle", "200", "--top-k", "1",
        "--window-size", "3", "--onset-cycle", "5",
        "--burst-size", "30", "--community", "Advanced",
    ],
    "slow_drift": [
        "--cycles", "10", "--benign-per-cycle", "150", "--top-k", "1",
        "--window-size", "3", "--onset-cycle", "3",
        "--ramp-pct", "0.015", "--harmbench-category", "cybercrime_intrusion",
    ],
}

ABLATIONS = ["full", "no_density", "no_growth", "no_novelty"]

RESULTS_DIR = Path(__file__).parent / "ablation_realistic"
CACHE_PATH = Path(__file__).parent / "master_score_cache.json"

REALISTIC_ARGS = [
    "--traffic-source", "realistic", "--attack-tier", "bc",
    "--n-per-tier", "50",
]


def run_one(regime: str, ablation: str, regime_args: list[str]) -> dict | None:
    out_path = RESULTS_DIR / f"{regime}_{ablation}.json"
    cmd = [
        sys.executable, str(Path(__file__).parent / "run_experiment.py"),
        "--regime", regime,
        "--seeds", "5",
        "--ablation", ablation,
        "--score-cache", str(CACHE_PATH),
        "--out", str(out_path),
        *regime_args,
        *REALISTIC_ARGS,
    ]
    print(f"\n{'='*60}")
    print(f"  {regime} / {ablation}")
    print(f"  -> {out_path}")
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

    all_results = {}
    for regime, regime_args in REGIMES.items():
        for ablation in ABLATIONS:
            key = f"{regime}/{ablation}"
            data = run_one(regime, ablation, regime_args)
            if data:
                all_results[key] = data.get("summary", {})

    print(f"\n\n{'='*70}")
    print("ABLATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Regime':<22} {'Ablation':<14} {'Det%':>6} {'MedLat':>7} {'Purity':>7} {'Cover':>7}")
    print("-" * 70)
    for key, summary in all_results.items():
        regime, ablation = key.split("/")
        gl = summary.get("guardlens", {})
        det = f"{gl.get('detection_rate', 0):.0%}"
        lat = gl.get("median_latency")
        lat_s = f"{lat:.1f}" if lat is not None else "N/A"
        pur = gl.get("mean_purity_at_detection")
        pur_s = f"{pur:.2f}" if pur is not None else "N/A"
        cov = gl.get("mean_coverage_at_detection")
        cov_s = f"{cov:.2f}" if cov is not None else "N/A"
        print(f"{regime:<22} {ablation:<14} {det:>6} {lat_s:>7} {pur_s:>7} {cov_s:>7}")

    combined_path = RESULTS_DIR / "ablation_summary_realistic.json"
    combined_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {combined_path}")


if __name__ == "__main__":
    main()
