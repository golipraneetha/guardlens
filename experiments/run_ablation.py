"""Run ablation study: leave-one-out on each emergence score component.

Produces results for: full, no_density, no_growth, no_novelty
across all 3 regimes. Uses --score-cache so DeBERTa scoring only
happens once per regime (~20 min), then ablation variants are instant.

Default seed count is 20, not 5 (Section V-D reviewer ask: some ablation
deltas -- notably R2 novelty removal, 60%->80% detection at n=5 -- were
borderline enough at n=5 that they could plausibly be sampling noise
rather than a real effect. n=20 narrows confidence intervals enough to
tell the two apart; raw hit counts (e.g. 16/20) are reported alongside
percentages since detection is a binary per-seed outcome and a count is
more informative than a rate alone at this sample size.

--parallel runs the 3 non-default ablations for a regime concurrently
*after* that regime's score cache has been warmed by the "full" run --
parallelizing across ablations within a regime is safe (traffic and
DeBERTa scores are identical across ablations, so once cached it's a
read-only cache hit); parallelizing across regimes is NOT attempted here
because they share one score-cache file, and run_experiment.py's cache
write on a miss overwrites the file with only the current run's texts
(fine for the existing fully-sequential regime-outer loop, but a race
under cross-regime parallelism).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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


def run_one(regime: str, ablation: str, regime_args: list[str], seeds: int) -> dict | None:
    out_path = RESULTS_DIR / f"{regime}_{ablation}.json"
    cmd = [
        sys.executable, str(Path(__file__).parent / "run_experiment.py"),
        "--regime", regime,
        "--seeds", str(seeds),
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--parallel", type=int, default=1,
                    help="Max concurrent ablation runs within a regime, after that "
                         "regime's score cache is warmed. 1 = fully sequential.")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    all_results = {}
    for regime, regime_args in REGIMES.items():
        # "full" first, sequentially -- this is what warms the shared score
        # cache for this regime; the remaining ablations only ever read it.
        key = f"{regime}/full"
        data = run_one(regime, "full", regime_args, args.seeds)
        if data:
            all_results[key] = data.get("summary", {})

        remaining = [a for a in ABLATIONS if a != "full"]
        if args.parallel > 1:
            with ThreadPoolExecutor(max_workers=min(args.parallel, len(remaining))) as ex:
                futures = {ex.submit(run_one, regime, ablation, regime_args, args.seeds): ablation
                          for ablation in remaining}
                for fut in futures:
                    ablation = futures[fut]
                    data = fut.result()
                    if data:
                        all_results[f"{regime}/{ablation}"] = data.get("summary", {})
        else:
            for ablation in remaining:
                data = run_one(regime, ablation, regime_args, args.seeds)
                if data:
                    all_results[f"{regime}/{ablation}"] = data.get("summary", {})

    print(f"\n\n{'='*70}")
    print(f"ABLATION SUMMARY (n={args.seeds} seeds)")
    print(f"{'='*70}")
    print(f"{'Regime':<22} {'Ablation':<14} {'Det (hits/n)':>14} {'Det%':>6} "
         f"{'MedLat':>7} {'Purity':>7} {'Cover':>7}")
    print("-" * 84)
    for key in (f"{r}/{a}" for r in REGIMES for a in ABLATIONS):
        summary = all_results.get(key)
        if not summary:
            continue
        regime, ablation = key.split("/")
        gl = summary.get("guardlens", {})
        hits = gl.get("detection_hits")
        n = gl.get("n_seeds")
        hits_s = f"{hits}/{n}" if hits is not None and n is not None else "N/A"
        det = f"{gl.get('detection_rate', 0):.0%}"
        lat = gl.get("median_latency")
        lat_s = f"{lat:.1f}" if lat is not None else "N/A"
        pur = gl.get("mean_purity_at_detection")
        pur_s = f"{pur:.2f}" if pur is not None else "N/A"
        cov = gl.get("mean_coverage_at_detection")
        cov_s = f"{cov:.2f}" if cov is not None else "N/A"
        print(f"{regime:<22} {ablation:<14} {hits_s:>14} {det:>6} "
             f"{lat_s:>7} {pur_s:>7} {cov_s:>7}")

    combined_path = RESULTS_DIR / f"ablation_summary_realistic_n{args.seeds}.json"
    combined_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {combined_path}")


if __name__ == "__main__":
    main()
