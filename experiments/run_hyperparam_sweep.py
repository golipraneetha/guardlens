"""Hyperparameter sensitivity sweep for GuardLens (Issue #6: reviewers ask
why window=3, top_k=3, min_cluster_size=5, match_threshold=0.85 were
chosen, with no supporting sensitivity data beyond Table VIII's
match-threshold sweep on R2 -- which is architecturally a no-op there
since R2 is a single-burst regime with no cross-cycle matching to test).

This sweeps window_size, top_k, and min_cluster_size on R1 (Novel
Family) instead, where temporal tracking is actually exercised across
multiple cycles (onset at cycle 5, gradual ramp) -- unlike R2, where the
attack fully arrives in one cycle and cross-cycle mechanics never
engage.

Per user instruction, this sweep runs against the realistic-traffic,
unseen-attack-variant condition (Tier B+C: LLM-paraphrased +
novel-intent AdvBench variants, realistic Alpaca+OASST1+UltraChat
benign pool) established in run_realistic_eval.py / Table IX, not the
original raw-benchmark-replay condition -- so sensitivity conclusions
aren't confounded with the "known benchmark" evaluation weakness
Table IX was built to address.

Traffic generation (build_batches) does not depend on window_size,
top_k, or min_cluster_size -- only on regime/seed/benign_per_cycle/
onset_cycle/cycles/attack pool -- so every run in this sweep draws
IDENTICAL batches per seed, and the existing Tier B+C DeBERTa score
cache (experiments/realistic_eval/score_cache/novel_family_realistic_bc.json)
covers 100% of the text ever scored here. No DeBERTa rescoring occurs.

Usage:
    python3 experiments/run_hyperparam_sweep.py
    python3 experiments/run_hyperparam_sweep.py --params window_size top_k
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REGIME = "novel_family"
SEEDS = 5
CYCLES = 10
TRAFFIC_SOURCE = "realistic"
ATTACK_TIER = "bc"
N_PER_TIER = 50
VARIANT_MODEL = "qwen3:8b"

SCORE_CACHE = (Path(__file__).parent / "realistic_eval" / "score_cache"
               / "novel_family_realistic_bc.json")

OUT_DIR = Path(__file__).parent / "hyperparam_sweep"

DEFAULTS = dict(window_size=3, top_k=3, min_cluster_size=5)

SWEEPS = {
    "window_size": [2, 3, 4, 5],
    "top_k": [1, 3, 5],
    "min_cluster_size": [3, 5, 8, 10],
}

CLI_FLAG = {
    "window_size": "--window-size",
    "top_k": "--top-k",
    "min_cluster_size": "--min-cluster-size",
}


def run_one(param: str, value: int, benign_per_cycle: int) -> Path:
    config = dict(DEFAULTS)
    config[param] = value

    suffix = f"_bpc{benign_per_cycle}" if benign_per_cycle != 200 else ""
    out_file = OUT_DIR / f"{param}_{value}{suffix}.json"

    cmd = [
        sys.executable, str(Path(__file__).parent / "run_experiment.py"),
        "--regime", REGIME,
        "--seeds", str(SEEDS),
        "--cycles", str(CYCLES),
        "--benign-per-cycle", str(benign_per_cycle),
        "--traffic-source", TRAFFIC_SOURCE,
        "--attack-tier", ATTACK_TIER,
        "--n-per-tier", str(N_PER_TIER),
        "--variant-model", VARIANT_MODEL,
        "--window-size", str(config["window_size"]),
        "--top-k", str(config["top_k"]),
        "--min-cluster-size", str(config["min_cluster_size"]),
        "--score-cache", str(SCORE_CACHE),
        "--out", str(out_file),
    ]

    print(f"\n{'='*70}")
    print(f"  sweep: {param}={value}  benign_per_cycle={benign_per_cycle}  "
          f"(window={config['window_size']}, "
          f"top_k={config['top_k']}, min_cluster_size={config['min_cluster_size']})")
    print(f"{'='*70}\n")

    subprocess.run(cmd, check=True)
    return out_file


def print_table(param: str, values: list[int], results_files: dict[int, Path],
                benign_per_cycle: int = 200):
    label = {
        "window_size": "Window Size (cycles)",
        "top_k": "Top-K (queue budget)",
        "min_cluster_size": "min_cluster_size (HDBSCAN)",
    }[param]

    print(f"\n**Table — {label} Sensitivity "
          f"(R1: Novel Family, unseen variants, realistic traffic, "
          f"benign_per_cycle={benign_per_cycle}, 5 seeds)**\n")
    print(f"| {label} | Detection Rate | Median Latency | Mean Purity | Mean Coverage |")
    print("|---|---|---|---|---|")

    rows = []
    for v in values:
        f = results_files[v]
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        gl = data["summary"]["guardlens"]
        default_marker = " (default)" if v == DEFAULTS[param] else ""
        det = gl["detection_rate"]
        lat = gl["median_latency"]
        pur = gl.get("mean_purity_at_detection")
        cov = gl.get("mean_coverage_at_detection")

        det_str = f"{det:.0%}"
        lat_str = f"{lat}" if lat is not None else "—"
        pur_str = f"{pur:.2f}" if isinstance(pur, (int, float)) else "—"
        cov_str = f"{cov:.2f}" if isinstance(cov, (int, float)) else "—"

        row = f"| {v}{default_marker} | {det_str} | {lat_str} | {pur_str} | {cov_str} |"
        print(row)
        rows.append((v, det, lat, pur, cov))

    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", nargs="+", default=list(SWEEPS.keys()),
                    choices=list(SWEEPS.keys()))
    ap.add_argument("--benign-per-cycle", type=int, default=200)
    args = ap.parse_args()
    bpc = args.benign_per_cycle

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SCORE_CACHE.exists():
        print(f"WARNING: expected score cache not found at {SCORE_CACHE}. "
              f"Run run_realistic_eval.py for novel_family first, or this "
              f"sweep will re-score DeBERTa from scratch for every run.")

    all_results: dict[str, dict[int, Path]] = {}
    for param in args.params:
        all_results[param] = {}
        for value in SWEEPS[param]:
            out = run_one(param, value, bpc)
            all_results[param][value] = out

    print(f"\n\n{'='*70}")
    print(f"HYPERPARAMETER SENSITIVITY SUMMARY (benign_per_cycle={bpc})")
    print(f"{'='*70}")

    summary = {}
    for param in args.params:
        rows = print_table(param, SWEEPS[param], all_results[param], bpc)
        summary[param] = rows

    suffix = f"_bpc{bpc}" if bpc != 200 else ""
    summary_path = OUT_DIR / f"summary{suffix}.json"
    summary_path.write_text(json.dumps(
        {p: [{"value": v, "detection_rate": d, "median_latency": l,
              "mean_purity": pu, "mean_coverage": c}
             for v, d, l, pu, c in rows]
         for p, rows in summary.items()},
        indent=2))
    print(f"\nWrote summary to {summary_path}")


if __name__ == "__main__":
    main()
