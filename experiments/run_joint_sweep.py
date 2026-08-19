"""Joint hyperparameter sweep (Section V-H / reviewer ask #6): sweep
match_threshold x growth_floor x score_weights jointly, rather than one
parameter at a time (Table VIII / run_hyperparam_sweep.py), to check
whether the fixed default priors sit in a stable neighborhood of the
response surface -- not to find an "optimal" config.

Framing matters here: if this sweep instead searched for whichever point
maximizes detection rate, a reviewer could reasonably ask why the paper
doesn't just use that point. The claim this sweep supports is narrower
and more defensible: performance stays within a tight band across a
broad neighborhood around the defaults, so the fixed choice isn't
fragile, without claiming it's optimal.

Grid (80 configs per regime):
  match_threshold: [0.75, 0.80, 0.85, 0.90, 0.95]   (default 0.85)
  growth_floor:     [0.00, 0.05, 0.10, 0.20]          (default 0.10)
  score_weights:    equal(1,1,1) / density_heavy(2,1,1) /
                    growth_heavy(1,2,1) / novelty_heavy(1,1,2)  (default equal)

Run on R1 (novel_family) and R2 (coordinated_attack) with realistic
traffic, Tier B+C unseen variants (the Table IX condition) -- the two
regimes where cross-cycle matching (match_threshold) and score-weight
choices can actually matter. R3 (slow_drift) is excluded: its
per-cycle attack volume is small enough that most sweep points behave
similarly regardless of these parameters, which would dilute the
stability signal rather than sharpen it.

None of match_threshold/growth_floor/score_weights affect traffic
generation or DeBERTa scoring, so all 80 configs for a regime share one
score cache -- the default-config run (first in the sweep) populates it.

Usage:
    python3 experiments/run_joint_sweep.py
    python3 experiments/run_joint_sweep.py --regimes novel_family --seeds 10
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

REGIMES = ["novel_family", "coordinated_attack"]
SEEDS = 10
CYCLES = 10
TRAFFIC_SOURCE = "realistic"
ATTACK_TIER = "bc"
N_PER_TIER = 50
VARIANT_MODEL = "qwen3:8b"
COMMUNITY = "Advanced"
BURST_SIZE = 30

MATCH_THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.95]
GROWTH_FLOORS = [0.0, 0.05, 0.10, 0.20]
SCORE_WEIGHTS = {
    "equal": (1, 1, 1),
    "density_heavy": (2, 1, 1),
    "growth_heavy": (1, 2, 1),
    "novelty_heavy": (1, 1, 2),
}

DEFAULT_MATCH_THRESHOLD = 0.85
DEFAULT_GROWTH_FLOOR = 0.10
DEFAULT_WEIGHTS_NAME = "equal"

OUT_DIR = Path(__file__).parent / "joint_sweep"


def regime_extra_args(regime: str) -> list[str]:
    if regime == "coordinated_attack":
        return ["--burst-size", str(BURST_SIZE), "--community", COMMUNITY]
    return []


def run_one(regime: str, seeds: int, match_threshold: float, growth_floor: float,
           weights_name: str, score_cache: Path) -> Path:
    weights = SCORE_WEIGHTS[weights_name]
    weights_str = ",".join(str(w) for w in weights)
    tag = f"mt{match_threshold}_gf{growth_floor}_w{weights_name}"
    out_file = OUT_DIR / regime / f"{tag}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(Path(__file__).parent / "run_experiment.py"),
        "--regime", regime,
        "--seeds", str(seeds),
        "--cycles", str(CYCLES),
        "--traffic-source", TRAFFIC_SOURCE,
        "--attack-tier", ATTACK_TIER,
        "--n-per-tier", str(N_PER_TIER),
        "--variant-model", VARIANT_MODEL,
        "--match-threshold", str(match_threshold),
        "--growth-floor", str(growth_floor),
        "--score-weights", weights_str,
        "--score-cache", str(score_cache),
        "--out", str(out_file),
    ] + regime_extra_args(regime)

    print(f"\n{'='*70}\n  {regime}: match_threshold={match_threshold} "
         f"growth_floor={growth_floor} weights={weights_name}{weights}\n{'='*70}\n")
    subprocess.run(cmd, check=True)
    return out_file


def stability_summary(regime: str, results: dict[tuple, Path]) -> dict:
    default_key = (DEFAULT_MATCH_THRESHOLD, DEFAULT_GROWTH_FLOOR, DEFAULT_WEIGHTS_NAME)
    rows = []
    for key, path in results.items():
        data = json.loads(path.read_text())
        gl = data["summary"]["guardlens"]
        rows.append(dict(
            match_threshold=key[0], growth_floor=key[1], weights=key[2],
            detection_rate=gl["detection_rate"],
            median_latency=gl["median_latency"],
            mean_purity=gl.get("mean_purity_at_detection"),
        ))

    det_rates = [r["detection_rate"] for r in rows]
    default_row = next(r for r in rows if
                       (r["match_threshold"], r["growth_floor"], r["weights"]) == default_key)

    import numpy as np
    return dict(
        regime=regime,
        n_configs=len(rows),
        default=default_row,
        detection_rate_min=min(det_rates),
        detection_rate_max=max(det_rates),
        detection_rate_median=float(np.median(det_rates)),
        n_configs_at_or_above_default=sum(1 for d in det_rates if d >= default_row["detection_rate"]),
        n_configs_below_default=sum(1 for d in det_rates if d < default_row["detection_rate"]),
        all_rows=rows,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", nargs="+", default=REGIMES, choices=REGIMES)
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--match-thresholds", type=float, nargs="+", default=MATCH_THRESHOLDS)
    ap.add_argument("--growth-floors", type=float, nargs="+", default=GROWTH_FLOORS)
    ap.add_argument("--weights-names", nargs="+", default=list(SCORE_WEIGHTS.keys()),
                    choices=list(SCORE_WEIGHTS.keys()),
                    help="Subset of the weights grid -- useful for a quick smoke test.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    grid = list(itertools.product(args.match_thresholds, args.growth_floors, args.weights_names))
    default_key = (DEFAULT_MATCH_THRESHOLD, DEFAULT_GROWTH_FLOOR, DEFAULT_WEIGHTS_NAME)
    print(f"Grid: {len(grid)} configs x {len(args.regimes)} regimes x {args.seeds} seeds")

    all_summaries = {}
    for regime in args.regimes:
        score_cache = OUT_DIR / f"{regime}_score_cache.json"

        print(f"\n{'#'*70}\n# {regime}: warming score cache with default config\n{'#'*70}")
        results = {default_key: run_one(regime, args.seeds, *default_key, score_cache)}

        for key in grid:
            if key == default_key:
                continue
            results[key] = run_one(regime, args.seeds, *key, score_cache)

        summary = stability_summary(regime, results)
        all_summaries[regime] = summary

        print(f"\n{'='*70}\n{regime} STABILITY SUMMARY\n{'='*70}")
        print(f"  default (match_threshold={DEFAULT_MATCH_THRESHOLD}, "
             f"growth_floor={DEFAULT_GROWTH_FLOOR}, weights={DEFAULT_WEIGHTS_NAME}): "
             f"detection_rate={summary['default']['detection_rate']:.1%}")
        print(f"  grid range: [{summary['detection_rate_min']:.1%}, "
             f"{summary['detection_rate_max']:.1%}]  median={summary['detection_rate_median']:.1%}")
        print(f"  {summary['n_configs_at_or_above_default']}/{summary['n_configs']} configs "
             f"at or above default detection rate")

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(all_summaries, indent=2))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
