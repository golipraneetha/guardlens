"""Adversarial evasion sweep: an attacker aware of clustering-based
detection splits the same burst volume across N semantically distinct
jailbreak communities. Sweep N in {1, 2, 3, 4, 5} and observe how
GuardLens detection latency, cluster purity, coverage, and fragmentation
change as diversity increases.

N=1 corresponds to the coordinated_attack regime baseline (Regime 2).
Higher N is a direct evasion move: each sub-family gets 30 // N templates,
which for N=5 falls below HDBSCAN's min_cluster_size default of 5, so a
naive per-family cluster cannot form.

Uses a single shared DeBERTa score cache -- the community-attack texts are
the same across sweep points (only the community selection changes), so a
single warmed cache serves the whole sweep.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

# Chosen for size (>=30 items each in jailbreak_llms) and to give the
# attacker a plausible attack budget across 5 semantically distinct
# communities. Order matters: index i is used for the N=i+1 sweep point.
COMMUNITIES = ["Advanced", "Toxic", "Anarchy", "Narrative", "Exception"]

SWEEP_N = [1, 2, 3, 4, 5]

BASE_ARGS = [
    "--seeds", "5", "--cycles", "10", "--benign-per-cycle", "200",
    "--top-k", "3", "--window-size", "3", "--onset-cycle", "5",
    "--burst-size", "30",
    "--attack-bypass-guardrail",
]

RESULTS_DIR = Path(__file__).parent / "evasion"
CACHE_DIR = Path(__file__).parent / "score_cache"


def run_one(n: int) -> dict | None:
    communities = ",".join(COMMUNITIES[:n])
    out_path = RESULTS_DIR / f"diverse_N{n}.json"
    cache_path = CACHE_DIR / "diverse_attack.json"
    cmd = [
        sys.executable, str(Path(__file__).parent / "run_experiment.py"),
        "--regime", "diverse_attack",
        "--communities", communities,
        "--score-cache", str(cache_path),
        "--out", str(out_path),
        *BASE_ARGS,
    ]
    print(f"\n{'='*60}")
    print(f"  N={n}  communities=[{communities}]")
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
    CACHE_DIR.mkdir(exist_ok=True)

    all_results = {}
    for n in SWEEP_N:
        data = run_one(n)
        if data:
            all_results[f"N={n}"] = data.get("summary", {})

    print(f"\n\n{'='*80}")
    print("EVASION SWEEP SUMMARY (5 seeds per point, K=3, burst_size=30)")
    print(f"{'='*80}")
    print(f"{'N':>3} {'GL Det%':>8} {'MedLat':>7} {'Purity':>7} {'Coverage':>9} "
          f"{'Fragmt':>7} {'RA Det%':>8} {'MMD Det%':>9} {'OS Det%':>8}")
    print("-" * 80)
    for key, summary in all_results.items():
        n = int(key.split("=")[1])
        gl = summary.get("guardlens", {})
        ra = summary.get("random_audit", {})
        mmd = summary.get("mmd_drift", {})
        os_c = summary.get("one_shot_cluster", {})
        det = f"{gl.get('detection_rate', 0):.0%}"
        lat = gl.get("median_latency")
        lat_s = f"{lat:.1f}" if lat is not None else "N/A"
        pur = gl.get("mean_purity_at_detection")
        pur_s = f"{pur:.2f}" if pur is not None else "N/A"
        cov = gl.get("mean_coverage_at_detection")
        cov_s = f"{cov:.2f}" if cov is not None else "N/A"
        frag = gl.get("mean_fragmentation_at_detection")
        frag_s = f"{frag:.2f}" if frag is not None else "N/A"
        ra_det = f"{ra.get('detection_rate', 0):.0%}"
        mmd_det = f"{mmd.get('detection_rate', 0):.0%}"
        os_det = f"{os_c.get('detection_rate', 0):.0%}"
        print(f"{n:>3} {det:>8} {lat_s:>7} {pur_s:>7} {cov_s:>9} {frag_s:>7} "
              f"{ra_det:>8} {mmd_det:>9} {os_det:>8}")

    combined_path = RESULTS_DIR / "evasion_summary.json"
    combined_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {combined_path}")


if __name__ == "__main__":
    main()
