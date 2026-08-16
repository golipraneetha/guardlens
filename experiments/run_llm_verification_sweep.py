"""LLM verification sweep: measure precision improvement from LLM-based
cluster triage across all 3 regimes.

Runs each regime with and without --llm-verify, then compares precision@K,
detection latency, and recall.  Uses existing DeBERTa score caches so the
only slow step is the Ollama LLM calls (cached after first run).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REGIMES = {
    "novel_family": [],
    "coordinated_attack": ["--community", "Advanced"],
    "slow_drift": ["--harmbench-category", "cybercrime_intrusion"],
}

BASE_ARGS = [
    "--seeds", "5", "--cycles", "10", "--benign-per-cycle", "200",
    "--top-k", "3", "--window-size", "3", "--onset-cycle", "5",
    "--burst-size", "30",
    "--traffic-source", "realistic", "--attack-tier", "bc",
    "--n-per-tier", "50",
]

LLM_MODEL = "llama3.1:latest"

RESULTS_DIR = Path(__file__).parent / "llm_verification_realistic_llama31"
CACHE_DIR = Path(__file__).parent / "llm_cache"
SCORE_CACHE_PATH = Path(__file__).parent / "master_score_cache.json"


def run_one(regime: str, regime_args: list[str], use_llm: bool) -> dict | None:
    tag = f"{regime}_llm" if use_llm else f"{regime}_baseline"
    out_path = RESULTS_DIR / f"{tag}.json"

    cmd = [
        sys.executable, str(Path(__file__).parent / "run_experiment.py"),
        "--regime", regime,
        "--score-cache", str(SCORE_CACHE_PATH),
        "--out", str(out_path),
        *regime_args,
        *BASE_ARGS,
    ]
    if use_llm:
        llm_cache = CACHE_DIR / f"{regime}_{LLM_MODEL.replace(':', '_')}_realistic.json"
        cmd.extend([
            "--llm-verify",
            "--llm-model", LLM_MODEL,
            "--llm-cache", str(llm_cache),
        ])

    label = f"{regime} ({'+ LLM' if use_llm else 'baseline'})"
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  -> {out_path}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    print(f"  finished in {elapsed:.1f}s (exit {result.returncode})")

    if result.returncode != 0:
        return None
    return json.loads(out_path.read_text())


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    all_results: dict[str, dict] = {}

    for regime, extra_args in REGIMES.items():
        for use_llm in [False, True]:
            tag = f"{regime}_{'llm' if use_llm else 'baseline'}"
            data = run_one(regime, extra_args, use_llm)
            if data:
                all_results[tag] = data.get("summary", {})

    print(f"\n\n{'='*80}")
    print("LLM VERIFICATION SWEEP SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Regime':<22} {'Mode':<10} {'Det%':>5} {'Lat':>5} {'Pur':>6} "
          f"{'Prec@K':>7} {'VDet%':>6} {'VLat':>5} {'VPrec':>7}")
    print("-" * 80)

    for regime in REGIMES:
        bl = all_results.get(f"{regime}_baseline", {})
        ll = all_results.get(f"{regime}_llm", {})

        gl_bl = bl.get("guardlens", {})
        gl_ll = ll.get("guardlens", {})
        gv = ll.get("guardlens_verified", {})
        prec = ll.get("precision", {})

        det = f"{gl_bl.get('detection_rate', 0):.0%}"
        lat = gl_bl.get("median_latency")
        lat_s = f"{lat:.1f}" if lat is not None else "N/A"
        pur = gl_bl.get("mean_purity_at_detection")
        pur_s = f"{pur:.2f}" if pur is not None else "N/A"

        uprec = prec.get("unverified_mean")
        uprec_s = f"{uprec:.2f}" if uprec is not None else "N/A"
        vdet = f"{gv.get('detection_rate', 0):.0%}"
        vlat = gv.get("median_latency")
        vlat_s = f"{vlat:.1f}" if vlat is not None else "N/A"
        vprec = prec.get("verified_mean")
        vprec_s = f"{vprec:.2f}" if vprec is not None else "N/A"

        print(f"{regime:<22} {'BL+LLM':<10} {det:>5} {lat_s:>5} {pur_s:>6} "
              f"{uprec_s:>7} {vdet:>6} {vlat_s:>5} {vprec_s:>7}")

    combined_path = RESULTS_DIR / "verification_summary_realistic_llama31.json"
    combined_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {combined_path}")


if __name__ == "__main__":
    main()
