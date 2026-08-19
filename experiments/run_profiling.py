"""Computational profile (Section V-F / reviewer ask #2): per-cycle
wall-clock and peak-memory cost, broken down by pipeline stage (embed,
cluster, registry, score, queue), across increasing cycle sizes -- to back
the paper's "no inference-time latency" claim with a measured
asynchronous-monitoring cost instead of an absence claim alone.

Two passes per cycle size, run separately:
  timing  -- no tracemalloc (which adds real overhead to hot paths and
             would bias the numbers it's supposedly measuring alongside),
             median/p95/mean per stage over cycles x seeds. Embedder cache
             is cleared before every cycle so repeated draws from a bounded
             benign pool (cycling back on itself at large cycle sizes)
             don't understate embedding cost via cache hits -- real
             production traffic doesn't repeat verbatim text.
  memory  -- tracemalloc peak, one representative seed (allocator behavior
             at a given cycle size is reproducible enough across seeds
             that memory doesn't need the same replication as timing).

Cycle sizes default to [100, 200, 500, 1000, 2000]. 100/200/500 overlap
the paper's studied traffic-volume range (Section V-C); 1000/2000 are
scaling extrapolation beyond it and should be reported as such (Section
VII), not as a validated deployment point.

Usage:
    python3 experiments/run_profiling.py
    python3 experiments/run_profiling.py --cycle-sizes 100 500 2000 --cycles 5 --seeds 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guardlens.monitor import GuardLensMonitor
from guardlens.embedder import Embedder
from traffic.datasets import load_benign_pool
from traffic.streams import BenignPool

OUT_DIR = Path(__file__).parent / "profiling"
CYCLE_SIZES = [100, 200, 500, 1000, 2000]
N_CYCLES = 10
N_SEEDS = 3
STAGES = ["embed", "cluster", "registry", "score", "queue", "total"]


def run_timing_pass(embedder: Embedder, benign_pool_texts: list[str],
                    cycle_size: int, n_cycles: int, n_seeds: int) -> list[dict]:
    records = []
    for seed in range(n_seeds):
        monitor = GuardLensMonitor(embedder=embedder)
        pool = BenignPool(benign_pool_texts, seed=seed)
        for c in range(1, n_cycles + 1):
            embedder.clear_cache()
            texts = pool.draw(cycle_size)
            result = monitor.process_cycle(c, texts)
            records.append(dict(
                embed=result.timing.embed_seconds,
                cluster=result.timing.cluster_seconds,
                registry=result.timing.registry_seconds,
                score=result.timing.score_seconds,
                queue=result.timing.queue_seconds,
                total=result.timing.total_seconds,
            ))
    return records


def run_memory_pass(embedder: Embedder, benign_pool_texts: list[str],
                    cycle_size: int, n_cycles: int) -> float:
    monitor = GuardLensMonitor(embedder=embedder)
    pool = BenignPool(benign_pool_texts, seed=0)
    tracemalloc.start()
    peak_bytes = 0
    for c in range(1, n_cycles + 1):
        embedder.clear_cache()
        texts = pool.draw(cycle_size)
        monitor.process_cycle(c, texts)
        _current, peak = tracemalloc.get_traced_memory()
        peak_bytes = max(peak_bytes, peak)
    tracemalloc.stop()
    return peak_bytes / (1024 * 1024)


def summarize_stage(records: list[dict], stage: str) -> dict:
    vals_ms = [r[stage] * 1000 for r in records]
    return dict(
        median_ms=float(np.median(vals_ms)),
        p95_ms=float(np.percentile(vals_ms, 95)),
        mean_ms=float(np.mean(vals_ms)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle-sizes", type=int, nargs="+", default=CYCLE_SIZES)
    ap.add_argument("--cycles", type=int, default=N_CYCLES)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading benign pool...")
    benign_pool_texts = load_benign_pool()
    print(f"  pool size: {len(benign_pool_texts)}")

    print("Loading embedder...")
    embedder = Embedder()

    results = {}
    for cs in args.cycle_sizes:
        print(f"\n{'='*60}\ncycle_size={cs}\n{'='*60}")

        print("  timing pass...")
        t0 = time.time()
        records = run_timing_pass(embedder, benign_pool_texts, cs, args.cycles, args.seeds)
        print(f"  ({time.time()-t0:.1f}s wall, {len(records)} cycle observations)")

        print("  memory pass...")
        peak_mb = run_memory_pass(embedder, benign_pool_texts, cs, args.cycles)

        stage_summary = {stage: summarize_stage(records, stage) for stage in STAGES}
        results[cs] = dict(stages=stage_summary, peak_memory_mb=peak_mb,
                           n_observations=len(records))

        print(f"  total: median={stage_summary['total']['median_ms']:.1f}ms "
             f"p95={stage_summary['total']['p95_ms']:.1f}ms  peak_mem={peak_mb:.1f}MB")
        for stage in STAGES[:-1]:
            s = stage_summary[stage]
            print(f"    {stage:10s} median={s['median_ms']:8.2f}ms  p95={s['p95_ms']:8.2f}ms")

    out_path = OUT_DIR / "profile_results.json"
    out_path.write_text(json.dumps(dict(config=vars(args), results=results), indent=2))
    print(f"\nWrote {out_path}")

    print(f"\n{'='*60}\nTable III: Computational profile (median, ms unless noted)\n{'='*60}")
    print(f"{'Cycle Size':>10} | {'Embed':>9} | {'HDBSCAN':>9} | {'Registry':>9} | "
         f"{'Score':>8} | {'Total':>9} | {'Peak Mem (MB)':>13}")
    for cs in args.cycle_sizes:
        r = results[cs]["stages"]
        extrapolation = " *" if cs > 500 else "  "
        print(f"{cs:>10}{extrapolation}| {r['embed']['median_ms']:>9.1f} | "
             f"{r['cluster']['median_ms']:>9.1f} | {r['registry']['median_ms']:>9.2f} | "
             f"{r['score']['median_ms']:>8.2f} | {r['total']['median_ms']:>9.1f} | "
             f"{results[cs]['peak_memory_mb']:>13.1f}")
    if any(cs > 500 for cs in args.cycle_sizes):
        print("\n* beyond the paper's studied traffic-volume range (200/500, Section "
             "V-C) -- scaling extrapolation, not a validated deployment point.")


if __name__ == "__main__":
    main()
