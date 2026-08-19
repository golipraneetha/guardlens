"""R4 -- benign demand shift study (Section V-E).

GuardLens is designed to surface persistent semantic emergence, not to
classify intent -- so a legitimate benign topic that appears and grows
(a product launch, a seasonal spike) should score highly on the same
Emergence Score that flags attacks. That's expected behavior, not a
defect, but it has a real cost: every benign-trend cluster that reaches
the fixed-budget Top-K queue is a slot an analyst spends on a false
positive. This script quantifies that cost rather than assuming it away.

Two sub-experiments, both using traffic.regimes.benign_demand_shift_regime
(R4). The "trend" is a coding-related topical subset drawn from across
all three realistic-traffic sources (Alpaca+OASST1+UltraChat, keyword-
filtered -- see _CODING_PATTERN below), standing in for a product-launch-
style adoption spike (e.g. a new coding-assistant feature). This was
chosen empirically, not assumed: treating a whole source (e.g. all of
UltraChat) as "the trend" doesn't work -- each of the three sources is
itself a diverse, multi-topic corpus, so HDBSCAN can't form one coherent
cluster from a random subsample of any single source (verified: mean
pairwise cosine within a random UltraChat sample is ~0.07, indistinguishable
from cross-source ~0.03-0.06). A keyword-narrowed topical subset is
~2x tighter (within-topic cosine ~0.13 vs. ~0.06 for the non-topical
remainder), which is what actually lets a real cluster form. The baseline
pool is the non-topical remainder of all three sources.

  1. pure_benign  -- no attacks, 15 cycles, 5 seeds. How many
     benign-trend clusters reach Top-K, and what Emergence Scores do
     they get, from natural topic diversity alone?
  2. with_attacks -- same benign demand shift, plus an AdvBench attack
     family injected starting cycle 8 (R1-style ramp). Does GuardLens's
     attack detection rate hold under queue competition with benign
     emerging clusters, and how much analyst load do the benign
     clusters add on top?

Usage:
    python3 experiments/run_benign_study.py
    python3 experiments/run_benign_study.py --seeds 5 --cycles 15
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guardlens.monitor import GuardLensMonitor
from guardlens.embedder import Embedder
from traffic.datasets import load_advbench
from traffic.realistic_traffic import load_realistic_benign_sources
from traffic.regimes import benign_demand_shift_regime, TrendSpec
from traffic.streams import BenignPool
from experiments.metrics import (cluster_purity, coverage, cluster_origin_label,
                                build_cycle_ledger, benign_cluster_rate,
                                queue_pollution, analyst_burden)

THRESHOLD = 0.5
OUT_DIR = Path(__file__).parent / "benign_study"

_CODING_KEYWORDS = ["python", "code", "function", "programming", "debug",
                    "script", "algorithm", "javascript", "sql", "api"]
_CODING_PATTERN = re.compile("|".join(_CODING_KEYWORDS), re.IGNORECASE)


def _split_topical(all_texts: list[str]) -> tuple[list[str], list[str]]:
    """Returns (topical, non_topical) -- see module docstring for why a
    keyword-narrowed subset, not a whole source, is used as the trend."""
    topical, non_topical = [], []
    for t in all_texts:
        (topical if _CODING_PATTERN.search(t) else non_topical).append(t)
    return topical, non_topical


def approved_view_with_origins(classifier, batch, bypass_attacks=False):
    apr_texts, apr_labels, apr_origins = [], [], []
    for t, l, o in zip(batch.texts, batch.labels, batch.origins):
        if bypass_attacks and l == 1:
            apr_texts.append(t)
            apr_labels.append(l)
            apr_origins.append(o)
            continue
        s = classifier.confidence(t)
        if s < THRESHOLD:
            apr_texts.append(t)
            apr_labels.append(l)
            apr_origins.append(o)
    return apr_texts, apr_labels, apr_origins


def build_seed_batches(seed: int, cycles: int, benign_per_cycle: int, with_attacks: bool):
    sources = load_realistic_benign_sources()
    all_texts = sources["alpaca"] + sources["oasst1"] + sources["ultrachat"]
    topical, non_topical = _split_topical(all_texts)

    baseline_pool = BenignPool(non_topical, seed=seed)
    trend_pool = BenignPool(topical, seed=seed + 1000)
    trend = TrendSpec(pool=trend_pool, name="coding_topic", onset_cycle=5,
                      peak_cycle=9, end_cycle=13, peak_frac=0.35)

    attack_pool = load_advbench() if with_attacks else None
    attack_onset_cycle = 8
    return benign_demand_shift_regime(
        baseline_pool, [trend], n_cycles=cycles, benign_per_cycle=benign_per_cycle,
        attack_pool=attack_pool, attack_onset_cycle=attack_onset_cycle, seed=seed)


def run_one_seed(classifier, embedder, seed: int, batches, meta, top_k: int = 3,
                 window_size: int = 3, min_cluster_size: int = 5) -> dict:
    with_attacks = meta["attack_onset_cycle"] is not None
    monitor = GuardLensMonitor(embedder=embedder, window_size=window_size,
                              top_k=top_k, min_cluster_size=min_cluster_size)

    origins_history: list[list[str]] = []
    labels_history: list[list[int]] = []
    ledger = []
    all_cluster_records = []   # every scored cluster, every cycle -- for Figure 5
    attack_detection_cycle = None
    per_cycle_log = []

    for batch in batches:
        apr_texts, apr_labels, apr_origins = approved_view_with_origins(classifier, batch)

        origins_history.append(apr_origins)
        origins_history = origins_history[-window_size:]
        labels_history.append(apr_labels)
        labels_history = labels_history[-window_size:]
        window_origins = [o for ct in origins_history for o in ct]
        window_is_attack = np.array([l for ct in labels_history for l in ct])

        result = monitor.process_cycle(batch.cycle, apr_texts)

        for c in result.clusters:
            label = cluster_origin_label(c.indices, window_origins)
            all_cluster_records.append(dict(
                cycle=batch.cycle, uid=c.uid, label=label,
                emergence=c.emergence, density=c.density,
                growth=c.growth, novelty=c.novelty, size=c.size,
            ))

        cycle_ledger = build_cycle_ledger(batch.cycle, result.queue, window_origins)
        ledger.extend(cycle_ledger)

        if with_attacks and len(window_is_attack) > 0:
            has_attack_cluster = any(
                cluster_purity(e.cluster.indices, window_is_attack) >= 0.5
                for e in result.queue)
            if has_attack_cluster and attack_detection_cycle is None:
                attack_detection_cycle = batch.cycle

        per_cycle_log.append(dict(
            cycle=batch.cycle, n_queue=len(result.queue),
            n_clusters=len(result.clusters),
            benign_trend_in_queue=sum(1 for e in cycle_ledger if e.label == "benign_trend"),
            attack_in_queue=sum(1 for e in cycle_ledger if e.label == "attack"),
        ))

    out = dict(
        seed=seed, trend_windows=meta["trend_windows"],
        attack_onset_cycle=meta["attack_onset_cycle"],
        attack_detection_cycle=attack_detection_cycle,
        benign_cluster_rate=benign_cluster_rate(ledger),
        queue_pollution=queue_pollution(ledger),
        analyst_burden=analyst_burden(ledger),
        n_topk_slots=len(ledger),
        per_cycle_log=per_cycle_log,
        cluster_records=all_cluster_records,
    )
    return out


def summarize(results: list[dict], with_attacks: bool) -> dict:
    n = len(results)
    summary = dict(
        n_seeds=n,
        mean_benign_cluster_rate=float(np.mean([r["benign_cluster_rate"] for r in results])),
        mean_queue_pollution=float(np.mean([r["queue_pollution"] for r in results])),
        mean_analyst_burden=float(np.mean([r["analyst_burden"] for r in results])),
    )
    if with_attacks:
        hits = [r for r in results if r["attack_detection_cycle"] is not None]
        summary["attack_detection_rate"] = len(hits) / n
        summary["attack_detection_hits"] = f"{len(hits)}/{n}"
        lats = [r["attack_detection_cycle"] - r["attack_onset_cycle"] for r in hits]
        summary["mean_attack_latency"] = float(np.mean(lats)) if lats else None

    # Emergence Score distributions by ground-truth label, pooled across
    # seeds -- the raw material for Figure 5.
    by_label: dict[str, list[float]] = {}
    for r in results:
        for c in r["cluster_records"]:
            by_label.setdefault(c["label"], []).append(c["emergence"])
    summary["emergence_by_label"] = {
        label: dict(n=len(vals), mean=float(np.mean(vals)), median=float(np.median(vals)),
                   p25=float(np.percentile(vals, 25)), p75=float(np.percentile(vals, 75)))
        for label, vals in by_label.items() if vals
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--cycles", type=int, default=15)
    ap.add_argument("--benign-per-cycle", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building traffic for all seeds x conditions...")
    all_batches = {}   # (with_attacks, seed) -> (batches, meta)
    for with_attacks in (False, True):
        for seed in range(args.seeds):
            all_batches[(with_attacks, seed)] = build_seed_batches(
                seed, args.cycles, args.benign_per_cycle, with_attacks)

    all_texts = list(dict.fromkeys(
        t for (batches, _meta) in all_batches.values() for b in batches for t in b.texts
    ))

    score_cache_path = OUT_DIR / "score_cache.json"
    cached_scores = {}
    if score_cache_path.exists():
        cached_scores = json.loads(score_cache_path.read_text())
        print(f"Loaded {len(cached_scores)} cached DeBERTa scores")

    if cached_scores and all(t in cached_scores for t in all_texts):
        print("All texts found in score cache -- skipping DeBERTa entirely.")

        class CachedClassifier:
            def __init__(self, cache): self._cache = cache
            def confidence(self, text): return self._cache[text]

        classifier = CachedClassifier(cached_scores)
    else:
        print("Loading DeBERTa injection classifier...")
        from real_classifiers import DebertaInjectionClassifier
        classifier = DebertaInjectionClassifier()
        print(f"Scoring {len(all_texts)} unique texts (batched)...")
        t0 = time.time()
        classifier.warm_cache(all_texts)
        print(f"  done in {time.time()-t0:.1f}s")
        scores_dict = {t: classifier.confidence(t) for t in all_texts}
        score_cache_path.write_text(json.dumps(scores_dict))
        print(f"Saved score cache ({len(scores_dict)} entries)")

    print("Loading embedder...")
    embedder = Embedder()

    for with_attacks, name in [(False, "pure_benign"), (True, "with_attacks")]:
        print(f"\n{'='*60}\n{name}\n{'='*60}")
        results = []
        for seed in range(args.seeds):
            batches, meta = all_batches[(with_attacks, seed)]
            t0 = time.time()
            r = run_one_seed(classifier, embedder, seed, batches, meta, top_k=args.top_k)
            results.append(r)
            print(f"  seed {seed}: benign_cluster_rate={r['benign_cluster_rate']:.2f} "
                  f"queue_pollution={r['queue_pollution']:.2f} "
                  f"analyst_burden={r['analyst_burden']:.2f} "
                  f"({time.time()-t0:.1f}s)")
            if with_attacks:
                print(f"    attack detected at cycle {r['attack_detection_cycle']} "
                     f"(onset={r['attack_onset_cycle']})")

        summary = summarize(results, with_attacks)
        print(f"\n{name} summary:")
        print(json.dumps({k: v for k, v in summary.items() if k != "emergence_by_label"},
                         indent=2))

        out_path = OUT_DIR / f"{name}.json"
        out_path.write_text(json.dumps(dict(config=vars(args), results=results,
                                            summary=summary), indent=2))
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
