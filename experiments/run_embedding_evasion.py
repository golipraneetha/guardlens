"""Embedding-aware adversary stress test (Section III-C.1 / V-G).

Reviewer ask: bring the embedding-aware adversary into the threat model
with at least a preliminary empirical result, rather than leaving it to
future work. traffic/regimes.py already has diverse_attack_regime and
experiments/run_evasion_sweep.py, which model an adversary who knows
GuardLens clusters and fragments a burst across N known jailbreak
communities to stay under HDBSCAN's min_cluster_size -- a real but
clustering-oblivious-to-the-embedding-model evasion move. This script
goes one step further: the adversary here knows the deployed embedding
model itself and uses it (via traffic.attack_variants.generate_diverse_
embeddings) to explicitly keep every attack variant's embedding far apart
from every other, rather than just producing surface-diverse paraphrases
that happen not to cluster.

Compares, under identical R1 (novel_family) ramp-onset traffic:
  standard        -- Tier B+C unseen variants (paraphrase + novel-intent,
                      the existing realistic-eval condition, Section IV-D)
  embedding_aware -- variants from generate_diverse_embeddings, explicitly
                      optimized to stay below max_cosine similarity to
                      every other accepted variant

Metrics: detection rate, detection latency, cluster purity, coverage,
mean cluster size at detection -- same headline metrics as Table IX.

Requires Ollama running locally with the configured --variant-model.

Usage:
    python3 experiments/run_embedding_evasion.py
    python3 experiments/run_embedding_evasion.py --seeds 5 --n-variants 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guardlens.embedder import Embedder
from traffic.datasets import load_advbench, load_benign_pool
from traffic.regimes import novel_family_regime, embedding_evasion_regime
from traffic.streams import BenignPool
from traffic.attack_variants import build_tiered_attack_pool, generate_diverse_embeddings
from experiments.run_experiment import run_one_seed
from experiments.metrics import cluster_purity

OUT_DIR = Path(__file__).parent / "embedding_evasion"
VARIANT_CACHE_DIR = Path(__file__).parent / "variant_cache"


def mean_cluster_size_at_detection(results: list[dict]) -> float | None:
    sizes = []
    for r in results:
        dc = r["guardlens"]["detection_cycle"]
        if dc is None:
            continue
        for entry in r["per_cycle_log"]:
            if entry["cycle"] == dc:
                # per_cycle_log doesn't carry raw cluster sizes, only
                # queue length -- approximate via n_queue as a proxy note.
                sizes.append(entry.get("n_queue"))
                break
    return float(np.mean(sizes)) if sizes else None


def build_condition_batches(attack_pool: list[str], benign_pool_texts: list[str],
                            seeds: int, cycles: int, benign_per_cycle: int,
                            onset_cycle: int) -> dict[int, tuple]:
    out = {}
    for seed in range(seeds):
        benign_pool = BenignPool(benign_pool_texts, seed=seed)
        out[seed] = embedding_evasion_regime(
            benign_pool, attack_pool, n_cycles=cycles,
            benign_per_cycle=benign_per_cycle, onset_cycle=onset_cycle, seed=seed)
    return out


def run_condition(name: str, batches_by_seed: dict[int, tuple], classifier,
                  embedder, top_k: int, window_size: int,
                  min_cluster_size: int) -> list[dict]:
    results = []
    for seed, (batches, onset) in batches_by_seed.items():
        t0 = time.time()
        r = run_one_seed(classifier, embedder, seed, batches, onset,
                         top_k, window_size, min_cluster_size)
        results.append(r)
        print(f"  [{name}] seed {seed}: detected at cycle "
             f"{r['guardlens']['detection_cycle']} (onset={onset}, "
             f"purity={r['guardlens']['purity_at_detection']}) "
             f"({time.time()-t0:.1f}s)")
    return results


def summarize(results: list[dict]) -> dict:
    n = len(results)
    hits = [r for r in results if r["guardlens"]["detection_cycle"] is not None]
    lats = [r["guardlens"]["detection_cycle"] - r["onset_cycle"] for r in hits]
    purities = [r["guardlens"]["purity_at_detection"] for r in hits
               if r["guardlens"]["purity_at_detection"] is not None]
    coverages = [r["guardlens"]["coverage_at_detection"] for r in hits
                if r["guardlens"]["coverage_at_detection"] is not None]
    return dict(
        n_seeds=n,
        detection_rate=len(hits) / n,
        detection_hits=f"{len(hits)}/{n}",
        mean_latency=float(np.mean(lats)) if lats else None,
        mean_purity_at_detection=float(np.mean(purities)) if purities else None,
        mean_coverage_at_detection=float(np.mean(coverages)) if coverages else None,
        mean_cluster_size_at_detection=mean_cluster_size_at_detection(results),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--benign-per-cycle", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--window-size", type=int, default=3)
    ap.add_argument("--min-cluster-size", type=int, default=5)
    ap.add_argument("--onset-cycle", type=int, default=5)
    ap.add_argument("--n-variants", type=int, default=30)
    ap.add_argument("--max-cosine", type=float, default=0.6)
    ap.add_argument("--max-attempts-per-variant", type=int, default=5)
    ap.add_argument("--variant-model", type=str, default="qwen3:8b")
    ap.add_argument("--n-per-tier", type=int, default=50)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    raw_attacks = load_advbench()
    benign_pool_texts = load_benign_pool()
    print(f"  advbench: {len(raw_attacks)}  benign pool: {len(benign_pool_texts)}")

    print("Loading embedder...")
    embedder = Embedder()

    print("\nBuilding standard (Tier B+C) attack pool...")
    tiers = build_tiered_attack_pool(
        raw_attacks, category="harmful_behavior", n_per_tier=args.n_per_tier,
        model=args.variant_model, cache_dir=VARIANT_CACHE_DIR)
    standard_pool = tiers["tier_b"] + tiers["tier_c"]
    print(f"  standard pool: {len(standard_pool)} variants")

    print("\nBuilding embedding-aware (dispersed) attack pool...")
    dispersed_pool = generate_diverse_embeddings(
        raw_attacks, embedder, n_variants=args.n_variants,
        max_cosine=args.max_cosine,
        max_attempts_per_variant=args.max_attempts_per_variant,
        model=args.variant_model,
        cache_path=VARIANT_CACHE_DIR / "harmful_behavior_disperse.json")
    print(f"  dispersed pool: {len(dispersed_pool)}/{args.n_variants} variants "
         f"(shortfall is itself a result -- see summary)")

    if not dispersed_pool:
        print("ERROR: dispersion produced zero variants (Ollama unavailable or "
             "misconfigured?). Falling back to threat-model text only -- see "
             "Section III-C.1 discussion; no empirical run possible.")
        sys.exit(1)

    print("\nBuilding traffic for all seeds x conditions...")
    standard_batches = build_condition_batches(
        standard_pool, benign_pool_texts, args.seeds, args.cycles,
        args.benign_per_cycle, args.onset_cycle)
    embedding_aware_batches = build_condition_batches(
        dispersed_pool, benign_pool_texts, args.seeds, args.cycles,
        args.benign_per_cycle, args.onset_cycle)

    all_texts = list(dict.fromkeys(
        t for (batches, _onset) in list(standard_batches.values()) + list(embedding_aware_batches.values())
        for b in batches for t in b.texts
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

    print(f"\n{'='*60}\nStandard (Tier B+C)\n{'='*60}")
    standard_results = run_condition(
        "standard", standard_batches, classifier, embedder,
        args.top_k, args.window_size, args.min_cluster_size)

    print(f"\n{'='*60}\nEmbedding-aware (dispersed)\n{'='*60}")
    embedding_aware_results = run_condition(
        "embedding_aware", embedding_aware_batches, classifier, embedder,
        args.top_k, args.window_size, args.min_cluster_size)

    standard_summary = summarize(standard_results)
    embedding_aware_summary = summarize(embedding_aware_results)
    embedding_aware_summary["dispersed_pool_shortfall"] = (
        f"{len(dispersed_pool)}/{args.n_variants}")

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print("standard:", json.dumps(standard_summary, indent=2))
    print("embedding_aware:", json.dumps(embedding_aware_summary, indent=2))

    out = dict(
        config=vars(args),
        standard=dict(results=standard_results, summary=standard_summary),
        embedding_aware=dict(results=embedding_aware_results,
                            summary=embedding_aware_summary),
    )
    out_path = OUT_DIR / "results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
