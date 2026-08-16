"""GuardLens benchmark harness.

Wires together: real deployed classifier (DeBERTa injection detector) ->
a traffic regime -> GuardLensMonitor + three baselines, all under the same
per-cycle review budget, across multiple seeds.

Regimes:
    novel_family        -- advbench items (Regime 1)
    coordinated_attack   -- a burst of near-duplicate templates from one
                            jailbreak_llms community (Regime 2)
    slow_drift           -- harmbench items, ramping in gradually (Regime 3)

Usage:
    python3 experiments/run_experiment.py --regime novel_family --seeds 3 --cycles 10
    python3 experiments/run_experiment.py --regime coordinated_attack --community Advanced
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guardlens.monitor import GuardLensMonitor
from guardlens.embedder import Embedder
from guardlens.emergence import ABLATION_MODES
from traffic.datasets import (load_benign_pool, load_advbench, load_harmbench,
                              load_jailbreak_community_templates)
from traffic.streams import BenignPool
from traffic.regimes import (novel_family_regime, coordinated_attack_regime,
                             slow_drift_regime, diverse_attack_regime)
from traffic.realistic_traffic import load_realistic_benign_pool
from traffic.attack_variants import build_tiered_attack_pool
from baselines.random_audit import RandomAuditBaseline
from baselines.stratified_random import StratifiedRandomBaseline
from baselines.mmd_drift import MMDDriftBaseline
from baselines.one_shot_cluster import OneShotClusterBaseline
from baselines.isolation_forest import IsolationForestBaseline
from experiments.metrics import (cluster_purity, coverage, attack_cluster_fragmentation,
                                precision_at_k, false_positive_reduction,
                                recall_preservation)

THRESHOLD = 0.5


def approved_view(classifier, batch, bypass_attacks=False):
    apr_texts, apr_labels, apr_scores = [], [], []
    for t, l in zip(batch.texts, batch.labels):
        if bypass_attacks and l == 1:
            apr_texts.append(t)
            apr_labels.append(l)
            apr_scores.append(0.0)
        else:
            s = classifier.confidence(t)
            if s < THRESHOLD:
                apr_texts.append(t)
                apr_labels.append(l)
                apr_scores.append(s)
    return apr_texts, apr_labels, apr_scores


def run_one_seed(classifier, embedder, seed: int, batches, onset_cycle: int,
                 top_k: int, window_size: int, min_cluster_size: int,
                 ref_cycles: int = 3, ablation: str = "full",
                 verifier=None, bypass_attacks: bool = False,
                 match_threshold: float = 0.85) -> dict:
    monitor = GuardLensMonitor(embedder=embedder, window_size=window_size,
                              top_k=top_k, min_cluster_size=min_cluster_size,
                              ablation=ablation, match_threshold=match_threshold)
    random_audit = RandomAuditBaseline(budget=top_k, seed=seed)
    stratified = StratifiedRandomBaseline(budget=top_k, seed=seed)
    mmd = MMDDriftBaseline(alpha=0.05, n_perm=100, seed=seed)
    one_shot = OneShotClusterBaseline(top_k=top_k, min_cluster_size=min_cluster_size)
    iforest = IsolationForestBaseline(top_k=top_k, random_state=seed)

    per_cycle_log = []
    gl_detection_cycle = None
    gl_purity_at_detection = None
    gl_coverage_at_detection = None
    gl_fragmentation_at_detection = None
    os_detection_cycle = None
    os_purity_at_detection = None
    os_coverage_at_detection = None
    ref_embeddings = []
    approved_labels_history = []   # mirrors the monitor's own sliding window
    approved_texts_history = []
    gl_verified_detection_cycle = None
    gl_verified_purity_at_detection = None
    gl_verified_coverage_at_detection = None
    gl_verified_precision_at_k = []
    gl_unverified_precision_at_k = []
    llm_call_latencies = []  # wall-clock seconds, cache hits excluded (0.0)

    for batch in batches:
        apr_texts, apr_labels, apr_scores = approved_view(classifier, batch,
                                                         bypass_attacks=bypass_attacks)

        # cluster.indices are positions into the monitor's internal sliding
        # window (last `window_size` cycles' approved texts concatenated),
        # not just this cycle's items -- reconstruct the same window here.
        approved_labels_history.append(apr_labels)
        approved_labels_history = approved_labels_history[-window_size:]
        approved_texts_history.append(apr_texts)
        approved_texts_history = approved_texts_history[-window_size:]
        window_is_attack = np.array(
            [l for cycle_labels in approved_labels_history for l in cycle_labels]
        )
        window_flat_texts = [t for ct in approved_texts_history for t in ct]

        t0 = time.time()
        result = monitor.process_cycle(batch.cycle, apr_texts)
        cycle_wall_seconds = time.time() - t0

        verified_queue = result.queue
        verdicts = []
        if verifier is not None and result.queue:
            embs = embedder.encode(window_flat_texts)
            verified_queue, verdicts = verifier.verify_queue(
                result.queue, window_flat_texts, embs)
            llm_call_latencies.extend(
                v.latency_seconds for v in verdicts if v.latency_seconds > 0.0)

        top1_purity = None
        if result.queue:
            top1_purity = cluster_purity(result.queue[0].cluster.indices, window_is_attack)
        cov = coverage(result.queue, window_is_attack) if len(window_is_attack) else 0.0

        if result.queue and len(window_is_attack) > 0:
            gl_unverified_precision_at_k.append(
                precision_at_k(result.queue, window_is_attack))
        if verifier is not None and verified_queue and len(window_is_attack) > 0:
            gl_verified_precision_at_k.append(
                precision_at_k(verified_queue, window_is_attack))

        has_attack_cluster = any(
            cluster_purity(e.cluster.indices, window_is_attack) >= 0.5 for e in result.queue
        )
        if has_attack_cluster and gl_detection_cycle is None:
            gl_detection_cycle = batch.cycle
            gl_purity_at_detection = max(
                cluster_purity(e.cluster.indices, window_is_attack) for e in result.queue
            )
            gl_coverage_at_detection = cov
            gl_fragmentation_at_detection = attack_cluster_fragmentation(
                result.queue, window_is_attack)

        if verifier is not None:
            has_verified_attack = any(
                cluster_purity(e.cluster.indices, window_is_attack) >= 0.5
                for e in verified_queue
            )
            if has_verified_attack and gl_verified_detection_cycle is None:
                gl_verified_detection_cycle = batch.cycle
                gl_verified_purity_at_detection = max(
                    cluster_purity(e.cluster.indices, window_is_attack)
                    for e in verified_queue
                )
                gl_verified_coverage_at_detection = coverage(
                    verified_queue, window_is_attack)

        random_audit.process_cycle(batch.cycle, apr_texts, apr_labels)
        stratified.process_cycle(batch.cycle, apr_texts, apr_labels, apr_scores)

        if apr_texts:
            embs = embedder.encode(apr_texts)
            if batch.cycle <= ref_cycles:
                ref_embeddings.append(embs)
                if batch.cycle == ref_cycles:
                    mmd.set_reference(np.vstack(ref_embeddings))
            elif mmd.reference is not None:
                mmd.process_cycle(batch.cycle, embs)

            # one-shot clustering: this cycle's items only, no window --
            # cluster indices are positions into apr_labels directly.
            cycle_is_attack = np.array(apr_labels)
            os_entries = one_shot.process_cycle(batch.cycle, embs)
            os_has_attack_cluster = any(
                cluster_purity(e.cluster.indices, cycle_is_attack) >= 0.5 for e in os_entries
            )
            if os_has_attack_cluster and os_detection_cycle is None:
                os_detection_cycle = batch.cycle
                one_shot.first_detection_cycle = batch.cycle
                os_purity_at_detection = max(
                    cluster_purity(e.cluster.indices, cycle_is_attack) for e in os_entries
                )
                os_coverage_at_detection = coverage(os_entries, cycle_is_attack)

            # Isolation Forest: per-cycle fit on this cycle's embeddings,
            # flags top-K most anomalous points as candidate attacks.
            iforest.process_cycle(batch.cycle, embs, apr_labels)

        cycle_entry = dict(
            cycle=batch.cycle, n_approved=len(apr_texts),
            n_attacks_approved=int(sum(apr_labels)),
            top1_purity=top1_purity, coverage=cov,
            n_queue=len(result.queue),
            cycle_wall_seconds=cycle_wall_seconds,
        )
        if verifier is not None:
            cycle_entry["n_queue_verified"] = len(verified_queue)
            cycle_entry["n_rejected_by_llm"] = len(result.queue) - len(verified_queue)
        per_cycle_log.append(cycle_entry)

    result_dict = dict(
        seed=seed, onset_cycle=onset_cycle,
        guardlens=dict(detection_cycle=gl_detection_cycle,
                       purity_at_detection=gl_purity_at_detection,
                       coverage_at_detection=gl_coverage_at_detection,
                       fragmentation_at_detection=gl_fragmentation_at_detection),
        random_audit=dict(detection_cycle=random_audit.first_detection_cycle),
        stratified_random=dict(detection_cycle=stratified.first_detection_cycle),
        mmd_drift=dict(detection_cycle=mmd.first_detection_cycle),
        one_shot_cluster=dict(detection_cycle=os_detection_cycle,
                              purity_at_detection=os_purity_at_detection,
                              coverage_at_detection=os_coverage_at_detection),
        isolation_forest=dict(detection_cycle=iforest.first_detection_cycle),
        per_cycle_log=per_cycle_log,
    )
    if verifier is not None:
        result_dict["guardlens_verified"] = dict(
            detection_cycle=gl_verified_detection_cycle,
            purity_at_detection=gl_verified_purity_at_detection,
            coverage_at_detection=gl_verified_coverage_at_detection,
        )
        result_dict["precision"] = dict(
            unverified_mean=float(np.mean(gl_unverified_precision_at_k))
                if gl_unverified_precision_at_k else None,
            verified_mean=float(np.mean(gl_verified_precision_at_k))
                if gl_verified_precision_at_k else None,
        )
        result_dict["llm_call_latencies_seconds"] = llm_call_latencies
    return result_dict


def summarize_cycle_wall_clock(results: list[dict]) -> dict:
    all_times = [c["cycle_wall_seconds"] for r in results
                 for c in r["per_cycle_log"] if "cycle_wall_seconds" in c]
    if not all_times:
        return {}
    return dict(
        n_cycles=len(all_times),
        mean_seconds=float(np.mean(all_times)),
        median_seconds=float(np.median(all_times)),
        p95_seconds=float(np.percentile(all_times, 95)),
        max_seconds=float(np.max(all_times)),
    )


def _t_ci_half_width(values: list[float], alpha: float = 0.05) -> float | None:
    """Half-width of a t-based (1-alpha) confidence interval, or None if
    fewer than 2 values.  Reported alongside means so tables can show
    'mean ± half_width' compactly."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return None
    from scipy import stats
    n = len(vals)
    std = float(np.std(vals, ddof=1))
    t_crit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    return t_crit * std / np.sqrt(n)


def summarize(results: list[dict]) -> dict:
    def latencies(method_key):
        return [r[method_key]["detection_cycle"] - r["onset_cycle"]
                for r in results if r[method_key]["detection_cycle"] is not None]

    def detection_rate(method_key):
        return sum(1 for r in results if r[method_key]["detection_cycle"] is not None) / len(results)

    def detection_hits(method_key):
        return sum(1 for r in results if r[method_key]["detection_cycle"] is not None)

    n_seeds = len(results)

    summary = {}
    for method_key in ["guardlens", "random_audit", "stratified_random", "mmd_drift",
                       "one_shot_cluster", "isolation_forest"]:
        lat = latencies(method_key)
        summary[method_key] = dict(
            detection_rate=detection_rate(method_key),
            detection_hits=detection_hits(method_key),
            n_seeds=n_seeds,
            median_latency=float(np.median(lat)) if lat else None,
            mean_latency=float(np.mean(lat)) if lat else None,
            latency_ci95_halfwidth=_t_ci_half_width(lat),
            latencies=lat,
        )
    for method_key in ["guardlens", "one_shot_cluster"]:
        purities = [r[method_key]["purity_at_detection"] for r in results
                    if r[method_key]["purity_at_detection"] is not None]
        coverages = [r[method_key]["coverage_at_detection"] for r in results
                     if r[method_key]["coverage_at_detection"] is not None]
        summary[method_key]["mean_purity_at_detection"] = float(np.mean(purities)) if purities else None
        summary[method_key]["purity_ci95_halfwidth"] = _t_ci_half_width(purities)
        summary[method_key]["mean_coverage_at_detection"] = float(np.mean(coverages)) if coverages else None
        summary[method_key]["coverage_ci95_halfwidth"] = _t_ci_half_width(coverages)
    # GuardLens-only: fragmentation (how many distinct clusters the flagged
    # attack was split across at first detection)
    frags = [r["guardlens"].get("fragmentation_at_detection") for r in results
             if r["guardlens"].get("fragmentation_at_detection") is not None]
    summary["guardlens"]["mean_fragmentation_at_detection"] = float(np.mean(frags)) if frags else None
    summary["cycle_wall_clock"] = summarize_cycle_wall_clock(results)

    if any("guardlens_verified" in r for r in results):
        vl = [r["guardlens_verified"]["detection_cycle"] - r["onset_cycle"]
              for r in results
              if r.get("guardlens_verified", {}).get("detection_cycle") is not None]
        vdr = sum(1 for r in results
                  if r.get("guardlens_verified", {}).get("detection_cycle") is not None) / len(results)
        vpur = [r["guardlens_verified"]["purity_at_detection"] for r in results
                if r.get("guardlens_verified", {}).get("purity_at_detection") is not None]
        vcov = [r["guardlens_verified"]["coverage_at_detection"] for r in results
                if r.get("guardlens_verified", {}).get("coverage_at_detection") is not None]
        summary["guardlens_verified"] = dict(
            detection_rate=vdr,
            median_latency=float(np.median(vl)) if vl else None,
            mean_latency=float(np.mean(vl)) if vl else None,
            mean_purity_at_detection=float(np.mean(vpur)) if vpur else None,
            mean_coverage_at_detection=float(np.mean(vcov)) if vcov else None,
        )
        uprec = [r["precision"]["unverified_mean"] for r in results
                 if r.get("precision", {}).get("unverified_mean") is not None]
        vprec = [r["precision"]["verified_mean"] for r in results
                 if r.get("precision", {}).get("verified_mean") is not None]
        summary["precision"] = dict(
            unverified_mean=float(np.mean(uprec)) if uprec else None,
            verified_mean=float(np.mean(vprec)) if vprec else None,
        )
        all_llm_latencies = [
            lat for r in results
            for lat in r.get("llm_call_latencies_seconds", [])
        ]
        summary["llm_wall_clock"] = dict(
            n_real_calls=len(all_llm_latencies),
            mean_seconds=float(np.mean(all_llm_latencies)) if all_llm_latencies else None,
            median_seconds=float(np.median(all_llm_latencies)) if all_llm_latencies else None,
            max_seconds=float(np.max(all_llm_latencies)) if all_llm_latencies else None,
            total_seconds=float(np.sum(all_llm_latencies)) if all_llm_latencies else None,
        )
    return summary


def build_batches(regime: str, seed: int, args, benign_pool_texts, attack_pools):
    benign_pool = BenignPool(benign_pool_texts, seed=seed)
    if regime == "novel_family":
        return novel_family_regime(
            benign_pool, attack_pools["advbench"], n_cycles=args.cycles,
            benign_per_cycle=args.benign_per_cycle, onset_cycle=args.onset_cycle, seed=seed)
    elif regime == "coordinated_attack":
        return coordinated_attack_regime(
            benign_pool, attack_pools["community"], n_cycles=args.cycles,
            benign_per_cycle=args.benign_per_cycle, onset_cycle=args.onset_cycle,
            burst_size=args.burst_size, seed=seed)
    elif regime == "slow_drift":
        return slow_drift_regime(
            benign_pool, attack_pools["harmbench"], n_cycles=args.cycles,
            benign_per_cycle=args.benign_per_cycle, onset_cycle=args.onset_cycle,
            ramp_per_cycle_pct=args.ramp_pct, seed=seed)
    elif regime == "diverse_attack":
        community_names = [c.strip() for c in args.communities.split(",") if c.strip()]
        return diverse_attack_regime(
            benign_pool, attack_pools["communities"], community_names,
            n_cycles=args.cycles, benign_per_cycle=args.benign_per_cycle,
            onset_cycle=args.onset_cycle, burst_size=args.burst_size, seed=seed)
    raise ValueError(f"unknown regime: {regime}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["novel_family", "coordinated_attack",
                                          "slow_drift", "diverse_attack"],
                    default="novel_family")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--benign-per-cycle", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--window-size", type=int, default=3)
    ap.add_argument("--min-cluster-size", type=int, default=5)
    ap.add_argument("--match-threshold", type=float, default=0.85,
                    help="Cosine similarity threshold for matching a cluster to its "
                         "predecessor across cycles (ClusterRegistry)")
    ap.add_argument("--embedding-model", type=str,
                    default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Sentence-transformers model name for embeddings")
    ap.add_argument("--onset-cycle", type=int, default=5)
    ap.add_argument("--burst-size", type=int, default=30)     # coordinated_attack only
    ap.add_argument("--community", type=str, default="Advanced")  # coordinated_attack only
    ap.add_argument("--communities", type=str, default="Advanced,Toxic,Anarchy",
                    help="Comma-separated community names for diverse_attack regime.")
    ap.add_argument("--ramp-pct", type=float, default=0.015)  # slow_drift only
    ap.add_argument("--harmbench-category", type=str, default="cybercrime_intrusion")  # slow_drift only
    ap.add_argument("--ablation", choices=list(ABLATION_MODES), default="full")
    ap.add_argument("--score-cache", type=str, default=None,
                    help="Path to DeBERTa score cache JSON. Saves after first run, "
                         "loads on subsequent runs to skip the ~20min scoring phase.")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--attack-bypass-guardrail", action="store_true",
                    help="Attack texts skip DeBERTa filtering (evasion experiment: "
                         "isolates clustering fragmentation from guardrail effectiveness)")
    ap.add_argument("--traffic-source", choices=["benchmark", "realistic"],
                    default="benchmark",
                    help="Benign traffic source: 'benchmark' uses jailbreak_llms_regular "
                         "(original), 'realistic' uses Alpaca+OASST1+UltraChat mix")
    ap.add_argument("--attack-tier", choices=["a", "b", "c", "bc"],
                    default="a",
                    help="Attack variant tier: 'a'=raw benchmark (original), "
                         "'b'=paraphrased, 'c'=novel-intent, 'bc'=b+c combined")
    ap.add_argument("--variant-model", type=str, default="qwen3:8b",
                    help="Ollama model for generating attack variants (tier b/c)")
    ap.add_argument("--n-per-tier", type=int, default=50,
                    help="Number of attack variants per tier")
    ap.add_argument("--llm-verify", action="store_true",
                    help="Enable LLM cluster verification (requires Ollama)")
    ap.add_argument("--llm-model", type=str, default="llama3.1",
                    help="Ollama model for cluster verification")
    ap.add_argument("--llm-cache", type=str, default=None,
                    help="Path to LLM verification cache JSON")
    ap.add_argument("--llm-samples", type=int, default=6,
                    help="Number of texts to sample per cluster for LLM")
    args = ap.parse_args()

    if args.out is None:
        args.out = str(Path(__file__).parent / f"{args.regime}_results.json")

    print("Loading datasets...")
    if args.traffic_source == "realistic":
        print("  Using realistic benign traffic (Alpaca + OASST1 + UltraChat)...")
        benign_pool_texts = load_realistic_benign_pool()
    else:
        benign_pool_texts = load_benign_pool()
    print(f"  benign pool: {len(benign_pool_texts)} ({args.traffic_source})")

    attack_pools = {}
    variant_cache_dir = Path(__file__).parent / "variant_cache"

    if args.regime == "novel_family":
        raw_attacks = load_advbench()
        if args.attack_tier == "a":
            attack_pools["advbench"] = raw_attacks
        else:
            tiers = build_tiered_attack_pool(
                raw_attacks, category="harmful_behavior",
                n_per_tier=args.n_per_tier, model=args.variant_model,
                cache_dir=variant_cache_dir)
            if args.attack_tier == "b":
                attack_pools["advbench"] = tiers["tier_b"]
            elif args.attack_tier == "c":
                attack_pools["advbench"] = tiers["tier_c"]
            else:  # "bc"
                attack_pools["advbench"] = tiers["tier_b"] + tiers["tier_c"]
        print(f"  advbench attacks: {len(attack_pools['advbench'])} (tier={args.attack_tier})")

    elif args.regime == "coordinated_attack":
        communities = load_jailbreak_community_templates()
        print(f"  available communities: { {k: len(v) for k, v in communities.items()} }")
        raw_community = communities[args.community]
        if args.attack_tier == "a":
            attack_pools["community"] = raw_community
        else:
            tiers = build_tiered_attack_pool(
                raw_community, category="jailbreak_template",
                n_per_tier=args.n_per_tier, model=args.variant_model,
                cache_dir=variant_cache_dir)
            if args.attack_tier == "b":
                attack_pools["community"] = tiers["tier_b"]
            elif args.attack_tier == "c":
                attack_pools["community"] = tiers["tier_c"]
            else:
                attack_pools["community"] = tiers["tier_b"] + tiers["tier_c"]
        print(f"  community '{args.community}': {len(attack_pools['community'])} (tier={args.attack_tier})")

    elif args.regime == "slow_drift":
        raw_harmbench = load_harmbench(semantic_category=args.harmbench_category)
        if args.attack_tier == "a":
            attack_pools["harmbench"] = raw_harmbench
        else:
            tiers = build_tiered_attack_pool(
                raw_harmbench, category=args.harmbench_category,
                n_per_tier=args.n_per_tier, model=args.variant_model,
                cache_dir=variant_cache_dir)
            if args.attack_tier == "b":
                attack_pools["harmbench"] = tiers["tier_b"]
            elif args.attack_tier == "c":
                attack_pools["harmbench"] = tiers["tier_c"]
            else:
                attack_pools["harmbench"] = tiers["tier_b"] + tiers["tier_c"]
        print(f"  harmbench[{args.harmbench_category}]: {len(attack_pools['harmbench'])} (tier={args.attack_tier})")

    elif args.regime == "diverse_attack":
        communities = load_jailbreak_community_templates()
        print(f"  available communities: { {k: len(v) for k, v in communities.items()} }")
        selected = [c.strip() for c in args.communities.split(",") if c.strip()]
        attack_pools["communities"] = {name: communities[name] for name in selected}
        print(f"  selected communities: { {n: len(v) for n,v in attack_pools['communities'].items()} }")

    print(f"\nGenerating '{args.regime}' traffic for {args.seeds} seeds x {args.cycles} cycles...")
    batches_by_seed = {}
    onset_by_seed = {}
    for seed in range(args.seeds):
        batches, onset = build_batches(args.regime, seed, args, benign_pool_texts, attack_pools)
        batches_by_seed[seed] = batches
        onset_by_seed[seed] = onset

    if args.attack_bypass_guardrail:
        all_texts = list(dict.fromkeys(
            t for batches in batches_by_seed.values() for b in batches
            for t, l in zip(b.texts, b.labels) if l == 0
        ))
    else:
        all_texts = list(dict.fromkeys(
            t for batches in batches_by_seed.values() for b in batches for t in b.texts
        ))

    score_cache_path = Path(args.score_cache) if args.score_cache else None
    cached_scores: dict[str, float] | None = None
    if score_cache_path and score_cache_path.exists():
        cached_scores = json.loads(score_cache_path.read_text())
        print(f"Loaded {len(cached_scores)} cached DeBERTa scores from {score_cache_path}")

    if cached_scores and all(t in cached_scores for t in all_texts):
        print("All texts found in score cache — skipping DeBERTa entirely.")

        class CachedClassifier:
            def __init__(self, cache): self._cache = cache
            def confidence(self, text): return self._cache[text]

        classifier = CachedClassifier(cached_scores)
    else:
        print("Loading DeBERTa injection classifier...")
        from real_classifiers import DebertaInjectionClassifier
        classifier = DebertaInjectionClassifier()
        print(f"Scoring {len(all_texts)} unique texts...")
        t0 = time.time()
        classifier.warm_cache(all_texts)
        print(f"  done in {time.time() - t0:.1f}s")
        if score_cache_path:
            scores_dict = {t: classifier.confidence(t) for t in all_texts}
            score_cache_path.write_text(json.dumps(scores_dict))
            print(f"Saved score cache ({len(scores_dict)} entries) to {score_cache_path}")

    print(f"Loading embedder ({args.embedding_model})...")
    embedder = Embedder(model_name=args.embedding_model)

    verifier = None
    if args.llm_verify:
        from guardlens.llm_verifier import ClusterVerifier
        llm_cache_path = Path(args.llm_cache) if args.llm_cache else None
        verifier = ClusterVerifier(
            model=args.llm_model,
            n_samples=args.llm_samples,
            cache_path=llm_cache_path,
        )
        print(f"LLM verifier enabled: model={args.llm_model}, samples={args.llm_samples}")

    results = []
    for seed in range(args.seeds):
        print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")
        r = run_one_seed(classifier, embedder, seed, batches_by_seed[seed],
                         onset_by_seed[seed], args.top_k, args.window_size,
                         args.min_cluster_size, ablation=args.ablation,
                         verifier=verifier,
                         bypass_attacks=args.attack_bypass_guardrail,
                         match_threshold=args.match_threshold)
        results.append(r)
        print(f"  GuardLens detected at cycle {r['guardlens']['detection_cycle']} "
              f"(onset={r['onset_cycle']}, purity={r['guardlens']['purity_at_detection']})")
        print(f"  Random audit detected at cycle {r['random_audit']['detection_cycle']}")
        print(f"  Stratified random detected at cycle {r['stratified_random']['detection_cycle']}")
        print(f"  MMD drift detected at cycle {r['mmd_drift']['detection_cycle']}")
        print(f"  One-shot cluster detected at cycle {r['one_shot_cluster']['detection_cycle']} "
              f"(purity={r['one_shot_cluster']['purity_at_detection']})")
        print(f"  Isolation Forest detected at cycle {r['isolation_forest']['detection_cycle']}")
        if "guardlens_verified" in r:
            gv = r["guardlens_verified"]
            print(f"  GuardLens+LLM detected at cycle {gv['detection_cycle']} "
                  f"(purity={gv['purity_at_detection']})")
            print(f"  Precision: unverified={r['precision']['unverified_mean']:.2f}, "
                  f"verified={r['precision']['verified_mean']}")

    if verifier:
        verifier.save_cache()

    summary = summarize(results)

    print(f"\n{'='*60}\nSUMMARY ({args.seeds} seeds, regime={args.regime})\n{'='*60}")
    method_keys = ["guardlens", "random_audit", "stratified_random", "mmd_drift",
                   "one_shot_cluster", "isolation_forest"]
    if "guardlens_verified" in summary:
        method_keys.append("guardlens_verified")
    for method in method_keys:
        s = summary[method]
        print(f"\n{method}:")
        print(f"  detection rate: {s['detection_rate']:.1%}")
        print(f"  median latency: {s['median_latency']}")
        print(f"  mean latency: {s['mean_latency']}")
        if method in ("guardlens", "one_shot_cluster", "guardlens_verified"):
            print(f"  mean purity at detection: {s.get('mean_purity_at_detection')}")
            print(f"  mean coverage at detection: {s.get('mean_coverage_at_detection')}")
    if summary.get("cycle_wall_clock"):
        cwc = summary["cycle_wall_clock"]
        print(f"\ncycle wall-clock (embed+cluster+score, incremental cache):")
        print(f"  n_cycles: {cwc.get('n_cycles')}")
        print(f"  mean:   {cwc.get('mean_seconds')}")
        print(f"  median: {cwc.get('median_seconds')}")
        print(f"  p95:    {cwc.get('p95_seconds')}")
        print(f"  max:    {cwc.get('max_seconds')}")
    if "precision" in summary:
        p = summary["precision"]
        print(f"\nprecision@K:")
        print(f"  unverified: {p.get('unverified_mean')}")
        print(f"  verified:   {p.get('verified_mean')}")
    if "llm_wall_clock" in summary:
        w = summary["llm_wall_clock"]
        print(f"\nllm wall-clock (real calls only, cache hits excluded):")
        print(f"  n_real_calls: {w.get('n_real_calls')}")
        print(f"  mean:   {w.get('mean_seconds')}")
        print(f"  median: {w.get('median_seconds')}")
        print(f"  max:    {w.get('max_seconds')}")
        print(f"  total:  {w.get('total_seconds')}")

    out_path = Path(args.out)
    out_path.write_text(json.dumps(dict(config=vars(args), results=results,
                                        summary=summary), indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
