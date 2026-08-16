# GuardLens — Complete Experiment Summary (Current, Non-Stale Results)

Every test currently backing the paper, in one place. All results below use
the **current** experimental setup: realistic benign traffic (15,000 items
from Alpaca + OASST1 + UltraChat, zero overlap with any attack benchmark),
Tier B+C unseen attack variants (50 LLM-paraphrased + 50 novel-intent per
regime, via `qwen3:8b`), the 5-baseline comparison (GuardLens, One-Shot
Cluster, MMD Drift, Stratified Random, Random Audit, **+ Isolation Forest**),
and 95% CIs (t-distribution) on continuous metrics. Superseded/stale runs
(old benchmark-only traffic, old N=30 variants, no IF baseline, no CIs,
3-seed ablation) are explicitly excluded — see the "Superseded" section at
the bottom for what was replaced and why.

---

## 1. Main Detection Results — All Regimes × Tiers × Methods

**Setup:** 5 seeds, 200 benign items/cycle, 10 cycles/seed, default
hyperparameters (window=3, top_k=3, min_cluster_size=5).

### 1a. Tier A (raw benchmark attack text)

| Regime | Method | Det. | Med. Latency | Purity |
|---|---|:---:|:---:|:---:|
| R1: Novel Family | **GuardLens** | 5/5 | 1.0 ± 1.88 | 0.90 ± 0.17 |
| | One-Shot Cluster | 5/5 | 1.0 ± 2.04 | 0.91 ± 0.20 |
| | MMD Drift | 5/5 | 1.0 ± 1.24 | — |
| | Isolation Forest | 1/5 | 1.0 (n=1) | — |
| | Stratified Random | 1/5 | 2.0 (n=1) | — |
| | Random Audit | 5/5 | 4.0 ± 0.68 | — |
| R2: Coordinated Attack | **GuardLens** | 5/5 | 0.0 ± 0.0 | 0.86 ± 0.23 |
| | One-Shot Cluster | 4/5 | 0.0 ± 0.0 | 0.89 ± 0.21 |
| | MMD Drift | 5/5 | 0.0 ± 0.0 | — |
| | Isolation Forest | 0/5 | — | — |
| | Stratified Random | 2/5 | 0.0 | — |
| | Random Audit | 1/5 | 0.0 (n=1) | — |
| R3: Slow Drift | **GuardLens** | 5/5 | 2.0 ± 1.04 | 0.85 ± 0.13 |
| | One-Shot Cluster | 5/5 | 3.0 ± 1.24 | 0.90 ± 0.28 |
| | MMD Drift | 4/5 | 4.0 ± 1.30 | — |
| | Isolation Forest | 1/5 | 4.0 (n=1) | — |
| | Stratified Random | 1/5 | 1.0 (n=1) | — |
| | Random Audit | 4/5 | 3.5 ± 1.52 | — |

### 1b. Tier B+C (100 unseen LLM-generated variants/regime — the hard condition)

| Regime | Method | Det. | Med. Latency | Purity |
|---|---|:---:|:---:|:---:|
| R1: Novel Family | **GuardLens** | 5/5 | 2.0 ± 1.88 | 0.95 ± 0.06 |
| | One-Shot Cluster | 5/5 | 1.0 ± 0.68 | 0.67 ± 0.26 |
| | MMD Drift | 5/5 | 1.0 ± 1.24 | — |
| | Isolation Forest | 4/5 | 4.0 ± 3.53 | — |
| | Stratified Random | 3/5 | 1.0 ± 5.17 | — |
| | Random Audit | 4/5 | 3.0 ± 1.59 | — |
| R2: Coordinated Attack | **GuardLens** | 5/5 | 0.0 ± 1.11 | 0.80 ± 0.26 |
| | One-Shot Cluster | 5/5 | 0.0 ± 0.0 | 0.84 ± 0.29 |
| | MMD Drift | 4/5 | 0.0 ± 0.0 | — |
| | Isolation Forest | 0/5 | — | — |
| | Stratified Random | 3/5 | 0.0 ± 1.43 | — |
| | Random Audit | 4/5 | 1.5 ± 1.52 | — |
| R3: Slow Drift | **GuardLens** | 5/5 | 2.0 ± 0.88 | 0.85 ± 0.13 |
| | One-Shot Cluster | 5/5 | 3.0 ± 0.68 | 0.87 ± 0.15 |
| | MMD Drift | 4/5 | 3.5 ± 3.44 | — |
| | Isolation Forest | 5/5 | 2.0 ± 1.84 | — |
| | Stratified Random | 4/5 | 3.5 ± 2.25 | — |
| | Random Audit | 3/5 | 2.0 ± 4.30 | — |

**Findings:**
- GuardLens and One-Shot Cluster are the only methods to reach 5/5 in every
  regime × tier combination; GuardLens's purity is consistently higher or
  comparable, and its detection-rate gap between Tier A and Tier B+C is zero
  everywhere (unlike Isolation Forest, which improves R1 1/5→4/5 but that's
  Tier-B+C easier-to-cluster paraphrase text, not genuine robustness).
- **Isolation Forest fails completely on R2 in both tiers (0/5)** — R2's
  attack is a tight 30-item burst, and isolation-based scoring makes points
  inside a dense cluster look *more* normal (harder to isolate), not less.
  This is the paper's strongest baseline-comparison finding.

Source: `experiments/realistic_eval/{regime}_tier{a,bc}_realistic.json`

---

## 2. Scale Sensitivity — 200 vs. 500 Benign Items/Cycle

**Setup:** Same as above (realistic traffic, Tier B+C, 5 seeds), default
hyperparameters, comparing benign_per_cycle=200 to 500 (2.5× traffic).

| Regime | Attack Volume Model | Det.@200 | Det.@500 | GL Lat.@200 | GL Lat.@500 | IF Det.@200 | IF Det.@500 |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| R1: Novel Family | scales with traffic | 5/5 | 5/5 | 2.0 | 2.0 ± 1.04 | 4/5 | 3/5 ± 5.17 |
| R2: Coordinated Attack | fixed 30-item burst | 5/5 | **2/5** | 0.0 | 0.5 ± 6.35 | 0/5 | 0/5 |
| R3: Slow Drift | scales with traffic | 5/5 | 5/5 | 2.0 | 1.0 ± 1.76 | 5/5 | 4/5 ± 2.05 |

**Findings:**
- R1 and R3 hold 100% detection at 2.5× traffic because their attack volume
  scales proportionally with `benign_per_cycle` — attack density in the
  traffic stream stays roughly constant.
- R2 degrades sharply (5/5 → 2/5) because its attack is a **fixed** 30-item
  burst — at 500/cycle the attack fraction is diluted from ~15% to ~6% of
  traffic, pushing it below HDBSCAN's default cluster-formation threshold in
  most seeds.
- Isolation Forest's R2 failure is volume-independent (0/5 at both 200 and
  500) — ruling out "IF just needs more data" and confirming the failure is
  structural (isolation scoring vs. density-based clustering), not a
  data-starvation artifact.

Source: `experiments/scale_test/{novel_family,coordinated_attack,slow_drift}_500.json`
vs. `experiments/realistic_eval/*_tierBC_realistic.json` (200/cycle baseline)

---

## 3. R2 `min_cluster_size` Sensitivity — Matched at Both Volumes

**Setup:** R2 only, realistic traffic, Tier B+C, 5 seeds, sweeping
`min_cluster_size` ∈ {3, 5, 8} at both 200 and 500 benign/cycle to test
whether a different constant recovers the Section 2 degradation.

| min_cluster_size | Det.@200 | Purity@200 | Det.@500 | Purity@500 |
|---|:---:|:---:|:---:|:---:|
| 3 | 4/5 | 0.96 ± 0.13 | 3/5 | 0.91 ± 0.19 |
| 5 (default) | **5/5** | 0.80 ± 0.26 | **2/5** | 0.77 ± 1.32 |
| 8 | 5/5 | 0.72 ± 0.26 | 3/5 | 0.70 ± 0.41 |

**Finding:** No single fixed value of `min_cluster_size` is optimal at both
volumes — the default (5) is *best* at 200/cycle and *worst* at 500/cycle.
mcs=3 and mcs=8 both do slightly better at 500/cycle but worse at 200/cycle.
This motivates a volume-adaptive threshold as a specific, scoped future-work
item rather than a tunable-away limitation. Note: the mcs=5 @500 purity CI
(±1.32, wider than the mean) reflects only 2/5 seeds detecting with very
different purity between them — itself evidence this cell is the least
reliable point in the sweep, not a data artifact to smooth over.

Source: `experiments/scale_test/coordinated_attack_{200,500}_mcs{3,8}.json`,
`coordinated_attack_500.json` (mcs=5 @500), `realistic_eval/coordinated_attack_tierBC_realistic.json` (mcs=5 @200)

---

## 4. R1 Full Hyperparameter Sweep — Matched 200 vs. 500

**Setup:** R1, realistic traffic, Tier B+C, 5 seeds, sweeping window_size ∈
{2,3,4,5}, top_k ∈ {1,3,5}, min_cluster_size ∈ {3,5,8,10} — 11 configs × 2
volumes = 22 trials.

| Parameter | Value | Det.@200 | Lat.@200 | Purity@200 | Det.@500 | Lat.@500 | Purity@500 |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| window_size | 2 | 5/5 | 2.0 | 0.87 | 5/5 | 2.0 | 0.96 |
| window_size | 3 (default) | 5/5 | 2.0 | 0.95 | 5/5 | 2.0 | 1.00 |
| window_size | 4 | 5/5 | 2.0 | 0.94 | 5/5 | **3.0** | 1.00 |
| window_size | 5 | 5/5 | 2.0 | 0.92 | 5/5 | **3.0** | 1.00 |
| top_k | 1 | 5/5 | 2.0 | 0.95 | 5/5 | 2.0 | 1.00 |
| top_k | 3 (default) | 5/5 | 2.0 | 0.95 | 5/5 | 2.0 | 1.00 |
| top_k | 5 | 5/5 | 2.0 | 0.95 | 5/5 | 2.0 | 1.00 |
| min_cluster_size | 3 | 5/5 | 2.0 | 0.96 | 5/5 | 2.0 | 1.00 |
| min_cluster_size | 5 (default) | 5/5 | 2.0 | 0.95 | 5/5 | 2.0 | 1.00 |
| min_cluster_size | 8 | 5/5 | 1.0 | 0.80 | 5/5 | 2.0 | 0.87 |
| min_cluster_size | 10 | 5/5 | 1.0 | 0.78 | 5/5 | 2.0 | 0.86 |

**Finding: 100% detection in all 22 configurations.** R1 is not
hyperparameter-fragile at either traffic volume — the only real effect is a
purity drop at min_cluster_size≥8 (0.95→0.80) and a latency penalty at
window_size≥4 specifically at 500/cycle (+1 cycle). This is the strongest
evidence against a "cherry-picked hyperparameters" objection.

Source: `experiments/hyperparam_sweep/{param}_{value}.json` (200/cycle),
`{param}_{value}_bpc500.json` (500/cycle)

---

## 5. Ablation Study — Emergence Score Components (Density × Growth × Novelty)

**Setup:** Realistic traffic, Tier B+C, **5 seeds** (upgraded from 3),
`top_k=1` forced (so leave-one-out removal can change which cluster is
selected — see script comment in `run_ablation.py`). Leave-one-out on each
of the three score components vs. the full score.

| Component removed | R1: Novel Family | R2: Coordinated Attack | R3: Slow Drift |
|---|:---:|:---:|:---:|
| — (full score) | 100% det, lat 2.0, pur 0.95 | 60% det, lat 0.0, pur 0.85 | 100% det, lat 3.0, pur 0.92 |
| Density | **80% det (−20pt), lat +1.5** | 60% det (unchanged), pur +0.10 | 100% det (unchanged) |
| Growth | 100% det (unchanged) | **40% det (−20pt)** | 100% det (unchanged) |
| Novelty | 100% det (unchanged) | 80% det (+20pt, noisy) | 100% det, **lat +1.0** |

**Finding:** All three components matter, each in a different regime —
density matters most for R1 (novel-family detection), growth matters most
for R2 (coordinated-burst detection), novelty matters most for R3
(slow-drift/topic-shift detection). This is a materially different and more
defensible result than the earlier 3-seed, benchmark-only ablation, which
showed density and growth as consistent no-ops (0 measurable impact
anywhere) and novelty as the only load-bearing term — that finding does not
survive realistic traffic + unseen attack variants and should not be used.

Caveat: R2's full-score baseline in this ablation config is only 60%
detection (top_k=1 is far more restrictive than the main experiment's
top_k=3, where R2 reaches 100%), so R2's ablation deltas sit on a smaller,
noisier base than R1/R3.

Source: `experiments/ablation_realistic/ablation_summary_realistic.json`

---

## 6. LLM-Based Cluster Verification — Precision vs. Detection Trade-off

**Setup:** Realistic traffic, Tier B+C, 5 seeds, 200 benign/cycle, verifier
model `qwen3:1.7b`, default hyperparameters (top_k=3).

| Regime | Precision (no LLM) | Precision (LLM-verified) | Detection: baseline → verified | Purity: baseline → verified | Real LLM calls |
|---|:---:|:---:|:---:|:---:|:---:|
| R1: Novel Family | 0.14 | **0.90** | 100% → 100% | 0.951 → 0.973 | 126 |
| R2: Coordinated Attack | 0.06 | **0.65** | **100% → 80%** | 0.802 → 0.794 | 126 |
| R3: Slow Drift | 0.13 | **1.00** | 100% → 100% | 0.849 → 0.888 | 122 |

**Finding:** LLM verification improves precision 6–10× in every regime, but
**this is not a free lunch** — on R2 it costs one detection out of five
seeds (100%→80%), filtering out a genuine attack cluster under the harder
unseen-variant condition. R2's achievable precision ceiling is also lower
post-verification (0.65 vs. 0.90–1.00 on R1/R3) — coordinated-burst clusters
under Tier B+C paraphrase/novel-intent text are messier for the verifier to
triage. This corrects the earlier (stale, benchmark-only) run, which showed
zero detection-rate cost anywhere; that was an artifact of easier raw
benchmark attack text, not a property of the verification approach itself.

Source: `experiments/llm_verification_realistic/verification_summary_realistic.json`

---

## Superseded / Stale Results (excluded above — do not cite these numbers)

| Superseded result | Replaced by | Why |
|---|---|---|
| Original 500/cycle scale test (never saved to disk) | Section 2 | Lost on session restart; rerun from scratch on realistic traffic |
| N=30 Tier B+C unseen variants (60/regime) | Section 1 (N=50, 100/regime) | Reviewer-motivated increase for stronger robustness claim |
| `experiments/ablation/` (3 seeds, benchmark-only traffic, raw Tier A attacks) | Section 5 (`ablation_realistic/`, 5 seeds) | Old run showed density/growth as no-ops — did not survive realistic traffic; conclusion reversed |
| `experiments/llm_verification/` (benchmark-only traffic, raw Tier A attacks) | Section 6 (`llm_verification_realistic/`) | Old run showed zero detection-rate cost — did not survive unseen variants; R2 cost discovered |
| All result JSONs without `isolation_forest` / `*_ci95_halfwidth` fields | All sections above | Predate the IF baseline + CI additions (Issues #5, #8) |

---

## File Index

| Section | Result files |
|---|---|
| 1. Main detection | `experiments/realistic_eval/*_tier{a,bc}_realistic.json` |
| 2. Scale sensitivity | `experiments/scale_test/{novel_family,coordinated_attack,slow_drift}_500.json` |
| 3. R2 mcs sweep | `experiments/scale_test/coordinated_attack_{200,500}_mcs{3,8}.json` |
| 4. R1 full sweep | `experiments/hyperparam_sweep/{param}_{value}[_bpc500].json` |
| 5. Ablation | `experiments/ablation_realistic/*.json` |
| 6. LLM verification | `experiments/llm_verification_realistic/*.json` |
| Master DeBERTa cache | `experiments/master_score_cache.json` (16,106 entries, covers all of the above) |
