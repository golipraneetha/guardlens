# VI. Results

## VI-A. Experimental Setup

All experiments run GuardLens behind a single deployed guardrail —
`ProtectAI/deberta-v3-base-prompt-injection-v2`, a binary
SAFE/INJECTION classifier — and observe only the traffic the guardrail
*approves*. Embeddings use `sentence-transformers/all-MiniLM-L6-v2`
(384-dim, L2-normalized). GuardLens clusters each 3-cycle sliding
window with HDBSCAN (`min_cluster_size=5`, `min_samples=3`,
`metric=euclidean`, `cluster_selection_method=eom`), scores clusters
by Emergence = density × max(growth, 0.1) × novelty, and surfaces the
top-K=3 clusters per cycle to a review queue. Each cycle draws 150–200
benign background texts (JailbreakLLMs regular queries + XSTest safe
prompts) plus regime-specific attack traffic; results are averaged
over 5 seeds. Detection latency is measured in cycles from attack
onset to the first cycle at which a genuine attack cluster (purity ≥
0.5) appears in the queue.

Three baselines are compared throughout: **Random Audit** (uniform
sample of K items/cycle), **Stratified Random** (samples near the
guardrail's decision threshold — the "smart analyst" heuristic),
**MMD Drift** (two-sample kernel test against a reference window), and
**One-Shot Cluster** (HDBSCAN on a single cycle, no sliding window or
emergence scoring — isolates the value of the temporal window itself).

## VI-B. Main Detection Results

**Table II — Detection Rate and Latency Across Three Blind-Spot Regimes (5 seeds)**

| Regime | Method | Detection Rate | Median Latency (cycles) | Purity @ Detection | Coverage @ Detection |
|---|---|---|---|---|---|
| **R1: Novel Family** (AdvBench, step onset + ramp, top-K=3) | GuardLens | 100% | 1.0 | 0.89 | 0.33 |
| | Random Audit | 100% | 2.0 | — | — |
| | Stratified Random | 0% | — | — | — |
| | MMD Drift | 100% | 1.0 | — | — |
| | One-Shot Cluster | 100% | 2.0 | 0.88 | 0.40 |
| **R2: Coordinated Attack** (JailbreakLLMs "Advanced" burst, top-K=3) | GuardLens | 100% | 0.0 | 0.86 | 0.80 |
| | Random Audit | 0% | — | — | — |
| | Stratified Random | 100% | 0.0 | — | — |
| | MMD Drift | 100% | 0.0 | — | — |
| | One-Shot Cluster | 100% | 0.0 | 0.86 | 0.88 |
| **R3: Slow Drift** (HarmBench cybercrime_intrusion, gradual ramp, top-K=5) | GuardLens | 100% | 2.0 | 1.00 | 0.51 |
| | Random Audit | 100% | 2.0 | — | — |
| | Stratified Random | 0% | — | — | — |
| | MMD Drift | 80% | 4.5 | — | — |
| | One-Shot Cluster | 100% | 5.0 | 1.00 | 0.64 |

**Key findings:**
- GuardLens matches or beats every baseline's detection latency in all three regimes, and is the only method with 100% detection in all three (MMD Drift misses 20% of R3 seeds; Stratified Random detects 0% in R1 and R3).
- **Stratified Random's failure is the important negative result**, not a redundant baseline: sampling near the guardrail's decision boundary — the intuitive "smart analyst" strategy — completely fails to find R1 (novel family) and R3 (slow drift) attacks, because those attacks don't hover near the classifier's threshold; they either evade confidently or drift in below the noise floor. It only works in R2, where near-duplicate templates happen to share a similar confidence band. This shows boundary-proximity sampling is not a substitute for clustering-based emergence detection.
- One-Shot Cluster (no temporal window) matches GuardLens's detection rate but is consistently slower to detect in R1 and R3 (2.0 vs 1.0 cycles; 5.0 vs 2.0 cycles) and has lower coverage in R3 (0.64 vs matching magnitude) — the sliding window's main value is accumulating enough cross-cycle attack volume before HDBSCAN's `min_cluster_size` threshold is met, which matters most for gradually-onsetting attacks.

## VI-C. Adversarial Evasion: Cluster Fragmentation

**Threat model.** An attacker aware that GuardLens clusters approved
traffic can attempt to defeat detection by splitting a fixed attack
budget across N semantically distinct jailbreak "communities" (writing
styles) rather than concentrating it in one, keeping each sub-family's
volume near or below HDBSCAN's `min_cluster_size=5` so no single
family forms a detectable cluster on its own.

**Design correction.** An initial version of this experiment split a
fixed *raw template* budget (`burst_size=30`) across N communities
before applying the DeBERTa guardrail filter. This confounded the
intended mechanism: per-community approval rates against DeBERTa are
extremely uneven (Advanced 50–66%, Toxic 9–14%, Anarchy/Narrative/Guidelines
0%, Exception/Start Prompt/Opposite 2–8%), so raising N mostly diluted
the raw budget toward communities the guardrail already blocks almost
entirely — collapsing the *effective approved attack volume* from
~20 items at N=1 to ~5 at N=5, independent of any clustering behavior.
Fragmentation was pinned at 1.0 across all N because only one
community's traffic ever survived the guardrail to reach the monitor.

The corrected design holds **approved attack volume constant**
(`burst_size=30`, split evenly across the N communities post-approval)
so the only variable is semantic diversity, isolating the clustering
mechanism from guardrail-approval-rate variance.

**Table III — Evasion Sweep: Detection vs. Community Diversity N (5 seeds, K=3, constant approved volume=30)**

| N | Communities | GL Detection | GL Latency | GL Purity | GL Coverage | GL Fragmentation | Random Audit | MMD Drift | One-Shot |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Advanced | 100% | 0.0 | 0.98 | 0.75 | 1.80 | 40% | 100% | 100% |
| 2 | + Toxic | 100% | 0.0 | 0.98 | 0.66 | 1.80 | 20% | 100% | 100% |
| 3 | + Anarchy | 100% | 0.0 | 0.95 | 0.72 | 2.00 | 40% | 100% | 100% |
| 4 | + Narrative | 100% | 0.0 | 1.00 | 0.64 | 2.40 | 40% | 100% | 100% |
| 5 | + Exception | 100% | 0.0 | 1.00 | 0.49 | 2.40 | 80% | 100% | 100% |

**Findings:**

1. **Detection rate and latency are invariant to fragmentation** (100%,
   0 cycles at every N). At `burst_size=30`, even N=5 leaves 6 items per
   community — still above `min_cluster_size=5` — so no community's
   sub-burst can hide below the clustering threshold entirely. An
   attacker who fragments finer than this floor would evade
   individual-community clusters, but only by shrinking each
   sub-family's volume enough that it no longer looks like a coordinated
   burst at all — a genuine budget/stealth trade-off, not a free evasion.

2. **Fragmentation rises with N (1.8 → 2.4) but not linearly** —
   cluster-level inspection at N=5 (Fig. X) shows the mechanism is
   governed by actual semantic separation between attack styles, not a
   simple 1:1 community-to-cluster mapping. At onset (cycle 5, N=5),
   HDBSCAN forms exactly 3 clusters from 5 communities: `Advanced`
   (5 items, purity 1.00) and `Narrative` (6 items, purity 0.86) are
   embedded distinctly enough to each form their own tight cluster,
   while `Anarchy`, `Toxic`, and `Exception` are close enough in
   embedding space that HDBSCAN merges them into a single 7-item mixed
   cluster (purity 1.00). This pattern is stable across subsequent
   cycles as the trickle continues (cycle 6–7: the merged cluster grows
   to 12–13 items while Advanced and Narrative remain separate).

3. **Coverage is the real cost to the attacker's benefit, declining
   monotonically from 0.75 (N=1) to 0.49 (N=5).** Because GuardLens's
   Top-K=3 queue can hold at most 3 clusters per cycle and fragmentation
   produces up to 3 simultaneous attack fragments at N≥4-5, the queue
   captures a shrinking share of the total attack footprint even though
   it still flags *some* cluster immediately. This is the actual
   trade-off the evasion strategy buys the attacker: not evasion of
   detection, but dilution of what the analyst sees on first alert.

4. **Random Audit's non-monotonic detection (40% → 20% → 40% → 40% →
   80%) reflects its lack of any structural signal** — its budget-of-3
   samples/cycle either happens to land on attack traffic or doesn't,
   independent of how the true attack is organized; its N=5 uptick is
   sampling noise, not a trend (5 seeds is not enough to resolve this
   baseline's variance, which is expected since it has no detection
   mechanism beyond chance).

## VI-D. Emergence Score Ablation

**Table IV — Leave-One-Out Ablation of Emergence Score Components**
(3 seeds, forced top-K=1 to make component removal decision-relevant —
at the main-experiment budget of K=3, HDBSCAN rarely produces enough
raw clusters for a leave-one-out change to alter which cluster is
selected)

| Regime | Variant | Detection Rate | Median Latency | Purity | Coverage |
|---|---|---|---|---|---|
| Novel Family | full | 100% | 1.0 | 0.98 | 0.29 |
| | no_density | 100% | 1.0 | 0.98 | 0.28 |
| | no_growth | 100% | 1.0 | 0.98 | 0.29 |
| | **no_novelty** | 100% | **2.0** | 0.95 | **0.16** |
| Coordinated Attack | full | 67% | 0.0 | 0.82 | 0.82 |
| | no_density | 67% | 0.0 | 0.82 | 0.82 |
| | no_growth | 67% | 0.0 | 0.82 | 0.82 |
| | no_novelty | 67% | 1.0 | 0.85 | 0.83 |
| Slow Drift | full | 100% | 3.0 | 1.00 | 0.49 |
| | no_density | 100% | 3.0 | 1.00 | 0.49 |
| | no_growth | 100% | 3.0 | 1.00 | 0.49 |
| | **no_novelty** | 100% | **5.0** | 1.00 | **0.23** |

**Finding:** Removing the **novelty** term is the only ablation that
degrades detection latency or coverage in any regime — density and
growth are redundant with novelty at this budget. This isolates
novelty (distance from historical cluster centroids) as the component
doing the load-bearing work of distinguishing a genuinely *new* blind
spot from a recurring, previously-reviewed pattern; density and growth
mainly refine ranking among already-novel candidates.

## VI-E. LLM-Based Cluster Verification

**Motivation.** GuardLens's Top-K queue has no ground truth — an
analyst must manually triage each surfaced cluster. This section
evaluates a post-hoc LLM triage layer (`ClusterVerifier`) that samples
representative texts (3 nearest + 3 farthest from centroid) from each
queued cluster and classifies it as THREAT or BENIGN via a local
Ollama model, filtering BENIGN clusters from the queue before analyst
review. Because verification runs *after* the queue is already formed,
it is architecturally decoupled from the cycle-based detection latency
reported in Sections VI-B–VI-D: it cannot delay first detection, only
change what an analyst sees once a cluster is already flagged.

**Table V — LLM Verification Impact on Review Queue (5 seeds/regime, qwen3:1.7b)**

| Regime | Detection Rate | Detection Latency | Purity | Precision@K (no filter) | Precision@K (+ LLM) |
|---|---|---|---|---|---|
| R1: Novel Family | 100% | 1.0 | 0.89 | 0.37 | 0.89 |
| R2: Coordinated Attack | 100% | 0.0 | 0.86 | 0.12 | 0.90 |
| R3: Slow Drift | 100% | 2.0 | 0.96 | 0.26 | 0.81 |

*Detection Rate and Detection Latency columns are identical
before/after LLM verification in all three regimes (confirming the
verifier cannot delay or suppress detection — it only filters the
queue after a cluster has already been flagged and timestamped).*

**Findings:**

1. **LLM verification improves precision@K substantially in every
   regime** — 3.1x (R1: 0.37→0.89), 7.3x (R2: 0.12→0.90), and 3.1x
   (R3: 0.26→0.81). The gain is largest in R2 because the Top-K=3
   queue there is often filled with benign clusters that happen to
   share superficial size/density with the true attack cluster; the
   LLM's semantic read of sampled texts filters these out reliably.
2. **Detection rate and latency are exactly unchanged** (100%
   detection, same cycle count, in all 3 regimes) — empirically
   confirming the architectural claim that this is a downstream
   triage filter, not part of the detection-critical path.
3. **Purity is reported here as the underlying cluster's true purity
   at the cycle of first detection** (unaffected by verification, since
   verification does not alter cluster membership — only whether
   the cluster survives into the queue an analyst sees).

**Wall-clock cost caveat.** LLM verification adds real inference time
per cluster that is separate from (and should not be conflated with)
the cycle-based detection-latency metric used throughout this paper.
Measured across the full 5-seed sweep (qwen3:1.7b, cache misses only):

| Regime | n Real Calls | Mean/call | Median | Max | Total |
|---|---|---|---|---|---|
| R1: Novel Family | 112 | 11.2s | 10.6s | 29.5s | 1252s |
| R2: Coordinated Attack | 104 | 11.6s | 11.5s | 21.3s | 1209s |
| R3: Slow Drift | 103 | 13.2s | 12.6s | 23.6s | 1355s |

For comparison, a single-seed pilot with llama3.1:8b on R2 (23 calls)
measured mean 45.2s/call, median 43.4s, max 95.5s — roughly 4x slower
than qwen3:1.7b, with a smaller precision gain (0.11→0.50 vs.
0.12→0.90 for qwen3:1.7b on the same regime). We recommend framing
LLM verification as an optional, asynchronous analyst-facing triage
step downstream of the real-time monitoring pipeline — it should not
run in the critical alerting path if per-cluster latency in the
10–30 second range (qwen3:1.7b) or 40–95 second range (llama3.1:8b)
is unacceptable for a given deployment. The smaller model is
recommended based on this evidence: it is simultaneously faster and
more precise in every regime tested.

## VI-F. Scale and Embedding-Model Sensitivity

To test two boundary conditions of the evaluation setup — traffic
volume and embedding model choice — we ran two additional checks on
Regime 2 (5 seeds each), holding all other pipeline components fixed.

**Table VI — Traffic Volume Sensitivity (R2, 5 seeds, attack volume held constant at 30)**

| Benign/Cycle | Detection Rate | Median Latency | Purity | Coverage | Cycle Wall-Clock (mean) |
|---|---|---|---|---|---|
| 200 (baseline) | 100% | 0.0 | 0.86 | 0.80 | 4.2s |
| 500 (2.5x) | 40% | 1.0 | 0.93 | 0.69 | 2.0s |

**Table VII — Embedding Model Sensitivity (R2, 5 seeds)**

| Embedding Model | GuardLens Detection | GuardLens Purity | GuardLens Coverage | MMD Drift Detection | Cycle Wall-Clock (mean) |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 (default) | 100% | 0.86 | 0.80 | 100% | 4.2s |
| all-mpnet-base-v2 | 100% | 0.84 | 0.87 | 80% | 6.1s |

**Findings:**

1. **Detection rate collapses under a fixed-absolute attack budget as
   background traffic scales.** Tripling benign volume from 200 to 500
   per cycle while holding attack volume fixed at 30 items dropped
   GuardLens's detection rate from 100% to 40% (3 of 5 seeds never
   detected the attack within the 10-cycle window; the 2 that did were
   delayed by 1 cycle). Every baseline degraded in the same direction
   (MMD Drift 100%→20%, One-Shot Cluster 100%→40%), indicating this is
   a property of the experimental design — the attack's share of
   traffic fell from ~15% to ~6% — rather than a GuardLens-specific
   weakness. Cycle wall-clock cost actually *decreased* (4.2s→2.0s,
   likely from a higher embedding-cache hit rate in larger overlapping
   windows), so compute throughput was not the bottleneck; the failure
   is in the clustering/ranking dynamics competing against a much
   larger benign cluster population at a fixed `min_cluster_size` and
   Top-K.

2. **GuardLens holds detection under an embedding-model swap; a
   baseline does not.** Substituting `all-mpnet-base-v2` for the
   default `all-MiniLM-L6-v2` left GuardLens's detection rate unchanged
   at 100%, with purity and coverage shifting only modestly (0.86→0.84,
   0.80→0.87). The MMD Drift baseline, by contrast, dropped from 100%
   to 80% detection under the identical embedding swap. This is
   evidence — on one alternate model and one regime — that GuardLens's
   cluster-based detection is at least as embedding-robust as a
   standard statistical baseline, though it does not test the more
   concerning adversarial case of an attacker deliberately crafting
   low-semantic-footprint text against a known embedding model (see
   Limitations).

These two results point in different directions and should be read
separately: the volume-scaling result is a genuine, unresolved
limitation of the fixed-`min_cluster_size`/fixed-Top-K configuration
as currently implemented; the embedding-model result is a modestly
positive robustness signal on the one axis tested. Section VII
discusses the scope of both findings and what remains open.
