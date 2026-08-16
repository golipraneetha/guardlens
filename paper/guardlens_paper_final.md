# GuardLens: Temporal Monitoring of Emerging LLM Safety Blind Spots

**Author:** Praneeth Agoli
**Affiliation:** praneethagoli@gmail.com

---

## Abstract

Large language model deployments rely on safety classifiers to filter harmful requests, yet these guardrails inevitably approve some adversarial inputs, creating undetected safety blind spots. We present GuardLens, a lightweight, model-agnostic monitoring layer that observes guardrail-approved traffic for emerging semantic clusters, combining density-based clustering within overlapping temporal windows, cross-cycle cluster persistence, and an interpretable Emergence Score to prioritize novel, growing clusters for analyst review. GuardLens requires no classifier access, retraining, or labeled attack data, and adds no inference-time latency.

We evaluate GuardLens across three attack-emergence regimes, matching detection rates across five baseline strategies while producing significantly purer analyst-facing clusters where attacks are semantically diverse. Beyond controlled benchmark evaluation, GuardLens maintains full detection under realistic conversational traffic from independent datasets and previously unseen, LLM-generated attack variants, consistent with detection tracking semantic emergence rather than benchmark-specific lexical artifacts. Traffic-volume experiments show detection holds at 2.5× scale wherever attack prevalence scales with traffic, degrading only under a deliberately fixed-volume attacker burst — a boundary condition we report directly. Matched hyperparameter sweeps show default settings are robust across a wide operating range. These results position GuardLens as a practical, deployable complement to existing guardrails rather than a replacement for classifier improvement.

**Keywords:** LLM safety, guardrail monitoring, blind spot detection, HDBSCAN clustering, temporal emergence detection

---

## I. Introduction

The rapid adoption of Large Language Models (LLMs) in production systems has made safety classifiers, commonly referred to as guardrails, a critical component of responsible AI deployment [1]. These classifiers evaluate incoming requests and block prompts associated with known attack patterns such as prompt injection, jailbreak attempts, and other policy violations. However, no classifier achieves perfect recall: automated red-teaming reliably elicits failures even from well-aligned systems [2], adversarial techniques continually evolve [3], and entirely new attack strategies can exploit gaps beyond a classifier's training distribution [4], [5]. When a guardrail mistakenly approves a harmful request, the failure is typically silent: the request is treated as benign, with no automated mechanism to flag it for review. Over time, these undetected failures accumulate into safety blind spots that can be systematically exploited.

Existing mitigation approaches address this problem only partially. Manual auditing is labor-intensive and unlikely to catch rare emerging patterns. Distribution-shift methods such as MMD [6] detect whether traffic has changed but cannot localize responsible requests or present reviewable clusters. Production clustering tools such as Clio [7] characterize user behavior but do not track emerging safety threats across time. No existing method maintains persistent semantic identities across monitoring cycles to prioritize emerging guardrail blind spots.

We present GuardLens, a lightweight, model-agnostic layer that introduces persistent semantic emergence monitoring for guardrail-approved traffic. GuardLens embeds approved prompts into a semantic space, clusters them within overlapping sliding windows using HDBSCAN [8], tracks cluster identity across monitoring cycles through centroid matching, and prioritizes emerging behaviors via an interpretable Emergence Score integrating density, temporal growth, and historical novelty. It requires no access to guardrail internals and adds no inference-time latency. A fixed-budget top-K analyst queue surfaces the most suspicious emerging clusters for investigation before they become widespread.

This work makes three contributions. First, we introduce persistent semantic emergence monitoring, combining temporal clustering, cross-cycle tracking, and an interpretable Emergence Score to prioritize emerging attack patterns without labeled data or classifier modifications. Second, we evaluate beyond benchmark replay: testing spans three attack-emergence regimes under realistic traffic, previously unseen LLM-generated attack variants, and volume and hyperparameter robustness checks, distinguishing genuine semantic emergence detection from reliance on benchmark-specific lexical artifacts (Section V-A). Third, we add an optional LLM-based verification stage that improves analyst queue precision without affecting detection latency.

---

## II. Related Work

### A. LLM Safety Classifiers

Adversarial jailbreak techniques [3] and transferable attacks [4] demonstrate persistent vulnerabilities in LLM safety alignment, exploiting gaps in classifier training distributions that persist even in well-aligned systems [5]. Production guardrails address this problem through classifier-based filtering: general-purpose moderation models such as Llama Guard [1] and WildGuard [9], specialized detectors such as the DeBERTa-based prompt-injection classifier used as the deployed guardrail throughout this work [10], and content-safety classifiers such as ShieldGemma [11]. These systems operate at the individual-request level, however, with no mechanism for monitoring the collective behavior of approved traffic after deployment. Concurrent work applies semantic-drift analysis to post-deployment safety monitoring via model-behavior probing rather than traffic clustering [12]. GuardLens is complementary: it monitors approved traffic rather than replacing classifiers or probing model behavior directly — the gap this work fills.

### B. Production Traffic Analysis

Clio [7] clusters production LLM conversations using hierarchical topic modeling to discover usage patterns. Its focus is descriptive—characterizing what users are doing—rather than identifying whether emerging activity represents a safety concern. GuardLens addresses a different question: temporal emergence detection, tracking cluster births, growth, and novelty across monitoring cycles. Recent work replaces offline, batch-oriented clustering with online variants for real-time stream monitoring [13]; online methods optimize for continuous per-item updates, whereas GuardLens's fixed-cycle batching is designed to align with periodic analyst review, making the two approaches a mismatch of operating assumptions rather than directly comparable baselines — a direction GuardLens does not yet adopt (Section VI-A). Broader production-ML-monitoring surveys report a similar gap between distributional health checks and actionable, prioritized alerts [14].

### C. Distribution Shift Detection

Statistical drift methods, including MMD [6], classifier-based approaches to detecting dataset shift [15], and broader concept-drift and online anomaly-detection surveys [16], [17], test whether new traffic differs from a historical reference distribution. Such aggregate tests, however, cannot localize the specific requests responsible for a detected shift or organize them into interpretable, reviewable clusters. As our experiments show, MMD is comparatively less sensitive to gradual drift below its per-cycle threshold, consistent with its design as a global drift detector rather than a cluster-localization method.

---

## III. Methodology

### A. GuardLens Overview

Existing safety infrastructure treats guardrail approval as a terminal decision, leaving the gap identified in Section II unaddressed: no mechanism monitors collective behavior in approved traffic over time. GuardLens treats persistent semantic emergence — the sustained growth of a coherent request pattern across time — as the unit of detection, because a single anomalous request is rarely actionable but a growing, unfamiliar cluster of them is.

As shown in Fig. 1, GuardLens embeds guardrail-approved requests, clusters them within an overlapping temporal window, tracks cluster identity across cycles via a persistent registry, and ranks clusters with an Emergence Score before surfacing the top-K to analyst review. An optional LLM-based verification stage refines this queue post hoc without sitting in the detection path.

**Algorithm 1 — GuardLens Per-Cycle Update**
```
Input: approved traffic for cycle t, sliding window W (size w),
       Cluster Registry R, match threshold τ=0.85, budget K
1. W ← W ∪ {embed(x) for x in approved_traffic_t}; drop cycle t-w
2. C ← HDBSCAN(W)                          # raw clusters this cycle
3. for each cluster c in C:
4.     match ← argmax_{r in R} cosine(centroid(c), centroid(r))
5.     if cosine(centroid(c), centroid(match)) ≥ τ:
6.         update age, size, growth history of match in R
7.     else:
8.         register c as new entry in R
9. for each tracked cluster r in R active this cycle:
10.    score(r) ← density(r) × max(growth(r), 0.1) × novelty(r)
11. Q ← top-K clusters in R by score
12. return Q                               # surfaced to analyst
```

### B. Semantic Representation and Temporal Clustering

Each approved request is embedded using Sentence-BERT [18]. These embeddings are accumulated over a sliding window of the *w* most recent monitoring cycles, *W_t = A_(t−w+1) ∪ A_(t−w+2) ∪ ... ∪ A_t*, where *A_i* is the set of approved requests observed in cycle *i*. The window exists because accumulating evidence across cycles helps distinguish a genuinely emerging cluster from background noise — an improvement in cluster purity relative to single-cycle clustering, quantified in Section V-B.

Embeddings within *W_t* are clustered using HDBSCAN [8] rather than DBSCAN, since attack clusters vary widely in density across regimes (a tight coordinated burst vs. a diffuse, drifting family) and DBSCAN's single global density threshold cannot accommodate both without per-regime tuning. Applied independently to a single cycle, however, clustering alone lacks temporal context to distinguish a novel emerging pattern from a recurring one — motivating the persistent tracking mechanism described next.

### C. Persistent Cluster Tracking

Raw per-cycle clusters carry no identity across cycles, which would make growth undefined. The Cluster Registry resolves this by matching each new cluster to historical centroids via cosine similarity. This threshold (0.85) reflects a common high-similarity range for SBERT-family embeddings [18] rather than a validated, task-specific choice; Section VI-A reports its sensitivity. Matched clusters accumulate age and growth history; unmatched clusters are registered as new. This lets GuardLens distinguish a slowly-building campaign from recurring noise, unlike stateless single-cycle clustering.

### D. Emergence Scoring

Density alone cannot separate an emerging threat from an established, stable topic, since both can be dense. GuardLens combines density (*D*), growth (*G*), and novelty (*N*) into a single score:

**E = D × max(G, 0.1) × N**

The multiplicative form treats the three signals as joint requirements. An additive score would let a dense, long-established cluster dominate on density alone despite near-zero novelty. Multiplication requires simultaneous evidence across all three before a cluster ranks highly. The 0.1 growth floor prevents fully zeroing out a cluster whose growth is flat but whose density and novelty are strong. Equal implicit weighting across the three terms is a deliberate simplicity choice: learned weights would require labeled emerging-attack data, which the method is designed to operate without. Section V-D's ablation validates this design empirically.

### E. LLM-Based Cluster Verification

The Top-K queue has no ground truth; an analyst must triage each cluster manually. To reduce this burden, a local LLM optionally classifies each queued cluster as THREAT or BENIGN from representative samples nearest and farthest from centroid. Because this runs after the queue is fixed, it can only filter what an analyst sees — it cannot alter detection latency, which keeps the real-time path unaffected by inference cost.

---

## IV. Experimental Setup

### A. Deployed Guardrail and Semantic Embeddings

All experiments evaluate GuardLens behind a deployed production-style guardrail, the ProtectAI DeBERTa Prompt Injection Classifier v2 [10]; only prompts classified SAFE are forwarded to GuardLens. Approved prompts are embedded with Sentence-Transformers all-MiniLM-L6-v2 (384-dim, L2-normalized) [18], chosen over larger encoders for its lower per-cycle latency and memory footprint given continuous rather than batch operation, and used throughout all experiments.

### B. Monitoring Configuration

A monitoring cycle is a fixed-size batch of approved requests — 150–200 prompts in the benchmark experiments, 200 or 500 in the traffic-volume experiments — rather than a fixed wall-clock interval, letting deployments map cycles to their own polling cadence. GuardLens maintains a sliding window of the three most recent cycles, clusters embeddings with HDBSCAN [8] (min_cluster_size=5, min_samples=3, Euclidean distance, EOM selection), tracks clusters across cycles via centroid matching, ranks them by the Emergence Score, and surfaces the Top-K=3 clusters for review. Unless stated otherwise, each experiment is averaged across five random seeds.

### C. Threat Model and Regimes

GuardLens assumes the deployed guardrail is fixed and monitors only what it already approved, with the goal of post-deployment visibility. A more sophisticated adversary with knowledge of the deployed embedding model, crafting deliberately low-semantic-footprint prompts, is outside this black-box threat model and is not evaluated here. Because no public production dataset contains temporally-labeled emerging prompt-injection attacks, we construct three threat regimes via controlled injection into benign traffic, consistent with standard practice in drift-detection evaluation. Evaluation spans both raw benchmark attacks and previously unseen LLM-generated semantic variants within the same harm categories, approximating how attacker behavior evolves after a benchmark-trained guardrail is deployed.

- **R1 — Novel Attack Family.** Beginning at monitoring cycle five, AdvBench prompts [4] are introduced at increasing volume (5–25% of approved traffic), representing a previously unseen attack family gaining adoption.
- **R2 — Coordinated Burst.** A pool of 58 near-duplicate prompts from the "Advanced" subset of JailbreakLLMs [3] supplies a fixed-size burst of 30 injected simultaneously at onset, followed by a smaller trickle (5/cycle) for two more cycles, modeling coordinated reuse of a successful jailbreak template. The burst size (30), not the pool size (58), is the volume held fixed in the traffic-scaling experiments of Section V-C.
- **R3 — Slow Semantic Drift.** Cybercrime-intrusion prompts from HarmBench [19] are introduced gradually across cycles, modeling prevalence that increases slowly rather than as a sudden burst.

### D. Realistic Traffic and Previously Unseen Attack Variants

The regimes above use publicly available attack datasets. To assess generalization beyond benchmark replay, we additionally evaluate against an independent benign pool of 15,000 prompts drawn from Alpaca [20], OASST1 [21], and UltraChat [22] — sources with no overlap with any attack benchmark used — eliminating the possibility that detection is driven by shared data-collection artifacts between benign and attack sources. For each regime, we further generate 100 previously unseen attack variants (50 semantic paraphrases, 50 novel-intent items within the same harm category) via a locally hosted LLM, cached deterministically for reproducibility, with no exact-text overlap with any public benchmark release.

### E. Baselines and Evaluation Metrics

GuardLens is compared against five baselines under the same per-cycle review budget: **Random Audit** (uniform sampling), **Stratified Random** (sampling near the guardrail's decision boundary), **MMD Drift** [6], **One-Shot Cluster** (per-cycle HDBSCAN, no temporal persistence), and **Isolation Forest** [23], a standard unsupervised anomaly detector applied to the same embeddings. Four metrics are reported: **Detection Rate**, the fraction of runs detecting at least one attack cluster (purity ≥0.5); **Detection Latency**, cycles from attack onset to first detection; **Purity**, the fraction of a detected cluster's members that are true attack items; and **Coverage**, the fraction of all attack items in the current window captured by that cluster. Given n=5 seeds, significance is assessed via non-overlapping 95% confidence intervals (t-distribution) rather than a parametric test. We report cluster-level metrics rather than per-request Precision/Recall because the analyst reviews Top-K clusters, not individual requests.

---

## V. Results

### A. Detection Under Unseen Attack Variants

GuardLens and One-Shot Cluster both reach 5/5 detection in all three regimes (Fig. 2) — no other method does. Detection rate alone therefore does not separate them; the differentiator is cluster quality (Section V-B). Isolation Forest's 0/5 on R2 is the most striking result here: R2's coordinated burst forms a tight cluster in embedding space, and isolation-based scoring rates tightly clustered points as more normal, not less — a documented failure mode of IF on collective anomalies that density-based clustering is designed to handle. This provides evidence against the "no strong unsupervised baseline" objection: given identical embeddings and review budget, a standard method fails structurally on exactly the regime it's least suited for.

### B. Cluster Purity at Detection

On R1 — diverse, gradually emerging attacks — GuardLens selects a substantially purer cluster (0.95 vs. 0.67, Fig. 3), with non-overlapping 95% confidence intervals. GuardLens's top cluster contains 5% noise versus 33% for One-Shot Cluster. On R2 and R3, where the attack cluster is structurally obvious, the two methods converge and their CIs overlap. The temporal mechanism adds value when attack diversity makes cluster selection non-trivial, not uniformly across all conditions. Coverage is lower for GuardLens (mean 0.44 vs. 0.74), reflecting deliberate selectivity: one clean cluster flagged early is more useful than a larger, noisier one.

### C. Volume Sensitivity and Hyperparameter Robustness

R1 and R3 are volume-invariant because their attack count scales proportionally with traffic (Table I). R2 degrades because its fixed burst size (30, Section IV-C) falls from ~15% to ~6% of the cycle at 500/cycle, dropping below HDBSCAN's density threshold. An 11-configuration R1 sweep (window_size, top_k, min_cluster_size), repeated at both volumes, showed 100% detection in all 22 trials — including variation in top_k itself, indicating detection is not an artifact of the fixed K=3 review budget. Fig. 4 shows the R2 min_cluster_size sensitivity is not fixable by tuning a constant: the default (5) is best at 200 and worst at 500; 3 is worst at 200 but ties for best at 500 — no single value reaches 5/5 at both volumes, motivating volume-adaptive thresholds as a specific future-work item. Isolation Forest's 0/5 on R2 persists at both volumes, consistent with the failure being structural rather than a data-starvation artifact.

**Table I — Detection at 200 vs. 500 Items/Cycle (Default Hyperparameters, 5 Seeds)**

| Regime | Attack Scaling | Items/Cycle | GuardLens | Isolation Forest |
|---|---|---|---|---|
| R1: Novel Family | ∝ traffic | 200 | 5 | 4 |
| | | 500 | 5 | 3 |
| R2: Coordinated | fixed burst | 200 | 5 | 0 |
| | | 500 | 2 | 0 |
| R3: Slow Drift | ∝ traffic | 200 | 5 | 5 |
| | | 500 | 5 | 4 |

*Values are number of seeds (of 5) with detection.*

**Fig. 2 — Detection rate by method and regime.** *(prefer `figure2a_detection_heatmap.svg` — true vector; `figure2a_detection_heatmap.png` also provided at exactly 300 DPI (1005×780px). Both in this same folder, Times New Roman throughout, no caption baked in — add the caption below using the `figurecaption` style.)* Heatmap of detection rate (of 5 seeds) by method (rows) and threat regime (columns) — GuardLens and One-Shot Cluster reach 5/5 everywhere; Isolation Forest fails completely on R2 (white cell).

**Fig. 3 — Cluster purity at detection.** *(prefer `figure2b_purity_bars.svg` — true vector; `figure2b_purity_bars.png` also provided at exactly 300 DPI (1005×780px). Both in this same folder, Times New Roman throughout, no caption baked in.)* Bar chart of cluster purity, mean ± 95% CI (t-distribution, n=5 seeds), GuardLens vs. One-Shot Cluster across the three regimes — GuardLens is significantly purer on R1 only; R2/R3 CIs overlap.

**Fig. 4 — R2 min_cluster_size sweep at both volumes.** *(prefer `figure3b_mincluster_sweep.svg` — true vector; `figure3b_mincluster_sweep.png` also provided at exactly 300 DPI (1005×780px). Both in this same folder, Times New Roman throughout, no caption baked in.)* GuardLens detection rate on R2 as min_cluster_size varies (3, 5, 8), compared at 200 and 500 items/cycle — no single value is optimal at both volumes.

### D. Emergence Score Ablation

**Table II — Leave-One-Component-Out Ablation: Detection Rate After Removal (realistic traffic, Tier B+C unseen variants, 5 seeds)**

| Component Removed | R1 (base 100%) | R2 (base 60%) | R3 (base 100%) |
|---|---|---|---|
| Density | 80% | 60% | 100% |
| Growth | 100% | 40% | 100% |
| Novelty | 100% | 80% | 100% |

*Latency increases only for: Density removal on R1 (+1.5 cycles); Novelty removal on R2 (+1.5 cycles) and R3 (+1.0 cycle). All other removals: +0.0 cycles.*

Density and growth both prove necessary in at least one regime (Table II): removing density costs 20 points of detection on R1 (100%→80%) and 1.5 cycles of latency; removing growth costs 20 points on R2 (60%→40%). Novelty's removal costs 1.0 cycle of latency on R3, and pairs with a detection-rate increase on R2 (60%→80%) that, at n=5 seeds, is more plausibly sampling noise than a genuine effect. Density, growth, and novelty are each necessary in at least one regime, not novelty alone — evidence the Emergence Score's multiplicative design is doing real work across all three signals rather than carrying two redundant terms. R2's full-score baseline in this configuration is only 60% detection (top_k=1, forced to make component removal decision-relevant), a smaller, noisier base than R1/R3's, and its ablation deltas should be read with that caveat.

### E. LLM-Based Cluster Verification

An optional post-hoc LLM verification stage (qwen3:1.7b) improves precision substantially across all three regimes, roughly 6× to 11×, but is not cost-free: on R2, verification costs one detection out of five seeds (100%→80%), filtering out a genuine attack cluster under the harder unseen-variant condition; R1 and R3 remain unaffected.

---

## VI. Discussion

**Temporal memory drives cluster-quality gains where it matters.** GuardLens and One-Shot Cluster differ only in the sliding window and persistent registry, yet this produces a 28-percentage-point purity gap on R1 (Section V-B). The benefit appears when attack diversity makes cluster selection non-trivial; when the attack cluster is structurally obvious (R2, R3), both methods converge. The ablation (Section V-D) confirms all three score components carry independent weight, with density and growth each proving necessary in different regimes.

**Boundary-proximity sampling underperforms semantic clustering.** Stratified Random — sampling near the guardrail's decision threshold — detects only 3/5, 3/5, and 4/5 seeds across regimes (Fig. 2), consistently behind GuardLens and One-Shot Cluster's 5/5. Novel and slowly-drifting attacks do not reliably hover near the classifier's confidence boundary; some evade confidently or drift in below the training distribution entirely, making boundary-proximity sampling inconsistent.

**Detection tracks semantic emergence, not benchmark text or raw volume.** Detection is unchanged when benchmark prompts are replaced with unseen LLM-generated variants in realistic traffic (Section V-A), consistent with semantic novelty detection rather than lexical-artifact matching. The volume study (Section V-C) shows detection holds at 2.5× traffic wherever attack prevalence scales proportionally, degrading only under R2's fixed-count burst — a threat-model property, not a general scaling failure.

**Default hyperparameters are robust across R1's full 22-configuration sweep at both volumes.** R2's min_cluster_size is the exception: no single value is optimal at both volumes (Section V-C), motivating a volume-adaptive threshold as future work.

### A. Limitations

These results should be read within the scope in which they were produced. Detection rates of 100% describe the specific attack mechanisms, traffic volumes, and five-seed averages tested here, not a general safety guarantee. The volume study establishes only two points (200 and 500 items/cycle), not a full scaling curve. The cross-cycle match threshold (cosine 0.85) was chosen as an embedding-space prior rather than tuned; a joint sweep across embedding models and thresholds remains open. Novel-intent attack variants (Section IV-D) test robustness to surface-form variation within the same harm categories, not detection of entirely new categories. Multilingual traffic and cross-model embedding robustness are untested. The "no inference-time latency" claim refers to GuardLens sitting outside the guardrail's request path, not to measured wall-clock cost; a computational profile of embedding, clustering, registry matching, and scoring across cycle sizes was not collected and remains future work. As a monitoring rather than blocking layer, these gaps are mitigated by defense-in-depth with the guardrail and human review. Detecting new harm categories, multilingual traffic, low-semantic-footprint adversarial text, monitoring model outputs rather than only inputs, and adopting online clustering [13] remain future work.

---

## VII. Conclusion

We presented GuardLens, a lightweight, model-agnostic monitoring layer that observes guardrail-approved traffic for emerging semantic clusters representing potential safety blind spots, combining density-based clustering, cross-cycle cluster persistence, and an interpretable Emergence Score to prioritize a bounded analyst review queue without labeled attack data or guardrail retraining.

Across three attack-emergence regimes evaluated under realistic traffic and previously unseen attack variants, GuardLens matched five baseline strategies in detection rate, including a standard unsupervised anomaly detector (Isolation Forest) that fails structurally on coordinated bursts. Against the strongest baseline, One-Shot Cluster, GuardLens ties on detection rate but surfaces significantly purer clusters where attacks are semantically diverse. Ablation results (Section V-D) validate the Emergence Score's multiplicative design, while LLM verification yields large precision gains (roughly 6×–11×) at the cost of one missed detection in five seeds on the coordinated-attack regime.

Scope caveats apply (Section VI-A): these results describe specific attack mechanisms, traffic volumes, and five-seed averages, not a general guarantee. Future work should prioritize a volume-adaptive min_cluster_size, joint threshold and embedding-model robustness studies, online clustering, multilingual validation, and detection of new harm categories. GuardLens is intended as a practical complement to existing guardrails, not a replacement for classifier improvement.

---

## Acknowledgment

*[Identify applicable funding agency here. If none, delete this line.]*

**AI Tools Used.** Claude (Anthropic) assisted in drafting and revising prose across the Introduction, Related Work, Methodology, Experimental Setup, Results, Discussion, Limitations, and Conclusion, translating author-collected experimental data and design decisions into manuscript text, and in verifying reference accuracy against primary sources. All experimental design, data collection, and analysis are the authors' own; all AI-assisted text was reviewed and edited by the authors, who are fully responsible for its accuracy and originality. Claude is not an author.

---

## Figures

**Fig. 1 — GuardLens Architecture** *(prefer `figure1_architecture.svg` — true vector; `figure1_architecture.png` also provided at 1050×2042px, exactly 300 DPI at the intended 3.5in single-column print width, real Times New Roman, lossless. Both in this same folder. Diagram only — no caption baked into the image; add the caption below it in Word using the `figurecaption` style, per the template's convention that "figure captions should be below the figures." Vertical, single-column layout (~0.51:1 aspect ratio), sized to fit within one column rather than spanning both.)*

**Top-to-bottom flow:** Incoming User Prompts → **Existing Safety Guardrail Classifier** (dashed box, frozen/external — "Frozen · DeBERTa Prompt Injection v2") → branches left to *Blocked (discarded)*, or continues down as *Approved* traffic into the dashed "GuardLens — Post-Deployment Monitoring Layer" region → **Sentence Embedding Model** ("all-MiniLM-L6-v2 · 384-dim, L2-norm") → **Sliding Window Buffer** ("Last W = 3 cycles of approved traffic") → **HDBSCAN Semantic Clustering** ("min_cluster_size = 5 · EOM selection") → **Cluster Registry** ("Cross-cycle identity via cosine ≥ 0.85") → **Emergence Score** ("E = Density × max(Growth, 0.1) × Novelty") → **Top-K Analyst Queue** ("Fixed budget · K = 3 clusters/cycle").

A dashed feedback curve labeled "historical centroids" runs from Cluster Registry to Emergence Score. The Top-K queue branches right to an optional, dotted-border **LLM Verifier** box ("Optional", no model name shown in-figure) and feeds **Analyst** ("Human review") directly below via a solid arrow; the LLM Verifier feeds back down into the same path via a dashed "filtered queue" arrow.

*Legend: dashed border = frozen/external; solid border = GuardLens core (real-time); dotted border = optional/asynchronous; dashed line = feedback/optional path.*

*Caption: "Fig. 1. GuardLens architecture. Approved traffic is embedded, clustered within a sliding window, tracked by a persistent Cluster Registry, and ranked by an Emergence Score before reaching a fixed-budget analyst queue, with optional LLM verification downstream."*

---

## References

[1] H. Inan et al., "Llama Guard: LLM-based input-output safeguard for human-AI conversations," arXiv:2312.06674, 2023.

[2] E. Perez et al., "Red teaming language models with language models," in *Proc. EMNLP*, 2022, pp. 3419–3428.

[3] Y. Liu et al., "Jailbreaking ChatGPT via prompt engineering: An empirical study," arXiv:2305.13860, 2023.

[4] A. Zou, Z. Wang, J. Z. Kolter, and M. Fredrikson, "Universal and transferable adversarial attacks on aligned language models," arXiv:2307.15043, 2023.

[5] N. Carlini et al., "Are aligned neural networks adversarially aligned?" arXiv:2306.15447, 2023.

[6] A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Schölkopf, and A. Smola, "A kernel two-sample test," *J. Mach. Learn. Res.*, vol. 13, pp. 723–773, 2012. Available: https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf

[7] A. Tamkin et al., "Clio: Privacy-preserving insights into real-world AI use," arXiv:2412.13678, 2024.

[8] R. J. G. B. Campello, D. Moulavi, and J. Sander, "Density-based clustering based on hierarchical density estimates," in *Proc. PAKDD*, 2013, pp. 160–172. Available: https://doi.org/10.1007/978-3-642-37456-2_14

[9] S. Han et al., "WildGuard: Open one-stop moderation tools for safety risks, jailbreaks, and refusals of LLMs," arXiv:2406.18495, 2024.

[10] ProtectAI, "DeBERTa-v3-base-prompt-injection-v2," Hugging Face model repository, 2024. Available: https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2

[11] W. Zeng et al., "ShieldGemma: Generative AI content moderation based on Gemma," arXiv:2407.21772, 2024.

[12] S. Zanbaghi, R. Rostampour, F. Abid, and S. Al Jarmakani, "Detecting sleeper agents in large language models via semantic drift analysis," arXiv:2511.15992, 2025.

[13] O. Vykhopen, V. Skorik, M. Tereshchenko, and V. Solopova, "Online density-based clustering for real-time narrative evolution monitoring," arXiv:2601.20680, 2026.

[14] H. Naveed, S. Barnett, C. Arora, J. Grundy, H. Khalajzadeh, and O. Haggag, "Monitoring machine learning systems: A multivocal literature review," arXiv:2509.14294, 2025.

[15] S. Rabanser, S. Günnemann, and Z. Lipton, "Failing loudly: An empirical study of methods for detecting dataset shift," in *Proc. NeurIPS*, 2019.

[16] F. Hinder, V. Vaquet, and B. Hammer, "One or two things we know about concept drift — a survey on monitoring evolving environments," arXiv:2310.15826, 2023.

[17] L. Correia, J.-C. Goos, P. Klein, T. Bäck, and A. V. Kononova, "Online model-based anomaly detection in multivariate time series: Taxonomy, survey, research challenges and future directions," arXiv:2408.03747, 2024.

[18] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," in *Proc. EMNLP*, 2019, pp. 3982–3992.

[19] M. Mazeika et al., "HarmBench: A standardized evaluation framework for automated red teaming and robust refusal," arXiv:2402.04249, 2024.

[20] R. Taori et al., "Stanford Alpaca: An instruction-following LLaMA model," GitHub repository, 2023. Available: https://github.com/tatsu-lab/stanford_alpaca

[21] A. Köpf et al., "OpenAssistant conversations—Democratizing large language model alignment," arXiv:2304.07327, 2023.

[22] N. Ding et al., "Enhancing chat language models by scaling high-quality instructional conversations," arXiv:2305.14233, 2023.

[23] F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation forest," in *Proc. IEEE ICDM*, 2008, pp. 413–422.
