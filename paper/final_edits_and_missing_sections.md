# GuardLens Paper — Final Edit List + Missing Section Content

## PART 1: Final Edit List (mapped to existing draft)

### Abstract
- **Change**: "GuardLens achieves 100% detection across all regimes with median latency of 0–2 cycles" → add one clause scoping this: *"...within the controlled traffic regimes and volumes tested (150–200 items/cycle); see Limitations for scale boundary conditions."* Do this once here; don't repeat the caveat every time the number appears in-body.
- **Change**: "adds no inference-time latency" → qualify: *"adds no inference-time latency to the guardrail itself; an optional LLM-based verification extension (Section V-D) adds analyst-facing latency outside the detection path."*

### Section I. Introduction
- **Fix**: Contributions list and regime description both say "four representative attack-emergence regimes," but Table II (Results) only covers three (R1–R3), with R4 (fragmentation) evaluated separately in Table III under a different framing. Add one sentence after the four-regime list: *"R4 is evaluated as a stress test of R2 specifically — an attacker fragmenting a coordinated burst — rather than as an independent regime, and is reported separately in Section V-B."*
- **Fix**: Third contribution bullet says "LLM-based semantic verification to evaluate the effectiveness of post-deployment guardrail monitoring." Decide framing: is this a contribution (a novel verification mechanism) or an evaluation method (a way of checking cluster quality)? Recommend: call it a contribution — *"An LLM-based post-hoc verification stage that filters the analyst review queue by semantic threat classification, evaluated for its effect on queue precision without impacting detection latency."* This matches how it's actually framed in Results.

### Section III-E. Persistent Cluster Tracking
- **Add** one sentence after the 0.85 threshold definition: *"This threshold was not tuned against a validation set; Section VII reports a sensitivity analysis and its scope."*

### Section IV-A. Experimental Setup
- **Add** cycle definition sentence: *"A monitoring cycle here is a fixed-size batch of approved traffic (150–200 items), not a fixed time interval; deployments should map cycle count to their own polling cadence. Section VII reports measured wall-clock compute cost per cycle."*
- **Fix**: R1 description in Section IV-C says attack traffic "gradually introduced, increasing from 5% to 25%." Per the actual regime implementation, this is a step onset at cycle 5 followed by a ramp — not a smooth gradual increase from the start. Reword: *"Beginning with a step onset at monitoring cycle five, AdvBench prompts are introduced at increasing volume across subsequent cycles (5%–25% of benign traffic), modeling a previously unseen attack family that is adopted with increasing frequency after its first appearance."*
- **Fix**: R2 description — name the community: *"...originating from a single JailbreakLLMs community (Advanced, 58 near-duplicate templates sharing one jailbreak style)..."*

### Section V. Results
- **Add methodology sentence to RQ2 (Adversarial Fragmentation)**: An earlier version of this experiment split the *raw* attack template budget across communities before guardrail filtering, which confounded fragmentation with each community's wildly different DeBERTa approval rate (50–66% for Advanced vs. 0–14% for the other four communities used). The corrected design (used in Table III) holds *approved* attack volume constant across all N by letting attack-labeled items bypass the guardrail filter, isolating the clustering mechanism. Add: *"To isolate the clustering-evasion mechanism from guardrail-approval-rate variance across communities (which differs by up to 66 percentage points), approved attack volume was held constant at 30 items across all fragmentation levels; see Section VII for details."*
- **Add methodology note to RQ3 (Ablation)**: explain why Table IV's R2 numbers (67% detection, purity 0.82) differ from Table II's R2 numbers (100% detection, purity 0.86) for the same regime. Add: *"This experiment forces Top-K=1 (vs. Top-K=3 in Table II) to make leave-one-out component removal decision-relevant — at Top-K=3, HDBSCAN rarely produces enough competing clusters for a ranking-function change to alter queue membership, so all ablation variants would score identically. The stricter budget explains the lower absolute detection rate and purity relative to Table II; only the *relative* differences across ablation variants (not absolute values) should be compared to Table II."*
- **Add new subsection V-D** — full content below in Part 2.

### New sections needed after Results
- **V-D. RQ4: LLM Verification** — new, content below (fills the "four research questions" gap).
- **V-E** *(renumber from V-D if V-D goes above)*, or keep as **VI-F. Scale and Embedding-Model Sensitivity** if the existing Results section stays numbered as "VI" — new subsection with Tables VI and VII (traffic-volume sensitivity, embedding-model sensitivity), full content already in `paper/results_section_draft.md` VI-F — paste as-is. This holds the primary data; Limitations references it rather than duplicating it.
- **VI. Discussion** — new, content below.
- **VII. Limitations** — new, full content in `paper/limitations_section_draft.md` (now trimmed to interpretation only for the scale/embedding-model findings, referencing Tables VI/VII in Results rather than restating numbers; threshold sensitivity data — Table VIII — still lives here since it doesn't have a natural home in the main Results narrative). Paste as-is.
- **VIII. Conclusion** — new, content below.

**Note on section numbering**: the existing draft's Results section is
labeled "V. RESULTS" but my working drafts used "VI." throughout —
reconcile to whichever numbering the final draft uses; the *relative*
order (Results → Discussion → Limitations → Conclusion) and table
numbers (II–VIII in sequence) are what matters, not the absolute
Roman numeral.

### References
- **Add** citations for LlamaGuard, ShieldGemma, and LLM-as-a-judge (named in Section II-A but uncited).
- **Add** citation for Ollama or the specific model card for qwen3/llama3.1 if the verification model choice needs a reference (or cite as a footnote/URL if no formal paper exists).

### Formatting (docx pass, not content)
- Math currently renders as inline broken text (`R384`, `Wt =i=t−W+1⋃t Ai`, `EPS=D×max(G,0.1)×N`) — convert to proper display equations.
- Confirm Figures 5, 6, 7 are actually embedded in the DOCX (not present in this PDF extract).

---

## PART 2: Missing Section Content (ready to paste)

### V-D. RQ4: Does LLM-Based Verification Improve Review Queue Precision Without Cost to Detection?

GuardLens's Top-K queue has no ground truth — an analyst must manually
triage each surfaced cluster. This section evaluates a post-hoc LLM
triage layer that samples representative texts (three nearest and
three farthest from centroid) from each queued cluster and classifies
it as THREAT or BENIGN via a local Ollama model (qwen3:1.7b),
filtering BENIGN clusters from the queue before analyst review.
Because verification runs after the queue is already formed, it is
architecturally decoupled from the cycle-based detection latency
reported in Sections V-A–V-C: it cannot delay first detection, only
change what an analyst sees once a cluster is already flagged.

**Table V. LLM Verification Impact on Review Queue (5 seeds/regime)**

| Regime | Detection Rate | Detection Latency | Purity | Precision@K (no filter) | Precision@K (+ LLM) |
|---|---|---|---|---|---|
| R1: Novel Family | 100% | 1.0 | 0.89 | 0.37 | 0.89 |
| R2: Coordinated Attack | 100% | 0.0 | 0.86 | 0.12 | 0.90 |
| R3: Slow Drift | 100% | 2.0 | 0.96 | 0.26 | 0.81 |

Detection Rate and Detection Latency are identical before and after
LLM verification in all three regimes, confirming the verifier cannot
delay or suppress detection. Precision@K improves substantially in
every regime: 3.1x in R1 (0.37→0.89), 7.3x in R2 (0.12→0.90), and
3.1x in R3 (0.26→0.81). The gain is largest in R2 because the Top-K=3
queue there is frequently filled with benign clusters that
superficially match the true attack cluster's size and density; the
LLM's semantic read of sampled texts reliably filters these out.

**Wall-clock cost.** LLM verification adds real inference time per
cluster that must not be conflated with the cycle-based detection
latency above. Measured across the full sweep (qwen3:1.7b, cache
misses only): mean 11.2–13.2 seconds per call across regimes (103–112
real calls per regime), with p95 in the 20–30 second range. A
single-seed comparison with the larger llama3.1:8b model showed
roughly 4x slower inference (mean 45.2s/call) with a smaller precision
gain (0.11→0.50 on R2, vs. 0.12→0.90 for qwen3:1.7b on the same
regime). We recommend qwen3:1.7b for this task based on this
evidence — it is simultaneously faster and more precise — and
recommend deploying LLM verification as an asynchronous,
analyst-facing triage step outside the real-time monitoring path
rather than in any latency-critical alerting flow.

---

### VI. Discussion

The results across Sections V-A through V-D support the paper's core
hypothesis: semantically coherent attack families that evade a fixed
safety classifier manifest as persistent, growing semantic clusters,
and tracking these clusters across monitoring cycles enables earlier
detection than approaches lacking temporal memory. Three findings
merit specific discussion.

**Temporal memory, not clustering alone, drives the latency
advantage.** GuardLens and One-Shot Clustering use identical HDBSCAN
parameters; the only difference is the sliding window and persistent
Cluster Registry. GuardLens's latency advantage over One-Shot Cluster
was most pronounced in the two regimes with gradual onset (R1: 1.0 vs.
2.0 cycles; R3: 2.0 vs. 5.0 cycles) and negligible in R2's sharp burst
(0.0 vs. 0.0 cycles, since the full attack volume arrives in a single
cycle with nothing to accumulate). This is consistent with the
mechanism as designed: temporal tracking helps precisely when evidence
must accumulate across cycles before crossing a density threshold, and
adds nothing when the threshold is already crossed at first
appearance.

**Boundary-proximity sampling is not a substitute for semantic
clustering.** Stratified Random — sampling near the guardrail's
decision threshold, a heuristic that mirrors how many production
safety teams currently allocate review effort — detected 0% of R1 and
R3 attacks. This is because novel and slowly-drifting attacks do not
necessarily hover near the classifier's confidence boundary; they
either evade confidently (scoring well below threshold) or drift in
below the classifier's training distribution entirely. The heuristic
only worked in R2, where near-duplicate templates happened to share a
similar confidence band. This is a negative result worth stating
plainly: teams currently using threshold-proximity sampling as their
primary post-hoc review strategy have a coverage gap for exactly the
attack types GuardLens is designed to catch.

**Novelty, not density or growth, is the load-bearing signal.**
The ablation study (Section V-C) isolates historical novelty — a
cluster's semantic distance from previously observed cluster
centroids — as the component responsible for GuardLens's latency and
coverage advantage. Density and growth mainly refine ranking among
already-novel candidates rather than driving detection outcomes
independently. This has a practical implication for deployment
tuning: teams adapting GuardLens to a new domain should prioritize
correctly calibrating the historical centroid comparison (window
length, similarity threshold) over tuning HDBSCAN's density parameters.

**Fragmentation degrades queue coverage, not detection.** The
corrected adversarial evasion analysis (Section V-B) shows that an
attacker who fragments a coordinated burst across more semantically
distinct communities does not evade detection — GuardLens still
surfaces a cluster within the same cycle regardless of fragmentation
level — but does reduce the fraction of total attack volume visible to
the analyst on first alert (coverage: 0.75 at N=1 → 0.49 at N=5). This
reframes the practical risk of this evasion strategy: it is not a way
to avoid triggering an alert, but a way to make the alert less
informative, requiring the analyst to actively investigate rather than
assume the queue's contents represent the full attack.

Taken together, these findings suggest GuardLens is best understood
not as a replacement for guardrail retraining or threshold tuning, but
as a complementary early-warning layer whose value is concentrated in
exactly the cases existing operational practice (threshold-proximity
sampling, static per-cycle clustering) handles poorly: gradual,
temporally-extended, and adversarially-fragmented blind spots.

---

### VII. Limitations

*(Full content already drafted and finalized with real experimental
data in `paper/limitations_section_draft.md` — paste directly. It
covers, in order: scope of the "100% detection" claim; scale, including
a real 2.5x scale-up result showing detection rate dropping from 100%
to 40% when attack volume is held constant in absolute terms as
background volume triples; the cycle-as-traffic-unit definition with a
real per-cycle wall-clock measurement; the 0.85 match-threshold
sensitivity sweep, reported honestly as an untested gap for the
multi-cycle cluster-drift failure mode rather than as robustness; and
embedding-model dependency, including a real mpnet-base-v2 substitution
result where GuardLens held 100% detection while the MMD Drift baseline
degraded to 80% under the same swap.)*

---

### VIII. Conclusion

We presented GuardLens, a lightweight, model-agnostic monitoring layer
that observes guardrail-approved traffic for emerging semantic
clusters representing potential safety blind spots. By combining
HDBSCAN clustering within overlapping temporal windows with persistent
cross-cycle cluster identity and an Emergence Score that prioritizes
historically novel, growing, and semantically cohesive clusters,
GuardLens surfaces a fixed-budget analyst review queue that requires
no access to the underlying classifier, no retraining, and no labeled
attack data.

Across three attack-emergence regimes, GuardLens matched or exceeded
four baseline monitoring strategies in detection rate and latency, and
ablation analysis identified historical novelty as the primary driver
of this advantage. A corrected adversarial fragmentation analysis
showed GuardLens's detection is robust to an attacker splitting a
coordinated attack across semantically distinct communities, though
analyst-visible coverage of the full attack degrades as fragmentation
increases. An optional LLM-based verification extension improved
review queue precision by 3–7x across all three regimes without
measurably affecting detection latency, though it introduces real
wall-clock cost (10–15 seconds per cluster with a lightweight local
model) that should be scoped to an asynchronous, analyst-facing triage
step rather than the real-time detection path.

These results should be read within the scope in which they were
produced. Detection rate figures of 100% describe performance on the
specific attack mechanisms and traffic volumes (150–200 items/cycle)
tested here, not a general safety guarantee. A direct test of scale
sensitivity — tripling background traffic while holding attack volume
fixed in absolute terms — showed detection rate falling from 100% to
40%, indicating the current fixed `min_cluster_size` and Top-K
configuration does not scale gracefully without further tuning; the
more realistic question of whether detection holds when attack volume
scales *proportionally* with traffic remains open. Similarly, the
cross-cycle match threshold's sensitivity was only meaningfully tested
on a single-burst regime where cross-cycle matching is architecturally
inactive, leaving the multi-cycle cluster-drift failure mode
untested, and a single alternate embedding model was checked rather
than a systematic robustness study against adversarially-crafted
low-semantic-footprint attacks.

Future work should prioritize, in order of how directly each addresses
an open question raised by this evaluation: (1) re-running the scale
experiment with attack volume held proportional to traffic rather than
fixed in absolute count, and evaluating whether `min_cluster_size` and
Top-K should scale with window size; (2) extending the match-threshold
sensitivity sweep to R1 and R3, where cross-cycle matching is actually
exercised across multiple cycles; (3) testing GuardLens against
attacks specifically crafted to have low semantic footprint under a
known embedding model, and across a broader set of embedding models;
and (4) validating detection latency against a real deployment's
traffic cadence to convert cycle-count latency into an operational
SLA. GuardLens is intended as a practical, deployable complement to
existing guardrails — not a replacement for classifier improvement —
and these next steps are aimed at closing the gap between the
controlled evaluation presented here and the operational conditions
of a production deployment.
