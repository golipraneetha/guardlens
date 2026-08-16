# VII. Limitations

**Scope of "100% detection."** The detection-rate figures reported in
Section VI (100% across R1–R3, and across all N in the evasion sweep)
hold within the controlled parameters of these experiments — specific
attack families, a fixed 150–200 item background rate, 5 random seeds,
and a single guardrail. They should not be read as a claim that
GuardLens detects all attacks with certainty in production. Real
deployments face far greater traffic heterogeneity, attack diversity,
and adversaries actively probing the monitor itself (Section VI-C's
evasion sweep is one instance of this, not an exhaustive one). We
report 100% detection as a property of these specific regimes, and
recommend it be read as "detects the blind-spot mechanisms tested
here, reliably, at this scale" rather than a general safety guarantee.

**Scale.** Each simulated cycle draws 150–200 benign background items
plus attack traffic — three to four orders of magnitude below the
per-interval volume of enterprise LLM traffic. This scale was chosen
so that DeBERTa scoring and embedding generation stayed within the
throughput of the local hardware used for all experiments (Ollama +
CPU-bound classifier, no GPU).

We tested a 2.5x scale-up (Table VI, Section VI-F) to get an initial
read on this question rather than leave it purely speculative, and the
result is a genuine limitation, not a reassuring one: detection rate
dropped from 100% to 40% when background traffic tripled with attack
volume held fixed in absolute terms. This does not answer the scaling
question so much as sharpen it: it shows the current
fixed-`min_cluster_size`, fixed-top-K configuration does not scale
gracefully when attack volume is held constant in absolute terms as
background volume grows. It leaves open whether detection would hold
if attack volume scaled *proportionally* with traffic instead (i.e.,
a fixed attack *fraction* rather than a fixed attack *count* —
arguably the more realistic enterprise scenario, since a viral
jailbreak template's absolute volume would likely also grow with total
traffic, not stay pinned at 30 items). We did not have time to run
that variant in this paper; it is the most direct and important piece
of follow-up work this finding motivates, and `min_cluster_size` and
`top_k` may need to scale with window size in a production deployment
rather than remain fixed constants.

**"Cycle" is a traffic-volume unit, not a time unit.** All latency
results in this paper are reported in cycles, not wall-clock time,
because a cycle is defined as one batch of `benign_per_cycle` +
attack items, not a fixed time window. This was a deliberate choice to
keep the detection-latency metric independent of any particular
deployment's polling cadence, but it means "0–2 cycles" cannot be
directly compared to a production SLA without first fixing how
frequently a real deployment batches traffic into a "cycle" (e.g., a
5-minute polling interval at enterprise volume vs. an hourly batch at
lower volume would give very different wall-clock latencies for the
same cycle-count result). We did not have a target deployment's actual
traffic cadence to calibrate against, so we leave cycle-to-wall-clock
conversion as an explicit deployment-time parameter rather than
assuming one. Note this is distinct from the wall-clock cost of the
optional LLM verification step (Section VI-E), which we do report in
real seconds because that step's cost is deployment-independent
(fixed per-cluster LLM inference time).

To at least ground the *compute* cost side of this (separate from the
polling-cadence side, which remains deployment-defined), we measured
actual wall-clock time for GuardLens's own per-cycle pipeline (embed +
HDBSCAN + registry update + emergence scoring) on Regime 2, 50 cycles
across 5 seeds, on the local CPU-only hardware used for all
experiments in this paper: mean 4.2s/cycle, median 3.0s, p95 7.8s, max
16.5s. This measurement benefits from the sliding window's embedding
cache (most texts in a given cycle's window were already embedded in
a prior cycle), so it reflects incremental streaming cost rather than
cold-start cost. This means GuardLens's own processing overhead is on
the order of single-digit seconds per cycle at this traffic volume —
well under any reasonable polling interval — but this number should
not be assumed to hold at the traffic volumes discussed in the Scale
limitation above; embedding and clustering cost both scale with
window size, so a 10-100x increase in `benign_per_cycle` would need
separate re-measurement.

**Cross-cycle match threshold (cosine similarity 0.85).** The
`ClusterRegistry` component that gives clusters stable identity across
cycles — which growth and novelty scoring in Section VI-D's ablation
both depend on — uses a hardcoded cosine similarity threshold of 0.85
to decide whether a new cycle's cluster is "the same" cluster grown or
shrunk, versus a newly-born one. We ran a sensitivity sweep over
{0.70, 0.75, 0.80, 0.85, 0.90, 0.95} on Regime 2 (5 seeds each).

**Table VIII — Match Threshold Sensitivity (Regime 2, 5 seeds)**

| Threshold | Detection Rate | Latency | Purity | Coverage | Fragmentation |
|---|---|---|---|---|---|
| 0.70 | 100% | 0.0 | 0.861 | 0.797 | 1.0 |
| 0.75 | 100% | 0.0 | 0.861 | 0.797 | 1.0 |
| 0.80 | 100% | 0.0 | 0.861 | 0.797 | 1.0 |
| 0.85 (default) | 100% | 0.0 | 0.861 | 0.797 | 1.0 |
| 0.90 | 100% | 0.0 | 0.861 | 0.797 | 1.0 |
| 0.95 | 100% | 0.0 | 0.861 | 0.797 | 1.0 |

**All six thresholds produce identical results on Regime 2.** This is
not evidence of general robustness — it is a limitation of the test
itself. R2's attack arrives as a single burst that first appears at
the onset cycle, so the very first attack cluster detected has no
prior-cycle cluster to match against (`prev_size=None`, age=1 for
every attack-bearing cluster at first detection). Cross-cycle matching
is architecturally a no-op for a regime whose detection event happens
at the first cycle attack traffic exists at all. The threshold
therefore remains functionally untested by this sweep for the failure
mode we actually care about: a slowly-evolving or gradually-drifting
attack (R1, R3) that must be correctly re-matched to its own prior
identity across multiple cycles for growth and novelty scoring to
work as intended. Testing the same sweep on R1/R3 — where the
temporal-tracking mechanism is actually exercised across 5+ cycles —
is necessary before any claim about threshold robustness can be made,
and is left as follow-up work. This threshold was not tuned against a
validation set in the first place; it was chosen as a reasonable prior
for `all-MiniLM-L6-v2` embeddings specifically, and we did not test
whether 0.85 generalizes to other embedding models. It remains
plausible that a poorly-calibrated threshold in either direction
causes real failure modes on multi-cycle regimes: too low, and a
malicious cluster could be incorrectly matched to (and its
growth/novelty scores diluted by) an unrelated benign cluster from a
prior cycle ("cluster drift" merging two semantically distinct
groups); too high, and a slowly-evolving attack that paraphrases
itself cycle-to-cycle would be treated as a sequence of "newly born"
clusters, resetting its growth signal to zero each time and
undermining the temporal-tracking mechanism that Section VI-B shows
gives GuardLens its latency advantage over One-Shot Cluster on R1/R3.

**Core clustering hyperparameters (window size, Top-K, min_cluster_size).**
Beyond the match threshold, three other hyperparameters are fixed
throughout this paper without a stated justification: sliding
`window_size=3`, review budget `top_k=3`, and HDBSCAN's
`min_cluster_size=5`. A reviewer can reasonably ask whether these
values were tuned to the specific attack volumes tested here, or
whether detection is robust to reasonable variation around them. We
ran three sensitivity sweeps — window size {2, 3, 4, 5}, Top-K {1, 3,
5}, and `min_cluster_size` {3, 5, 8, 10} — each varied independently
with the other two held at their default value, 5 seeds per
configuration. Unlike Table VIII's match-threshold sweep, these run on
**R1 (Novel Family)**, where the attack ramps in gradually over
several cycles and cross-cycle tracking is actually exercised — R2's
single-burst onset would again make several of these parameters
architecturally inert (e.g., `window_size` cannot matter if the whole
attack is visible and clusterable within one cycle). We also ran this
sweep against the **realistic-traffic, unseen-attack-variant condition**
from Section VI-G (Table IX) — Alpaca/OASST1/UltraChat benign traffic
and LLM-paraphrased/novel-intent AdvBench variants — rather than raw
benchmark replay, so this sensitivity result is not confounded by the
same "known benchmark" concern that motivated Table IX in the first
place.

**Table X — Window Size Sensitivity (R1, unseen variants, realistic traffic, 5 seeds)**

| Window Size (cycles) | Detection Rate | Median Latency | Mean Purity | Mean Coverage |
|---|---|---|---|---|
| 2 | 100% | 1.0 | 0.87 | 0.52 |
| 3 (default) | 100% | 1.0 | 0.84 | 0.50 |
| 4 | 100% | 1.0 | 0.78 | 0.42 |
| 5 | 100% | 1.0 | 0.87 | 0.44 |

**Table XI — Top-K Sensitivity (R1, unseen variants, realistic traffic, 5 seeds)**

| Top-K (queue budget) | Detection Rate | Median Latency | Mean Purity | Mean Coverage |
|---|---|---|---|---|
| 1 | 100% | 1.0 | 0.84 | 0.50 |
| 3 (default) | 100% | 1.0 | 0.84 | 0.50 |
| 5 | 100% | 1.0 | 0.84 | 0.50 |

**Table XII — min_cluster_size Sensitivity (R1, unseen variants, realistic traffic, 5 seeds)**

| min_cluster_size (HDBSCAN) | Detection Rate | Median Latency | Mean Purity | Mean Coverage |
|---|---|---|---|---|
| 3 | 100% | 1.0 | 0.89 | 0.43 |
| 5 (default) | 100% | 1.0 | 0.84 | 0.50 |
| 8 | 100% | 1.0 | 0.78 | 0.56 |
| 10 | 100% | 1.0 | 0.87 | 0.55 |

**Detection rate and median latency are unchanged across all eleven
configurations tested** — 100% detection at 1.0-cycle median latency
holds for every window size, every Top-K, and every `min_cluster_size`
value swept, on the harder unseen-attack-variant condition. Purity and
coverage move modestly (purity spans 0.78–0.89 across all three
sweeps; coverage spans 0.42–0.56), but neither shows a clear monotonic
trend with any parameter, and no configuration causes a detection
failure. The Top-K sweep is a degenerate case worth naming explicitly:
purity and coverage are byte-for-byte identical across Top-K=1, 3, and
5 because R1's attack cluster is consistently ranked first by the
Emergence Score — a lower Top-K never excludes it, and a higher Top-K
only adds lower-ranked benign clusters that don't affect the
detection or purity/coverage-at-first-detection metrics (which are
computed from the first genuine attack cluster found, not the full
queue). This means the Top-K sweep in this experiment demonstrates
*ranking* stability, not queue-composition sensitivity — a separate
question (how much benign material accompanies the attack cluster in
the queue) is more directly addressed by Section VI-D's precision
metrics.

This sweep is reassuring within its scope but should not be read as
"GuardLens is insensitive to hyperparameters in general." It shows
stability across a roughly 2–3x range around each default on one
regime, one traffic condition, and one embedding model. It does not
test values far outside this range (e.g., `min_cluster_size=50` on
150–200 item cycles, which would likely never trigger), does not test
parameter *interactions* (all three were varied one at a time, holding
the others at default — a joint sweep is combinatorially larger and
left as follow-up work), and does not test R2 or R3 under the same
protocol (R2 is architecturally insensitive to `window_size` for the
reason stated above, and R3's slower ramp was not re-run under this
sweep due to time constraints). The defaults used throughout this
paper were chosen as reasonable priors for the traffic volumes tested
(150–200 items/cycle) rather than tuned via grid search against a
validation set, and this sweep is evidence — not proof — that nearby
values would have produced substantively similar results.

**Dependency on a single embedding model.** GuardLens's entire
detection mechanism is downstream of `all-MiniLM-L6-v2` producing
embeddings where semantically related attack texts land close together
in cosine space. An adversary who constructs prompts specifically to
have a low-semantic-footprint under this embedding model — for
example, by heavily obfuscating or encoding malicious intent behind
surface text that reads as generic or benign in this particular vector
space, without necessarily fooling the downstream guardrail or a
human reviewer — could produce attack traffic that never clusters
tightly enough to clear `min_cluster_size` or accumulate density, and
would be invisible to GuardLens regardless of volume. We did not test
this specific low-semantic-footprint attack class in this paper (an
earlier design decision in this project prioritized the
LLM-verification extension over a full embedding-robustness study,
given time constraints — see Section VI-E). We did run a narrower
check (Table VII, Section VI-F): substituting `all-mpnet-base-v2` for
`all-MiniLM-L6-v2` on Regime 2. GuardLens held 100% detection under
the swap while the MMD Drift baseline dropped to 80% under the
identical swap — evidence, on this one alternate model and regime,
that GuardLens's cluster-based detection is at least as
embedding-robust as a standard statistical baseline. This is
reassuring but narrow evidence: it shows detection survives switching
to one different, generically-trained embedding model on one regime.
It says nothing about the adversarial case above (an attacker
specifically crafting low-semantic-footprint text against a *known*
embedding model), which remains untested and is the more concerning
version of this limitation. Because GuardLens is architecturally a
monitoring layer, not a blocking layer, over-reliance on a single
embedding model is mitigated in practice by defense-in-depth (the
underlying guardrail and any downstream human review remain the
actual safety backstops), but this adversarial blind spot in the
detection mechanism itself remains genuinely untested; testing against
deliberately-crafted low-semantic-footprint attacks, and across more
embedding models (e.g., one tuned specifically for adversarial-text
detection), is future work.
