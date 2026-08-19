# GuardLens

Temporal monitoring layer for emerging LLM safety blind spots. GuardLens observes traffic a
guardrail has already approved, clusters it in a sliding temporal window, tracks cluster
identity across monitoring cycles, and ranks clusters with an interpretable Emergence Score to
prioritize a fixed-budget analyst review queue. See [`paper/`](paper/) for the manuscript.

## Setup

Requires Python 3.11+ (developed on 3.14).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The deployed guardrail classifier (ProtectAI DeBERTa Prompt Injection v2) and the Sentence-BERT
embedder download automatically from Hugging Face on first run.

### Optional: LLM-based cluster verification and attack-variant generation

Sections of the pipeline that generate LLM-paraphrased attack variants (`traffic/attack_variants.py`)
or run the optional post-hoc cluster verifier (`guardlens/llm_verifier.py`) call a locally hosted
[Ollama](https://ollama.com) server over HTTP — not a pip dependency. Install Ollama separately and
pull the models referenced in the experiment you're running, e.g.:

```bash
ollama pull qwen3:8b      # attack-variant generation (Tier B/C)
ollama pull llama3.1      # cluster verification (independent-model rerun)
ollama pull qwen3:1.7b    # cluster verification (original run)
```

Experiments that don't pass `--llm-verify` or use `--attack-tier b/c/bc` don't need Ollama at all.

## Running the core experiments

All experiment scripts live in `experiments/` and write a JSON results file plus a printed summary.
The main benchmark harness:

```bash
python3 experiments/run_experiment.py --regime novel_family --seeds 5 --cycles 10
```

`--regime` is one of `novel_family` (R1), `coordinated_attack` (R2), `slow_drift` (R3), or
`diverse_attack`. Key flags:

- `--traffic-source {benchmark,realistic}` — benchmark replay vs. the realistic Alpaca+OASST1+UltraChat benign pool
- `--attack-tier {a,b,c,bc}` — raw benchmark attacks vs. LLM-generated unseen variants
- `--llm-verify --llm-model <name>` — enable the optional LLM cluster verifier
- `--score-cache <path>` — cache DeBERTa guardrail scores across runs to skip re-scoring (the
  guardrail-scoring pass is the slowest part of a cold run; a warm cache makes reruns fast)

Other scripts follow the same pattern for specific experiments:

| Script | What it runs |
|---|---|
| `run_ablation.py` | Leave-one-component-out Emergence Score ablation |
| `run_benign_study.py` | R4 benign demand-shift regime (queue competition) |
| `run_profiling.py` | Per-stage computational profile across cycle sizes |
| `run_embedding_evasion.py` | Embedding-aware adversary stress test |
| `run_joint_sweep.py` | Joint match-threshold / growth-floor / score-weight sweep |
| `run_llm_verification_sweep.py` | LLM cluster verifier sweep across regimes |

A full cold run across all regimes and traffic conditions takes on the order of hours, dominated by
DeBERTa guardrail scoring and (where enabled) local LLM calls; use `--score-cache` and the existing
caches under `experiments/*/` to avoid re-paying that cost on reruns.

## Tests

```bash
pytest tests/
```

Unit tests cover the clusterer, cluster registry, emergence scoring, baselines, LLM verifier
parsing, and monitor integration — independent of the guardrail model or Ollama.

## Reproducing the paper's results

Result JSONs and score/LLM caches backing the paper's tables (`experiments/ablation_realistic/`,
`experiments/joint_sweep/`, `experiments/embedding_evasion/`, `experiments/profiling/`,
`experiments/llm_verification_realistic_llama31/`) are checked into the repo rather than
regenerated on every clone, since several of these runs take hours. Each directory's `summary*.json`
is what the paper's tables are drawn from directly; the corresponding `run_*.py` script with the
same flags reproduces it from scratch.

## Layout

- `guardlens/` — core library: embedder, clusterer, cluster registry, emergence scoring, monitor, LLM verifier
- `baselines/` — comparison methods (random audit, stratified random, MMD drift, one-shot cluster, isolation forest)
- `traffic/` — traffic regime construction, dataset loaders, realistic benign pool, attack-variant generation
- `experiments/` — experiment scripts, metrics, and result outputs
- `tests/` — unit tests
- `paper/` — manuscript source and figures
