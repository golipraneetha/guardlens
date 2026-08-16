# Session Status (last updated before laptop restart, 2026-07-14)

## What's complete and saved

All of these are done, on disk, and reflected in `paper/all_experiments_summary.md`:

- Main detection results (`realistic_eval/`), scale test (`scale_test/`),
  hyperparameter sweeps (`hyperparam_sweep/`) — unchanged, current.
- Ablation study rerun on realistic traffic + Tier B/C, 5 seeds:
  `ablation_realistic/ablation_summary_realistic.json`
- LLM verification rerun on realistic traffic + Tier B/C, **qwen3:1.7b**
  verifier, 5 seeds: `llm_verification_realistic_qwen1.7b/verification_summary_realistic.json`
  (this dir was renamed from `llm_verification_realistic/` to make room for
  the llama3.1 attempt below — same data, just moved).

## Incomplete — do not trust, needs rerun

**LLM verification with `llama3.1:latest` as verifier** was attempted to
replace qwen3:1.7b (stronger model, and independent of the qwen3:8b variant
generator — avoids "verifier judges its own paraphrases" critique). It was
**killed after 1h40m with zero completed verdicts** — CPU-only inference on
this machine + heavy system contention (load avg spiked to 200+) meant the
first `novel_family_llm` run never finished even one Ollama call.

- Only file that exists: `llm_verification_realistic_llama31/novel_family_baseline.json`
  (the fast no-LLM baseline, which did complete). **No `*_llm.json` files
  exist** — nothing to salvage, the run produced no cached verdicts.
- `run_llm_verification_sweep.py` is currently configured for
  `LLM_MODEL = "llama3.1:latest"`, writing to `llm_verification_realistic_llama31/`.
  If retrying, first check system load (`uptime`) is reasonable before
  launching — this machine cannot run an 8B model via Ollama CPU inference
  under concurrent load without effectively stalling.
- If not retrying: revert `LLM_MODEL` to `"qwen3:1.7b"` and `RESULTS_DIR` to
  `llm_verification_realistic_qwen1.7b` in `run_llm_verification_sweep.py`
  to restore the working configuration, and rely on the qwen3:1.7b results
  already in hand (see above) — those are complete and valid.

## Paper draft status

`paper/all_experiments_summary.md` currently documents the **qwen3:1.7b**
LLM verification results as Section 6 (this is correct/current — it was
written before the llama3.1 attempt and does not need updating unless the
llama3.1 rerun eventually succeeds and the user wants it swapped in).

No other paper files were changed this session beyond what's already
reflected in `all_experiments_summary.md`.
