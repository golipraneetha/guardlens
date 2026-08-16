"""Dataset loaders. Reuses the benchmark CSVs already vetted in the
tenant-calibration project rather than re-downloading anything.

Three distinct attack pools are kept deliberately separate so the three
evaluation regimes don't share attack text between them:
  - advbench   -> Regime 1 (novel adversarial family)
  - jailbreak_llms (jailbreak=True) -> Regime 2 (coordinated attack: narrow
    template variants drawn from a single community)
  - harmbench  -> Regime 3 (slow drift)
"""
from __future__ import annotations

import csv
from pathlib import Path

BENCH_DIR = Path("/Users/chaituprani/Downloads/tenant-calibration/benchmarks")
MAX_CHARS = 600  # truncate to keep classifier/embedding scoring fast


def _load_col(path: Path, col: str, where=None) -> list[str]:
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if where is not None and not where(r):
                continue
            v = r.get(col, "").strip()
            if v:
                rows.append(v[:MAX_CHARS])
    return rows


def load_benign_pool() -> list[str]:
    """'Regular' chat prompts (jailbreak_llms) + xstest safe-but-tricky
    prompts, so the benign pool includes some hard negatives (prompts that
    superficially resemble attacks but aren't)."""
    regular = _load_col(BENCH_DIR / "jailbreak_llms_regular.csv", "prompt")
    xstest_safe = _load_col(BENCH_DIR / "xstest_prompts.csv", "prompt",
                            where=lambda r: r.get("label") == "safe")
    return regular + xstest_safe


def load_advbench() -> list[str]:
    return _load_col(BENCH_DIR / "advbench_harmful_behaviors.csv", "goal")


def load_harmbench(semantic_category: str | None = None) -> list[str]:
    """harmbench spans several unrelated harm categories (copyright,
    cybercrime, chemical_biological, misinformation, ...). For regimes that
    represent ONE coherent attack family gradually emerging, sampling
    across all categories injects an incoherent mix that won't cluster
    regardless of clustering quality -- pass semantic_category to restrict
    to a single coherent family."""
    where = None
    if semantic_category is not None:
        where = lambda r: r.get("SemanticCategory", "").strip() == semantic_category
    return _load_col(BENCH_DIR / "harmbench_behaviors.csv", "Behavior", where=where)


def load_jailbreak_community_templates(min_community_size: int = 20) -> dict[str, list[str]]:
    """Groups confirmed jailbreak prompts by their originating Discord/Reddit
    community. Communities with many near-duplicate submissions are good
    stand-ins for a 'coordinated attack': many variants of one template
    hitting the guardrail in a short span."""
    by_community: dict[str, list[str]] = {}
    with open(BENCH_DIR / "jailbreak_llms_prompts.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            community = r.get("community", "").strip()
            prompt = r.get("prompt", "").strip()
            if not community or not prompt:
                continue
            by_community.setdefault(community, []).append(prompt[:MAX_CHARS])
    return {c: texts for c, texts in by_community.items() if len(texts) >= min_community_size}
