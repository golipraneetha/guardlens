"""Realistic benign traffic from established open-source conversation datasets.

Mixes three sources to simulate diverse enterprise LLM traffic:
  - Alpaca (instruction-following tasks)
  - OpenAssistant OASST1 (multi-turn human conversations)
  - UltraChat 200k (synthetic but diverse dialog)

This replaces the single-source jailbreak_llms_regular benign pool with
traffic that a reviewer cannot dismiss as "from the same distribution as
the attack benchmark."
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / "realistic_cache"


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.json"


def _load_alpaca(n: int = 5000, seed: int = 42) -> list[str]:
    cached = _cache_path("alpaca")
    if cached.exists():
        texts = json.loads(cached.read_text())
        return texts[:n]

    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    texts = []
    for row in ds:
        instr = (row.get("instruction") or "").strip()
        inp = (row.get("input") or "").strip()
        if instr:
            t = f"{instr} {inp}".strip() if inp else instr
            texts.append(t[:600])
    rng = random.Random(seed)
    rng.shuffle(texts)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(texts[:n]))
    return texts[:n]


def _load_oasst1(n: int = 5000, seed: int = 42) -> list[str]:
    cached = _cache_path("oasst1")
    if cached.exists():
        texts = json.loads(cached.read_text())
        return texts[:n]

    from datasets import load_dataset
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    texts = []
    for row in ds:
        if row.get("role") == "prompter" and row.get("lang") == "en":
            t = (row.get("text") or "").strip()
            if t and len(t) > 20:
                texts.append(t[:600])
    rng = random.Random(seed)
    rng.shuffle(texts)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(texts[:n]))
    return texts[:n]


def _load_ultrachat(n: int = 5000, seed: int = 42) -> list[str]:
    cached = _cache_path("ultrachat")
    if cached.exists():
        texts = json.loads(cached.read_text())
        return texts[:n]

    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    texts = []
    for row in ds:
        msgs = row.get("messages", [])
        if msgs and msgs[0].get("role") == "user":
            t = msgs[0]["content"].strip()
            if t and len(t) > 20:
                texts.append(t[:600])
    rng = random.Random(seed)
    rng.shuffle(texts)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(texts[:n]))
    return texts[:n]


def load_realistic_benign_pool(
    n_per_source: int = 5000,
    seed: int = 42,
) -> list[str]:
    """Returns a shuffled mix of Alpaca + OASST1 + UltraChat prompts.

    Default 15,000 total (5k each), comparable in size to the original
    jailbreak_llms_regular pool (13,985) but from three independent
    sources with no overlap with attack benchmarks.
    """
    alpaca = _load_alpaca(n_per_source, seed)
    oasst = _load_oasst1(n_per_source, seed)
    ultra = _load_ultrachat(n_per_source, seed)

    pool = alpaca + oasst + ultra
    rng = random.Random(seed)
    rng.shuffle(pool)

    return pool
