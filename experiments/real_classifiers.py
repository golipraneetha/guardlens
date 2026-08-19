"""
Deployed prompt-injection guardrail wrapper.

Adapted from the tenant-calibration project's real_classifiers.py, trimmed
to just the classifier GuardLens experiments sit behind: ProtectAI's
DeBERTa prompt-injection detector. It is frozen and pretrained — GuardLens
never retrains or fine-tunes it, only forwards the prompts it approves.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _BatchedHFClassifier:
    """Loads a HF sequence-classification model once (cached for the life
    of the process) and caches per-text confidence scores. The rest of the
    pipeline only ever calls confidence(text) one item at a time (it's
    simulating a live decision stream), so batching happens via
    warm_cache(texts): callers that already know the full item list up
    front (baseline scoring, cycle-chunk streaming) pass it in and get
    batched forward passes; confidence() then becomes a cache hit. A cold
    confidence() call (cache miss) still works — it just falls back to a
    batch of one.

    Batch size defaults to 16 (not 32) and warm_cache bisects + retries on
    OOM as a safety net, clearing the MPS cache between batches to reduce
    fragmentation. Subclasses can set _AVOID_MPS = True to skip MPS
    entirely (DeBERTa-v2/v3's disentangled attention materializes large
    relative-position bias tensors that are known to be memory-inefficient
    on MPS specifically, well beyond what CUDA/CPU need for the same
    model — this repeatedly OOM'd here even at batch size 1)."""
    _MODEL_NAME: str
    _AVOID_MPS = False

    def __init__(self, device: str | None = None, batch_size: int = 16):
        if device is None:
            device = _pick_device()
            if self._AVOID_MPS and device == "mps":
                device = "cpu"
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self._MODEL_NAME)
        self.model = (AutoModelForSequenceClassification
                      .from_pretrained(self._MODEL_NAME)
                      .to(self.device).eval())
        # Dynamic INT8 quantization via qnnpack (the only backend available
        # on this ARM machine; fbgemm is x86-only) was tested and found to
        # badly miscalibrate this specific DeBERTa-v3 model: mean |score
        # delta| of 0.44 vs fp32 on a 6,320-prompt sample, with 40.7% of
        # prompts flipping their approve/block decision at THRESHOLD=0.5
        # (experiments/quantization_diagnostic.py). Running fp32 throughout
        # instead -- slower (~1 item/s vs ~3-4 item/s on CPU) but correct.
        self._cache: dict[str, float] = {}

    def _score_batch(self, texts: list[str]) -> list[float]:
        raise NotImplementedError

    def _score_batch_safe(self, texts: list[str]) -> list[float]:
        try:
            return self._score_batch(texts)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower() or len(texts) == 1:
                raise
            if self.device == "mps":
                torch.mps.empty_cache()
            mid = len(texts) // 2
            return (self._score_batch_safe(texts[:mid])
                    + self._score_batch_safe(texts[mid:]))

    def warm_cache(self, texts) -> None:
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        for i in range(0, len(todo), self.batch_size):
            batch = todo[i:i + self.batch_size]
            for t, s in zip(batch, self._score_batch_safe(batch)):
                self._cache[t] = float(np.clip(s, 0.02, 0.98))
            if self.device == "mps":
                torch.mps.empty_cache()

    def confidence(self, text: str) -> float:
        if text not in self._cache:
            self.warm_cache([text])
        return self._cache[text]


class DebertaInjectionClassifier(_BatchedHFClassifier):
    """ProtectAI/deberta-v3-base-prompt-injection-v2 — DeBERTa fine-tuned
    specifically for prompt-injection detection (binary SAFE/INJECTION
    head). confidence() is P(INJECTION). Frozen and pretrained: there is
    no fit step. Forced to CPU (_AVOID_MPS) — see _BatchedHFClassifier
    docstring."""
    _MODEL_NAME = "ProtectAI/deberta-v3-base-prompt-injection-v2"
    _AVOID_MPS = True

    def _score_batch(self, texts: list[str]) -> list[float]:
        enc = self.tokenizer(texts, padding=True, truncation=True,
                             max_length=512, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        inj_idx = self.model.config.label2id.get("INJECTION", 1)
        return probs[:, inj_idx].tolist()
