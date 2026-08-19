"""Diagnostic: does INT8 dynamic quantization backend (qnnpack, forced on
this ARM machine) shift DeBERTa confidence scores enough to flip
approve/block decisions at THRESHOLD=0.5, relative to unquantized fp32?

Reuses master_score_cache.json's prompts (already scored under qnnpack
this session) and rescores the same texts in fp32, comparing directly.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

THRESHOLD = 0.5
MODEL_NAME = "ProtectAI/deberta-v3-base-prompt-injection-v2"

cache_path = Path(__file__).parent / "master_score_cache.json"
qnnpack_scores = json.loads(cache_path.read_text())
print(f"Loaded {len(qnnpack_scores)} cached qnnpack-quantized scores")

# sample: all texts near the threshold (most diagnostic) + a random spread
items = list(qnnpack_scores.items())
near_threshold = [t for t, s in items if 0.3 <= s <= 0.7]
rng = np.random.default_rng(0)
random_sample_idx = rng.choice(len(items), size=min(300, len(items)), replace=False)
random_sample = [items[i][0] for i in random_sample_idx]
sample_texts = list(dict.fromkeys(near_threshold + random_sample))
print(f"Testing {len(sample_texts)} texts ({len(near_threshold)} near threshold 0.3-0.7, "
      f"rest random sample)")

print("\nLoading fp32 model (no quantization)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to("cpu").eval()
inj_idx = model.config.label2id.get("INJECTION", 1)

fp32_scores = {}
batch_size = 16
for i in range(0, len(sample_texts), batch_size):
    batch = sample_texts[i:i + batch_size]
    enc = tokenizer(batch, padding=True, truncation=True, max_length=512,
                    return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[:, inj_idx].tolist()
    for t, s in zip(batch, probs):
        fp32_scores[t] = float(np.clip(s, 0.02, 0.98))
    if i % 160 == 0:
        print(f"  scored {i}/{len(sample_texts)}...")

print("\n=== Comparison: qnnpack (cached) vs fp32 (fresh) ===")
deltas = []
flips = []
for t in sample_texts:
    q = qnnpack_scores[t]
    f = fp32_scores[t]
    deltas.append(abs(q - f))
    q_approved = q < THRESHOLD
    f_approved = f < THRESHOLD
    if q_approved != f_approved:
        flips.append((t, q, f))

deltas = np.array(deltas)
print(f"n={len(sample_texts)}")
print(f"mean |delta|:   {deltas.mean():.4f}")
print(f"median |delta|: {np.median(deltas):.4f}")
print(f"max |delta|:    {deltas.max():.4f}")
print(f"p95 |delta|:    {np.percentile(deltas, 95):.4f}")
print(f"\nThreshold flips (approve/block decision changes): {len(flips)} / {len(sample_texts)} "
      f"({100*len(flips)/len(sample_texts):.1f}%)")

if flips:
    print("\nFirst 10 flips (text[:80], qnnpack_score, fp32_score):")
    for t, q, f in flips[:10]:
        print(f"  q={q:.3f} f={f:.3f}  {t[:80]!r}")

out = {
    "n_tested": len(sample_texts),
    "mean_abs_delta": float(deltas.mean()),
    "median_abs_delta": float(np.median(deltas)),
    "max_abs_delta": float(deltas.max()),
    "p95_abs_delta": float(np.percentile(deltas, 95)),
    "n_threshold_flips": len(flips),
    "flip_rate": len(flips) / len(sample_texts),
    "flips": [{"text": t[:200], "qnnpack": q, "fp32": f} for t, q, f in flips],
}
out_path = Path(__file__).parent / "quantization_diagnostic_results.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\nWrote {out_path}")
