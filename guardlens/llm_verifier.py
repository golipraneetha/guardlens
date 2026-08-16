"""LLM-based cluster verification: post-queue triage filter.

After GuardLens produces a ranked review queue of top-K clusters, the
verifier samples representative texts from each cluster, sends them to
a local LLM (via Ollama), and asks whether the cluster represents a
genuine safety concern or a benign anomaly.  Clusters the LLM rejects
are filtered from the queue before analyst inspection.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

from .queue import QueueEntry
from ._math import cosine

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a safety analyst reviewing text clusters that passed through "
    "an AI content safety filter. Your job is to determine whether a cluster "
    "of similar texts represents a genuine safety concern (e.g., coordinated "
    "jailbreak attempts, prompt injections, harmful content requests) or a "
    "benign pattern (e.g., similar but harmless user queries)."
)

CLUSTER_PROMPT_TEMPLATE = (
    "Below are {n} representative texts from a cluster of {size} similar items "
    "that were approved by an automated safety classifier. These texts were "
    "grouped together because they are semantically similar to each other.\n\n"
    "--- CLUSTER TEXTS ---\n{numbered_texts}\n--- END ---\n\n"
    "Analyze these texts and respond in EXACTLY this format (no other text):\n\n"
    "VERDICT: THREAT or BENIGN\n"
    "CATEGORY: <one or two word category label>\n"
    "SEVERITY: HIGH or MEDIUM or LOW or NONE\n"
    "REASONING: <one sentence explanation>"
)


@dataclass
class LLMVerdict:
    cluster_uid: int
    is_threat: bool
    category: str
    severity: str
    reasoning: str
    latency_seconds: float = 0.0  # 0.0 for cache hits; wall-clock for real calls


class ClusterVerifier:
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        model: str = "llama3.1",
        n_samples: int = 6,
        temperature: float = 0.0,
        max_tokens: int = 256,
        cache_path: Path | None = None,
    ):
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.model = model
        self.n_samples = n_samples
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._cache: dict[str, dict] = {}
        self._cache_path = cache_path
        if cache_path and cache_path.exists():
            self._cache = json.loads(cache_path.read_text())
            logger.info("Loaded %d cached LLM verdicts from %s",
                        len(self._cache), cache_path)

    def verify_queue(
        self,
        queue: list[QueueEntry],
        window_texts: list[str],
        embeddings: np.ndarray | None = None,
    ) -> tuple[list[QueueEntry], list[LLMVerdict]]:
        verdicts: list[LLMVerdict] = []
        filtered: list[QueueEntry] = []
        for entry in queue:
            samples = self._sample_texts(entry, window_texts, embeddings)
            verdict = self._verify_cluster(entry.cluster.uid,
                                           entry.cluster.size, samples)
            verdicts.append(verdict)
            if verdict.is_threat:
                filtered.append(entry)
        for i, entry in enumerate(filtered):
            entry.rank = i + 1
        return filtered, verdicts

    def save_cache(self) -> None:
        if self._cache_path:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._cache, indent=2))

    def _sample_texts(
        self,
        entry: QueueEntry,
        window_texts: list[str],
        embeddings: np.ndarray | None,
    ) -> list[str]:
        indices = entry.cluster.indices
        n = min(self.n_samples, len(indices))
        if n <= 0:
            return []

        if embeddings is not None and len(embeddings) > 0:
            centroid = entry.cluster.centroid
            sims = np.array([cosine(embeddings[i], centroid) for i in indices])
            order = np.argsort(sims)
            n_far = n // 2
            n_near = n - n_far
            selected_idx = np.concatenate([order[:n_far], order[-n_near:]])
            selected = indices[selected_idx.astype(int)]
        else:
            step = max(1, len(indices) // n)
            selected = indices[::step][:n]

        return [window_texts[i] for i in selected]

    def _cache_key(self, texts: list[str]) -> str:
        content = self.model + "\n" + "\n".join(sorted(texts))
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _verify_cluster(self, cluster_uid: int, cluster_size: int,
                        texts: list[str]) -> LLMVerdict:
        key = self._cache_key(texts)
        if key in self._cache:
            c = self._cache[key]
            return LLMVerdict(
                cluster_uid=cluster_uid,
                is_threat=c["is_threat"],
                category=c["category"],
                severity=c["severity"],
                reasoning=c["reasoning"],
                latency_seconds=0.0,
            )

        prompt = self._build_prompt(texts, cluster_size)
        t0 = time.time()
        raw_response = self._call_ollama(prompt)
        elapsed = time.time() - t0
        verdict = self._parse_response(cluster_uid, raw_response)
        verdict.latency_seconds = elapsed

        self._cache[key] = dict(
            is_threat=verdict.is_threat,
            category=verdict.category,
            severity=verdict.severity,
            reasoning=verdict.reasoning,
        )
        return verdict

    def _build_prompt(self, texts: list[str], cluster_size: int) -> str:
        numbered = "\n".join(f"{i+1}. {t[:200]}" for i, t in enumerate(texts))
        return (
            SYSTEM_PROMPT + "\n\n" +
            CLUSTER_PROMPT_TEMPLATE.format(
                n=len(texts),
                size=cluster_size,
                numbered_texts=numbered,
            )
        )

    def _call_ollama(self, prompt: str, retries: int = 2) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        for attempt in range(retries + 1):
            try:
                resp = requests.post(
                    f"{self.ollama_base_url}/api/generate",
                    json=payload,
                    timeout=300,
                )
                resp.raise_for_status()
                return resp.json().get("response", "").strip()
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError,
                    requests.exceptions.HTTPError) as e:
                if attempt < retries:
                    logger.warning("Ollama call failed (attempt %d/%d): %s",
                                   attempt + 1, retries + 1, e)
                    continue
                raise

    def _parse_response(self, cluster_uid: int, raw: str) -> LLMVerdict:
        fields: dict[str, str] = {}
        for line in raw.strip().split("\n"):
            for key in ("VERDICT", "CATEGORY", "SEVERITY", "REASONING"):
                if line.upper().startswith(key + ":"):
                    fields[key] = line.split(":", 1)[1].strip()
                    break

        missing = [key for key in ("VERDICT", "CATEGORY", "SEVERITY", "REASONING")
                  if key not in fields]
        if missing:
            logger.warning("LLM response missing field(s) %s for cluster %d; "
                           "raw response: %r", missing, cluster_uid, raw[:500])

        verdict_str = fields.get("VERDICT", "THREAT").upper()
        is_threat = "BENIGN" not in verdict_str

        return LLMVerdict(
            cluster_uid=cluster_uid,
            is_threat=is_threat,
            category=fields.get("CATEGORY", "unknown"),
            severity=fields.get("SEVERITY", "unknown"),
            reasoning=fields.get("REASONING", "parse failed"),
        )
