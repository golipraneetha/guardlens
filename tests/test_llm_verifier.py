"""Tests for LLM cluster verification with mocked Ollama responses."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from guardlens.llm_verifier import ClusterVerifier, LLMVerdict
from guardlens.registry import TrackedCluster
from guardlens.queue import QueueEntry


def _make_entry(uid: int, indices: list[int], centroid: np.ndarray | None = None,
                size: int | None = None) -> QueueEntry:
    idx = np.array(indices)
    if centroid is None:
        centroid = np.array([1.0, 0.0, 0.0])
    return QueueEntry(
        rank=1,
        cluster=TrackedCluster(
            uid=uid, local_id=0, indices=idx, centroid=centroid,
            density=0.5, size=size or len(indices), age=1, prev_size=None,
        ),
    )


THREAT_RESPONSE = """\
VERDICT: THREAT
CATEGORY: jailbreak
SEVERITY: HIGH
REASONING: These texts attempt to bypass safety filters using role-play framing."""

BENIGN_RESPONSE = """\
VERDICT: BENIGN
CATEGORY: cooking
SEVERITY: NONE
REASONING: These are normal recipe-related queries with no safety concern."""


class TestParseResponse:
    def test_parse_threat(self):
        v = ClusterVerifier()
        verdict = v._parse_response(42, THREAT_RESPONSE)
        assert verdict.is_threat is True
        assert verdict.category == "jailbreak"
        assert verdict.severity == "HIGH"
        assert verdict.cluster_uid == 42

    def test_parse_benign(self):
        v = ClusterVerifier()
        verdict = v._parse_response(7, BENIGN_RESPONSE)
        assert verdict.is_threat is False
        assert verdict.category == "cooking"
        assert verdict.severity == "NONE"

    def test_malformed_defaults_to_threat(self):
        v = ClusterVerifier()
        verdict = v._parse_response(1, "I cannot understand the format")
        assert verdict.is_threat is True
        assert verdict.reasoning == "parse failed"

    def test_partial_format(self):
        v = ClusterVerifier()
        verdict = v._parse_response(1, "VERDICT: BENIGN\ngarbage line")
        assert verdict.is_threat is False
        assert verdict.category == "unknown"


class TestSampleTexts:
    def test_with_embeddings_near_and_far(self):
        centroid = np.array([1.0, 0.0, 0.0])
        embeddings = np.array([
            [1.0, 0.0, 0.0],   # idx 0: identical to centroid (near)
            [0.9, 0.1, 0.0],   # idx 1: near
            [0.0, 1.0, 0.0],   # idx 2: far
            [0.0, 0.0, 1.0],   # idx 3: far
            [0.95, 0.05, 0.0], # idx 4: near
            [0.1, 0.9, 0.0],   # idx 5: far
        ])
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        entry = _make_entry(1, [0, 1, 2, 3, 4, 5], centroid=centroid)
        v = ClusterVerifier(n_samples=4)
        texts = ["t0", "t1", "t2", "t3", "t4", "t5"]
        sampled = v._sample_texts(entry, texts, embeddings)
        assert len(sampled) == 4
        # Should include some far texts (t2, t3) and some near texts (t0, t4)
        assert any(t in sampled for t in ["t2", "t3", "t5"])
        assert any(t in sampled for t in ["t0", "t1", "t4"])

    def test_without_embeddings_fallback(self):
        entry = _make_entry(1, [0, 1, 2, 3, 4, 5])
        v = ClusterVerifier(n_samples=3)
        texts = ["t0", "t1", "t2", "t3", "t4", "t5"]
        sampled = v._sample_texts(entry, texts, None)
        assert len(sampled) == 3

    def test_fewer_items_than_samples(self):
        entry = _make_entry(1, [0, 1])
        v = ClusterVerifier(n_samples=6)
        texts = ["t0", "t1"]
        sampled = v._sample_texts(entry, texts, None)
        assert len(sampled) == 2


class TestCacheKey:
    def test_order_independent(self):
        v = ClusterVerifier()
        k1 = v._cache_key(["alpha", "beta", "gamma"])
        k2 = v._cache_key(["gamma", "alpha", "beta"])
        assert k1 == k2

    def test_different_model_different_key(self):
        v1 = ClusterVerifier(model="llama3.1")
        v2 = ClusterVerifier(model="qwen3:1.7b")
        k1 = v1._cache_key(["alpha", "beta"])
        k2 = v2._cache_key(["alpha", "beta"])
        assert k1 != k2


class TestVerifyQueue:
    @patch.object(ClusterVerifier, "_call_ollama")
    def test_filters_benign_keeps_threat(self, mock_ollama):
        mock_ollama.side_effect = [THREAT_RESPONSE, BENIGN_RESPONSE]
        v = ClusterVerifier()
        texts = ["attack1", "attack2", "attack3", "benign1", "benign2", "benign3"]
        q = [
            _make_entry(1, [0, 1, 2]),
            _make_entry(2, [3, 4, 5]),
        ]
        filtered, verdicts = v.verify_queue(q, texts)
        assert len(filtered) == 1
        assert filtered[0].cluster.uid == 1
        assert filtered[0].rank == 1
        assert len(verdicts) == 2
        assert verdicts[0].is_threat is True
        assert verdicts[1].is_threat is False

    @patch.object(ClusterVerifier, "_call_ollama")
    def test_cache_prevents_duplicate_calls(self, mock_ollama):
        mock_ollama.return_value = THREAT_RESPONSE
        v = ClusterVerifier()
        texts = ["a", "b", "c"]
        q = [_make_entry(1, [0, 1, 2])]

        v.verify_queue(q, texts)
        assert mock_ollama.call_count == 1

        q2 = [_make_entry(99, [0, 1, 2])]
        v.verify_queue(q2, texts)
        assert mock_ollama.call_count == 1  # cached

    @patch.object(ClusterVerifier, "_call_ollama")
    def test_cache_persistence(self, mock_ollama, tmp_path):
        cache_file = tmp_path / "test_cache.json"
        mock_ollama.return_value = THREAT_RESPONSE

        v1 = ClusterVerifier(cache_path=cache_file)
        v1.verify_queue([_make_entry(1, [0, 1])], ["a", "b"])
        v1.save_cache()
        assert cache_file.exists()

        v2 = ClusterVerifier(cache_path=cache_file)
        v2.verify_queue([_make_entry(2, [0, 1])], ["a", "b"])
        assert mock_ollama.call_count == 1  # second verifier used cache


class TestBuildPrompt:
    def test_truncates_long_texts(self):
        v = ClusterVerifier()
        long_text = "x" * 500
        prompt = v._build_prompt([long_text], cluster_size=10)
        assert "x" * 200 in prompt
        assert "x" * 201 not in prompt

    def test_includes_cluster_size(self):
        v = ClusterVerifier()
        prompt = v._build_prompt(["hello"], cluster_size=42)
        assert "42" in prompt
