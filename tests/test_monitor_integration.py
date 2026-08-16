"""End-to-end pipeline test with a fake embedder (deterministic, no model
download) so this runs in CI in under a second while still exercising the
real cluster_window -> registry -> emergence -> queue chain.
"""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guardlens.monitor import GuardLensMonitor


class FakeEmbedder:
    """Maps a fixed vocabulary of synthetic texts to hand-picked unit
    vectors: 'benign-N' -> one blob, 'attack-N' -> a distinct blob."""
    dim = 4

    def __init__(self):
        rng = np.random.default_rng(0)
        self._benign_center = self._norm(np.array([1.0, 0.0, 0.0, 0.0]))
        self._attack_center = self._norm(np.array([0.0, 1.0, 0.0, 0.0]))
        self._rng = rng

    @staticmethod
    def _norm(v):
        return v / np.linalg.norm(v)

    def encode(self, texts, batch_size=64):
        out = []
        for t in texts:
            if t.startswith("attack"):
                v = self._attack_center + self._rng.normal(scale=0.02, size=self.dim)
            else:
                v = self._benign_center + self._rng.normal(scale=0.02, size=self.dim)
            out.append(self._norm(v))
        return np.array(out)


def make_texts(prefix, n, cycle):
    return [f"{prefix}-c{cycle}-{i}" for i in range(n)]


def test_no_alerts_during_benign_only_cycles():
    monitor = GuardLensMonitor(embedder=FakeEmbedder(), window_size=3,
                               top_k=3, min_cluster_size=5, min_samples=3)
    for cycle in range(1, 4):
        result = monitor.process_cycle(cycle, make_texts("benign", 20, cycle))
    assert result.n_items == 20
    # A single recurring benign blob should never dominate the queue with
    # high emergence once it's been seen more than once (novelty decays).
    if result.queue:
        assert result.queue[0].cluster.novelty < 0.5


def test_attack_cluster_surfaces_after_onset():
    window_size = 3
    monitor = GuardLensMonitor(embedder=FakeEmbedder(), window_size=window_size,
                               top_k=3, min_cluster_size=5, min_samples=3)
    cycle_texts = {}
    for cycle in range(1, 4):
        cycle_texts[cycle] = make_texts("benign", 30, cycle)
        monitor.process_cycle(cycle, cycle_texts[cycle])

    cycle_texts[4] = make_texts("benign", 30, 4) + make_texts("attack", 8, 4)
    monitor.process_cycle(4, cycle_texts[4])
    cycle_texts[5] = make_texts("benign", 30, 5) + make_texts("attack", 15, 5)
    result5 = monitor.process_cycle(5, cycle_texts[5])

    # Reconstruct the same sliding window the monitor just used, in the
    # same order, so cluster.indices (positions into that flat window) can
    # be mapped back to attack/benign labels.
    window_flat = [t for c in range(5 - window_size + 1, 6) for t in cycle_texts[c]]
    is_attack = np.array([t.startswith("attack") for t in window_flat])

    assert len(result5.queue) > 0
    ranks = [e.rank for e in result5.queue]
    assert ranks == list(range(1, len(result5.queue) + 1))

    top = result5.queue[0].cluster
    top_labels = is_attack[top.indices]
    assert top_labels.mean() >= 0.8, (
        "top-ranked cluster by Emergence Score should be attack-dominated "
        "once the attack family has appeared for two consecutive cycles"
    )


def test_window_size_bounds_memory():
    monitor = GuardLensMonitor(embedder=FakeEmbedder(), window_size=2,
                               top_k=3, min_cluster_size=5, min_samples=3)
    for cycle in range(1, 6):
        monitor.process_cycle(cycle, make_texts("benign", 10, cycle))
    assert len(monitor._window_texts) == 2
