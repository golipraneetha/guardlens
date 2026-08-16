import random
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.random_audit import RandomAuditBaseline
from baselines.stratified_random import StratifiedRandomBaseline
from baselines.mmd_drift import MMDDriftBaseline, mmd_squared


def test_random_audit_detects_first_attack_cycle():
    baseline = RandomAuditBaseline(budget=100, seed=0)  # budget=all -> deterministic
    texts = [f"t{i}" for i in range(20)]
    labels = [0] * 20
    baseline.process_cycle(1, texts, labels)
    assert baseline.first_detection_cycle is None

    labels_with_attack = [0] * 19 + [1]
    baseline.process_cycle(2, texts, labels_with_attack)
    assert baseline.first_detection_cycle == 2


def test_random_audit_never_detects_beyond_first_hit():
    baseline = RandomAuditBaseline(budget=100, seed=0)
    texts = [f"t{i}" for i in range(10)]
    baseline.process_cycle(1, texts, [1] * 10)
    baseline.process_cycle(2, texts, [1] * 10)
    assert baseline.first_detection_cycle == 1


def test_stratified_random_prefers_high_score_items():
    # fraction=0.2 over 10 items -> pool_size=2: the attack (score 0.49) and
    # one benign item (all tied at 0.1, stable sort keeps original order).
    # Sampling budget=1 from that 2-item pool should hit the attack ~50% of
    # the time -- far higher than the 10% a uniform draw over all 10 items
    # would give, which is the actual property this baseline should have.
    texts = [f"t{i}" for i in range(10)]
    labels = [0] * 9 + [1]  # attack is the last item
    scores = [0.1] * 9 + [0.49]  # attack has highest score (closest to threshold)
    hit_count = 0
    n_trials = 200
    for trial_seed in range(n_trials):
        b = StratifiedRandomBaseline(budget=1, seed=trial_seed, fraction=0.2)
        if b.process_cycle(1, texts, labels, scores):
            hit_count += 1
    hit_rate = hit_count / n_trials
    assert 0.35 < hit_rate < 0.65  # ~50% expected, not the ~10% uniform draw would give


def test_mmd_near_zero_for_two_draws_from_same_distribution():
    # Independent draws from the *same* distribution, not literal duplicate
    # points -- the unbiased MMD^2 estimator is only unbiased in this sense;
    # comparing X against a literal copy of itself inflates the cross term
    # (every point matches itself with kernel value 1) and is not the right
    # test of "no distributional difference."
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    Y = rng.normal(size=(200, 4))
    stat = mmd_squared(X, Y)
    shifted = rng.normal(loc=3.0, size=(200, 4))
    stat_shifted = mmd_squared(X, shifted)
    assert abs(stat) < abs(stat_shifted)
    assert abs(stat) < 0.05


def test_mmd_baseline_flags_shifted_distribution():
    rng = np.random.default_rng(0)
    ref = rng.normal(loc=0.0, scale=1.0, size=(60, 4))
    baseline = MMDDriftBaseline(alpha=0.05, n_perm=100, seed=1)
    baseline.set_reference(ref)

    same = rng.normal(loc=0.0, scale=1.0, size=(40, 4))
    detected_same, p_same = baseline.process_cycle(1, same)

    shifted = rng.normal(loc=3.0, scale=1.0, size=(40, 4))
    detected_shift, p_shift = baseline.process_cycle(2, shifted)

    assert p_shift < p_same
    assert detected_shift is True
