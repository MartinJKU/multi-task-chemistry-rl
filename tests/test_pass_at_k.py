from __future__ import annotations

import math

import pytest

from grpo_reasoning.common.passk import bootstrap_passk_ci, pass_at_k
from grpo_reasoning.multitask.passk_report import find_crossover


def test_pass_at_1_is_empirical_accuracy():
    """pass@1 equals the fraction of correct completions."""
    assert pass_at_k(n=10, c=3, k=1) == pytest.approx(0.3)
    assert pass_at_k(n=8, c=0, k=1) == 0.0
    assert pass_at_k(n=8, c=8, k=1) == 1.0


def test_pass_at_k_certain_when_enough_correct():
    """If fewer than k completions are wrong, at least one of k is always correct."""
    assert pass_at_k(n=10, c=9, k=2) == 1.0
    assert pass_at_k(n=4, c=4, k=4) == 1.0


def test_pass_at_k_zero_when_none_correct():
    """With no correct completion, pass@k is zero for any k."""
    assert pass_at_k(n=16, c=0, k=8) == 0.0


def test_pass_at_k_matches_closed_form():
    """Estimator matches 1 - C(n-c, k) / C(n, k)."""
    n, c, k = 10, 2, 3
    expected = 1.0 - math.comb(n - c, k) / math.comb(n, k)
    assert pass_at_k(n, c, k) == pytest.approx(expected)


def test_pass_at_k_monotonic_in_k():
    """pass@k is non-decreasing in k for fixed (n, c)."""
    n, c = 32, 4
    values = [pass_at_k(n, c, k) for k in (1, 2, 4, 8, 16)]
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))


def test_pass_at_k_validates_k():
    """k must be within [1, n]."""
    with pytest.raises(ValueError):
        pass_at_k(n=4, c=1, k=0)
    with pytest.raises(ValueError):
        pass_at_k(n=4, c=1, k=5)


def test_bootstrap_ci_brackets_point_estimate():
    """The bootstrap CI should contain the corpus mean pass@k."""
    counts = [0, 1, 2, 4, 8, 16, 16]
    n, k = 16, 4
    point = sum(pass_at_k(n, c, k) for c in counts) / len(counts)
    low, high = bootstrap_passk_ci(counts, n, k, iterations=1000, seed=0)
    assert low <= point <= high
    assert 0.0 <= low <= high <= 1.0


def _summary(label: str, pass_by_k: dict[int, float]) -> dict:
    """Build a minimal pass@k summary dict for crossover tests."""
    return {
        "model_label": label,
        "task_type": "single_count",
        "properties": ["ring_count"],
        "pass_at_k": {str(k): {"mean": v} for k, v in pass_by_k.items()},
    }


def test_find_crossover_detects_base_catching_up():
    """Elicitation: base starts lower but overtakes the RL model at large k."""
    base = _summary("base", {1: 0.2, 4: 0.5, 16: 0.8})
    grpo = _summary("grpo", {1: 0.5, 4: 0.6, 16: 0.7})
    assert find_crossover(base, grpo) == 16


def test_find_crossover_none_when_model_stays_ahead():
    """Expansion: RL model dominates at every k, so there is no crossover."""
    base = _summary("base", {1: 0.2, 4: 0.4, 16: 0.6})
    grpo = _summary("grpo", {1: 0.5, 4: 0.7, 16: 0.9})
    assert find_crossover(base, grpo) is None
