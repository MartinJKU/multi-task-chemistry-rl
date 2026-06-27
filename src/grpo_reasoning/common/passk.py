"""Pure (torch-free) pass@k estimators.

Kept separate from ``eval.py`` so the estimator math is importable and testable
without loading torch/transformers.
"""

from __future__ import annotations

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator from Chen et al. (2021).

    Estimates the probability that at least one of k completions drawn without
    replacement from n total completions is correct, given c of the n are
    correct. This is the standard low-variance estimator used to avoid the bias
    of simply checking the first k of n samples.

    Args:
        n: Total number of sampled completions for the item.
        c: Number of correct completions among the n.
        k: The k in pass@k; must satisfy 1 <= k <= n.

    Returns:
        Estimated pass@k in [0, 1].

    Raises:
        ValueError: If k is out of the supported range.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1 (got {k})")
    if k > n:
        raise ValueError(f"pass@k requires k <= n (got k={k}, n={n})")
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def bootstrap_passk_ci(
    correct_counts: list[int],
    n: int,
    k: int,
    iterations: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap a confidence interval for a corpus-level pass@k estimate.

    Args:
        correct_counts: Per-item correct-completion counts (each in [0, n]).
        n: Number of completions sampled per item.
        k: The k in pass@k.
        iterations: Number of bootstrap resamples over items.
        seed: RNG seed for reproducibility.
        alpha: Two-sided significance level (0.05 -> 95% CI).

    Returns:
        Tuple of (low, high) percentile bounds for the mean pass@k.
    """
    if not correct_counts:
        return 0.0, 0.0
    per_item = np.array([pass_at_k(n, c, k) for c in correct_counts])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_item), size=(iterations, len(per_item)))
    means = per_item[idx].mean(axis=1)
    low = float(np.percentile(means, 100 * alpha / 2))
    high = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return low, high
