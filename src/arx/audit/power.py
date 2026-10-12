"""Resolution bound: the smallest effect this sample could have detected.

A margin that is smaller than the minimum detectable effect (MDE) is not
evidence of "no difference" and it is certainly not evidence of a difference —
it is evidence that the experiment could not tell. Reporting it as a
distinction is the most common silent defect in benchmark write-ups.

No scipy: normal quantiles come from :class:`statistics.NormalDist`.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np

_N = NormalDist()


def z_for_confidence(alpha: float = 0.05) -> float:
    """Two-sided z at level ``alpha`` (the ``1 - alpha/2`` quantile)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return _N.inv_cdf(1.0 - alpha / 2.0)


def z_for_power(power: float = 0.8) -> float:
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    return _N.inv_cdf(power)


def mde_from_se(se: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable effect for a given standard error.

    ``delta = (z_{1-alpha/2} + z_{power}) * se``. The two z terms add because
    the test and the detection threshold sit on opposite sides of each other.
    """
    if se < 0:
        raise ValueError("se must be non-negative")
    if se == 0.0:
        return 0.0
    return (z_for_confidence(alpha) + z_for_power(power)) * float(se)


def power_at_effect(
    effect: float, se: float, alpha: float = 0.05
) -> float:
    """Probability of detecting ``effect`` at level ``alpha`` for this SE."""
    if se <= 0:
        raise ValueError("se must be positive")
    z_a = z_for_confidence(alpha)
    return float(1.0 - _N.cdf(z_a - abs(effect) / se))


def paired_diff_se(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    clusters: np.ndarray | None = None,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> float:
    """Cluster-bootstrap SE of the mean paired difference.

    Resampling happens at the cluster level (default: item level). Items
    sharing a gold generator are dependent, so an item-level bootstrap
    understates the SE — which in turn understates the MDE and lets
    unreadable leaderboards look readable.
    """
    a = np.asarray(scores_a, dtype=float).ravel()
    b = np.asarray(scores_b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError("scores_a and scores_b must have the same shape")
    if a.size == 0:
        raise ValueError("empty score arrays")
    if n_boot < 2:
        raise ValueError("n_boot must be >= 2")

    diff = a - b
    if clusters is None:
        clusters = np.arange(diff.size)
    clusters = np.asarray(clusters).ravel()
    if clusters.shape[0] != diff.shape[0]:
        raise ValueError("clusters must have one id per item")

    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    idx_by_cluster = [np.where(clusters == c)[0] for c in uniq]
    k = len(uniq)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        picked = rng.integers(0, k, size=k)
        idx = np.concatenate([idx_by_cluster[p] for p in picked])
        boots[i] = diff[idx].mean()
    return float(boots.std(ddof=1))


def cluster_bootstrap_mde(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    clusters: np.ndarray | None = None,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    n_boot: int = 2000,
    seed: int = 0,
) -> float:
    """MDE for the paired difference ``scores_a - scores_b``."""
    se = paired_diff_se(scores_a, scores_b, clusters, n_boot=n_boot, seed=seed)
    return mde_from_se(se, alpha=alpha, power=power)


def binomial_se(p: float, n: int) -> float:
    """SE of a proportion, clamped away from the degenerate p in {0, 1}.

    A leaderboard accuracy of exactly 0 or 1 is possible when a system gets
    every item right, and the plain formula gives SE=0 there, which would
    claim infinite resolution. The clamp to ``1/(2n)`` is the standard
    half-count correction for that case. Values outside ``[0, 1]`` are not
    corrected — they are impossible inputs and are refused.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"a proportion must be in [0, 1], got {p}")
    p = min(max(p, 1.0 / (2 * n)), 1.0 - 1.0 / (2 * n))
    return float(np.sqrt(p * (1.0 - p) / n))
