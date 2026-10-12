"""Tests for the power machinery.

Every function here feeds a verdict. If ``mde_from_se`` were off by a factor,
boards would be declared readable that are not, and no other test in the suite
would notice — so the arithmetic is checked against closed-form values rather
than against itself.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from arx.audit.power import (
    binomial_se,
    cluster_bootstrap_mde,
    mde_from_se,
    paired_diff_se,
    power_at_effect,
    z_for_confidence,
    z_for_power,
)


def test_normal_quantiles_match_the_tables():
    assert z_for_confidence(0.05) == pytest.approx(1.9599639845, abs=1e-9)
    assert z_for_confidence(0.01) == pytest.approx(2.5758293035, abs=1e-9)
    assert z_for_power(0.80) == pytest.approx(0.8416212336, abs=1e-9)
    assert z_for_power(0.90) == pytest.approx(1.2815515655, abs=1e-9)


def test_quantiles_reject_impossible_levels():
    with pytest.raises(ValueError):
        z_for_confidence(0.0)
    with pytest.raises(ValueError):
        z_for_confidence(1.0)
    with pytest.raises(ValueError):
        z_for_power(0.0)
    with pytest.raises(ValueError):
        z_for_power(1.0)


def test_mde_is_the_sum_of_the_two_quantiles_times_se():
    expected = (z_for_confidence(0.05) + z_for_power(0.8)) * 0.01
    assert mde_from_se(0.01) == pytest.approx(expected)
    assert mde_from_se(0.01) == pytest.approx(0.0280158522, abs=1e-9)


def test_mde_shrinks_as_the_sample_grows():
    small = mde_from_se(binomial_se(0.9, 100))
    large = mde_from_se(binomial_se(0.9, 10_000))
    assert large < small


def test_mde_of_a_zero_se_is_zero_not_an_error():
    """se=0 is the correct limit: zero uncertainty means zero MDE.

    Callers that can produce se=0 from degenerate data are responsible for
    not treating that as infinite resolution — see the cluster bootstrap,
    which returns 0.0 when every resample gives the same difference.
    """
    assert mde_from_se(0.0) == 0.0


def test_mde_rejects_a_negative_se():
    with pytest.raises(ValueError):
        mde_from_se(-0.01)


def test_binomial_se_is_the_textbook_formula():
    assert binomial_se(0.5, 100) == pytest.approx(0.05)
    assert binomial_se(0.9, 400) == pytest.approx(math.sqrt(0.9 * 0.1 / 400))


def test_binomial_se_rejects_bad_input():
    with pytest.raises(ValueError):
        binomial_se(-0.1, 100)
    with pytest.raises(ValueError):
        binomial_se(1.1, 100)
    with pytest.raises(ValueError):
        binomial_se(0.5, 0)


def test_binomial_se_clamps_the_perfect_score():
    """A system that gets every item right must not claim infinite resolution."""
    assert binomial_se(1.0, 400) > 0
    assert binomial_se(1.0, 400) == pytest.approx(binomial_se(1 - 1 / 800, 400))


def test_power_at_effect_inverts_the_mde():
    """An effect exactly at the MDE must have power at the target level."""
    se = 0.01
    mde = mde_from_se(se, alpha=0.05, power=0.8)
    assert power_at_effect(mde, se, alpha=0.05) == pytest.approx(0.8, abs=1e-6)


def test_power_increases_with_the_effect():
    assert power_at_effect(0.05, 0.01) > power_at_effect(0.02, 0.01)


def test_paired_diff_se_matches_the_analytic_value_when_clusters_are_items():
    rng = np.random.default_rng(0)
    n = 500
    a = (rng.random(n) < 0.9).astype(float)
    b = (rng.random(n) < 0.7).astype(float)
    diff = a - b
    se = paired_diff_se(a, b, n_boot=4000, seed=0)
    analytic = diff.std(ddof=1) / math.sqrt(n)
    assert se == pytest.approx(analytic, rel=0.08)


def test_cluster_bootstrap_widens_when_items_are_correlated_within_clusters():
    """Resampling at the cluster level must reflect the cluster structure.

    Items sharing a gold generator move together, so an item-level bootstrap
    treats them as independent evidence and understates the SE. Here the
    cluster effect dominates the item noise, so the two estimates must differ
    in the direction that matters: the cluster one is larger.
    """
    rng = np.random.default_rng(1)
    n_clusters, per_cluster = 20, 25
    cluster_effect = rng.normal(0.0, 0.20, size=n_clusters)
    items = np.repeat(np.arange(n_clusters), per_cluster)
    noise = rng.normal(0.0, 0.02, size=n_clusters * per_cluster)
    diff = cluster_effect[items] + noise

    a = 0.5 + diff / 2.0
    b = 0.5 - diff / 2.0

    coarse = paired_diff_se(a, b, items, n_boot=2000, seed=0)
    fine = paired_diff_se(a, b, np.arange(a.size), n_boot=2000, seed=0)

    # item-level sees 500 independent numbers; the truth is 20 clusters
    assert fine < coarse
    assert coarse / fine > 2.0

    coarse_mde = cluster_bootstrap_mde(a, b, items, n_boot=2000, seed=0)
    fine_mde = cluster_bootstrap_mde(a, b, np.arange(a.size), n_boot=2000, seed=0)
    assert coarse_mde > fine_mde


def test_cluster_bootstrap_agrees_with_item_level_when_there_is_no_structure():
    """With genuinely independent items the two must coincide."""
    rng = np.random.default_rng(11)
    n = 600
    diff = rng.normal(0.0, 0.3, size=n)
    a = 0.5 + diff / 2.0
    b = 0.5 - diff / 2.0
    item_level = paired_diff_se(a, b, None, n_boot=2000, seed=0)
    clustered = paired_diff_se(a, b, np.arange(n), n_boot=2000, seed=0)
    assert clustered == pytest.approx(item_level, rel=0.15)


def test_mde_is_deterministic_for_a_fixed_seed():
    rng = np.random.default_rng(2)
    a = (rng.random(300) < 0.9).astype(float)
    b = (rng.random(300) < 0.75).astype(float)
    first = cluster_bootstrap_mde(a, b, n_boot=500, seed=7)
    second = cluster_bootstrap_mde(a, b, n_boot=500, seed=7)
    assert first == second


def test_paired_helpers_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_diff_se(np.array([1.0, 0.0]), np.array([1.0]))


def test_cluster_bootstrap_rejects_an_empty_pair():
    with pytest.raises(ValueError):
        cluster_bootstrap_mde(np.array([]), np.array([]))
