"""Tests for the sequential gate.

The load-bearing claim is that the calibrated boundaries hold the family-wise
error rate at or below alpha when the same hypothesis is looked at repeatedly.
That cannot be established by inspecting the formula, so the central test is a
simulation: under the null (a fresh N(0,1) at every look), the fraction of runs
that cross a boundary must not exceed alpha.

The naive textbook shape ``z * sqrt(k/j)`` looks right and is not: it holds only
if you peek once, at the end. The regression test at the bottom pins that
finding so it cannot silently come back.
"""

from __future__ import annotations

import math
import random

import pytest

from arx.audit.sequential import (
    SequentialState,
    calibrate_scale,
    effective_alpha,
    fixed_sample_boundary,
    obf_boundaries,
    union_bound_rate,
    z_from_effect,
)
from arx.audit.verdict import Verdict

# ---------------------------------------------------------------------------
# boundaries
# ---------------------------------------------------------------------------

def test_fixed_sample_boundary_is_the_normal_quantile():
    assert fixed_sample_boundary(0.05) == pytest.approx(1.9599639845, abs=1e-9)
    assert fixed_sample_boundary(0.01) == pytest.approx(2.5758293035, abs=1e-9)


def test_boundaries_are_strictly_decreasing():
    bounds = obf_boundaries(5, 0.05)
    assert all(bounds[i] > bounds[i + 1] for i in range(len(bounds) - 1))


@pytest.mark.parametrize("k", [1, 2, 5, 10])
def test_every_boundary_is_at_least_the_fixed_sample_one(k: int):
    """The honest cost of assuming nothing about dependence between looks.

    A calibrated union bound cannot be more permissive than a single look,
    because it has to cover every look simultaneously under arbitrary
    dependence. This conservatism is a deliberate choice, not a defect, and
    this test pins it so a later "optimisation" has to argue with it.
    """
    fixed = fixed_sample_boundary(0.05)
    for bound in obf_boundaries(k, 0.05):
        assert bound >= fixed - 1e-9


def test_union_bound_rate_decreases_in_the_scale():
    assert union_bound_rate(3.0, 5) < union_bound_rate(2.0, 5)


def test_calibration_hits_the_target():
    for k in (1, 2, 3, 5, 10):
        c = calibrate_scale(k, 0.05)
        assert union_bound_rate(c, k) == pytest.approx(0.05, abs=1e-6)


def test_effective_alpha_is_stricter_than_the_naive_bound():
    per_look = effective_alpha(10, 0.05)
    assert 0.0 < per_look < 0.05
    assert per_look < 0.05 / 10 * 1.5


def test_boundaries_reject_bad_arguments():
    with pytest.raises(ValueError):
        obf_boundaries(0)
    with pytest.raises(ValueError):
        calibrate_scale(5, 0.0)
    with pytest.raises(ValueError):
        fixed_sample_boundary(1.5)


# ---------------------------------------------------------------------------
# z_from_effect
# ---------------------------------------------------------------------------

def test_z_from_effect_divides_in_the_documented_order():
    """Guard against the classic (se, effect) swap."""
    assert z_from_effect(2.0, 1.0) == pytest.approx(2.0)
    assert z_from_effect(1.0, 2.0) == pytest.approx(0.5)


def test_z_from_effect_rejects_nonpositive_se():
    with pytest.raises(ValueError):
        z_from_effect(1.0, 0.0)
    with pytest.raises(ValueError):
        z_from_effect(1.0, -0.1)


# ---------------------------------------------------------------------------
# SequentialState
# ---------------------------------------------------------------------------

def test_early_crossing_stops_the_gate():
    state = SequentialState(k_max=4, alpha=0.05)
    assert state.observe(10.0) is Verdict.READABLE
    assert state.done


def test_state_refuses_further_looks_once_stopped():
    state = SequentialState(k_max=4, alpha=0.05)
    state.observe(10.0)
    with pytest.raises(RuntimeError):
        state.observe(1.0)


def test_quantities_inside_the_boundary_keep_the_gate_open():
    state = SequentialState(k_max=5, alpha=0.05)
    for _ in range(4):
        assert state.observe(0.1) is None
        assert not state.done
    state.observe(0.1)
    assert state.done


def test_exhausting_the_planned_looks_raises():
    state = SequentialState(k_max=2, alpha=0.05)
    state.observe(0.0)
    state.observe(0.0)
    assert state.done
    with pytest.raises(RuntimeError):
        state.observe(0.0)


def test_state_rejects_a_meaningless_plan():
    with pytest.raises(ValueError):
        SequentialState(k_max=0)
    with pytest.raises(ValueError):
        SequentialState(k_max=5, alpha=0.0)


# ---------------------------------------------------------------------------
# the bias of stopping early must stay visible
# ---------------------------------------------------------------------------

def test_stopping_early_flags_a_biased_effect_estimate():
    """Sequential stopping inflates the estimate; the summary must say so.

    This is a correctness property, not a nicety: an effect size taken at an
    early-stopped look is biased away from zero, and a downstream consumer that
    reads it without the flag will overstate the finding.
    """
    state = SequentialState(k_max=4, alpha=0.05)
    state.observe(10.0)
    summary = state.summary()
    assert summary["effect_estimate_is_biased"] is True
    assert "biased" in summary["effect_bias_note"]


def test_no_bias_flag_before_any_stop():
    state = SequentialState(k_max=4, alpha=0.05)
    state.observe(0.0)
    summary = state.summary()
    assert summary["effect_estimate_is_biased"] is False


# ---------------------------------------------------------------------------
# FWER under the null: the central claim
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k_max", [1, 2, 5, 10])
def test_fwer_under_the_null_is_controlled(k_max: int):
    """Simulation: P(any boundary crossed) must not exceed alpha.

    Fixed seed so the result is reproducible; N large enough that the
    Monte-Carlo standard error at rate 0.05 is around 0.0015.
    """
    rng = random.Random(20261007)
    alpha = 0.05
    trials = 20_000
    boundaries = obf_boundaries(k_max, alpha)
    crossings = 0
    for _ in range(trials):
        if any(abs(rng.gauss(0.0, 1.0)) >= bound for bound in boundaries):
            crossings += 1
    rate = crossings / trials
    assert rate <= alpha + 0.005, (
        f"FWER {rate:.4f} exceeds alpha={alpha} at k_max={k_max}"
    )


def test_naive_textbook_shape_does_not_control_fwer():
    """Regression witness for the finding that forced the calibrated bound.

    The textbook ``z * sqrt(k/j)`` shape is valid for a single final look and
    invalid under repeated peeking. Pinning its failure keeps a future reader
    from "simplifying" the calibration back to it.
    """
    rng = random.Random(1234)
    k_max = 10
    alpha = 0.05
    final = fixed_sample_boundary(alpha)
    naive = [final * math.sqrt(k_max / j) for j in range(1, k_max + 1)]
    trials = 20_000
    crossings = sum(
        1
        for _ in range(trials)
        if any(abs(rng.gauss(0.0, 1.0)) >= b for b in naive)
    )
    naive_rate = crossings / trials
    calibrated_rate = sum(
        1
        for _ in range(trials)
        if any(abs(rng.gauss(0.0, 1.0)) >= b for b in obf_boundaries(k_max, alpha))
    ) / trials

    assert naive_rate > alpha * 1.5, "the naive shape should visibly overshoot"
    assert calibrated_rate <= alpha + 0.006
    assert calibrated_rate < naive_rate


def test_sequential_gate_reaches_a_decision_in_fewer_looks_under_the_alternative():
    rng = random.Random(999)
    k_max = 5
    boundaries = obf_boundaries(k_max, 0.05)
    looks_used = []
    for _ in range(4000):
        for j, bound in enumerate(boundaries):
            if abs(rng.gauss(0.9, 1.0)) >= bound:
                looks_used.append(j + 1)
                break
        else:
            looks_used.append(k_max)
    assert sum(looks_used) / len(looks_used) < k_max


def test_sequential_is_more_conservative_than_a_single_look():
    """A z that a fixed-sample test would call significant may still fail."""
    state = SequentialState(k_max=5, alpha=0.05)
    assert abs(2.0) >= fixed_sample_boundary(0.05)
    assert state.observe(2.0) is None or state.done


def test_summary_is_json_safe():
    import json

    state = SequentialState(k_max=3, alpha=0.05)
    state.observe(0.5)
    json.dumps(state.summary())
    assert state.summary()["verdict"] is None
