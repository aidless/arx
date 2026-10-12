"""Tests for the noise-floor estimators.

The properties under test are the ones the auditor's verdicts rest on. If
``loo_floor`` silently returned something plausible for a two-generator
board, every downstream verdict would be quietly meaningless — so the
degenerate cases are tested as first-class behaviour, not as edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from arx.audit.floor import (
    DawidSkeneResult,
    column_majority,
    dawid_skene,
    external_gold_floor,
    floor_estimate,
    identifiability_diagnostics,
    loo_floor,
)


def _votes(m: int, n: int, *, agreement: float = 0.9, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 2, size=n)
    return np.array(
        [np.where(rng.random(n) > agreement, 1 - truth, truth) for _ in range(m)],
        dtype=int,
    )


# ---------------------------------------------------------------------------
# column_majority
# ---------------------------------------------------------------------------

def test_majority_picks_the_plurality_label():
    votes = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 0]])
    assert list(column_majority(votes)) == [1, 1, 0]


def test_majority_tie_is_deterministic_and_low():
    votes = np.array([[0, 1], [1, 0]])
    out = column_majority(votes)
    assert list(out) == [0, 0]
    assert list(column_majority(votes)) == list(out)


def test_majority_marks_abstention_only_columns():
    votes = np.array([[-1, 1, -1], [-1, 1, -1]])
    assert list(column_majority(votes)) == [-1, 1, -1]


# ---------------------------------------------------------------------------
# loo_floor
# ---------------------------------------------------------------------------

def test_loo_floor_returns_one_value_per_generator():
    votes = _votes(5, 300)
    floors = loo_floor(votes)
    assert floors.shape == (5,)
    assert np.all(floors > 0.5)


def test_loo_floor_is_self_consistency_for_two_generators():
    """With m=2, leave-one-out has nothing to compare against.

    The function still returns a number so the degeneracy is visible rather
    than hidden behind a NaN, but that number is pairwise agreement, and the
    caller must refuse to treat it as a reproduction floor.
    """
    votes = np.array([[1, 0, 1, 1, 0], [1, 0, 0, 1, 0]])
    floors = loo_floor(votes)
    pairwise = np.mean(votes[0] == votes[1])
    assert floors[0] == pytest.approx(pairwise)
    assert floors[1] == pytest.approx(pairwise)


def test_loo_floor_ignores_abstentions_per_generator():
    votes = np.array(
        [
            [1, 1, 0, 1],
            [1, -1, 0, 1],
            [1, -1, 0, -1],
        ]
    )
    floors = loo_floor(votes)
    assert not np.any(np.isnan(floors))


# ---------------------------------------------------------------------------
# external gold
# ---------------------------------------------------------------------------

def test_external_gold_floor_scores_against_the_gold():
    votes = np.array([[1, 1, 0, 0], [1, 0, 0, 0]])
    gold = np.array([1, 0, 0, 0])
    floors = external_gold_floor(votes, gold)
    # generator 0 disagrees with the gold on item 1 only
    assert floors[0] == pytest.approx(0.75)
    assert floors[1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Dawid-Skene
# ---------------------------------------------------------------------------

def test_dawid_skene_recovers_the_truth_on_clean_data():
    votes = _votes(5, 400, agreement=0.95, seed=1)
    result = dawid_skene(votes)
    assert isinstance(result, DawidSkeneResult)
    assert result.converged
    assert result.usable()
    assert not result.complement_flipped


def test_dawid_skene_flags_a_complement_fit():
    """Global complement symmetry lets EM converge on the opposite labelling.

    m=2 with perfectly anti-correlated generators is exactly that case: the
    likelihood is identical for the truth and its complement, so without an
    anchor the fit is a coin toss. The anchor plus the flip flag make it
    visible instead of silently inverting every downstream accuracy.
    """
    votes = np.array([[1, 1, 0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0, 1, 1]])
    result = dawid_skene(votes)
    majority = column_majority(votes)
    agreement = np.mean(result.truth == majority)
    assert agreement >= 0.5
    if result.complement_flipped:
        assert "complement" in result.to_text().lower()


def test_dawid_skene_is_deterministic():
    votes = _votes(4, 200, seed=7)
    a, b = dawid_skene(votes), dawid_skene(votes)
    assert np.array_equal(a.truth, b.truth)
    assert a.n_iter == b.n_iter


def test_dawid_skene_accuracies_track_agreement():
    votes = _votes(3, 400, agreement=0.7, seed=3)
    result = dawid_skene(votes)
    assert result.accuracies.shape == (3,)
    assert np.all(result.accuracies > 0.5)
    assert np.all(result.accuracies <= 1.0)


def test_dawid_skene_rejects_bad_input():
    with pytest.raises(ValueError):
        dawid_skene(np.array([1, 0, 1]))


# ---------------------------------------------------------------------------
# floor_estimate
# ---------------------------------------------------------------------------

def test_floor_estimate_returns_floor_and_fit():
    votes = _votes(5, 300, seed=11)
    floor, result = floor_estimate(votes)
    assert 0.0 <= floor <= 1.0
    assert isinstance(result, DawidSkeneResult)


def test_floor_estimate_raises_on_a_collapsed_fit():
    """A degenerate board must raise, never yield a confident-looking number."""
    votes = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    with pytest.raises(RuntimeError):
        floor_estimate(votes)


def test_floor_estimate_refuses_two_generators_structurally():
    """m=2 must be refused by construction, not by lucky divergence.

    Whether the EM happens to diverge at m=2 depends on the data; a
    degenerate board that converges would hand back a confident number.
    """
    with pytest.raises(ValueError, match="at least 3 generators"):
        floor_estimate(_votes(2, 200, seed=4))
    with pytest.raises(ValueError):
        floor_estimate(_votes(1, 50))


# ---------------------------------------------------------------------------
# identifiability
# ---------------------------------------------------------------------------

def test_identifiability_flags_too_few_generators():
    diags = identifiability_diagnostics(_votes(2, 100))
    assert diags["identifiable"] is False
    assert "floor_degenerate" in diags["undecidable"]


def test_identifiability_flags_constant_generators():
    votes = _votes(4, 100)
    votes[2] = 1
    diags = identifiability_diagnostics(votes)
    assert 2 in diags["constant_generators"]
    assert diags["identifiable"] is False


def test_identifiability_passes_a_healthy_board():
    diags = identifiability_diagnostics(_votes(5, 400, agreement=0.9, seed=5))
    assert diags["identifiable"] is True
    assert diags["undecidable"] == []
    assert 0.8 < diags["pairwise_agreement_mean"] < 0.95
