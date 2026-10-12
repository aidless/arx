"""Tests for the evaluation gate.

The gate exists to make a failed audit *unignorable*. So the load-bearing
tests are about the refusal, not about the happy path: a gate that lets a
MARGINAL result through unwrapped is decoration.
"""

from __future__ import annotations

import numpy as np
import pytest

from arx.audit.auditor import AuditConfig
from arx.audit.verdict import Verdict
from arx.gate import (
    TOLERATED,
    WRITABLE,
    EvaluationGate,
    GateRejection,
    gate_report,
)


def _votes(m: int, n: int, agreement: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 2, size=n)
    return np.array([
        np.where(rng.random(n) > agreement, 1 - truth, truth) for _ in range(m)
    ])


def _readable_kwargs(seed: int = 0) -> dict:
    """A setup that genuinely clears the noise floor."""
    return dict(
        votes=_votes(5, 500, 0.93, seed),
        leaderboard_scores={"strong": 0.97, "weak": 0.60},
        judge_ids={"j1", "j2", "j3", "j4", "j5"},
        contestant_ids={"strong", "weak"},
        delta_declared=0.02,
    )


# ---------------------------------------------------------------------------
# the happy path
# ---------------------------------------------------------------------------

def test_a_readable_board_passes_and_yields_the_report():
    gate = EvaluationGate()
    with gate.check(**_readable_kwargs()) as report:
        assert report.verdict is Verdict.READABLE
    assert report.margin is not None and report.floor_estimate is not None


def test_decision_returns_allowed_and_report_without_raising():
    gate = EvaluationGate()
    allowed, report = gate.decision(**_readable_kwargs())
    assert allowed is True
    assert report.verdict is Verdict.READABLE


# ---------------------------------------------------------------------------
# the refusal
# ---------------------------------------------------------------------------

def test_an_undecidable_board_is_refused():
    gate = EvaluationGate()
    with pytest.raises(GateRejection) as excinfo:
        with gate.check(
            votes=_votes(2, 200, 0.9),
            leaderboard_scores={"a": 0.9, "b": 0.8},
        ):
            pytest.fail("the block after a refused audit must not run")

    assert excinfo.value.report.verdict is Verdict.UNDECIDABLE
    assert "Fix the design" in str(excinfo.value)


def test_a_unreadable_board_is_refused():
    """The leader does not clear the measured floor at all.

    Strong generators put the floor near 0.92; a leaderboard whose best system
    scores 0.60 is below the noise floor, so its rank means nothing.
    """
    gate = EvaluationGate()
    with pytest.raises(GateRejection) as excinfo:
        with gate.check(
            votes=_votes(5, 400, 0.93),
            leaderboard_scores={"a": 0.60, "b": 0.55},
            judge_ids={f"j{i}" for i in range(5)},
            contestant_ids={"a", "b"},
        ):
            pytest.fail("the block after a refused audit must not run")
    assert excinfo.value.report.verdict is Verdict.UNREADABLE


def test_a_marginal_board_is_refused_by_default():
    """MARGINAL is not a finding, and a strict gate must not pretend it is."""
    gate = EvaluationGate(strict=True)
    allowed, report = gate.decision(
        votes=_votes(5, 500, 0.80),
        leaderboard_scores={"a": 0.78, "b": 0.77},
        delta_declared=0.10,
    )
    assert report.verdict is Verdict.MARGINAL
    assert allowed is False


def test_a_marginal_board_passes_only_when_explicitly_tolerated():
    """Opting in must be a deliberate, visible act with two separate switches."""
    strict_gate = EvaluationGate(strict=False)
    allowed, report = strict_gate.decision(
        votes=_votes(5, 500, 0.80),
        leaderboard_scores={"a": 0.78, "b": 0.77},
        delta_declared=0.10,
    )
    assert report.verdict is Verdict.MARGINAL
    assert allowed is True

    by_flag = EvaluationGate(strict=True, allow_marginal_with_note=True)
    assert by_flag.decision(
        votes=_votes(5, 500, 0.80),
        leaderboard_scores={"a": 0.78, "b": 0.77},
        delta_declared=0.10,
    )[0] is True


def test_the_verdict_sets_are_disjoint_and_cover_the_taxonomy():
    assert WRITABLE == frozenset({Verdict.READABLE})
    assert TOLERATED == frozenset({Verdict.MARGINAL})
    assert not (WRITABLE & TOLERATED)


# ---------------------------------------------------------------------------
# the label travels with the number
# ---------------------------------------------------------------------------

def test_a_marginal_label_carries_its_caveat():
    """A downstream summary that quotes only the label must still see the
    weakness; a bare 'MARGINAL' is easy to read as a soft pass."""
    gate = EvaluationGate(strict=False)
    _, marginal = gate.decision(
        votes=_votes(5, 500, 0.80),
        leaderboard_scores={"a": 0.78, "b": 0.77},
        delta_declared=0.10,
    )
    label = gate.labelled_verdict(marginal)
    assert label.startswith("MARGINAL")
    assert "not separable" in label


def test_a_readable_label_is_left_alone():
    gate = EvaluationGate()
    _, report = gate.decision(**_readable_kwargs())
    assert gate.labelled_verdict(report) == "READABLE"


# ---------------------------------------------------------------------------
# configuration plumbing
# ---------------------------------------------------------------------------

def test_the_gate_config_reaches_the_auditor():
    """The gate must not silently audit with different settings than the
    caller asked for."""
    gate = EvaluationGate(config=AuditConfig(delta_declared=0.50))
    allowed, report = gate.decision(
        votes=_votes(5, 500, 0.80),
        leaderboard_scores={"a": 0.78, "b": 0.77},
    )
    assert report.delta_declared == 0.50
    assert report.verdict is Verdict.MARGINAL


def test_per_call_kwargs_override_the_gate_config():
    gate = EvaluationGate(config=AuditConfig(delta_declared=0.50))
    _, report = gate.decision(
        votes=_votes(5, 500, 0.80),
        leaderboard_scores={"a": 0.78, "b": 0.77},
        delta_declared=0.001,
    )
    assert report.delta_declared == 0.001


def test_gate_report_helper_does_not_raise():
    allowed, report = gate_report(**_readable_kwargs())
    assert allowed is True and report.verdict is Verdict.READABLE

    allowed, report = gate_report(
        votes=_votes(2, 200, 0.9), leaderboard_scores={"a": 0.9, "b": 0.8})
    assert allowed is False and report.verdict is Verdict.UNDECIDABLE


def test_the_rejection_message_names_the_next_step():
    """A refusal the caller cannot act on will be worked around."""
    gate = EvaluationGate()
    _, report = gate.decision(
        votes=_votes(2, 200, 0.9), leaderboard_scores={"a": 0.9, "b": 0.8})
    message = GateRejection(report)._default_message()
    assert "design" in message

    _, marginal = gate.decision(
        votes=_votes(5, 500, 0.80),
        leaderboard_scores={"a": 0.78, "b": 0.77},
        delta_declared=0.10,
    )
    message = GateRejection(marginal)._default_message()
    assert "collect enough data" in message or "restate" in message
