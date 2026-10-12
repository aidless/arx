"""Tests for pre-registration and multiplicity control.

Two distinct self-deception channels live here, and each gets a test that
would fail if the guard were removed:

* **k_max laundering** — run an experiment, miss, then claim you only ever
  intended one look. The guard is an ordering (the plan is written to the
  ledger before the run) plus a check that the run honoured it.
* **Family inflation** — N hypotheses each spending their own alpha gives a
  family-wise error rate near ``1 - (1 - alpha)^N``. The guard is an allowance
  that shrinks as the family grows.
"""

from __future__ import annotations

import pytest

from arx.audit.prereg import (
    MultiplicityLedger,
    Preregistration,
    check_preregistration,
    one_sided_p_from_z,
    plan_k_max,
    prereg_fingerprint,
    read_preregistration,
    sha256_text,
    two_sided_p_from_z,
)

# ---------------------------------------------------------------------------
# plan_k_max
# ---------------------------------------------------------------------------

def test_plan_is_deterministic_given_its_arguments():
    """Replay must reproduce the plan exactly, so it cannot depend on anything
    beyond its arguments."""
    a = plan_k_max(prior_confidence=0.5, effect_se_ratio=1.5)
    b = plan_k_max(prior_confidence=0.5, effect_se_ratio=1.5)
    assert a == b


def test_plan_grows_as_confidence_falls():
    confident = plan_k_max(prior_confidence=0.9)
    middling = plan_k_max(prior_confidence=0.5)
    unsure = plan_k_max(prior_confidence=0.1)
    assert confident <= middling <= unsure


def test_plan_extremes_are_bounded():
    assert plan_k_max(prior_confidence=1.0) == 1
    assert plan_k_max(prior_confidence=0.0) == 10


def test_plan_buys_looks_when_the_expected_effect_is_small():
    tight = plan_k_max(prior_confidence=0.5, effect_se_ratio=4.0)
    loose = plan_k_max(prior_confidence=0.5, effect_se_ratio=0.2)
    assert loose >= tight


def test_plan_rejects_impossible_arguments():
    with pytest.raises(ValueError):
        plan_k_max(prior_confidence=1.5)
    with pytest.raises(ValueError):
        plan_k_max(prior_confidence=-0.1)
    with pytest.raises(ValueError):
        plan_k_max(prior_confidence=0.5, alpha=0.0)
    with pytest.raises(ValueError):
        plan_k_max(prior_confidence=0.5, target_power=1.0)


# ---------------------------------------------------------------------------
# fingerprint and round-trip
# ---------------------------------------------------------------------------

def test_fingerprint_binds_the_script_hash():
    a = prereg_fingerprint(hypothesis_text="h", k_max=3, alpha=0.05,
                           script_sha256="aa" * 32)
    b = prereg_fingerprint(hypothesis_text="h", k_max=3, alpha=0.05,
                           script_sha256="bb" * 32)
    assert a != b


def test_fingerprint_binds_k_max_and_alpha():
    base = prereg_fingerprint(hypothesis_text="h", k_max=3, alpha=0.05)
    assert base != prereg_fingerprint(hypothesis_text="h", k_max=4, alpha=0.05)
    assert base != prereg_fingerprint(hypothesis_text="h", k_max=3, alpha=0.01)


def test_preregistration_round_trips_through_ledger_records():
    prereg = Preregistration.create(hypothesis_text="H1", k_max=4, alpha=0.05)
    record = {"kind": "pre_register", "payload": prereg.to_dict()}
    assert read_preregistration([record], "H1") == prereg


def test_the_latest_registration_wins():
    """A deliberate redesign supersedes an earlier plan."""
    first = Preregistration.create(hypothesis_text="H1", k_max=2, alpha=0.05)
    second = Preregistration.create(hypothesis_text="H1", k_max=7, alpha=0.05)
    records = [
        {"kind": "pre_register", "payload": first.to_dict()},
        {"kind": "pre_register", "payload": second.to_dict()},
    ]
    found = read_preregistration(records, "H1")
    assert found is not None and found.k_max == 7


def test_reading_a_missing_registration_returns_none():
    assert read_preregistration([], "H1") is None
    assert read_preregistration(
        [{"kind": "pre_register", "payload": Preregistration.create(
            hypothesis_text="other", k_max=2, alpha=0.05).to_dict()}],
        "H1",
    ) is None


# ---------------------------------------------------------------------------
# check_preregistration — the refusal
# ---------------------------------------------------------------------------

def test_missing_registration_is_refused():
    ok, reason = check_preregistration(None, blob_k_max=5, blob_script_sha256=None)
    assert not ok
    assert "no pre-registration" in reason


def test_k_max_laundering_is_caught():
    """The whole point: the plan said five looks, the run claims one."""
    prereg = Preregistration.create(hypothesis_text="H1", k_max=5, alpha=0.05)
    ok, reason = check_preregistration(prereg, blob_k_max=1, blob_script_sha256=None)
    assert not ok
    assert "k_max=1" in reason and "k_max=5" in reason


def test_honouring_the_plan_passes():
    prereg = Preregistration.create(hypothesis_text="H1", k_max=5, alpha=0.05)
    ok, reason = check_preregistration(prereg, blob_k_max=5, blob_script_sha256=None)
    assert ok, reason


def test_pasting_a_different_script_is_caught():
    prereg = Preregistration.create(hypothesis_text="H1", k_max=5, alpha=0.05,
                                    script_sha256="aa" * 32)
    ok, reason = check_preregistration(prereg, blob_k_max=5,
                                       blob_script_sha256="bb" * 32)
    assert not ok
    assert "hash" in reason


def test_pinned_hash_must_be_reported_by_the_run():
    prereg = Preregistration.create(hypothesis_text="H1", k_max=5, alpha=0.05,
                                    script_sha256="aa" * 32)
    ok, reason = check_preregistration(prereg, blob_k_max=5, blob_script_sha256=None)
    assert not ok
    assert "did not report" in reason


def test_a_tampered_record_is_internally_inconsistent():
    """Editing k_max in the ledger must not silently grant a new plan."""
    prereg = Preregistration.create(hypothesis_text="H1", k_max=5, alpha=0.05)
    payload = dict(prereg.to_dict())
    payload["k_max"] = 2
    found = read_preregistration([{"kind": "pre_register", "payload": payload}], "H1")
    ok, reason = check_preregistration(found, blob_k_max=2, blob_script_sha256=None)
    assert not ok
    assert "inconsistent" in reason


def test_sha256_text_is_stable():
    assert sha256_text("abc") == sha256_text("abc")
    assert len(sha256_text("abc")) == 64
    assert sha256_text("abc") != sha256_text("abd")


# ---------------------------------------------------------------------------
# p-value conversions
# ---------------------------------------------------------------------------

def test_two_sided_p_from_z_is_actually_two_sided():
    """Regression: the transcribed version was named one-sided and computed
    two-sided, which is a silent factor of two for every caller."""
    assert two_sided_p_from_z(1.9599639845) == pytest.approx(0.05, abs=1e-8)
    assert two_sided_p_from_z(0.0) == pytest.approx(1.0, abs=1e-12)
    assert two_sided_p_from_z(-1.9599639845) == pytest.approx(0.05, abs=1e-8)


def test_one_sided_p_is_half_the_two_sided_one():
    z = 1.5
    assert one_sided_p_from_z(z) == pytest.approx(two_sided_p_from_z(z) / 2.0)


# ---------------------------------------------------------------------------
# MultiplicityLedger
# ---------------------------------------------------------------------------

def test_the_first_test_gets_a_fraction_of_alpha_not_all_of_it():
    """Regression, and the reason the default is `spending` not `bonferroni`.

    The transcribed version handed the first hypothesis the whole of alpha.
    That rule has no spending-function interpretation: it is a heuristic whose
    guarantees depend on how many tests you later decide to run. The
    ``6/pi^2`` rule assigns the first test ``alpha * 6/pi^2`` — less than
    alpha, and the remainder provably covers the rest of the family for any
    future family size.
    """
    ledger = MultiplicityLedger(alpha=0.05)
    assert ledger.threshold_for_next() == pytest.approx(0.05 * 6 / 9.869604401089358)
    assert ledger.threshold_for_next() < ledger.alpha


def test_the_allowance_tightens_as_the_family_grows():
    ledger = MultiplicityLedger(alpha=0.05)
    allowances = [ledger.threshold_for_next()]
    for i in range(5):
        ledger.register(f"H{i}", 0.001)
        allowances.append(ledger.threshold_for_next())
    assert all(
        allowances[i] > allowances[i + 1] for i in range(len(allowances) - 1)
    )


def test_the_spending_family_never_exceeds_alpha():
    """The guarantee: the allowances sum to at most alpha, so no dependence
    assumption is needed for the family-wise bound."""
    ledger = MultiplicityLedger(alpha=0.05, mode="spending")
    for i in range(500):
        ledger.register(f"H{i}", 0.001)
    assert ledger.summary()["family_alpha_spent"] <= ledger.alpha


def test_bonferroni_mode_divides_evenly():
    ledger = MultiplicityLedger(alpha=0.05, mode="bonferroni")
    first = ledger.threshold_for_next()
    ledger.register("H0", 0.001)
    second = ledger.threshold_for_next()
    assert first == pytest.approx(0.05)
    assert second == pytest.approx(0.025)


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        MultiplicityLedger(mode="vibes")
    with pytest.raises(ValueError):
        MultiplicityLedger(alpha=0.0)


def test_register_reports_pass_and_fail():
    ledger = MultiplicityLedger(alpha=0.05)
    assert ledger.register("H0", 0.01)["passed"] is True
    assert ledger.register("H1", 0.9)["passed"] is False


def test_register_rejects_impossible_p_values():
    ledger = MultiplicityLedger()
    with pytest.raises(ValueError):
        ledger.register("H0", 1.5)
    with pytest.raises(ValueError):
        ledger.register("H0", -0.1)


def test_an_unscoreable_test_still_counts_against_the_family():
    """A test that produced no p-value was still tried.

    Counting only the scored ones would let an unscoreable test hide from the
    correction, which is the hole the family size exists to close.
    """
    ledger = MultiplicityLedger(alpha=0.05)
    ledger.register("H0", 0.001)
    before = ledger.threshold_for_next()
    record = ledger.register("H1", None)
    assert record["scored"] is False
    assert record["passed"] is False
    assert ledger.m == 2
    assert ledger.n_scored == 1
    assert ledger.n_unscored == 1
    assert ledger.threshold_for_next() < before


def test_duplicate_hypothesis_names_do_not_corrupt_the_thresholds():
    """Regression: the transcribed version used ``list.index(name)``, so a
    repeated name silently received the first occurrence's allowance."""
    ledger = MultiplicityLedger(alpha=0.05)
    ledger.register("same", 0.001)
    ledger.register("same", 0.9)
    ledger.register("other", 0.001)
    summary = ledger.summary()
    # the later registration for "same" failed and must be the one reported
    assert "same" in summary["rejected"]
    assert "other" in summary["accepted"]


def test_summary_records_the_allowance_each_test_actually_received():
    ledger = MultiplicityLedger(alpha=0.05)
    first = ledger.register("H0", 0.02)
    second = ledger.register("H1", 0.02)
    assert first["allowance"] > second["allowance"]
    assert first["index"] == 0 and second["index"] == 1


def test_summary_is_json_safe():
    import json

    ledger = MultiplicityLedger(alpha=0.05)
    ledger.register("H0", 0.01)
    ledger.register("H1", None)
    json.dumps(ledger.summary())


def test_reconstruct_from_ledger_matches_the_original():
    original = MultiplicityLedger(alpha=0.05)
    original.register("H0", 0.01)
    original.register("H1", None)
    original.register("H2", 0.04)
    records = [
        {"kind": "multiplicity", "payload": {"event": "register", **rec}}
        for rec in original.records
    ]
    rebuilt = MultiplicityLedger.from_ledger(records, alpha=0.05)
    assert rebuilt.m == original.m
    assert rebuilt.p_values == original.p_values
    assert rebuilt.summary()["next_allowance"] == pytest.approx(
        original.summary()["next_allowance"]
    )


def test_from_ledger_ignores_unrelated_records():
    rebuilt = MultiplicityLedger.from_ledger(
        [{"kind": "gate", "payload": {}}, {"kind": "scheduler", "payload": {}}]
    )
    assert rebuilt.m == 0


# ---------------------------------------------------------------------------
# bh mode
# ---------------------------------------------------------------------------

def test_bh_rejects_a_clear_signal_and_not_noise():
    ledger = MultiplicityLedger(alpha=0.05, mode="bh")
    ledger.register("clear", 0.0005)
    ledger.register("borderline", 0.30)
    ledger.register("noise", 0.80)
    rejected = ledger.bh_rejects()
    assert "clear" in rejected
    assert "noise" not in rejected
