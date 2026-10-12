"""Tests for the auditor, including the mutation battery.

The auditor's negative results are only worth something if they are
*non-vacuous*: it must reject genuinely broken boards and accept genuinely
sound ones. A checker that rejects everything looks identical to a rigorous
one under a test suite that only feeds it broken input. So every mutation here
is paired with a control that must still pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from arx.audit.auditor import AuditConfig, audit_item_sets, audit_leaderboard
from arx.audit.verdict import UndecidableReason, Verdict


def _votes(m: int, n: int, *, agreement: float = 0.93, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 2, size=n)
    return np.array(
        [np.where(rng.random(n) > agreement, 1 - truth, truth) for _ in range(m)],
        dtype=int,
    )


def _sound_kwargs(n: int = 400, seed: int = 0) -> dict:
    return dict(
        votes=_votes(5, n, agreement=0.93, seed=seed),
        leaderboard_scores={"padded": 0.965, "plain": 0.70},
        judge_ids={"j1", "j2", "j3", "j4", "j5"},
        contestant_ids={"padded", "plain"},
        delta_declared=0.02,
    )


# ---------------------------------------------------------------------------
# the control first: a sound board must pass
# ---------------------------------------------------------------------------

def test_readable_board_is_accepted():
    report = audit_leaderboard(**_sound_kwargs())
    assert report.verdict is Verdict.READABLE, report.to_text()
    assert report.margin is not None and report.margin > 0
    assert report.floor_estimate is not None
    assert report.threshold == max(0.02, report.mde or 0.0)


def test_below_floor_is_unreadable_not_undecidable():
    """A board where the leader does not beat the floor is a *finding*.

    UNREADABLE says "we measured it and the ranking does not hold".
    UNDECIDABLE says "we cannot measure it". Conflating them hides the result.
    """
    report = audit_leaderboard(
        votes=_votes(5, 400, agreement=0.93, seed=2),
        leaderboard_scores={"best": 0.80, "worse": 0.75},
        judge_ids={"j1", "j2", "j3", "j4", "j5"},
        contestant_ids={"best", "worse"},
    )
    assert report.verdict is Verdict.UNREADABLE


def test_small_margin_is_marginal():
    report = audit_leaderboard(
        votes=_votes(5, 400, agreement=0.75, seed=3),
        leaderboard_scores={"a": 0.80, "b": 0.79},
        delta_declared=0.25,  # a deliberately wide declared bound
    )
    assert report.verdict is Verdict.MARGINAL
    assert "resolution bound" in report.reasons[0]


# ---------------------------------------------------------------------------
# Tenet 1: actor-disjointness
# ---------------------------------------------------------------------------

def test_actor_overlap_is_undecidable():
    kwargs = _sound_kwargs()
    kwargs["judge_ids"] = {"padded", "j2", "j3", "j4", "j5"}
    report = audit_leaderboard(**kwargs)
    assert report.verdict is Verdict.UNDECIDABLE
    assert UndecidableReason.ACTORS_OVERLAP.value in report.reasons
    assert report.extra["overlapping_actors"] == ["padded"]


def test_disjoint_actors_pass_the_check():
    kwargs = _sound_kwargs()
    kwargs["judge_ids"] = {"judge-a", "judge-b", "judge-c", "judge-d", "judge-e"}
    assert audit_leaderboard(**kwargs).verdict is Verdict.READABLE


# ---------------------------------------------------------------------------
# Tenet 2: common denominator
# ---------------------------------------------------------------------------

def test_denominator_mismatch_is_undecidable():
    kwargs = _sound_kwargs()
    kwargs["leaderboard_item_ids"] = set(range(300))
    kwargs["floor_item_ids"] = set(range(400))
    report = audit_leaderboard(**kwargs)
    assert report.verdict is Verdict.UNDECIDABLE
    assert UndecidableReason.DOMAIN_MISMATCH.value in report.reasons
    assert report.extra["items_only_in_leaderboard"] == 0
    assert report.extra["items_only_in_floor"] == 100


def test_matching_denominators_pass():
    kwargs = _sound_kwargs()
    ids = set(range(400))
    kwargs["leaderboard_item_ids"] = ids
    kwargs["floor_item_ids"] = set(ids)
    assert audit_leaderboard(**kwargs).verdict is Verdict.READABLE


# ---------------------------------------------------------------------------
# Tenet 3: multi-generator sufficiency
# ---------------------------------------------------------------------------

def test_single_generator_is_undecidable():
    report = audit_leaderboard(
        votes=_votes(1, 200),
        leaderboard_scores={"a": 0.9, "b": 0.8},
    )
    assert report.verdict is Verdict.UNDECIDABLE
    assert UndecidableReason.SINGLE_GENERATOR.value in report.reasons


def test_two_generators_are_undecidable():
    """The structural result: two judges cannot define a reproduction floor."""
    report = audit_leaderboard(
        votes=_votes(2, 400),
        leaderboard_scores={"a": 0.9, "b": 0.8},
    )
    assert report.verdict is Verdict.UNDECIDABLE
    assert UndecidableReason.FLOOR_DEGENERATE.value in report.reasons
    assert "pairwise agreement" in report.extra["note"]


# ---------------------------------------------------------------------------
# dawid-skene floor path
# ---------------------------------------------------------------------------

def test_dawid_skene_floor_path_produces_a_verdict():
    kwargs = _sound_kwargs()
    report = audit_leaderboard(**kwargs, floor_method="dawid_skene")
    assert report.verdict is Verdict.READABLE
    assert report.floor_method == "dawid_skene"
    assert report.dk_text
    assert "complement_flipped" in report.extra


def test_unknown_floor_method_raises():
    with pytest.raises(ValueError, match="unknown floor_method"):
        audit_leaderboard(**_sound_kwargs(), floor_method="magic")


# ---------------------------------------------------------------------------
# power / MDE
# ---------------------------------------------------------------------------

def test_cluster_bootstrap_mde_is_used_when_paired_scores_given():
    rng = np.random.default_rng(0)
    n = 400
    a = rng.integers(0, 2, size=n).astype(float)
    b = np.where(rng.random(n) > 0.3, a, 1 - a)
    report = audit_leaderboard(
        votes=_votes(5, n, seed=5),
        leaderboard_scores={"a": float(a.mean()), "b": float(b.mean())},
        per_item_scores={"a": a, "b": b},
    )
    assert report.extra["mde_method"] == "cluster_bootstrap"


def test_binomial_mde_is_flagged_as_an_approximation():
    report = audit_leaderboard(**_sound_kwargs())
    assert report.extra["mde_method"] == "binomial_approx"
    assert report.mde is not None and report.mde > 0


def test_audit_config_defaults_are_the_documented_ones():
    cfg = AuditConfig()
    assert (cfg.delta_declared, cfg.alpha, cfg.power) == (0.02, 0.05, 0.8)
    assert cfg.floor_method == "loo"


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------

def test_bad_vote_shape_raises():
    with pytest.raises(ValueError, match="2-D"):
        audit_leaderboard(np.array([1, 0, 1]), {"a": 0.9})


def test_empty_leaderboard_raises():
    with pytest.raises(ValueError, match="empty"):
        audit_leaderboard(_votes(3, 50), {})


# ---------------------------------------------------------------------------
# both gold conventions
# ---------------------------------------------------------------------------

def test_audit_item_sets_compares_conventions():
    votes = _votes(5, 200, seed=9)
    scores = {"a": 0.95, "b": 0.6}
    majority = audit_item_sets(votes, scores, gold_mode="majority")
    loo = audit_item_sets(votes, scores, gold_mode="loo")
    assert majority.verdict is not None and loo.verdict is not None


def test_audit_item_sets_rejects_unknown_mode():
    with pytest.raises(ValueError, match="gold_mode"):
        audit_item_sets(_votes(3, 50), {"a": 0.9}, gold_mode="wishful")


# ---------------------------------------------------------------------------
# MUTATION BATTERY
# ---------------------------------------------------------------------------

MUTANTS = {
    "one_generator": lambda kw: kw | {"votes": _votes(1, 400)},
    "two_generators": lambda kw: kw | {"votes": _votes(2, 400)},
    "judge_is_contestant": lambda kw: kw | {"judge_ids": {"padded", "j2", "j3", "j4", "j5"}},
    "contestant_is_judge": lambda kw: kw | {"contestant_ids": {"j1", "plain"}},
    "floor_on_a_subset": lambda kw: kw | {
        "leaderboard_item_ids": set(range(250)),
        "floor_item_ids": set(range(400)),
    },
    "leader_below_floor": lambda kw: kw | {"leaderboard_scores": {"padded": 0.50, "plain": 0.40}},
    "noise_generators": lambda kw: kw | {"votes": _votes(5, 400, agreement=0.50)},
}


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_mutation_battery_catches_every_mutant(name: str):
    """Each mutation of an otherwise-sound board must change the verdict.

    A mutant that slips through means the auditor's refusals are vacuous.
    """
    baseline = _sound_kwargs()
    assert audit_leaderboard(**baseline).verdict is Verdict.READABLE, (
        "control failed: the unmutated board must be READABLE, otherwise this "
        "battery proves nothing"
    )
    mutated = MUTANTS[name](dict(baseline))
    report = audit_leaderboard(**mutated)
    assert report.verdict is not Verdict.READABLE, (
        f"mutant {name!r} was not caught — the auditor's negative results are "
        f"vacuous.\n{report.to_text()}"
    )


def test_battery_control_still_passes_after_every_mutant():
    """Guard against a checker that trivially rejects everything."""
    for _ in range(3):
        assert audit_leaderboard(**_sound_kwargs()).verdict is Verdict.READABLE


def test_noise_gold_is_refused_despite_a_huge_margin():
    """The loudest possible false positive, found by the battery above.

    Uninformative generators put the floor at chance. A board reporting 0.965
    against that floor shows a margin of +0.45 — and is meaningless, because
    the gold it was scored against carries no signal. Regression test for the
    gold-signal check.
    """
    report = audit_leaderboard(
        votes=_votes(5, 400, agreement=0.50, seed=42),
        leaderboard_scores={"padded": 0.965, "plain": 0.70},
    )
    assert report.verdict is Verdict.UNDECIDABLE, report.to_text()
    assert UndecidableReason.FLOOR_DEGENERATE.value in report.reasons
    assert "chance level" in report.extra["floor_error"]
    assert report.extra["chance_level"] == pytest.approx(0.5)
    assert report.extra["floor_vs_chance"]["above_chance"] is False


def test_noise_gold_is_refused_even_with_self_consistent_scores():
    """The refusal is about the gold, not about an implausible score.

    Same noise generators, but a leaderboard that agrees with the noise-level
    well (0.52). The margin looks fine and the gold is still worthless.
    """
    report = audit_leaderboard(
        votes=_votes(5, 400, agreement=0.50, seed=42),
        leaderboard_scores={"padded": 0.52, "plain": 0.50},
    )
    assert report.verdict is Verdict.UNDECIDABLE, report.to_text()


def test_strong_generators_clear_the_chance_check():
    """The check must not become a filter for annotators I approve of."""
    report = audit_leaderboard(
        votes=_votes(5, 400, agreement=0.70, seed=43),
        leaderboard_scores={"padded": 0.95, "plain": 0.60},
    )
    assert report.verdict is Verdict.READABLE, report.to_text()
    assert report.extra["floor_vs_chance"]["above_chance"] is True


def test_weak_generators_are_refused_when_there_are_too_few_items():
    """Floor resolution comes from item count, not generator count.

    Five 60%-accurate generators over 400 items put the floor around 0.55,
    which is not detectably above chance at that n. The refusal reports how
    many items would have been needed, so the fix is visible rather than a
    mystery.
    """
    report = audit_leaderboard(
        votes=_votes(5, 400, agreement=0.60, seed=43),
        leaderboard_scores={"padded": 0.95, "plain": 0.60},
    )
    assert report.verdict is Verdict.UNDECIDABLE, report.to_text()
    test = report.extra["floor_vs_chance"]
    assert test["above_chance"] is False
    assert test["n_items_needed"] > 400


def test_the_same_weak_generators_pass_with_enough_items():
    """Adding items is the fix; adding generators is not."""
    report = audit_leaderboard(
        votes=_votes(5, 2500, agreement=0.60, seed=43),
        leaderboard_scores={"padded": 0.95, "plain": 0.60},
    )
    assert report.verdict is Verdict.READABLE, report.to_text()
    assert report.extra["floor_vs_chance"]["above_chance"] is True


def test_mutations_produce_a_variety_of_verdicts():
    """The battery must exercise more than one refusal reason."""
    seen = set()
    for _name, mutate in MUTANTS.items():
        report = audit_leaderboard(**mutate(_sound_kwargs()))
        seen.add(report.verdict)
        if report.reasons:
            seen.add(tuple(report.reasons[:1]))
    assert Verdict.UNDECIDABLE in seen
    assert Verdict.UNREADABLE in seen
