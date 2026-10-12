"""Tests for the scheduler and the budget.

This module decides what the agent does next, so its behaviour *is* the run's
behaviour. Two properties matter more than the exact scores:

* **determinism** -- the same context must yield the same action, because the
  ledger records the decision and a replay has to reproduce it;
* **termination** -- the loop must reach WRITE, and a per-action quota running
  out must disqualify that action rather than ending the run.
"""

from __future__ import annotations

import json
import math

import pytest

from arx.scheduler import (
    ACTION_ORDER,
    COST,
    RESOURCE,
    TERMINAL_VERDICTS,
    Action,
    Budget,
    SchedulerContext,
    choose,
    expected_information_gain,
    feasible,
    log_decision,
    replay_contexts,
    score,
    stalled,
)


def _ctx(**kwargs) -> SchedulerContext:
    base = dict(
        n_papers=0,
        n_hypotheses=0,
        n_accepted=0,
        n_rejected=0,
        last_verdict=None,
        judge_available=False,
        mde_ratio=None,
        margin=None,
        stalled_steps=0,
        stall_limit=6,
        budget=Budget(),
    )
    base.update(kwargs)
    return SchedulerContext(**base)


# ---------------------------------------------------------------------------
# the shape of a run
# ---------------------------------------------------------------------------

def test_cold_start_reads_the_literature_first():
    assert choose(_ctx()) is Action.LITERATURE


def test_a_first_sweep_of_a_topic_that_has_never_been_read_is_worth_reading():
    assert expected_information_gain(Action.LITERATURE, _ctx()) == 1.0


def test_papers_in_hand_means_write_a_hypothesis():
    assert choose(_ctx(n_papers=3)) is Action.HYPOTHESIZE


def test_a_live_hypothesis_beats_a_second_sweep_of_the_same_topic():
    """The second sweep re-runs the same query and re-finds the same papers.
    Testing the candidate already in hand is the only thing that can change
    what we know, so it must win."""
    ctx = _ctx(n_papers=3, n_hypotheses=3)
    assert choose(ctx) is Action.EXPERIMENT
    assert score(Action.EXPERIMENT, ctx) > score(Action.LITERATURE, ctx)


def test_a_finished_run_is_gated_before_anything_else():
    ctx = _ctx(n_papers=3, n_hypotheses=3, last_verdict="RAN")
    assert choose(ctx) is Action.GATE
    assert expected_information_gain(Action.GATE, ctx) == pytest.approx(0.90)


def test_a_refusal_sends_the_loop_back_to_hypotheses():
    for verdict in sorted(TERMINAL_VERDICTS):
        ctx = _ctx(n_papers=3, n_hypotheses=2, last_verdict=verdict)
        assert choose(ctx) is Action.HYPOTHESIZE, verdict


def test_a_broken_setup_is_not_re_run():
    """Another attempt with the same design cannot help when the design is what
    failed; the scheduler must price that in."""
    broken = _ctx(n_papers=3, n_hypotheses=1, last_verdict="UNDECIDABLE")
    fresh = _ctx(n_papers=3, n_hypotheses=1, last_verdict=None)
    assert score(Action.EXPERIMENT, broken) < score(Action.EXPERIMENT, fresh)


def test_a_wide_mde_makes_another_run_worth_more():
    vague = _ctx(n_papers=3, n_hypotheses=1, mde_ratio=4.0)
    tight = _ctx(n_papers=3, n_hypotheses=1, mde_ratio=1.0)
    assert score(Action.EXPERIMENT, vague) > score(Action.EXPERIMENT, tight)


def test_nothing_to_test_means_no_experiment(  ):
    assert expected_information_gain(Action.EXPERIMENT, _ctx(n_papers=5)) == 0.0


def test_gating_without_a_run_is_worthless():
    assert expected_information_gain(Action.GATE, _ctx(n_papers=5, last_verdict="GATED")) == 0.0


def test_choices_are_deterministic():
    ctx = _ctx(n_papers=3, n_hypotheses=2, last_verdict="RAN")
    assert len({choose(ctx) for _ in range(100)}) == 1


def test_write_is_never_chosen_while_work_remains():
    """It scores +inf and would win an argmax, which is how a cold start kills
    itself. It must only be chosen on the terminal conditions."""
    for ctx in (_ctx(), _ctx(n_papers=3), _ctx(n_papers=3, n_hypotheses=2),
                _ctx(n_papers=3, n_hypotheses=2, last_verdict="RAN")):
        assert choose(ctx) is not Action.WRITE
        assert score(Action.WRITE, ctx) == math.inf


def test_the_stall_counter_forces_the_run_to_end():
    ctx = _ctx(n_papers=99, n_hypotheses=0, stalled_steps=6, stall_limit=6)
    assert stalled(ctx)
    assert choose(ctx) is Action.WRITE


def test_stalling_below_the_limit_does_not_end_the_run():
    ctx = _ctx(n_papers=3, stalled_steps=5, stall_limit=6)
    assert not stalled(ctx)
    assert choose(ctx) is not Action.WRITE


def test_action_order_covers_every_action():
    assert set(ACTION_ORDER) == set(Action)
    assert set(COST) == set(Action)
    assert set(RESOURCE) == set(Action)


# ---------------------------------------------------------------------------
# budget: exhausted vs halting
# ---------------------------------------------------------------------------

def test_a_per_action_quota_running_out_does_not_halt_the_run():
    """The bug this pins: one literature sweep consuming the literature quota,
    read as 'the whole run is over'."""
    budget = Budget(max_literature_sweeps=1)
    budget.charge("literature_sweeps")
    assert "literature_sweeps" in budget.exhausted()
    assert budget.halting() == []


def test_the_wall_clock_running_out_does_halt_the_run():
    budget = Budget(max_wall_seconds=10.0)
    budget.charge("wall_seconds", 10.0)
    assert budget.halting() == ["wall_seconds"]


def test_an_exhausted_action_is_infeasible_rather_than_chosen():
    budget = Budget(max_experiment_runs=1)
    budget.charge("experiment_runs")
    ctx = _ctx(n_papers=3, n_hypotheses=1, budget=budget)
    assert not feasible(Action.EXPERIMENT, ctx)
    assert score(Action.EXPERIMENT, ctx) == -math.inf
    assert choose(ctx) is not Action.EXPERIMENT


def test_an_unbounded_budget_reports_infinite_remaining():
    budget = Budget(max_llm_calls=None, max_wall_seconds=None,
                    max_experiment_runs=None, max_literature_sweeps=None)
    budget.charge("llm_calls", 10_000)
    assert budget.remaining("llm_calls") == math.inf
    assert budget.is_unbounded()
    assert budget.halting() == []


def test_remaining_is_never_negative():
    budget = Budget(max_llm_calls=2)
    budget.charge("llm_calls", 5)
    assert budget.remaining("llm_calls") == 0.0


def test_an_unbounded_budget_never_exhausts():
    budget = Budget(max_llm_calls=None, max_wall_seconds=None,
                    max_experiment_runs=None, max_literature_sweeps=None)
    budget.charge("experiment_runs", 1000)
    assert budget.exhausted() == []


def test_charging_a_negative_amount_is_refused():
    with pytest.raises(ValueError):
        Budget().charge("llm_calls", -1)


def test_charging_an_unknown_resource_is_refused():
    with pytest.raises(KeyError):
        Budget().charge("unicorns")


def test_quotas_are_keyed_by_resource_not_by_action_name():
    """``Action.LITERATURE`` is ``"literature"`` but the counter is
    ``"literature_sweeps"``. Quotas follow the counter, so a caller that passes
    the action's value gets a KeyError from ``charge`` rather than a silently
    untracked quota."""
    assert RESOURCE[Action.LITERATURE] == "literature_sweeps"
    assert RESOURCE[Action.LITERATURE] != Action.LITERATURE.value
    budget = Budget()
    budget.set_quota(RESOURCE[Action.LITERATURE], 3)
    assert budget.quota_left(RESOURCE[Action.LITERATURE]) == 3.0
    # Charging the counter also consumes the matching quota, so the two cannot
    # drift apart when the caller forgets one of them.
    budget.charge(RESOURCE[Action.LITERATURE])
    assert budget.quota_left(RESOURCE[Action.LITERATURE]) == 2.0
    assert budget.literature_sweeps == 1
    # An explicit quota charge is separate and additive.
    budget.charge_quota(RESOURCE[Action.LITERATURE])
    assert budget.quota_left(RESOURCE[Action.LITERATURE]) == 1.0
    assert budget.literature_sweeps == 1


def test_an_unset_quota_is_unbounded():
    assert Budget().quota_left("anything") == math.inf


def test_the_budget_round_trips_through_its_dict_form():
    budget = Budget(max_llm_calls=7, max_wall_seconds=None, max_experiment_runs=4)
    budget.charge("llm_calls", 3)
    budget.charge("wall_seconds", 12.5)
    budget.set_quota(RESOURCE[Action.EXPERIMENT], 2)
    again = Budget.from_dict(budget.to_dict())
    assert again == budget
    assert again.max_llm_calls == 7
    assert again.max_wall_seconds is None
    assert again.llm_calls == 3
    assert again.quota_left(RESOURCE[Action.EXPERIMENT]) == 2.0


# ---------------------------------------------------------------------------
# context and decision records
# ---------------------------------------------------------------------------

def test_the_context_round_trips_through_its_dict_form():
    ctx = _ctx(n_papers=12, n_hypotheses=3, mde_ratio=2.5, margin=0.03,
               last_verdict="RAN", stalled_steps=2, stall_limit=6)
    ctx.budget.charge("llm_calls", 4)
    again = SchedulerContext.from_dict(ctx.to_dict())
    assert again == ctx
    assert again.budget == ctx.budget


def test_the_context_dict_is_strict_json_safe():
    ctx = _ctx(n_papers=1, budget=Budget(max_llm_calls=None, max_wall_seconds=None,
                                         max_experiment_runs=None,
                                         max_literature_sweeps=None))
    text = json.dumps(ctx.to_dict(), allow_nan=False)
    assert "Infinity" not in text and "NaN" not in text


def test_a_decision_log_is_json_safe_even_with_infinite_scores():
    """WRITE scores +inf and an infeasible action scores -inf. Emitting bare
    ``Infinity`` is invalid JSON for strict parsers, which is exactly the kind
    of silent corruption a ledger must not contain."""
    unbounded = Budget(max_llm_calls=None, max_wall_seconds=None,
                       max_experiment_runs=None, max_literature_sweeps=None)
    log = log_decision(1, _ctx(n_papers=3, n_hypotheses=1, budget=unbounded))
    text = json.dumps(log.to_dict(), allow_nan=False)
    assert "Infinity" not in text
    assert log.to_dict()["scores"]["write"] is None          # +inf -> null
    assert log.to_dict()["remaining"]["llm_calls"] is None   # unbounded -> null


def test_an_unbounded_budget_survives_being_recorded_and_restored():
    """``None`` means unbounded. Restoring it must not quietly reinstate the
    default cap, which would make a resumed run halt for a reason the original
    run was never given."""
    original = Budget(max_llm_calls=None, max_wall_seconds=None)
    restored = Budget.from_dict(original.to_dict())
    assert restored.max_llm_calls is None
    assert restored.max_wall_seconds is None
    assert restored.remaining("wall_seconds") == math.inf
    assert restored.halting() == []


def test_a_decision_log_records_the_action_and_its_context():
    log = log_decision(3, _ctx(n_papers=3))
    payload = log.to_dict()
    assert payload["action"] == Action.HYPOTHESIZE.value
    assert payload["step"] == 3
    assert set(payload["scores"]) == {a.value for a in Action}


def test_replay_contexts_pulls_only_scheduler_records():
    records = [
        {"kind": "scheduler", "payload": {"step": 1, "context": {"n_papers": 0}}},
        {"kind": "gate", "payload": {"verdict": "READABLE"}},
        {"kind": "scheduler", "payload": {"step": 2, "context": {"n_papers": 5}}},
    ]
    assert replay_contexts(records) == [(1, {"n_papers": 0}), (2, {"n_papers": 5})]


def test_replay_reproduces_the_recorded_actions():
    """The trust property: decisions must be a pure function of the recorded
    context, or a third party cannot check the run."""
    contexts = [
        _ctx(),
        _ctx(n_papers=3),
        _ctx(n_papers=3, n_hypotheses=2),
        _ctx(n_papers=3, n_hypotheses=2, last_verdict="RAN"),
        _ctx(n_papers=3, n_hypotheses=1, last_verdict="UNDECIDABLE"),
        _ctx(n_papers=3, stalled_steps=6, stall_limit=6),
    ]
    records = []
    for step, ctx in enumerate(contexts, start=1):
        log = log_decision(step, ctx)
        records.append({"kind": "scheduler", "payload": log.to_dict()})

    for step, payload in replay_contexts(records):
        rebuilt = SchedulerContext.from_dict(payload)
        assert choose(rebuilt).value == records[step - 1]["payload"]["action"]
