"""Budget-aware action selection.

A fixed state machine (PLAN → LITERATURE → HYPOTHESIZE → EXPERIMENT → GATE →
WRITE) is correct but not *autonomous*: it cannot decide that one more
experiment is worth more than another literature sweep, because it has no cost
model. This module scores candidate actions by expected information gain per
unit cost:

    utility(action) = E[IG(action)] / cost(action)

Design constraints, in order of importance:

1. **Deterministic, and reproducible from the ledger.** Ties break by a fixed
   order. Two runs on identical inputs must produce identical ledgers. A sharp
   estimator that depends on an LLM would break replay, so the estimator is
   deliberately coarse — it only has to *rank* actions.

2. **Feasibility gates before scores.** An action whose resource is exhausted
   scores ``-inf`` regardless of how informative it would be.

3. **WRITE is terminal, not high-scoring.** Giving WRITE ``+inf`` and letting it
   win argmax means a cold start immediately terminates — a suicide bug. WRITE
   is selected only when the stall counter fires or nothing else is feasible.
   The stall counter is what guarantees the loop converges.

4. **A broken setup must not be re-run.** If the last verdict was UNDECIDABLE,
   another run of the same entrypoint burns budget without changing the answer,
   so EXPERIMENT's gain collapses. The fix is to change the setup, not rerun it.

The constants below are ratios. Only their ordering is claimed to mean anything;
they are not calibrated to wall-clock or dollars. ``docs/architecture.md``
records that, and ``arx.scheduler.calibrate`` is where a fitted replacement
would go — with the caveat that a fitted scheduler becomes its own Goodhart
target, which ``arx.audit.goodhart`` monitors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

from .context import SchedulerContext


class Action(StrEnum):
    LITERATURE = "literature"
    HYPOTHESIZE = "hypothesize"
    EXPERIMENT = "experiment"
    GATE = "gate"
    WRITE = "write"


# Coarse unit costs. One experiment costs about two literature sweeps; gating is
# nearly free. These are ratios, not measurements.
COST: dict[Action, float] = {
    Action.LITERATURE: 3.0,
    Action.HYPOTHESIZE: 2.0,
    Action.EXPERIMENT: 6.0,
    Action.GATE: 0.5,
    Action.WRITE: 0.0,
}

# The resource each action spends, checked for feasibility.
RESOURCE: dict[Action, str] = {
    Action.LITERATURE: "literature_sweeps",
    Action.HYPOTHESIZE: "llm_calls",
    Action.EXPERIMENT: "experiment_runs",
    Action.GATE: "llm_calls",
    Action.WRITE: "llm_calls",
}

# Tie-break order, and the order the original state machine executed in.
ACTION_ORDER: dict[Action, int] = {
    Action.GATE: 0,
    Action.EXPERIMENT: 1,
    Action.HYPOTHESIZE: 2,
    Action.LITERATURE: 3,
    Action.WRITE: 4,
}

TERMINAL_VERDICTS = frozenset({"UNDECIDABLE", "MARGINAL", "UNREADABLE"})


def expected_information_gain(action: Action, ctx: SchedulerContext) -> float:
    """Expected information gain in [0, 1]. Monotone, coarse, reproducible.

    Scale: 1.0 means "resolves the pending decision", 0.0 means "tells us
    nothing we do not already know".
    """
    if action is Action.WRITE:
        return 0.0

    if action is Action.LITERATURE:
        # The first sweep opens the problem. A later sweep re-runs the same
        # query and therefore re-finds the same papers, so it is close to
        # worthless while candidates are already queued -- testing them is the
        # only thing that can change what we know.
        if ctx.n_papers == 0:
            return 1.0
        if ctx.n_hypotheses > 0:
            return 0.10
        return max(0.0, 0.45 * math.exp(-ctx.n_papers / 40.0))

    if action is Action.HYPOTHESIZE:
        if ctx.n_papers == 0:
            return 0.0                      # nothing to condition on yet
        return 0.10 if ctx.n_hypotheses > 0 else 0.80

    if action is Action.EXPERIMENT:
        if ctx.n_hypotheses == 0:
            return 0.0
        if ctx.last_verdict == "RAN":
            return 0.0                      # gate it before running again
        if ctx.last_verdict in TERMINAL_VERDICTS:
            # The *previous* design failed. A different hypothesis may still
            # fare better, so this is not zero -- but it is not a fresh start
            # either, and saying so is what stops a broken setup being re-run
            # at full price.
            return 0.10
        # Testing a candidate that has never been run is the primary way the
        # loop learns anything, so this is a substantial baseline.
        #
        # ``mde_ratio`` is a one-sided bonus and deliberately never subtracts.
        # It describes the *last* accepted result, not the candidate waiting in
        # the queue; letting a previous test's narrow MDE push this below the
        # value of writing yet another hypothesis makes the agent propose
        # instead of measure, which is the failure this whole project is about.
        ratio = ctx.mde_ratio if ctx.mde_ratio is not None else 1.0
        return min(1.0, 0.60 + 0.5 * max(0.0, ratio - 1.0))

    if action is Action.GATE:
        # Gating is cheap and decisive right after a run; worthless otherwise.
        return 0.90 if ctx.last_verdict == "RAN" else 0.0

    return 0.0


def feasible(action: Action, ctx: SchedulerContext) -> bool:
    if action is Action.WRITE:
        return True
    if ctx.budget.remaining(RESOURCE[action]) <= 0:
        return False
    if action is Action.LITERATURE and ctx.budget.remaining("wall_seconds") <= 1.0:
        return False
    return True


def score(action: Action, ctx: SchedulerContext) -> float:
    """Utility = IG / cost, after feasibility. ``-inf`` means infeasible."""
    if not feasible(action, ctx):
        return -math.inf
    cost = COST[action]
    if cost == 0.0:
        return math.inf                     # terminal marker, never a ranking
    return expected_information_gain(action, ctx) / cost


def stalled(ctx: SchedulerContext) -> bool:
    return ctx.stall_limit > 0 and ctx.stalled_steps >= ctx.stall_limit


def choose(ctx: SchedulerContext) -> Action:
    """Highest-utility feasible action. WRITE only on stall or last resort."""
    if stalled(ctx):
        return Action.WRITE
    ranked = [
        (score(a, ctx), -ACTION_ORDER[a], a)
        for a in Action
        if a is not Action.WRITE
    ]
    ranked = [t for t in ranked if not math.isinf(t[0]) or t[0] > 0]
    if not ranked:
        return Action.WRITE
    best = max(ranked, key=lambda t: (t[0], t[1]))
    if best[0] <= 0.0:
        # Nothing informative is affordable. Continue only if a gate or an
        # experiment is still pending; otherwise stop rather than spin.
        progress = any(
            expected_information_gain(a, ctx) > 0.0 and feasible(a, ctx)
            for a in Action if a is not Action.WRITE
        )
        if not progress:
            return Action.WRITE
    return best[2]


@dataclass
class DecisionLog:
    step: int
    action: Action
    score: float
    scores: dict[str, float]
    remaining: dict[str, float | None]
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "action": self.action.value,
            "score": _finite(self.score),
            "scores": {k: _finite(v) for k, v in self.scores.items()},
            "remaining": self.remaining,
            "context": self.context,
        }


def _finite(v: float) -> float | None:
    """Map ``inf``/``NaN`` to ``None`` — strict JSON has no infinity."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isinf(f) or math.isnan(f):
        return None
    return round(f, 6)


def log_decision(step: int, ctx: SchedulerContext) -> DecisionLog:
    action = choose(ctx)
    return DecisionLog(
        step=step,
        action=action,
        score=score(action, ctx),
        scores={a.value: score(a, ctx) for a in Action},
        remaining=ctx.remaining_table(),
        context=ctx.to_dict(),
    )


def replay_contexts(records) -> list[tuple[int, dict]]:
    """Pull ``(step, context)`` pairs out of ledger records, in order."""
    out: list[tuple[int, dict]] = []
    for rec in records:
        kind = rec.get("kind") if isinstance(rec, dict) else getattr(rec, "kind", None)
        payload = rec.get("payload") if isinstance(rec, dict) else getattr(rec, "payload", None)
        if kind != "scheduler" or not isinstance(payload, dict):
            continue
        ctx = payload.get("context")
        if isinstance(ctx, dict):
            out.append((int(payload.get("step", len(out))), ctx))
    return out


__all__ = [
    "ACTION_ORDER",
    "Action",
    "COST",
    "DecisionLog",
    "RESOURCE",
    "TERMINAL_VERDICTS",
    "choose",
    "expected_information_gain",
    "feasible",
    "log_decision",
    "replay_contexts",
    "score",
    "stalled",
]
