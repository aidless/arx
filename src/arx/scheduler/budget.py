"""Hard resource ceilings for an autonomous run.

The loop must terminate on its own, and the way it terminates has to be
inspectable. Two questions that look alike and are not:

* ``exhausted()`` — *which* resource is at or over its cap. Used for reporting,
  and by the scheduler to disqualify a single class of action.
* ``halting()`` — *which* resource, when exhausted, must stop the whole loop.

Conflating them is a real bug with a real signature: spend the literature
quota once, and the loop halts after a single sweep. Literature sweeps are
capped; running out of them is not a reason to stop researching. Only the
wall-clock ceiling is terminal.

A third distinction matters just as much: a budget that the audited system
reports about itself is not an audit. ``charge`` counts calls this process made.
Cross-checking it against the gateway's own count is the job of
``arx.oracles.auditor`` — see ``docs/evaluation-gate.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Resources whose exhaustion ends the run. Everything else only disqualifies
# the actions that consume it.
_HALTING: tuple[str, ...] = ("wall_seconds",)

# (cap attribute, used attribute) for every accounted resource.
_ACCOUNTED: tuple[tuple[str, str], ...] = (
    ("max_llm_calls", "llm_calls"),
    ("max_wall_seconds", "wall_seconds"),
    ("max_experiment_runs", "experiment_runs"),
    ("max_literature_sweeps", "literature_sweeps"),
)


@dataclass
class Budget:
    """Caps (``None`` = unbounded) and the counters spent against them."""

    max_llm_calls: int | None = 200
    max_wall_seconds: float | None = 1800.0
    max_experiment_runs: int | None = 12
    max_literature_sweeps: int | None = 3

    llm_calls: int = 0
    wall_seconds: float = 0.0
    experiment_runs: int = 0
    literature_sweeps: int = 0

    # Action-specific quotas, keyed by *resource* name (``RESOURCE[Action.X]``,
    # e.g. ``"literature_sweeps"``) -- not by the action's own value, which is
    # ``"literature"``. They are a second, finer ceiling on top of the caps
    # above, and they round-trip through ``to_dict``/``from_dict`` so a
    # resumed run cannot silently acquire a quota it never had.
    #
    # Nothing in the default flow sets them: the scheduler prices actions by
    # their caps (``remaining``), so ``quotas`` arrives empty and
    # ``quota_left`` returns ``inf``. They exist for callers that want to
    # ration one action independently of the global caps. Do not read their
    # presence as evidence that a quota is being enforced -- check
    # ``quota_left``.
    quotas: dict[str, int] = field(default_factory=dict)
    _quota_used: dict[str, int] = field(default_factory=dict, repr=False)

    # -- spending -----------------------------------------------------------
    def charge(self, resource: str, amount: float = 1.0) -> None:
        if amount < 0:
            raise ValueError("cannot charge a negative amount")
        if resource == "llm_calls":
            self.llm_calls += int(amount)
        elif resource == "wall_seconds":
            self.wall_seconds += float(amount)
        elif resource == "experiment_runs":
            self.experiment_runs += int(amount)
        elif resource == "literature_sweeps":
            self.literature_sweeps += int(amount)
        else:
            raise KeyError(f"unknown resource: {resource!r}")
        if resource in self.quotas:
            self._quota_used[resource] = self._quota_used.get(resource, 0) + int(amount)

    # -- reading ------------------------------------------------------------
    def remaining(self, resource: str) -> float:
        """Allowance left, or ``math.inf`` when unbounded.

        Always returns a float, so callers can do arithmetic without
        special-casing ``None``; ``inf`` behaves correctly under ``min``/``max``.
        """
        cap_of = {used: cap for cap, used in _ACCOUNTED}
        try:
            cap = getattr(self, cap_of[resource])
        except KeyError:
            raise KeyError(f"unknown resource: {resource!r}") from None
        used = getattr(self, resource)
        if cap is None:
            return math.inf
        return max(0.0, float(cap) - float(used))

    def exhausted(self) -> list[str]:
        """Every accounted resource at or over its cap. Reporting only."""
        out: list[str] = []
        for cap_attr, used_attr in _ACCOUNTED:
            cap = getattr(self, cap_attr)
            if cap is None:
                continue
            if getattr(self, used_attr) >= cap:
                out.append(used_attr)
        return out

    def halting(self) -> list[str]:
        """Resources whose exhaustion must stop the loop outright."""
        out: list[str] = []
        for used_attr in _HALTING:
            cap_attr = f"max_{used_attr}"
            cap = getattr(self, cap_attr)
            if cap is not None and getattr(self, used_attr) >= cap:
                out.append(used_attr)
        return out

    def is_unbounded(self) -> bool:
        return all(getattr(self, cap) is None for cap, _ in _ACCOUNTED)

    # -- per-action quotas (orthogonal to the global caps) ------------------
    def set_quota(self, action: str, n: int) -> None:
        self.quotas[action] = int(n)

    def quota_left(self, action: str) -> float:
        if action not in self.quotas:
            return math.inf
        return max(0.0, float(self.quotas[action] - self._quota_used.get(action, 0)))

    def charge_quota(self, action: str, amount: int = 1) -> None:
        self._quota_used[action] = self._quota_used.get(action, 0) + int(amount)

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "max_llm_calls": self.max_llm_calls,
            "max_wall_seconds": self.max_wall_seconds,
            "max_experiment_runs": self.max_experiment_runs,
            "max_literature_sweeps": self.max_literature_sweeps,
            "llm_calls": self.llm_calls,
            "wall_seconds": round(self.wall_seconds, 6),
            "experiment_runs": self.experiment_runs,
            "literature_sweeps": self.literature_sweeps,
            "quotas": dict(self.quotas),
            "quota_used": dict(self._quota_used),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Budget:
        budget = cls()
        # ``None`` in the serialized form means *unbounded*, which is a
        # deliberate setting and not the same as "key absent". Skipping it
        # would silently re-impose the constructor's default cap: a run that
        # was granted no wall-clock limit would resume with the 1800-second
        # default and halt for a reason that never applied to it. So the test
        # is membership, not truthiness.
        for key in ("max_llm_calls", "max_wall_seconds", "max_experiment_runs",
                    "max_literature_sweeps", "llm_calls", "wall_seconds",
                    "experiment_runs", "literature_sweeps"):
            if key in data:
                setattr(budget, key, data[key])
        budget.quotas = dict(data.get("quotas") or {})
        budget._quota_used = dict(data.get("quota_used") or {})
        return budget

    def snapshot(self) -> dict:
        """The subset that belongs in a ledger decision record."""
        return {
            "llm_calls": self.llm_calls,
            "wall_seconds": round(self.wall_seconds, 6),
            "experiment_runs": self.experiment_runs,
            "literature_sweeps": self.literature_sweeps,
        }


__all__ = ["Budget"]
