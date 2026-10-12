"""The scheduler's view of the world.

Everything the scheduler needs to rank actions, and nothing else. Two
properties are load-bearing:

* **Round-trippable.** ``SchedulerContext.from_dict(ctx.to_dict()) == ctx``.
  A decision recorded in the ledger must be replayable from the ledger alone,
  so the context has to survive JSON without losing or silently coercing
  fields. (An earlier version serialised the budget as a nested dict and could
  not read it back; the round-trip test exists because of that.)

* **JSON-safe.** No ``inf``, no ``NaN``. Strict parsers reject them, and a
  ledger that cannot be parsed by a strict parser is not a ledger. Unbounded
  remaining is represented as ``None`` in the serialised form.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .budget import Budget


@dataclass
class SchedulerContext:
    n_papers: int = 0
    n_hypotheses: int = 0
    n_accepted: int = 0
    n_rejected: int = 0
    last_verdict: str | None = None
    judge_available: bool = False
    mde_ratio: float | None = None
    margin: float | None = None
    stalled_steps: int = 0
    stall_limit: int = 6
    budget: Budget = field(default_factory=Budget)
    extra: dict = field(default_factory=dict)

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "n_papers": int(self.n_papers),
            "n_hypotheses": int(self.n_hypotheses),
            "n_accepted": int(self.n_accepted),
            "n_rejected": int(self.n_rejected),
            "last_verdict": self.last_verdict,
            "judge_available": bool(self.judge_available),
            "mde_ratio": self.mde_ratio,
            "margin": self.margin,
            "stalled_steps": int(self.stalled_steps),
            "stall_limit": int(self.stall_limit),
            "budget": self.budget.to_dict(),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict) -> SchedulerContext:
        budget = data.get("budget") or {}
        return cls(
            n_papers=int(data.get("n_papers", 0)),
            n_hypotheses=int(data.get("n_hypotheses", 0)),
            n_accepted=int(data.get("n_accepted", 0)),
            n_rejected=int(data.get("n_rejected", 0)),
            last_verdict=data.get("last_verdict"),
            judge_available=bool(data.get("judge_available", False)),
            mde_ratio=data.get("mde_ratio"),
            margin=data.get("margin"),
            stalled_steps=int(data.get("stalled_steps", 0)),
            stall_limit=int(data.get("stall_limit", 6)),
            budget=Budget.from_dict(budget) if isinstance(budget, dict) else Budget(),
            extra=dict(data.get("extra") or {}),
        )

    # -- readability --------------------------------------------------------
    def describe(self) -> str:
        parts = [
            f"papers={self.n_papers}",
            f"hyps={self.n_hypotheses}",
            f"accepted={self.n_accepted}",
            f"rejected={self.n_rejected}",
            f"last={self.last_verdict or '-'}",
            f"stall={self.stalled_steps}/{self.stall_limit}",
        ]
        if self.mde_ratio is not None:
            parts.append(f"mde_ratio={self.mde_ratio:.2f}")
        return " ".join(parts)

    def remaining_table(self) -> dict[str, float | None]:
        """Remaining budget as JSON-safe values (``None`` means unbounded)."""
        out: dict[str, float | None] = {}
        for name in ("llm_calls", "wall_seconds", "experiment_runs", "literature_sweeps"):
            rem = self.budget.remaining(name)
            out[name] = None if math.isinf(rem) else round(rem, 3)
        return out


__all__ = ["SchedulerContext"]
