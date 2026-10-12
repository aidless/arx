"""Verdict taxonomy for benchmark audits.

A leaderboard is only *readable* if the top system beats the measured noise
floor by a margin the sample can actually resolve. Four ways that can fail
are distinguished, because they call for different fixes:

    READABLE     margin exceeds the resolution bound
    MARGINAL     positive but inside the resolution bound (needs more data)
    UNREADABLE   the leader does not clear the floor at all
    UNDECIDABLE  the premises are violated; no floor is estimable

`UNDECIDABLE` is not a failure of the analysis. It is the analysis finding
that the design cannot support *any* floor estimate, which is a strictly
stronger and more useful statement than reporting a number that means
nothing.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class Verdict(enum.StrEnum):
    READABLE = "READABLE"
    MARGINAL = "MARGINAL"
    UNREADABLE = "UNREADABLE"
    UNDECIDABLE = "UNDECIDABLE"
    STOP_EARLY = "STOP_EARLY"


class UndecidableReason(enum.StrEnum):
    """Why no floor is estimable. Kept separate from UNREADABLE on purpose."""

    SINGLE_GENERATOR = "single_generator"
    FLOOR_DEGENERATE = "floor_degenerate"
    ACTORS_OVERLAP = "actors_overlap"
    DOMAIN_MISMATCH = "domain_mismatch"


@dataclass
class AuditReport:
    verdict: Verdict
    reasons: list[str] = field(default_factory=list)

    # Scores and the margin over the floor.
    leader_system: str | None = None
    leader_score: float | None = None
    floor_estimate: float | None = None
    margin: float | None = None

    # Resolution.
    delta_declared: float | None = None
    mde: float | None = None
    threshold: float | None = None

    # Provenance of the numbers.
    n_items: int | None = None
    n_floor_items: int | None = None
    n_generators: int | None = None
    n_contestants: int | None = None
    floor_method: str = ""

    dk_text: str = ""
    sequential: dict | None = None
    extra: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.verdict in (Verdict.READABLE, Verdict.STOP_EARLY)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "reasons": list(self.reasons),
            "leader_system": self.leader_system,
            "leader_score": self.leader_score,
            "floor_estimate": self.floor_estimate,
            "margin": self.margin,
            "delta_declared": self.delta_declared,
            "mde": self.mde,
            "threshold": self.threshold,
            "n_items": self.n_items,
            "n_floor_items": self.n_floor_items,
            "n_generators": self.n_generators,
            "n_contestants": self.n_contestants,
            "floor_method": self.floor_method,
            "sequential": self.sequential,
            "extra": self.extra,
        }

    def to_text(self) -> str:
        lines = [f"Verdict: {self.verdict.value}"]
        if self.reasons:
            lines.append("Reasons: " + ", ".join(self.reasons))
        if self.leader_system is not None:
            lines.append(f"Leader: {self.leader_system} = {self.leader_score:.4f}")
        if self.floor_estimate is not None:
            lines.append(f"Noise floor ({self.floor_method}): {self.floor_estimate:.4f}")
        if self.margin is not None:
            lines.append(f"Margin over floor: {self.margin:+.4f}")
        if self.threshold is not None:
            parts = []
            if self.delta_declared is not None:
                parts.append(f"delta={self.delta_declared:.4f}")
            if self.mde is not None:
                parts.append(f"MDE={self.mde:.4f}")
            lines.append(f"Resolution bound: {self.threshold:.4f} ({', '.join(parts)})")
        if self.n_items is not None:
            lines.append(f"Items: {self.n_items} (floor on {self.n_floor_items})")
        if self.n_generators is not None:
            lines.append(f"Gold generators: {self.n_generators}")
        if self.dk_text:
            lines.extend(["", self.dk_text])
        if self.sequential:
            lines.extend(["", "Sequential gate:"])
            for k, v in self.sequential.items():
                lines.append(f"  {k}: {v}")
        if self.extra:
            lines.extend(["", f"Extra: {self.extra}"])
        return "\n".join(lines)
