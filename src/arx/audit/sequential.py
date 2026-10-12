"""Sequential evaluation gate with family-wise error control.

The problem this solves: an agent that looks at the same hypothesis k times
and reports the first look where p < 0.05 has an effective type-I rate of
``1 - (1-alpha)^k``. At k=10, alpha=0.05 that is about 40%. A gate that
ignores this is not a gate; it is a p-hacking engine with a paper trail.

Boundary calibration
--------------------
The textbook O'Brien-Fleming shape ``z_{alpha/2} * sqrt(k/j)`` does **not**
control the family-wise error rate on its own — simulated k=5 gives ~0.088 and
k=10 gives ~0.145 against a nominal 0.05, because the shape is a spending
*plan*, not a calibrated set of critical values.

Here the shape is kept and the scale is **solved** so that the union bound
holds:

    sum_{j=1..k} 2 (1 - Phi(c sqrt(k/j))) = alpha

The union bound needs no assumption about the dependence between looks, so
FWER <= alpha holds for arbitrary dependence. The price is conservatism: every
boundary, including the last, sits at or above the fixed-sample 1.96
(k=10 gives c ~= 2.34). That price is paid knowingly — see
``docs/sequential-gate.md`` — and ``tests/test_sequential.py`` pins it.

No scipy: normal quantiles come from :class:`statistics.NormalDist`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist

from .verdict import Verdict

_N = NormalDist()
_TOL = 1e-10


def fixed_sample_boundary(alpha: float = 0.05) -> float:
    """Two-sided critical z for a single, unpeeked look."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return _N.inv_cdf(1.0 - alpha / 2.0)


def union_bound_rate(c: float, k_max: int) -> float:
    """Union-bound FWER for scale ``c`` over ``k_max`` looks."""
    if k_max < 1:
        raise ValueError("k_max must be >= 1")
    if c <= 0:
        return 1.0
    return sum(
        2.0 * (1.0 - _N.cdf(c * math.sqrt(k_max / j)))
        for j in range(1, k_max + 1)
    )


def calibrate_scale(k_max: int, alpha: float = 0.05) -> float:
    """Solve for the scale ``c`` that makes the union bound equal ``alpha``.

    Monotone in ``c`` (the rate is strictly decreasing), so bisection is exact
    and needs no derivatives.
    """
    if k_max < 1:
        raise ValueError("k_max must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    lo, hi = 1e-6, 40.0
    if union_bound_rate(lo, k_max) < alpha:
        return lo
    if union_bound_rate(hi, k_max) > alpha:
        return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if union_bound_rate(mid, k_max) > alpha:
            lo = mid
        else:
            hi = mid
        if hi - lo < _TOL:
            break
    return 0.5 * (lo + hi)


def obf_boundaries(k_max: int, alpha: float = 0.05) -> list[float]:
    """Calibrated O'Brien-Fleming boundaries, one per look.

    Strictly decreasing in ``j``, and every entry is >= the fixed-sample
    boundary.
    """
    if k_max < 1:
        raise ValueError("k_max must be >= 1")
    c = calibrate_scale(k_max, alpha)
    return [c * math.sqrt(k_max / j) for j in range(1, k_max + 1)]


def effective_alpha(k_max: int, alpha: float = 0.05) -> float:
    """The naive per-look alpha that would give the same family-wise rate.

    Reported alongside the gate so a reader can see what the gate bought.
    """
    if k_max < 1:
        raise ValueError("k_max must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return 1.0 - (1.0 - alpha) ** (1.0 / k_max)


def z_from_effect(effect: float, se: float) -> float:
    """``z = effect / SE``. Centralised so the argument order cannot drift."""
    if se <= 0:
        raise ValueError("SE must be positive")
    return float(effect) / float(se)


@dataclass
class SequentialState:
    """Running state for one hypothesis across looks."""

    k_max: int
    alpha: float = 0.05
    label: str = ""
    looks: list[float] = field(default_factory=list)
    stopped_at: int | None = None
    verdict: Verdict | None = None
    boundaries: list[float] = field(init=False)

    def __post_init__(self) -> None:
        if self.k_max < 1:
            raise ValueError("k_max must be >= 1")
        self.boundaries = obf_boundaries(self.k_max, self.alpha)

    @property
    def n_looks(self) -> int:
        return len(self.looks)

    @property
    def done(self) -> bool:
        return self.stopped_at is not None or len(self.looks) >= self.k_max

    @property
    def next_boundary(self) -> float | None:
        if self.done:
            return None
        return self.boundaries[len(self.looks)]

    def observe(self, z: float) -> Verdict | None:
        """Record one look.

        Returns ``READABLE`` if a boundary was crossed, ``MARGINAL`` at the
        final look without a crossing, and ``None`` while the gate continues.
        """
        if self.stopped_at is not None:
            raise RuntimeError("gate already stopped")
        j = len(self.looks)
        if j >= self.k_max:
            raise RuntimeError("exhausted k_max looks")
        self.looks.append(float(z))
        if abs(z) >= self.boundaries[j]:
            self.stopped_at = j
            self.verdict = Verdict.READABLE if j == self.k_max - 1 else Verdict.STOP_EARLY
            return Verdict.READABLE
        if j == self.k_max - 1:
            self.stopped_at = j
            self.verdict = Verdict.MARGINAL
            return Verdict.MARGINAL
        return None

    def summary(self) -> dict:
        """JSON-safe state, including the sequential-stopping bias flag.

        Stopping early inflates the magnitude of the observed effect:
        conditional on having crossed a boundary, the estimate is on average
        larger than the truth. The flag is always present so a downstream
        consumer cannot miss it by simply not looking for it.
        """
        out: dict = {
            "label": self.label,
            "k_max": self.k_max,
            "alpha": self.alpha,
            "looks": list(self.looks),
            "boundaries": list(self.boundaries),
            "n_looks": self.n_looks,
            "stopped_at": self.stopped_at,
            "verdict": self.verdict.value if self.verdict else None,
            "stopped_early": self.stopped_at is not None,
            "effect_estimate_is_biased": self.stopped_at is not None,
        }
        if self.stopped_at is not None:
            out["effect_bias_note"] = (
                "estimate is conditional on early stopping and is biased away "
                "from zero; report the final-look estimate or a bias-adjusted "
                "interval, not this value"
            )
        return out


__all__ = [
    "SequentialState",
    "calibrate_scale",
    "effective_alpha",
    "fixed_sample_boundary",
    "obf_boundaries",
    "union_bound_rate",
    "z_from_effect",
]
