"""Online monitoring of the scheduler's own predictive validity.

The scheduler scores actions by an expected-information-gain function. If that
function was fitted to history, history can move out from under it — a textbook
Goodhart loop. This module turns the loop into something observable:

    every gate event logs (predicted_ig, observed_outcome),
    and we track the rank correlation between them over a sliding window.

When the correlation falls below ``threshold`` the scheduler is no longer
tracking reality, and a recalibration event is emitted into the ledger.

Nothing here calls an LLM. It is arithmetic over ledger records, so a third
party can recompute every number — which is the only reason the monitor is
worth having at all.

The utility mapping below is deliberately coarse and **fixed**. Letting it be
fitted would make the monitor itself a Goodhart target.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

VERDICT_UTILITY: dict[str, float] = {
    "READABLE": 1.0,
    "STOP_EARLY": 1.0,
    "MARGINAL": 0.4,
    "UNDECIDABLE": 0.0,
    "REJECTED": 0.0,
    "UNREADABLE": 0.0,
    "NO_DATA": 0.0,
}


def _rankdata(xs: list[float]) -> list[float]:
    """Average ranks with ties handled. Avoids a scipy dependency."""
    n = len(xs)
    order = sorted(range(n), key=lambda i: xs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    """Tie-aware Spearman rank correlation.

    Returns 0.0 when either series is constant. That is the honest answer — a
    constant predictor carries no ranking information — and it keeps the
    ledger's JSON contract free of NaN.
    """
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have equal length")
    if len(xs) < 2:
        return 0.0
    rx, ry = _rankdata(list(xs)), _rankdata(list(ys))
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0.0 or dy == 0.0:
        return 0.0
    return float(num / (dx * dy))


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have equal length")
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys, strict=True)) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in xs) / n)
    sy = math.sqrt(sum((b - my) ** 2 for b in ys) / n)
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return float(cov / (sx * sy))


@dataclass
class GoodhartMonitor:
    """Windowed monitor over ``(predicted, observed)`` pairs."""

    window: int = 30
    threshold: float = 0.3
    min_samples: int = 10
    window_long: int = 100
    use_spearman: bool = True
    pairs: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError("window must be >= 2")
        if self.min_samples < 2:
            raise ValueError("min_samples must be >= 2")

    # -- observation --------------------------------------------------------
    def observe(self, predicted_ig: float, verdict: str) -> None:
        self.observe_pair(predicted_ig, VERDICT_UTILITY.get(verdict, 0.0))

    def observe_pair(self, predicted: float, observed: float) -> None:
        self.pairs.append((float(predicted), float(observed)))

    # -- statistics ---------------------------------------------------------
    def _windowed(self) -> list[tuple[float, float]]:
        return self.pairs[-self.window:]

    def _corr(self, pairs: list[tuple[float, float]]) -> float:
        if len(pairs) < 2:
            return 0.0
        xs = [p for p, _ in pairs]
        ys = [o for _, o in pairs]
        return spearman(xs, ys) if self.use_spearman else pearson(xs, ys)

    @property
    def correlation(self) -> float:
        return self._corr(self._windowed())

    @property
    def long_correlation(self) -> float:
        return self._corr(self.pairs[-self.window_long:])

    @property
    def degraded(self) -> bool:
        """True once there is enough evidence that the model has drifted."""
        return (
            len(self._windowed()) >= self.min_samples
            and self.correlation < self.threshold
        )

    def status(self) -> dict:
        w = self._windowed()
        return {
            "n": len(w),
            "n_total": len(self.pairs),
            "correlation": round(self.correlation, 4),
            "long_correlation": round(self.long_correlation, 4),
            "threshold": self.threshold,
            "degraded": self.degraded,
            "insufficient_data": len(w) < self.min_samples,
            "spearman": self.use_spearman,
        }

    # -- reconstruction -----------------------------------------------------
    @classmethod
    def from_ledger(
        cls,
        records: Iterable[dict],
        window: int = 30,
        threshold: float = 0.3,
        min_samples: int = 10,
    ) -> GoodhartMonitor:
        """Rebuild the monitor from ledger records, in order.

        Pairs a ``scheduler`` decision that chose ``gate`` with the ``gate``
        record that follows it. A gate with no preceding scheduler record
        carries no prediction and is skipped — counting it would bias the
        correlation toward zero.
        """
        mon = cls(window=window, threshold=threshold, min_samples=min_samples)
        pending: float | None = None
        for rec in records:
            kind = rec.get("kind")
            payload = rec.get("payload", {})
            if kind == "scheduler":
                if payload.get("action") == "gate":
                    pending = _score_of(payload, "gate")
                else:
                    pending = None
            elif kind == "gate":
                if pending is not None:
                    mon.observe(pending, str(payload.get("verdict", "")))
                    pending = None
        return mon


def _score_of(payload: dict, action: str) -> float:
    scores = payload.get("scores") or {}
    val = scores.get(action)
    if val is None:
        val = payload.get("chosen_score", 0.0)
    try:
        val = float(val)
    except (TypeError, ValueError):
        return 0.0
    if math.isinf(val) or math.isnan(val):
        return 0.0
    return val


@dataclass
class ProxyTargetMonitor:
    """Track a proxy metric against a target metric (RL / Goodhart studies).

    ``drift`` is the gap between the means. A near-zero correlation between
    proxy and target at plateau is the Goodhart signature: the policy is
    optimising the judge's residual, not the task.
    """

    window: int = 50
    corr_threshold: float = 0.2
    proxy: list[float] = field(default_factory=list)
    target: list[float] = field(default_factory=list)

    def record(self, proxy: float, target: float) -> None:
        self.proxy.append(float(proxy))
        self.target.append(float(target))

    def drift(self) -> float:
        if not self.proxy:
            return 0.0
        p = self.proxy[-self.window:]
        t = self.target[-self.window:]
        return float(abs(sum(p) / len(p) - sum(t) / len(t)))

    def divergence(self) -> float:
        if len(self.proxy) < self.window:
            return float("nan")
        return pearson(self.proxy[-self.window:], self.target[-self.window:])

    def is_goodharting(self) -> bool:
        d = self.divergence()
        return (not math.isnan(d)) and d < self.corr_threshold


__all__ = [
    "GoodhartMonitor",
    "ProxyTargetMonitor",
    "VERDICT_UTILITY",
    "pearson",
    "spearman",
]
