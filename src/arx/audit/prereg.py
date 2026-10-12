"""Pre-registration and cross-hypothesis multiplicity control.

Two distinct self-deception channels live here.

**k_max laundering.** If the experiment script declares its own number of
looks, it can run a fixed-sample test, miss, and then claim it only ever
intended one look. The fix is structural: the *scheduler* decides ``k_max``
before the experiment runs, writes it to the ledger, and the gate reads the
ledger — never the experiment's own report.

**Family inflation.** Each hypothesis spending its own 0.05 gives a
family-wise error rate of ``1 - (1-alpha)^m``: about 64% at m=20. The gate
hands out a decreasing per-test level instead.

Guarantees, stated precisely because they are easy to overclaim:

``spending`` (default, online)
    Threshold for the i-th test is ``gamma_i = alpha * (6/pi^2) / i^2``.
    Since ``sum_i gamma_i = alpha`` exactly, the union bound gives
    ``FWER <= alpha`` for **arbitrary dependence** between tests. This is
    the right rule for an agent that must decide as it goes and cannot know
    m in advance.

``bh`` (offline)
    Benjamini-Hochberg over the full p-value list, controlling FDR at alpha
    under independence or positive dependence (PRDS). Only usable once every
    p-value is in hand, so it is for the post-hoc report, not the live gate.

``bonferroni``
    ``alpha / (m+1)``. Kept for compatibility with the earliest design; it is
    a shrinking per-test level and is conservative, but the ``m+1`` is a
    heuristic, not a calibrated rule. Prefer ``spending``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from statistics import NormalDist

_N = NormalDist()
_SPENDING_TOTAL = 6.0 / (math.pi**2)  # sum_{i>=1} 1/i^2 = pi^2/6


# ---------------------------------------------------------------------------
# Look-plan pre-registration
# ---------------------------------------------------------------------------

def plan_k_max(
    *,
    prior_confidence: float,
    alpha: float = 0.05,
    target_power: float = 0.8,
    effect_se_ratio: float | None = None,
    k_min: int = 1,
    k_max: int = 10,
) -> int:
    """Decide the number of looks *before* any data is seen.

    Deterministic given its arguments, so a replay reproduces it exactly.
    The mapping is deliberately coarse and monotone:

    * low prior confidence -> more looks;
    * a small expected effect relative to its standard error -> more looks,
      because separating it will need them.

    ``effect_se_ratio`` is the expected effect divided by the standard error,
    in z units. If it is unknown the prior alone decides.
    """
    if not 0.0 <= prior_confidence <= 1.0:
        raise ValueError("prior_confidence must be in [0, 1]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0.0 < target_power < 1.0:
        raise ValueError("target_power must be in (0, 1)")
    if k_min < 1 or k_max < k_min:
        raise ValueError("require 1 <= k_min <= k_max")

    base = 1.0 + (k_max - k_min) * (1.0 - prior_confidence)

    if effect_se_ratio is not None:
        needed = _N.inv_cdf(1.0 - alpha / 2.0) + _N.inv_cdf(target_power)
        shortfall = max(0.0, needed - float(effect_se_ratio))
        base += shortfall

    return max(k_min, min(k_max, int(round(base))))


def prereg_fingerprint(
    *,
    hypothesis_text: str,
    k_max: int,
    alpha: float,
    script_sha256: str | None = None,
) -> str:
    """Stable id for a pre-registration record.

    The script hash closes the "run version A, paste version B" hole: change
    the script after registering and the fingerprint no longer matches.
    """
    payload = json.dumps(
        {
            "hypothesis": hypothesis_text.strip(),
            "k_max": int(k_max),
            "alpha": float(alpha),
            "script_sha256": script_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Preregistration:
    hypothesis_text: str
    k_max: int
    alpha: float
    script_sha256: str | None = None
    fingerprint: str = ""

    @classmethod
    def create(
        cls,
        *,
        hypothesis_text: str,
        k_max: int,
        alpha: float,
        script_sha256: str | None = None,
    ) -> Preregistration:
        return cls(
            hypothesis_text=hypothesis_text,
            k_max=int(k_max),
            alpha=float(alpha),
            script_sha256=script_sha256,
            fingerprint=prereg_fingerprint(
                hypothesis_text=hypothesis_text,
                k_max=k_max,
                alpha=alpha,
                script_sha256=script_sha256,
            ),
        )

    def to_dict(self) -> dict:
        return {
            "hypothesis": self.hypothesis_text,
            "k_max": self.k_max,
            "alpha": self.alpha,
            "script_sha256": self.script_sha256,
            "fingerprint": self.fingerprint,
        }


def read_preregistration(
    records: Iterable[dict],
    hypothesis_text: str,
) -> Preregistration | None:
    """Latest pre-registration for a hypothesis in the ledger.

    Latest wins so a deliberate re-registration (a new design) supersedes the
    old one, while an accidental duplicate is harmless.
    """
    found: Preregistration | None = None
    for rec in records:
        if rec.get("kind") != "pre_register":
            continue
        payload = rec.get("payload", {})
        if payload.get("hypothesis") != hypothesis_text:
            continue
        found = Preregistration(
            hypothesis_text=payload["hypothesis"],
            k_max=int(payload["k_max"]),
            alpha=float(payload["alpha"]),
            script_sha256=payload.get("script_sha256"),
            fingerprint=payload.get("fingerprint", ""),
        )
    return found


def check_preregistration(
    prereg: Preregistration | None,
    *,
    blob_k_max: int | None = None,
    blob_script_sha256: str | None = None,
) -> tuple[bool, str]:
    """Verify the experiment honoured its pre-registration.

    Returns ``(ok, reason)``. The gate treats ``False`` as UNDECIDABLE, which
    is correct: an experiment that broke its own contract is not evidence
    against the hypothesis, it is no evidence at all.
    """
    if prereg is None:
        return False, "no pre-registration in ledger for this hypothesis"

    expected = prereg_fingerprint(
        hypothesis_text=prereg.hypothesis_text,
        k_max=prereg.k_max,
        alpha=prereg.alpha,
        script_sha256=prereg.script_sha256,
    )
    if expected != prereg.fingerprint:
        return False, "pre-registration record is internally inconsistent (tampered)"

    if blob_k_max is not None and int(blob_k_max) != prereg.k_max:
        return False, (
            f"experiment declared k_max={int(blob_k_max)} but pre-registered "
            f"k_max={prereg.k_max}"
        )

    if prereg.script_sha256 is not None:
        if blob_script_sha256 is None:
            return False, (
                "pre-registration pinned a script hash but the run did not "
                "report one"
            )
        if blob_script_sha256 != prereg.script_sha256:
            return False, "experiment script hash does not match pre-registration"

    return True, "ok"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Multiplicity
# ---------------------------------------------------------------------------

def two_sided_p_from_z(z: float) -> float:
    """Two-sided p-value from a z statistic."""
    return float(2.0 * (1.0 - _N.cdf(abs(float(z)))))


def one_sided_p_from_z(z: float) -> float:
    """One-sided (upper-tail) p-value from a z statistic."""
    return float(1.0 - _N.cdf(float(z)))


@dataclass
class MultiplicityLedger:
    """Hands out a per-test level as the family grows.

    Every registration is stored whole (hypothesis, p, index, allowance,
    passed), so ``summary`` never has to recompute a threshold from a
    positional index — the bug that makes duplicate hypothesis names silently
    receive the wrong allowance.
    """

    alpha: float = 0.05
    mode: str = "spending"
    p_values: list[float] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    records: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if self.mode not in ("spending", "bh", "bonferroni"):
            raise ValueError(f"unknown mode {self.mode!r}")

    @property
    def m(self) -> int:
        """Family size: every hypothesis *tested*, scored or not.

        A test whose p-value could not be computed still consumes a slot in the
        family — the family is the set of things that were tried, not the set of
        things that produced a number. Counting only the scored ones would let
        an unscoreable test hide from the correction.
        """
        return len(self.records)

    @property
    def n_scored(self) -> int:
        return len(self.p_values)

    @property
    def n_unscored(self) -> int:
        return self.m - self.n_scored

    def threshold_for_next(self) -> float:
        """Per-test level the next hypothesis will be judged against."""
        i = self.m + 1
        if self.mode == "spending":
            return self.alpha * _SPENDING_TOTAL / (i * i)
        if self.mode == "bonferroni":
            return self.alpha / i
        # BH: offline, so the live threshold uses the current family size.
        return self.alpha  # tightened later, in bh_rejects()

    def bh_rejects(self) -> list[str]:
        """Benjamini-Hochberg rejections over the scored p-values."""
        m = self.n_scored
        if m == 0:
            return []
        order = sorted(range(m), key=lambda i: self.p_values[i])
        kmax = -1
        for rank, idx in enumerate(order, start=1):
            if self.p_values[idx] <= self.alpha * rank / m:
                kmax = rank
        if kmax < 0:
            return []
        return [self.hypotheses[idx] for idx in order[:kmax]]

    def register(self, hypothesis: str, p: float | None) -> dict:
        """Record one test.

        ``p=None`` means the test ran but produced no usable p-value. It is a
        real outcome (an experiment can be unscoreable), it still grows the
        family, and it is never treated as a pass.
        """
        if p is not None and not 0.0 <= float(p) <= 1.0:
            raise ValueError("p must be in [0, 1] or None")
        allowance = self.threshold_for_next()
        scored = p is not None
        passed = bool(scored and float(p) <= allowance)
        rec = {
            "hypothesis": hypothesis,
            "p": float(p) if scored else None,
            "index": self.m,
            "allowance": float(allowance),
            "passed": passed,
            "scored": scored,
            "mode": self.mode,
        }
        if scored:
            self.p_values.append(float(p))
            self.hypotheses.append(hypothesis)
        self.records.append(rec)
        return rec

    def summary(self) -> dict:
        last_by_h: dict[str, dict] = {}
        for rec in self.records:
            last_by_h[rec["hypothesis"]] = rec
        accepted = [h for h, r in last_by_h.items() if r["passed"]]
        rejected = [h for h, r in last_by_h.items() if not r["passed"]]
        out = {
            "alpha": self.alpha,
            "mode": self.mode,
            "m": self.m,
            "n_scored": self.n_scored,
            "n_unscored": self.n_unscored,
            "p_values": list(self.p_values),
            "accepted": accepted,
            "rejected": rejected,
            "next_allowance": self.threshold_for_next(),
        }
        if self.n_unscored:
            out["unscored_hypotheses"] = [
                r["hypothesis"] for r in self.records if not r["scored"]
            ]
        if self.mode == "bh":
            out["bh_rejects"] = self.bh_rejects()
        # The family-wise ceiling actually paid for, so a reader can see the
        # budget going out rather than being told it is under control.
        #
        # `spending` converges: sum 1/i^2 is finite, so the schedule can never
        # spend more than alpha however long the family grows.
        #
        # `bonferroni` does not, and saying so is the point of reporting it.
        # The textbook correction is alpha/m for a family whose size m is
        # fixed in advance; an online run does not know m, so handing out
        # alpha/i means the i-th test is judged against alpha/i and the total
        # is a partial harmonic sum -- which grows without bound, like
        # alpha * log(m). It matches Bonferroni only when the family is declared
        # up front. Reporting a number here is what stops "bonferroni" being
        # read as "FWER controlled" when it is not.
        if self.mode == "spending":
            out["family_alpha_spent"] = self.alpha * _SPENDING_TOTAL * sum(
                1.0 / ((i + 1) ** 2) for i in range(self.m)
            )
        elif self.mode == "bonferroni":
            total = sum(1.0 / i for i in range(1, self.m + 1))
            out["family_alpha_spent"] = self.alpha * total
            out["budget_note"] = (
                "per-test level is alpha/i with i the position in the family, "
                "which sums to alpha * (harmonic m) and grows without bound. "
                "The Bonferroni guarantee needs a family size fixed in advance; "
                "this is a shrinking allowance, not FWER control."
            )
        return out

    @classmethod
    def from_ledger(
        cls, records: Iterable[dict], alpha: float = 0.05, mode: str = "spending"
    ) -> MultiplicityLedger:
        led = cls(alpha=alpha, mode=mode)
        for rec in records:
            if rec.get("kind") != "multiplicity":
                continue
            payload = rec.get("payload", {})
            if payload.get("event") == "register":
                raw_p = payload.get("p")
                led.register(
                    str(payload["hypothesis"]),
                    float(raw_p) if raw_p is not None else None,
                )
        return led


__all__ = [
    "MultiplicityLedger",
    "Preregistration",
    "check_preregistration",
    "one_sided_p_from_z",
    "plan_k_max",
    "prereg_fingerprint",
    "read_preregistration",
    "sha256_text",
    "two_sided_p_from_z",
]
