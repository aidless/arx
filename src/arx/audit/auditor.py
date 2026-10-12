"""The auditor: decide whether a leaderboard is statistically readable.

Four tenets, applied in this order because each is a precondition for the
next:

1. **Actor-disjointness** — the noise floor must be measured on raters
   disjoint from the contestants. If the same models judge and compete, the
   "floor" measures peer spread, not measurement error.
2. **Common denominator** — a margin is only defined when the leaderboard and
   the floor are computed over the same item set. Comparing 1,084 items to
   1,520 is not a small error; it is arithmetic on two different populations.
3. **Multi-generator sufficiency** (m >= 3) — with two generators, removing
   one leaves one, so leave-one-out collapses into pairwise agreement. Two
   annotators cannot define an independent reproduction floor.
4. **Power-bounded resolution** — a margin below the minimum detectable
   effect is not a finding. It is a statement that this sample could not
   have detected anything this small.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .floor import (
    column_majority,
    dawid_skene,
    floor_estimate,
    loo_floor,
)
from .power import binomial_se, cluster_bootstrap_mde, mde_from_se
from .verdict import AuditReport, UndecidableReason, Verdict


@dataclass
class AuditConfig:
    delta_declared: float = 0.02
    alpha: float = 0.05
    power: float = 0.8
    n_boot: int = 2000
    seed: int = 0
    floor_method: str = "loo"  # "loo" | "dawid_skene"

    def as_kwargs(self) -> dict:
        """Settings as keyword arguments for :func:`audit_leaderboard`.

        Exists so a caller (the gate, the agent) can hand the same audit
        settings to the auditor without re-listing them, which is how the two
        drift apart and a run gets audited under settings nobody recorded.
        """
        return {
            "delta_declared": self.delta_declared,
            "alpha": self.alpha,
            "power": self.power,
            "n_boot": self.n_boot,
            "seed": self.seed,
            "floor_method": self.floor_method,
        }

    def replace(self, **changes) -> AuditConfig:
        """A copy with some settings changed; unknown keys are refused."""
        known = self.as_kwargs()
        unknown = set(changes) - set(known)
        if unknown:
            raise TypeError(f"unknown audit setting(s): {sorted(unknown)}")
        return AuditConfig(**{**known, **changes})

    def to_dict(self) -> dict:
        return self.as_kwargs()


def _blank(**kw) -> AuditReport:
    report = AuditReport(verdict=Verdict.UNDECIDABLE)
    for k, v in kw.items():
        setattr(report, k, v)
    return report


def audit_leaderboard(
    votes,
    leaderboard_scores: dict[str, float],
    *,
    leaderboard_item_ids=None,
    floor_item_ids=None,
    judge_ids=None,
    contestant_ids=None,
    per_item_scores: dict[str, np.ndarray] | None = None,
    delta_declared: float = 0.02,
    alpha: float = 0.05,
    power: float = 0.8,
    n_boot: int = 2000,
    seed: int = 0,
    floor_method: str = "loo",
) -> AuditReport:
    """Audit one leaderboard.

    Parameters
    ----------
    votes:
        ``(m, n)`` integer matrix of gold-generator labels; ``-1`` = abstention.
    leaderboard_scores:
        ``{system_name: accuracy}``.
    leaderboard_item_ids, floor_item_ids:
        Optional item-id sets. When both are given and differ, the audit is
        ``UNDECIDABLE`` for denominator mismatch.
    judge_ids, contestant_ids:
        Optional actor-id sets. A non-empty intersection is ``UNDECIDABLE``.
    per_item_scores:
        Optional ``{system_name: per_item_binary_array}``. When supplied for
        at least two systems, the MDE is computed by a cluster bootstrap on
        the paired difference instead of a binomial approximation.
    """
    votes = np.asarray(votes)
    if votes.ndim != 2:
        raise ValueError("votes must be 2-D (generators x items)")
    m, n = votes.shape

    if not leaderboard_scores:
        raise ValueError("leaderboard_scores is empty")

    report = _blank(n_items=n, n_floor_items=n, n_generators=m,
                    n_contestants=len(leaderboard_scores),
                    delta_declared=delta_declared)
    report.extra["alpha"] = alpha
    report.extra["power"] = power

    # --- Tenet 1: actor-disjointness --------------------------------------
    if judge_ids is not None and contestant_ids is not None:
        overlap = set(map(str, judge_ids)) & set(map(str, contestant_ids))
        if overlap:
            report.reasons.append(UndecidableReason.ACTORS_OVERLAP.value)
            report.extra["overlapping_actors"] = sorted(overlap)
            return report

    # --- Tenet 2: common denominator --------------------------------------
    if leaderboard_item_ids is not None and floor_item_ids is not None:
        lb = set(leaderboard_item_ids)
        fl = set(floor_item_ids)
        if lb != fl:
            report.reasons.append(UndecidableReason.DOMAIN_MISMATCH.value)
            report.extra["leaderboard_items"] = len(lb)
            report.extra["floor_items"] = len(fl)
            report.extra["items_only_in_leaderboard"] = len(lb - fl)
            report.extra["items_only_in_floor"] = len(fl - lb)
            report.n_floor_items = len(fl)
            return report

    # --- Tenet 3: multi-generator sufficiency ------------------------------
    if m < 2:
        report.reasons.append(UndecidableReason.SINGLE_GENERATOR.value)
        return report
    if m == 2:
        report.reasons.append(UndecidableReason.FLOOR_DEGENERATE.value)
        report.extra["note"] = (
            "with two generators, leave-one-out leaves one, so the floor "
            "degenerates into pairwise agreement; it cannot be estimated"
        )
        return report

    # --- Floor -------------------------------------------------------------
    report.floor_method = floor_method
    if floor_method == "dawid_skene":
        try:
            floor, fit = floor_estimate(votes)
            report.dk_text = fit.to_text()
            report.extra["complement_flipped"] = fit.complement_flipped
        except (RuntimeError, ValueError) as exc:
            # A collapsed fit, or a generator set too small to identify one.
            # Both mean the same thing for the caller: no floor exists here.
            report.reasons.append(UndecidableReason.FLOOR_DEGENERATE.value)
            report.extra["floor_error"] = str(exc)
            return report
    elif floor_method == "loo":
        floors = loo_floor(votes)
        valid = floors[~np.isnan(floors)]
        if valid.size == 0:
            report.reasons.append(UndecidableReason.FLOOR_DEGENERATE.value)
            return report
        floor = float(valid.mean())
        report.extra["per_generator_floor"] = [round(float(x), 4) for x in floors]
    else:
        raise ValueError(f"unknown floor_method {floor_method!r}")

    report.floor_estimate = floor

    # --- Gold-signal check -------------------------------------------------
    # A floor is only interpretable if it is *detectably* above chance. If the
    # generators reproduce nothing, the "gold" is noise, and a system scoring
    # 0.96 against it has not beaten a floor — it was scored against something
    # else. Without this check a board of pure-noise annotators audits as
    # READABLE with a margin of nearly +0.5, which is the loudest possible
    # false positive.
    #
    # Resolution comes from the number of *items*, not the number of
    # generators: the dominant source of variability in the floor is which
    # items were drawn, and that source is shared by every generator. So the
    # standard error is computed from a single generator's item count, which
    # is deliberately the conservative choice — averaging five generators over
    # the same items does not buy five times the power, and pretending it does
    # would be the same error this package exists to catch.
    chance = _chance_level(votes)
    report.extra["chance_level"] = chance
    floor_test = _floor_above_chance(floor, chance, n, alpha)
    report.extra["floor_vs_chance"] = floor_test
    if not floor_test["above_chance"]:
        report.reasons.append(UndecidableReason.FLOOR_DEGENERATE.value)
        report.extra["floor_error"] = (
            f"the measured floor ({floor:.4f}) is not detectably above the "
            f"chance level ({chance:.4f}) at n={n} items "
            f"(z={floor_test['z']:.2f} < {floor_test['critical_z']:.2f}): the "
            f"generators do not demonstrably reproduce anything, so there is "
            f"no gold for a score to be measured against"
        )
        return report

    # --- Margin ------------------------------------------------------------
    leader_system = max(leaderboard_scores, key=lambda k: leaderboard_scores[k])
    leader_score = float(leaderboard_scores[leader_system])
    report.leader_system = leader_system
    report.leader_score = leader_score
    margin = leader_score - floor
    report.margin = margin

    # --- Tenet 4: power-bounded resolution ---------------------------------
    mde, mde_method = _resolve_mde(
        leaderboard_scores,
        per_item_scores=per_item_scores,
        n_floor_items=report.n_floor_items or n,
        alpha=alpha,
        power=power,
        n_boot=n_boot,
        seed=seed,
    )
    report.mde = mde
    report.extra["mde_method"] = mde_method
    threshold = max(float(delta_declared), mde)
    report.threshold = threshold

    if margin < 0.0:
        report.verdict = Verdict.UNREADABLE
        report.reasons.append(
            f"leader {leader_system} ({leader_score:.4f}) is at or below the "
            f"measured noise floor ({floor:.4f})"
        )
    elif margin < threshold:
        report.verdict = Verdict.MARGINAL
        report.reasons.append(
            f"margin {margin:+.4f} is inside the resolution bound {threshold:.4f}"
        )
    else:
        report.verdict = Verdict.READABLE
    return report


def _chance_level(votes) -> float:
    """Agreement a generator would achieve with no information at all.

    For ``K`` observed labels this is ``1/K``. The floor's job is to represent
    what a rater reproduces; at chance level it reproduces nothing, so no
    score can be said to beat it.
    """
    observed = votes[votes >= 0]
    if observed.size == 0:
        return 1.0
    k = int(np.unique(observed).size)
    return 1.0 / max(k, 1)


def _floor_above_chance(
    floor: float, chance: float, n_items: int, alpha: float
) -> dict:
    """Is the measured floor detectably above chance at this sample size?

    Under the null that the generators carry no information, the floor sits at
    ``chance``. We test the difference with a normal approximation and report
    the statistic so a reader can see why the verdict was refused rather than
    having to trust it. The critical value is two-sided, which makes the check
    stricter about *claiming readability* — the error this package would
    rather make is saying "undecidable".
    """
    from .power import z_for_confidence

    critical = z_for_confidence(alpha)
    if n_items <= 0:
        return {"z": 0.0, "critical_z": critical, "above_chance": False,
                "se": None, "n_items": int(n_items)}
    se = (chance * (1.0 - chance) / n_items) ** 0.5
    if se <= 0.0:
        # A degenerate label set (single label) has no chance variance to test.
        return {"z": 0.0, "critical_z": critical, "above_chance": floor > chance,
                "se": 0.0, "n_items": int(n_items)}
    z = (floor - chance) / se
    if floor > chance:
        needed = int(np.ceil(chance * (1 - chance) * (critical / (floor - chance)) ** 2))
        note = f"{needed} items would put this floor above chance at alpha={alpha}"
    else:
        # At or below chance, more items do not help: the estimate would move
        # toward chance, not away from it. Reporting a sample size here would
        # invite the reader to collect more data to fix something that more
        # data cannot fix.
        needed = None
        note = (
            f"the floor ({floor:.4f}) is at or below the chance level "
            f"({chance:.4f}); no sample size makes it informative"
        )
    return {
        "z": float(z),
        "critical_z": float(critical),
        "se": float(se),
        "n_items": int(n_items),
        "n_items_needed": needed,
        "note": note,
        "above_chance": bool(z >= critical),
    }


def _resolve_mde(
    leaderboard_scores: dict[str, float],
    *,
    per_item_scores: dict[str, np.ndarray] | None,
    n_floor_items: int,
    alpha: float,
    power: float,
    n_boot: int,
    seed: int,
) -> tuple[float, str]:
    """MDE between the top two systems, by the best method available."""
    ranked = sorted(leaderboard_scores.items(), key=lambda kv: kv[1], reverse=True)

    if per_item_scores is not None and len(ranked) >= 2:
        a_name = ranked[0][0]
        b_name = ranked[1][0]
        if a_name in per_item_scores and b_name in per_item_scores:
            a = np.asarray(per_item_scores[a_name], dtype=float)
            b = np.asarray(per_item_scores[b_name], dtype=float)
            if a.shape == b.shape and a.size >= 2:
                return (
                    cluster_bootstrap_mde(a, b, alpha=alpha, power=power,
                                          n_boot=n_boot, seed=seed),
                    "cluster_bootstrap",
                )

    # Fallback: binomial SE of the leader's accuracy. This ignores pairing and
    # is therefore *not* the paired MDE; it is flagged as an approximation so a
    # reader is never misled about how strong the resolution claim is.
    if n_floor_items <= 0:
        return 0.0, "unavailable"
    se = binomial_se(ranked[0][1], int(n_floor_items))
    return mde_from_se(se, alpha=alpha, power=power), "binomial_approx"


def audit_item_sets(
    votes,
    leaderboard_scores: dict[str, float],
    *,
    gold_mode: str = "majority",
    answer_probs=None,
) -> AuditReport:
    """Convenience: compare the *two* gold-construction conventions.

    Readability is not a property of a dataset; it is a property of the
    parameters you declared. Running both conventions and diffing the verdicts
    is the only way to see how much of a ranking is an artefact of the rule.
    """
    del answer_probs  # reserved: probability-weighted gold
    if gold_mode not in ("majority", "loo"):
        raise ValueError("gold_mode must be 'majority' or 'loo'")
    if gold_mode == "majority":
        cons = column_majority(votes)
        observed = cons >= 0
        if not observed.any():
            return _blank(n_items=int(votes.shape[1]),
                          n_generators=int(votes.shape[0]))
        it_ids = set(np.where(observed)[0].tolist())
        fl_ids = set(it_ids)
        return audit_leaderboard(
            votes,
            leaderboard_scores,
            leaderboard_item_ids=it_ids,
            floor_item_ids=fl_ids,
        )
    return audit_leaderboard(votes, leaderboard_scores)


__all__ = [
    "AuditConfig",
    "audit_leaderboard",
    "audit_item_sets",
    "dawid_skene",
]
