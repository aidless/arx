"""Example 03 -- what a noise floor does to a learner, in plain numpy.

No RL library is needed: this is an epsilon-greedy bandit whose *observed*
reward is the true reward plus judge noise. It optimises the observation, which
is what a model trained against an LLM judge does.

The point is not that it learns badly. It is that some arm rankings are not
recoverable from this apparatus at any sample size, and the run reports which.

Run:  PYTHONPATH=src python3 examples/03_goodhart_rl.py
"""

from __future__ import annotations

import _support as S

from arx.audit import audit_leaderboard
from arx.audit.power import mde_from_se
from arx.rl.goodhart_env import simulate_bandit, unrankable_pairs


def main() -> None:
    S.rule("A bandit optimising an observed reward that carries judge noise")

    outcome = simulate_bandit(n_arms=10, n_rounds=4000, sigma=0.05, seed=0)
    true_quality = outcome.true_quality

    print("  true quality per arm (the thing we actually care about):")
    for i, q in enumerate(true_quality):
        print(f"    arm {i:>2}  {q:.4f}")
    print(f"\n  judge noise sigma        : {outcome.sigma:.4f}")
    print(f"  best arm (true)          : {outcome.best_arm}")
    print(f"  arm it settled on        : {outcome.chosen_arm}  "
          f"({outcome.true_quality[outcome.chosen_arm]:.4f})")
    print(f"  ranked correctly         : {outcome.ranked_correctly}")
    print(f"  regret at the end        : {outcome.regret:+.4f}")
    print(f"  empirical noise floor    : {outcome.noise_floor:.4f}")

    S.rule("The arms it cannot tell apart, at any sample size")
    pairs = unrankable_pairs(true_quality, outcome.sigma)
    print(f"  {len(pairs)} of {len(true_quality) * (len(true_quality) - 1) // 2} "
          f"arm pairs differ by less than sigma:")
    for a, b in pairs[:12]:
        print(f"    arm {a:>2} ({true_quality[a]:.4f})  vs  arm {b:>2} "
              f"({true_quality[b]:.4f})   gap={abs(true_quality[a] - true_quality[b]):.4f}")
    if len(pairs) > 12:
        print(f"    ... and {len(pairs) - 12} more")
    print("""
    These are not hard comparisons, they are impossible ones. With noise
    sigma, two arms whose true qualities differ by less than sigma cannot be
    ordered by any amount of data from this apparatus. A leaderboard that
    ranks them is reporting its own rounding, not the arms.""")

    S.rule("The same question through the auditor")
    # Five gold generators grading 400 items; two arms whose accuracies differ
    # by a hair. The apparatus is identical in both calls below -- only the gap
    # between the two arms changes.
    import numpy as np

    rng = np.random.default_rng(11)
    n_items = 400
    truth = rng.integers(0, 2, size=n_items)
    votes = np.array([
        np.where(rng.random(n_items) < 0.90, truth, 1 - truth) for _ in range(5)
    ])

    def two_arms(accuracy_a: float, accuracy_b: float) -> dict[str, float]:
        a = np.where(rng.random(n_items) < accuracy_a, truth, 1 - truth)
        b = np.where(rng.random(n_items) < accuracy_b, truth, 1 - truth)
        return {
            "armA": float((a == truth).mean()),
            "armB": float((b == truth).mean()),
        }

    for label, (acc_a, acc_b) in (
        ("a 0.3-point lead", (0.947, 0.944)),
        ("a 5-point lead", (0.950, 0.900)),
    ):
        scores = two_arms(acc_a, acc_b)
        # The floor is the same in both calls -- it depends on the generators,
        # not on the contestants -- so only the margin moves.
        report = audit_leaderboard(
            votes, scores,
            judge_ids={"j1", "j2", "j3", "j4", "j5"},
            contestant_ids={"armA", "armB"},
            delta_declared=0.02,
        )
        observed = abs(scores["armA"] - scores["armB"])
        print(f"\n  {label}: observed gap {observed:.4f}, "
              f"armA={scores['armA']:.4f} armB={scores['armB']:.4f}")
        print(f"    floor {report.floor_estimate:.4f}  margin {report.margin:+.4f}  "
              f"mde {report.mde:.4f}  ->  {report.verdict.value}")
        for reason in report.reasons:
            print(f"    {reason}")

    print("""
    Both came back READABLE, and that is worth sitting with rather than
    dressing up. The fallback MDE is a binomial standard error on the leader's
    own accuracy, so it measures how well the leader is resolved *against the
    floor* -- not against the runner-up. The 0.3-point lead is not a resolved
    comparison between two arms; the auditor simply was not asked that
    question.

    The honest fix is to hand the auditor per-item scores, which switches it
    to the paired cluster bootstrap. Example 02 does that; the flag to watch is
    ``mde_method``, which reads ``binomial_approx`` here and
    ``cluster_bootstrap`` there. An MDE that silently answers a different
    question than the one you asked is exactly the failure this package
    exists to make visible, so it is labelled rather than hidden.""")

    S.rule("What sample size would it take?")
    # The MDE here is set by the *floor's* accuracy and the item count, not by
    # the gap between the two arms: the fallback treats the leader's accuracy as
    # a binomial proportion, which is the resolution of the apparatus itself.
    se = (0.25 / 400) ** 0.5
    resolution = mde_from_se(se, alpha=0.05, power=0.8)
    print(f"  at n=400 the apparatus resolves differences of about "
          f"{resolution:.4f} or larger")
    print("  items needed to resolve a given gap:")
    for gap in (0.10, 0.05, 0.02, 0.01):
        items = int(np.ceil(400 * (resolution / gap) ** 2))
        print(f"    gap {gap:.3f}  ->  about {items:>7} items")
    print("""
    Note the shape: halving the gap quadruples the items. The comparison that
    actually binds here is between the leader and the *floor*, not between the
    two arms -- which is why the narrow lead above still cleared the bar once
    the leader was far enough above the noise.""")


if __name__ == "__main__":
    main()
