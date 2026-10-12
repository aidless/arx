"""Example 02 -- the audit on its own, one call per tenet.

``arx.audit`` has no dependency on the agent, the model or the network. This
example uses nothing else, and shows the four checks in the order they run.
Each one is a *reason to refuse*, so a refusal is the tool working.

Run:  PYTHONPATH=src python3 examples/02_audit_a_leaderboard.py
"""

from __future__ import annotations

import _support as S
import numpy as np

from arx.audit import audit_leaderboard, identifiability_diagnostics, loo_floor
from arx.audit.verdict import Verdict


def votes(n_generators: int, n_items: int = 400, agreement: float = 0.93, seed: int = 0):
    """A generator pool: each generator matches the latent label with
    probability ``agreement``."""
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 2, size=n_items)
    return np.array([
        np.where(rng.random(n_items) < agreement, truth, 1 - truth)
        for _ in range(n_generators)
    ])


BOARD = {"strong": 0.96, "weak": 0.60}


def show(title: str, report) -> None:
    print(f"\n  {title}")
    print(f"    verdict  : {report.verdict.value}")
    if report.margin is not None:
        print(f"    floor    : {report.floor_estimate:.4f} "
              f"({report.floor_method}, m={report.n_generators}, n={report.n_items})")
        print(f"    margin   : {report.margin:+.4f}   threshold: {report.threshold:.4f} "
              f"= max(delta={report.delta_declared}, mde={report.mde:.4f})")
    if report.reasons:
        for reason in report.reasons:
            print(f"    reason   : {reason}")


def main() -> None:
    S.rule("A leaderboard is readable only if four things hold")

    print("""
  1. actor-disjointness   the gold producers are not the contestants
  2. common denominator   the floor and the board are computed on the same items
  3. multi-generator      at least three gold producers
  4. power-bounded        the margin exceeds what this sample can resolve
""")

    # ------------------------------------------------------------------ 4 ok
    S.rule("Tenet 3 and 4: five generators, a clear separation")
    report = audit_leaderboard(
        votes(5, agreement=0.93),
        BOARD,
        judge_ids={"g1", "g2", "g3", "g4", "g5"},
        contestant_ids={"strong", "weak"},
        delta_declared=0.02,
    )
    show("five generators, agreement 0.93", report)
    print("\n    READABLE. But note what that word does not mean: it says the")
    print("    gap is larger than the measurement error. It says nothing about")
    print("    whether the measurement is the right one.")

    # --------------------------------------------------------------- tenet 3
    S.rule("Tenet 3: two generators are not enough")
    show("two generators", audit_leaderboard(
        votes(2),
        BOARD,
        judge_ids={"g1", "g2"},
        contestant_ids={"strong", "weak"},
    ))
    print("\n    With m=2, leave-one-out leaves a single rater: 'agreement with")
    print("    the others' degenerates into agreement with itself. There is no")
    print("    estimator, so the honest verdict is UNDECIDABLE, not a number.")

    S.rule("Tenet 1: the board and the floor must share their items")
    show("floor on 400 items, board on 150", audit_leaderboard(
        votes(5),
        BOARD,
        judge_ids={"g1", "g2", "g3", "g4", "g5"},
        contestant_ids={"strong", "weak"},
        leaderboard_item_ids=set(range(150)),
        floor_item_ids=set(range(400)),
    ))
    print("\n    Two different denominators. The difference between the two")
    print("    statistics is not an effect size, it is an artefact of which items")
    print("    each was computed on.")

    S.rule("Tenet 2: the judge cannot be one of the contestants")
    show("'strong' is also a gold generator", audit_leaderboard(
        votes(5),
        BOARD,
        judge_ids={"strong", "g2", "g3", "g4", "g5"},
        contestant_ids={"strong", "weak"},
    ))
    print("\n    If a contestant helped define the gold, its own score partly")
    print("    measures how well it agrees with itself.")

    # --------------------------------------------------------------- tenet 4
    S.rule("Tenet 4: the margin has to beat what the sample can resolve")
    thin = votes(5, n_items=120, agreement=0.93, seed=3)
    show("same board, only 120 items", audit_leaderboard(
        thin, BOARD,
        judge_ids={"g1", "g2", "g3", "g4", "g5"},
        contestant_ids={"strong", "weak"},
        delta_declared=0.02,
    ))
    print("\n    The gap did not change; the resolution did. A number below the")
    print("    minimum detectable effect is not a small effect, it is no")
    print("    measurement at all.")

    # -------------------------------------------- the case that looks best
    S.rule("The trap: gold producers that carry no signal")
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 2, size=(5, 400))          # pure coin flips
    report = audit_leaderboard(
        noise,
        {"padded": 0.965, "plain": 0.70},              # a very large margin
        judge_ids={"g1", "g2", "g3", "g4", "g5"},
        contestant_ids={"padded", "plain"},
        delta_declared=0.02,
    )
    show("five generators agreeing only by chance", report)
    print(f"\n    floor vs chance : {report.extra.get('floor_vs_chance')}")
    print("""
    This is the one that matters. Five independent coin-flippers agree with
    each other about half the time, so the floor sits near 0.5 -- and any
    score above that looks like a landslide. A margin-only check calls it
    READABLE with a huge margin. The floor is tested against the chance rate
    it would have if the generators carried no signal at all, and when it
    cannot be distinguished from chance the verdict is UNDECIDABLE.

    The resolution comes from the number of ITEMS, not the number of
    generators: the floor's variability is driven by which items were drawn,
    and that variation is shared by every generator.""")
    print(f"    n_items_needed to call this floor above chance: "
          f"{report.extra.get('floor_vs_chance', {}).get('n_items_needed')}")
    print(f"    {report.extra.get('floor_vs_chance', {}).get('note')}")

    # ----------------------------------------------------------- raw parts
    S.rule("Unpaired vs paired resolution: read mde_method")
    pool = votes(5, agreement=0.93)
    rng = np.random.default_rng(5)
    truth = rng.integers(0, 2, size=400)
    per_item = {
        "strong": (np.where(rng.random(400) < 0.96, truth, 1 - truth) == truth).astype(float),
        "weak": (np.where(rng.random(400) < 0.60, truth, 1 - truth) == truth).astype(float),
    }
    unpaired = audit_leaderboard(
        pool, BOARD,
        judge_ids={"g1", "g2", "g3", "g4", "g5"},
        contestant_ids={"strong", "weak"},
    )
    paired = audit_leaderboard(
        pool, BOARD,
        judge_ids={"g1", "g2", "g3", "g4", "g5"},
        contestant_ids={"strong", "weak"},
        per_item_scores=per_item,
    )
    print(f"  without per-item scores : mde={unpaired.mde:.4f} "
          f"({unpaired.extra.get('mde_method')})  -> {unpaired.verdict.value}")
    print(f"  with per-item scores    : mde={paired.mde:.4f} "
          f"({paired.extra.get('mde_method')})  -> {paired.verdict.value}")
    print(f"""
    Same board, same generators, same items. The verdict moved from
    {unpaired.verdict.value} to {paired.verdict.value} because the MDE moved from
    {unpaired.mde:.4f} to {paired.mde:.4f}, and nothing about the data changed.

    The fallback treats the leader's accuracy as an independent binomial
    proportion. The paired bootstrap resamples items and keeps the two systems'
    outcomes on the same item together, so it sees the correlation the fallback
    assumes away. When the two systems agree item-by-item more than chance
    would predict, the paired difference has a *larger* standard error than the
    independent approximation suggests -- the fallback was optimistic.

    An MDE that quietly answers a different question than the one asked is the
    failure this package exists to make visible. It is not resolved by picking
    the friendlier number; it is resolved by naming the method, which is why
    ``mde_method`` is always in the report.""")

    S.rule("The pieces, if you want them separately")
    pool = votes(5, agreement=0.93)
    floors = loo_floor(pool)
    print(f"  loo_floor per generator : {np.round(floors, 4).tolist()}")
    print(f"  median                  : {float(np.median(floors)):.4f}")
    diagnostics = identifiability_diagnostics(pool)
    print(f"  identifiability         : identifiable={diagnostics['identifiable']} "
          f"pairwise_agreement={diagnostics['pairwise_agreement_mean']:.4f}")
    print(f"  same call, noisy pool   : "
          f"{identifiability_diagnostics(noise)['undecidable']}")
    print(f"\n  Verdicts available      : {[v.value for v in Verdict]}")


if __name__ == "__main__":
    main()
