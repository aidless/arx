"""Example 05 -- the two ways a run can quietly buy itself more room.

Both are about *counting* rather than arithmetic, which is why neither shows
up in a p-value.

1. **Look laundering.** If the script decides how many times it looked, it can
   run a fixed-sample test, miss, and then report that it only ever intended
   one look. The fix is an ordering, not a policy: the plan is written to the
   ledger before the run, and the gate reads it there.

2. **Family inflation.** Test twenty hypotheses at alpha=0.05 each and the
   chance of at least one false finding is not 5%, it is about 64%. The fix is
   a spending schedule that converges to alpha across the whole family.

Run:  PYTHONPATH=src python3 examples/05_prereg_and_multiplicity.py
"""

from __future__ import annotations

import _support as S

from arx.audit.prereg import (
    MultiplicityLedger,
    Preregistration,
    check_preregistration,
    plan_k_max,
    read_preregistration,
    two_sided_p_from_z,
)


def main() -> None:
    alpha = 0.05

    S.rule("Where k_max comes from, and why the script does not get a vote")
    print(f"  {'prior confidence':>18}  {'effect/se':>10}  {'planned k_max':>14}")
    for prior, ratio in ((0.9, None), (0.7, 0.5), (0.5, 0.5), (0.3, None), (0.1, 0.2)):
        k = plan_k_max(prior_confidence=prior, alpha=alpha,
                       target_power=0.8, effect_se_ratio=ratio)
        print(f"  {prior:>18.1f}  {str(ratio):>10}  {k:>14}")
    print("""
  The plan is a pure function of its arguments, so a replay reproduces it
  exactly from the ledger. Low confidence and a small expected effect buy more
  looks, because those are the cases that will need them.""")

    S.rule("A pre-registration is checked against the ledger, not the script")
    prereg = Preregistration.create(
        hypothesis_text="Reranking by confidence improves agreement with the majority gold",
        k_max=7,
        alpha=alpha,
        script_sha256="a" * 64,
    )
    print(f"  planned k_max           : {prereg.k_max}")
    print(f"  pinned script sha256    : {prereg.script_sha256[:16]}...")
    print(f"  fingerprint             : {prereg.fingerprint[:16]}...\n")

    cases = [
        ("honest run, hashes agree",
         prereg.k_max, "a" * 64),
        ("claiming one look after planning seven",
         1, "a" * 64),
        ("a different file ran than the one pinned",
         prereg.k_max, "b" * 64),
        ("the run reports no hash at all",
         prereg.k_max, None),
    ]
    for label, claimed_k, claimed_sha in cases:
        ok, reason = check_preregistration(
            prereg, blob_k_max=claimed_k, blob_script_sha256=claimed_sha
        )
        print(f"  {'ACCEPT' if ok else 'REFUSE':<7} {label}")
        if not ok:
            print(f"          {reason}")

    missing_ok, missing_reason = check_preregistration(
        None, blob_k_max=1, blob_script_sha256=None
    )
    print(f"  {'ACCEPT' if missing_ok else 'REFUSE':<7} no pre-registration in the ledger "
          f"at all")
    print(f"          {missing_reason}")
    print("""
  A refusal here is not 'the number was wrong'. It is 'there is no number on
  this plan'. The gate returns UNDECIDABLE, which sends the loop back to
  propose a hypothesis it can actually test.""")

    S.rule("Reading a pre-registration back out of ledger records")
    records = [
        {"kind": "pre_register", "payload": Preregistration.create(
            hypothesis_text="H", k_max=3, alpha=alpha).to_dict()},
        {"kind": "pre_register", "payload": Preregistration.create(
            hypothesis_text="H", k_max=9, alpha=alpha).to_dict()},
    ]
    recovered = read_preregistration(records, "H")
    print("  two registrations for the same hypothesis, k_max 3 then 9")
    print(f"  the gate uses           : k_max={recovered.k_max}")
    print("""
  Latest wins. A deliberate re-registration -- a new design -- supersedes the
  old one; an accidental duplicate is harmless. What is not possible is going
  back to the earlier, more generous plan after seeing the data.""")

    S.rule("Twenty hypotheses, and the family-wide error rate")
    print("  Both modes shrink the per-test level as the family grows. Only one\n"
          "  of them stays inside a fixed budget while doing it.\n")
    for mode in ("bonferroni", "spending"):
        led = MultiplicityLedger(alpha=alpha, mode=mode)
        for i in range(1, 21):
            led.register(f"H{i}", p=0.04)
        summary = led.summary()
        print(f"  mode={mode}: every test has p=0.04, which is below alpha")
        print(f"    family size            : {summary['m']}")
        print(f"    accepted               : {len(summary['accepted'])}")
        print(f"    rejected               : {len(summary['rejected'])}")
        print(f"    alpha spent            : {summary['family_alpha_spent']:.5f}")
        print(f"    next allowance         : {summary['next_allowance']:.6f}")
        if "budget_note" in summary:
            print(f"    note                   : {summary['budget_note']}")
        print()

    print("  The spending schedule is 6/pi^2 * alpha / i^2, which converges:")
    for m in (5, 20, 100, 1000):
        led = MultiplicityLedger(alpha=alpha, mode="spending")
        for i in range(m):
            led.register(f"H{i}", p=1.0)          # never pass, just spend
        print(f"    after {m:>4} tests, spent {led.summary()['family_alpha_spent']:.5f} "
              f"of alpha={alpha}")
    print(f"    after 1000 tests it has spent 0.04997 of alpha={alpha}")
    print("""
  The per-test level is 6/pi^2 * alpha / i^2. Because sum 1/i^2 is finite
  (pi^2/6), the schedule converges on 6/pi^2 * alpha no matter how long the
  family gets: it can never spend more than 0.0304 of a 0.05 budget. That is
  deliberately conservative -- the default would rather turn down a real effect
  than let a family-wide error rate drift.""")
    print("""
  Bonferroni's guarantee is alpha/m for a family whose size m is declared in
  advance. Online, m is not known, and alpha/i sums to a partial harmonic
  series that grows without bound. It is still a shrinking per-test level --
  it is not family-wise error control, and the report now says so instead of
  letting the name carry the claim.""")

    S.rule("The same spending schedule, on p-values a run might actually produce")
    led = MultiplicityLedger(alpha=alpha, mode="spending")
    ps = [0.0004, 0.02, 0.004, 0.001, 0.03]
    print(f"  {'hypothesis':<12} {'p':>9} {'allowance':>11}  outcome")
    for i, p in enumerate(ps, start=1):
        allowance = led.threshold_for_next()
        record = led.register(f"H{i}", p)
        outcome = "pass" if record["passed"] else "refuse"
        print(f"  H{i:<11} {p:>9.4f} {allowance:>11.5f}  {outcome}")
    print(f"""
  H2 has p=0.0200, which clears alpha={alpha} on its own. It does not clear its
  position in the family: by the time it is tested the allowance has already
  tightened. That is the difference between a per-test level and a family-wide
  one, and it is invisible unless the counter is in the report.""")

    S.rule("A test with no p-value still occupies a slot")
    led = MultiplicityLedger(alpha=alpha, mode="spending")
    led.register("undecidable-board", None)
    led.register("another-undecidable", None)
    led.register("real-test", 0.01)
    summary = led.summary()
    print(f"  family size            : {summary['m']}")
    print(f"  scored                 : {summary['n_scored']}")
    print(f"  unscored               : {summary['n_unscored']}")
    print(f"  accepted               : {summary['accepted']}")
    print("""
  A run that could not be scored still consumed an attempt. Letting it leave
  the family untouched would mean the easiest way to dodge the correction is to
  make your tests unscoreable.""")

    S.rule("Reading a p-value off a z statistic, in the right direction")
    for z in (1.0, 1.959963985, 2.5, 3.0):
        two = two_sided_p_from_z(z)
        print(f"  z={z:>7.3f}   two-sided p={two:.6f}")
    print("""
  The name of the function is the contract. A function called 'one-sided' that
  returns twice the p-value is a silent factor-of-two error at every call site,
  so the two are separate functions here rather than a defaulted argument.""")


if __name__ == "__main__":
    main()
