"""Example 04 -- why a sequential gate has to be calibrated, not copied.

The textbook O'Brien-Fleming shape ``z * sqrt(k/j)`` is a *shape*, not a
calibrated boundary. Plugging it into ``alpha=0.05`` and looking at each look
individually gives a family-wise error rate well above 5%, because the looks
are not independent tests you get to keep all of.

This example measures the inflation by simulation, then shows the calibrated
alternative and how much it costs.

Run:  PYTHONPATH=src python3 examples/04_sequential_gate.py
"""

from __future__ import annotations

import math
from statistics import NormalDist

import _support as S
import numpy as np

from arx.audit.sequential import (
    SequentialState,
    calibrate_scale,
    effective_alpha,
    fixed_sample_boundary,
    obf_boundaries,
    union_bound_rate,
    z_from_effect,
)

_N = NormalDist()
TRIALS = 200_000


def naive_boundaries(k_max: int, alpha: float = 0.05) -> list[float]:
    """The shape everyone writes down: z_{alpha/2} * sqrt(k/j)."""
    z = _N.inv_cdf(1 - alpha / 2)
    return [z * math.sqrt(k_max / j) for j in range(1, k_max + 1)]


def simulated_fwer(boundaries: list[float], trials: int = TRIALS,
                   seed: int = 0) -> float:
    """Fraction of null runs that cross at least one boundary.

    Under the null every look is a standard normal draw. This is the whole
    question: does the family-wise error rate stay at alpha?
    """
    rng = np.random.default_rng(seed)
    k = len(boundaries)
    z = rng.standard_normal((trials, k))
    crossed = (np.abs(z) >= np.asarray(boundaries)).any(axis=1)
    return float(crossed.mean())


def main() -> None:
    alpha = 0.05

    S.rule("The shape that looks right and is not")
    print("  O'Brien-Fleming specifies a *spending* rule, not a critical value.")
    print("  Writing z*sqrt(k/j) and stopping when |z| crosses it gives:\n")
    print(f"  {'looks':>6}  {'nominal':>8}  {'naive FWER':>11}  {'calibrated FWER':>16}")
    for k in (1, 2, 5, 10):
        naive = naive_boundaries(k, alpha)
        calibrated = obf_boundaries(k, alpha)
        print(f"  {k:>6}  {alpha:>8.3f}  "
              f"{simulated_fwer(naive):>11.4f}  "
              f"{simulated_fwer(calibrated):>16.4f}")
    print("""
  The naive column is the number that matters. At ten looks it is roughly
  three times the nominal rate: an agent that checks often and reports the
  first crossing is not running a test at 5%, it is running one at ~15%.""")

    S.rule("Where the calibrated boundary comes from")
    print("  We want the smallest scale c such that the union bound holds:\n")
    print("      sum_j  2 * (1 - Phi(c * sqrt(k/j)))  <=  alpha\n")
    print("  That bound holds whatever the dependence between looks is, which")
    print("  is why it is the default: it costs power, and it does not need a")
    print("  modelling assumption to be true.\n")
    print(f"  {'looks':>6}  {'scale c':>8}  {'union bound':>12}  "
          f"{'first':>7}  {'last':>7}")
    for k in (1, 2, 5, 10):
        c = calibrate_scale(k, alpha)
        bounds = obf_boundaries(k, alpha)
        print(f"  {k:>6}  {c:>8.4f}  {union_bound_rate(c, k):>12.5f}  "
              f"{bounds[0]:>7.4f}  {bounds[-1]:>7.4f}")
    print(f"""
  Every boundary is at least the fixed-sample value {fixed_sample_boundary(alpha):.4f}.
  That is not a bug: a look taken before the data is complete must clear a
  higher bar to leave the same total error budget. The last look is the
  cheapest, and it is still not free.""")

    S.rule("What that costs, in z")
    print(f"  {'looks':>6}  {'naive final':>12}  {'calibrated final':>17}  "
          f"{'naive per-look alpha':>21}  {'one-shot alpha':>15}")
    for k in (2, 5, 10):
        print(f"  {k:>6}  {naive_boundaries(k, alpha)[-1]:>12.4f}  "
              f"{obf_boundaries(k, alpha)[-1]:>17.4f}  "
              f"{effective_alpha(k, alpha):>21.5f}  {alpha:>15.3f}")
    print("""
  At k=10 the final look needs |z| >= 2.34 rather than 1.96. A study that
  would have been significant at the fixed-sample boundary is not significant
  here -- and that is the price of having been allowed to look nine times.""")

    S.rule("One hypothesis, five looks -- the loop in miniature")
    state = SequentialState(k_max=5, alpha=alpha, label="reranker beats baseline by 0.03")
    effect, se = 0.060, 0.030
    print(f"  true effect {effect}, se {se} -- z grows as data accumulates\n")
    for j in range(1, 6):
        z = z_from_effect(effect * (j / 5), se)
        boundary = state.next_boundary
        verdict = state.observe(z)
        marker = "  <- STOP" if verdict is not None else ""
        state_word = verdict.value if verdict is not None else "(continue)"
        print(f"    look {j}  z={z:+.4f}  boundary={boundary:.4f}  "
              f"{state_word}{marker}")
        if state.done:
            break
    for key, value in state.summary().items():
        print(f"    {key:<26} {value}")
    print("""
  Read the last two lines. A sequential stop inflates the effect estimate --
  conditional on having crossed a boundary, the observed effect is on average
  larger than the true one. The summary always carries
  ``effect_estimate_is_biased`` so a downstream reader cannot pick up the
  number without also picking up that fact.""")

    S.rule("The same effect, one look instead of five")
    fixed = SequentialState(k_max=1, alpha=alpha)
    z = z_from_effect(effect, se)
    print(f"  z={z:+.4f}  boundary={fixed.next_boundary:.4f}  "
          f"-> {fixed.observe(z).value}")
    print(f"""
  With one look the design would have stopped at |z| >= {fixed_sample_boundary(alpha):.4f}.
  The sequential version asks for more. Both control the error rate at
  {alpha}; they differ in what they let you get away with along the way.""")


if __name__ == "__main__":
    main()
