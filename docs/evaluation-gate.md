# The evaluation gate

The gate decides whether a leaderboard means anything. It is four checks and a
verdict, and every verdict it can produce is a reason to *not* believe
something.

```
READABLE       the margin clears the floor by more than the apparatus can resolve
MARGINAL       positive, but inside the resolution bound
UNREADABLE     measured, and the ranking does not hold (leader at or below floor)
UNDECIDABLE    the premises fail; there is no estimate to interpret
STOP_EARLY     a sequential boundary was crossed before the final look
```

`UNDECIDABLE` is not a polite `MARGINAL`. It says the question was not asked in
a way that permits an answer — two generators, overlapping actors, mismatched
denominators, a floor indistinguishable from chance. `MARGINAL` says the answer
is "smaller than we can tell".

## The floor

`loo_floor(votes)` computes, for each generator `g`, the fraction of items where
`g` agrees with the majority of the *other* generators:

```
others = delete(votes, g)
floor[g] = mean(votes[g] == majority(others))
```

Three properties are load-bearing:

* **`m >= 3`.** With `m = 2`, deleting one generator leaves one, and agreement
  with "the rest" is agreement with itself. The method returns `NaN` and the
  auditor reports `floor_degenerate` rather than a number.
* **The denominator is the floor's items**, so `n_floor_items` is reported
  alongside the MDE. A floor computed on 1,520 items and a board computed on
  1,084 of them is `domain_mismatch`, not a negative effect.
* **It is compared against the chance rate.** `se = sqrt(chance*(1-chance)/n)`,
  `z = (floor - chance)/se`, refused when `z < 1.96`. When the floor is at or
  below chance the report says so in plain words, because more items will not
  help: the estimate would move *toward* chance, not away from it.

## The MDE, and why its method name matters

The MDE is the smallest margin this apparatus could detect at the declared
power. Two estimators are available:

| method | when | what it measures |
|---|---|---|
| `binomial_approx` | no per-item scores | the leader's own accuracy, as an independent binomial proportion |
| `cluster_bootstrap` | per-item scores supplied | the paired difference between the top two, resampled at item level |

They are not interchangeable, and the difference is not small. On a board with
a genuine separation:

```
without per_item_scores : mde=0.0274  (binomial_approx)  -> READABLE
with per_item_scores    : mde=0.0743  (cluster_bootstrap) -> MARGINAL
```

The paired estimator sees the correlation between the two systems' per-item
outcomes. When two systems agree item-by-item more than chance would predict,
the paired difference has a *larger* standard error than the independent
approximation assumes — the fallback was optimistic. Sampling noise in which
items you drew is shared by both systems, and the approximation throws that
away.

This is why `mde_method` is always in the report. A resolution bound that
quietly answers a different question than the one asked is exactly the failure
this package exists to make visible, and it is not fixed by picking the
friendlier number.

## Known limits of the current MDE

* `binomial_approx` uses `n_floor_items`, but the leader's accuracy is taken
  from `scores`, so the pairing is nominal. It is a resolution claim of the
  *apparatus*, not of the *comparison*.
* The two fallbacks disagree by a factor of ~2.7 on the same board. Until
  `per_item_scores` is the default, treat any `binomial_approx` verdict as an
  upper bound on readability.
* `per_item_scores` requires the experiment to emit them. `arx.examples`
  shows both branches so the difference is visible rather than assumed.

## The four tenets, as refusals

| tenet | refuse when | code |
|---|---|---|
| actor-disjointness | `judge_ids ∩ contestant_ids ≠ ∅` | `actors_overlap` |
| common denominator | `leaderboard_item_ids ≠ floor_item_ids` | `domain_mismatch` |
| multi-generator | `m < 3` | `floor_degenerate` |
| power-bounded | `0 ≤ margin < threshold` | `MARGINAL`, with the bound in the reason |

Plus the two that are not about the *board* but about the *apparatus*:
`floor_vs_chance` (the floor is not above chance) and `mde_method` (the
resolution bound is not what you think it is).

## Worked example

```python
from arx.audit import audit_leaderboard
import numpy as np

# five generators, 400 items, 93% agreement with the latent truth
rng = np.random.default_rng(0)
truth = rng.integers(0, 2, size=400)
votes = np.array([np.where(rng.random(400) < 0.93, truth, 1 - truth)
                  for _ in range(5)])

report = audit_leaderboard(
    votes,
    {"strong": 0.96, "weak": 0.60},
    judge_ids={"j1", "j2", "j3", "j4", "j5"},
    contestant_ids={"strong", "weak"},
    delta_declared=0.02,
)
```

```
Verdict: READABLE
Reasons: (none)
Floor estimate: 0.9210  (loo, m=5, n=400)
Margin over floor: 0.0390
Declared delta: 0.0200
Minimum detectable effect: 0.0274  (binomial_approx)
```

The same call with `generators=2` returns `UNDECIDABLE / floor_degenerate`.
The same board on 120 items returns `MARGINAL`, with
`margin +0.0267 is inside the resolution bound 0.0501`. The gap did not change;
the resolution did.

Note also that the *floor itself* carries uncertainty. At 120 items the floor's
own standard error is `sqrt(0.25/120) = 0.0456`, which is why the chance check
demands `n >= 228` before it will call a 0.93 floor above chance at all. Fewer
items and the same honest verdict is not `MARGINAL` but `UNDECIDABLE` —
the generators have not demonstrably reproduced anything.
