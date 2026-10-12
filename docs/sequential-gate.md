# The sequential gate

Same hypothesis, data accumulating, agent deciding when to report. The naive
version of this is a p-hacking engine with a paper trail.

## The failure it prevents

Run a test at `alpha=0.05`, look at the data after every new batch, and report
the first `p < 0.05`. The nominal rate is 5%. The actual rate, simulated under
the null over 200,000 runs:

| looks | nominal | naive `z·√(k/j)` | calibrated |
|---|---|---|---|
| 1 | 0.050 | 0.0502 | 0.0502 |
| 2 | 0.050 | 0.0552 | 0.0497 |
| 5 | 0.050 | 0.0893 | 0.0493 |
| 10 | 0.050 | **0.1462** | 0.0489 |

The textbook shape roughly triples the family-wise error rate at ten looks.
It is a *shape*, not a calibrated boundary: it comes from a normal
approximation that is only valid for a single terminal look, and applying it to
early looks hands out free error.

## The calibration

Solve for the smallest scale `c` such that the union bound holds:

```
sum_{j=1..k}  2 * (1 - Phi(c * sqrt(k / j)))  <=  alpha
```

The bound holds under **any** dependence structure between the looks. That is
the point, and it is also the cost: it needs no modelling assumption, and it is
conservative.

| looks | scale `c` | union bound | first boundary | last boundary |
|---|---|---|---|---|
| 1 | 1.9600 | 0.05000 | 1.9600 | 1.9600 |
| 2 | 2.0014 | 0.05000 | 2.8304 | 2.0014 |
| 5 | 2.1766 | 0.05000 | 4.8670 | 2.1766 |
| 10 | 2.3443 | 0.05000 | 7.4133 | 2.3443 |

At `k=10` the final look needs `|z| >= 2.34`, not `1.96`. A study that would
have been significant at the fixed-sample boundary is not significant here,
and that is the price of having been allowed to look nine times.

## The bias, stated in the report

Sequential stopping inflates the effect estimate. Conditional on having crossed
a boundary, the observed effect is on average larger than the true effect — a
study that stops early because it looked good looked good partly by luck.

`SequentialState.summary()` always carries:

```json
{
  "effect_estimate_is_biased": true,
  "effect_bias_note": "estimate is conditional on early stopping and is biased
                       away from zero; report the final-look estimate or a
                       bias-adjusted interval, not this value"
}
```

The field is always present, so a downstream consumer cannot pick up the number
without also picking up that fact.

## Reference

```python
from arx.audit.sequential import SequentialState, obf_boundaries, calibrate_scale

state = SequentialState(k_max=5, alpha=0.05, label="reranker beats baseline")
for j in range(5):
    verdict = state.observe(z_j)          # None while the gate continues
    if verdict is not None:
        break

state.summary()                          # looks, boundaries, verdict, bias flag
```

`obf_boundaries(k, alpha)` returns the full schedule. `calibrate_scale(k, alpha)`
returns the solved `c`. `union_bound_rate(c, k)` reports the achieved bound, so
the calibration can be checked rather than trusted. `effective_alpha(k, alpha)`
reports the per-look rate the naive approach would need to hit the nominal
family rate, which is the cleanest way to see how much it is actually spending.

## Correlation-aware spending

The union-bound calibration costs power because it assumes nothing about the
dependence between looks. For a *known* correlation structure — and for
bootstrap/count statistics over accumulating iid data, `corr(z_i, z_j) =
sqrt(min/max)` is derivable rather than assumed — an exact
Lan-DeMets spending function recovers most of that gap.

This package ships the assumption-free version only. The reason is in the
tests: `test_sequential.py` pins the simulated FWER of the calibrated boundary
to `<= alpha + 0.005` at `k ∈ {1,2,5,10}` over 20,000 null runs. A
correlation-aware boundary would need the same witness test, and shipping both
without it would mean shipping one path that is known to be conservative and
one that is merely believed to be exact.
