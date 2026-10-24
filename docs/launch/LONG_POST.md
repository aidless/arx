# Launch post (draft) — arx

> Status: draft. Do not publish until the pre-flight checklist below passes.
> The post deliberately avoids the word "AGI" and does not claim a research
> result; both are load-bearing for its credibility.

## A leaderboard difference is not a measurement until you know what it can resolve

Most evaluation debates ask whether an effect is big enough. `arx` starts one
question earlier: **can this measurement resolve this comparison at all?**

That sounds like statistics pedantry until you run the same board through two
defensible resolution methods. On the board in `examples/02_audit_a_leaderboard.py`:

```text
without per-item scores : mde=0.0274 (binomial_approx)   -> READABLE
with per-item scores    : mde=0.0743 (cluster_bootstrap) -> MARGINAL
```

Same board, same generators, same items. Nothing about the data changed. The
verdict moved because the resolution bound moved by ~2.7×. If a report only
prints "READABLE", it has quietly answered a different question from the one you
asked.

### Four ways to fail, not one

`arx` returns four verdicts, and the distinctions are the product:

- **READABLE** — the margin clears the floor by more than the apparatus can
  resolve.
- **MARGINAL** — the answer is smaller than we can tell. More items might fix it.
- **UNDECIDABLE** — the question was not asked in a way that permits an answer.
  More items will not fix it; the design has to change.
- **UNREADABLE** — the measurement itself is broken (e.g. the floor and the
  board are computed on different item sets).

There is a fifth check that runs before all of them: **is the floor above chance
at all?** Five generators who each flip a coin agree with each other half the
time, so their floor sits near 0.5 — and any score above that looks like a
landslide. If the floor cannot be distinguished from chance, there is no gold to
measure against, and the verdict is `UNDECIDABLE` no matter how large the margin.

### Why put this in front of an agent?

A loop that proposes its own hypotheses, runs its own experiments, and judges
its own results will optimise the appearance of progress. The gate is the
counterweight. Three design choices matter:

1. **The gate checks the apparatus, not the number.** It never asks whether an
   effect is big enough; it asks whether this design can resolve this
   comparison.
2. **Ordering, not policy, closes look-laundering.** `k_max` is written to the
   ledger before the experiment runs. No policy can stop "run a fixed-sample
   test, miss, then claim you intended one look" — ordering makes the claim
   impossible rather than forbidden.
3. **The scheduler is not fitted to the run it judges.** That would make the
   audit a Goodhart target.

### What it is not

- It does not generate a good hypothesis. The gate decides whether a hypothesis
  was *measured*, not whether it was worth testing.
- `READABLE` does not certify the result. It says the margin cleared the floor
  by more than the apparatus can resolve; it does not say the effect is real or
  that the apparatus measured the right thing.
- The RL/Goodhart demo is a bandit with judge noise, not a research result.
- It is not a general-purpose AGI system, and the README says so.

### Try it without a model or a key

```bash
git clone https://github.com/aidless/arx && cd arx
python3 -m pip install -e .
python3 examples/02_audit_a_leaderboard.py
```

Six examples run offline in seconds. The demo at
`https://aidless.github.io/arx/` shows eight synthetic audit cases with the
reason each refusal fired, plus a real ChaosNLI human-vs-LLM-floor case.

Status: the statistical core and loop are complete and tested (373 tests: 370
passing, 3 skipped; ruff clean; Python 3.12). Literature and Zotero
integrations are implemented but not exercised against live services — `arx
doctor` reports what the install can actually do.

Apache-2.0. I would rather be told where the gate is wrong than get a star.

## Pre-flight checklist (do these before publishing this post)

1. `python3 examples/01_quick_run.py` and `examples/02_...` exit 0 offline.
2. Fresh venv: install from a checkout (or wheel) and run `arx --help`.
3. `python3 -m pytest` reports 370 passed / 3 skipped.
4. `https://aidless.github.io/arx/` returns 200.
5. Decide whether to publish `arx-researcher` on PyPI first; if not, keep the
   install instructions clone-based (do not imply a PyPI package exists).
6. Post once, then wait a day before the short versions.
