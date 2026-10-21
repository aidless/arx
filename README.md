# arx

**An autonomous research loop that will not write a claim it cannot support.**

`arx` reads literature, proposes hypotheses, runs experiments in a sandbox, and
reports findings — but only after a statistical gate decides the measurement
could actually resolve the thing being claimed.

[![CI](https://github.com/aidless/arx/actions/workflows/ci.yml/badge.svg)](https://github.com/aidless/arx/actions/workflows/ci.yml)

```
pytest: 373 passed   |   ruff: clean   |   examples: 6, all runnable offline
```

## The one-sentence version

A leaderboard's rank difference is readable only if it exceeds the noise
floor of the labels it was measured against — and most of them don't.

## Install and run

```bash
cd arx
python3 -m pip install -e .
python3 examples/01_quick_run.py     # no API key, no network, no model
```

The examples are the fastest way to see what the gate does. Each one runs
offline, in a few seconds, and prints the numbers its own code produced.

The `arx` command lands wherever pip puts user scripts. On this machine that is
`~/.local/bin`, which is often not on `PATH`, so either add it or call the CLI as
a module:

```bash
export PATH="$HOME/.local/bin:$PATH"    # if pip installed to user site
arx --help
# or, no PATH change needed:
python3 -m arx.cli --help
```

Python 3.12+ is the only supported version and the only one required — the core
package imports nothing beyond `numpy` and the standard library.

## Audit a leaderboard

```bash
arx audit board.json --delta 0.02
arx audit board.json --delta 0.02 --html report.html
```

`board.json` needs `votes` (generators × items), `scores` (system → score), and
optionally `judge_ids`, `contestant_ids`, `item_ids`, `floor_item_ids`. The
`--html` report is a single self-contained file: the verdict, the floor, the
margin, the resolution bound, and the reason each refusal fired. No network, no
model, no JavaScript. A hostile leaderboard name is escaped, not rendered.

`examples/06_audit_report.py` writes a demo report to `examples/_out/`.

## What the gate refuses

Every scientific claim in a report has to clear four checks. Each check is a
**reason to refuse**, and refusing is the tool working:

| tenet | refuses when | verdict |
|---|---|---|
| **actor-disjointness** | the gold producers are also the contestants | `UNDECIDABLE` |
| **common denominator** | floor and board computed on different item sets | `UNDECIDABLE` |
| **multi-generator** | fewer than three gold producers | `UNDECIDABLE` |
| **power-bounded** | margin is smaller than the MDE at declared power | `MARGINAL` |

Plus a fifth, quieter check that runs first: **is the floor above chance at
all?** Five generators who each flip a coin agree with each other half the time,
so their floor sits near 0.5 — and any score above that looks like a landslide.
If the floor is not detectably above the chance rate, there is no gold to
measure against, and the verdict is `UNDECIDABLE` no matter how large the
margin.

`UNDECIDABLE` is not a polite `MARGINAL`. It says the question was not asked in
a way that permits an answer. `MARGINAL` says the answer is "smaller than we can
tell". The difference matters, because the fix is different: one needs a new
design, the other needs more items.

## Why this exists

The claim-gate separation is the whole point. A system that proposes
hypotheses, runs experiments, and judges its own results will optimise the
appearance of progress — the same failure mode it is supposed to detect. Three
design decisions keep that from happening:

1. **The gate is a check on the apparatus, not the number.** It never asks "is
   this effect big enough". It asks "can this measurement resolve this
   comparison", which is a question about the design.

2. **Ordering, not policy, closes look-laundering.** `k_max` is decided and
   written to the ledger *before* the experiment runs. No policy can prevent
   "run a fixed-sample test, miss, then claim you only intended one look" — the
   enforcement would have to be done by the party that wants to break it. The
   ordering makes the claim impossible rather than forbidden.

3. **The scheduler is not fitted to the run it judges.** `expected_information_
   gain` is a hand-written heuristic over a few coarse features. A learned
   scheduler cannot be verified by the audit it schedules, and a fitted one is
   a Goodhart target: optimise it and you optimise for appearances.

## What is new here

Not the statistics — leave-one-out floors, Dawid-Skene, the four tenets are
classical measurement theory. Not the loop either. The contribution is putting
a gate in front of an *agent that generates its own hypotheses*, which is what
makes the classical tools necessary rather than optional:

- **The refusal taxonomy is complete and explicit.** `READABLE` / `MARGINAL` /
  `UNREADABLE` / `UNDECIDABLE` are the four distinct failure modes, and the
  documentation says which one you are in and why. A system that only returns
  "significant" is a system that cannot tell you it failed.
- **Ordering enforces the pre-registration.** The script hash pinned before the
  run, read off the file after; `k_max` from the ledger, never from the
  experiment's self-report.
- **`mde_method` is always in the report.** `binomial_approx` and
  `cluster_bootstrap` disagree by ~2.7× on the same board. A resolution bound
  that quietly answers a different question is the failure this package exists
  to make visible.

## What it does not do

- **It does not generate a good hypothesis.** `HYPOTHESIZE` is a call to an
  LLM with the literature in context. The gate decides whether the hypothesis
  was *measured*, not whether it was *good*.
- **It does not certify a result, only the measurement.** A `READABLE` verdict
  says the margin cleared the floor by more than the apparatus can resolve. It
  does not say the effect is real, nor that the apparatus measured the right
  thing.
- **It does not fix a bad design.** `UNDECIDABLE` sends the loop back to
  `HYPOTHESIZE`. It will not add generators, fix denominators, or guess what
  you meant.
- **No claim survives contact with an unfalsifiable statement.** `arx theory`
  reports coverage — how many tested hypotheses could constrain a theory at
  all. A high number with weak theories is still weak science, and the report
  says so.

## Documentation

| document | what it covers |
|---|---|
| [docs/architecture.md](docs/architecture.md) | the loop, the five modules, the experiment contract |
| [docs/evaluation-gate.md](docs/evaluation-gate.md) | the four tenets, the floor, the MDE, worked example |
| [docs/sequential-gate.md](docs/sequential-gate.md) | why the naive boundary triples the error rate |
| [docs/ledger.md](docs/ledger.md) | what the hash chain proves, and what it does not |

## Status

The statistical core and the loop are complete and tested (373 tests, ruff
clean, Python 3.12). The literature and Zotero integrations are implemented but
have not been exercised against live services — `arx doctor` reports what your
install can actually do. The RL/Goodhart demonstration (`arx.rl.goodhart_env`) is a bandit with
judge noise, not a research result.

**This is not a general-purpose AGI system and does not claim to be.** It is a
loop that refuses to write a claim it cannot support, and every refusal is
recorded in a ledger you can verify.

## License

Apache-2.0.
