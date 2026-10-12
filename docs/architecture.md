# Architecture

`arx` is an autonomous research loop with a statistical evaluation gate in
front of every claim. The gate is not a plugin and it is not optional: a
hypothesis either clears it, or the loop goes back to proposing hypotheses.

```
                       ┌──────────────────────────────────────┐
                       │  LITERATURE   arXiv / Semantic       │
                       │                 Scholar / Zotero     │
                       └───────────────────┬──────────────────┘
                                           ▼
                       ┌──────────────────────────────────────┐
                       │  HYPOTHESIZE  LLM proposes, then     │
                       │                 dedup against every  │
                       │                 settled hypothesis   │
                       └───────────────────┬──────────────────┘
                                           ▼
                       ┌──────────────────────────────────────┐
                       │  EXPERIMENT    pre-register, then    │
                       │                 run the script in a  │
                       │                 sandbox              │
                       └───────────────────┬──────────────────┘
                                           ▼
                       ┌──────────────────────────────────────┐
                       │  GATE          four tenets → verdict │
                       │                 READABLE / MARGINAL /│
                       │                 UNREADABLE /          │
                       │                 UNDECIDABLE           │
                       └──────────┬─────────────────┬─────────┘
                       ACCEPTED   │                 │  anything else
                                  ▼                 ▼
                          ┌──────────────┐   back to HYPOTHESIZE
                          │  WRITE       │   (never a claim)
                          └──────────────┘
```

## The five modules

| module | responsibility | what it refuses to do |
|---|---|---|
| `arx.scheduler` | decides the next action from a budget-aware score | never uses a learned model to score |
| `arx.experiments` | runs a script in a subprocess; appends an immutable record | never executes model-generated code in-process |
| `arx.audit` | the four tenets, the noise floor, the MDE, the sequential gate | never reports a margin without its resolution |
| `arx.agent` | the loop, the pre-registration ordering, the state machine | never writes a claim that did not clear the gate |
| `arx.theory` | binds hypotheses to a falsifiable statement | never lets an unfalsifiable hypothesis count as a test |

## The gate is a check on the *design*, not on the number

The question the gate answers is not "is this effect big enough". It is:

> can this apparatus measure the thing the leaderboard claims to measure?

Four things have to hold, and each one is a reason to refuse:

1. **Actor-disjointness.** The systems that produced the gold are not the
   systems being ranked. If they are, the floor partly measures agreement with
   itself.
2. **Common denominator.** The floor and the board are computed on the same
   items. Two different denominators make their difference an artefact of
   subsetting, not an effect.
3. **Multi-generator sufficiency.** At least three gold producers. With two,
   leave-one-out leaves a single rater and "agreement with the others"
   collapses into self-consistency. There is no estimator.
4. **Power-bounded resolution.** The margin must exceed what the sample can
   resolve (`MDE` at the declared power), not merely exceed zero.

A fourth, quieter check runs first: **is the floor above chance at all?** Five
generators who each flip a coin agree with each other about half the time, so
the floor sits near 0.5 and any score above that looks like a landslide. If the
measured floor is not detectably above the chance rate, there is no gold for a
score to be measured against, and the verdict is `UNDECIDABLE` no matter how
large the margin.

## The two things that make it honest rather than thorough

### Ordering, not policy, closes k_max laundering

`do_experiment` decides `k_max` before the run, hashes the script it is about to
execute, writes both to the ledger, and only then launches the subprocess.
`do_gate` reads `k_max` from the ledger and never from the experiment's own
report.

This closes the hole where you run a fixed-sample test, miss, and then report
that you only ever intended one look. No policy does it, because the policy
would have to be enforced by the party that wants to break it. The ordering
makes the claim impossible rather than forbidden.

The same reasoning fixes the script hash: the authoritative hash is the one
measured off the file that ran, not the one the script prints about itself. A
script cannot be required to know its own hash, and a claim it makes about
itself cannot outweigh the bytes on disk.

### The scheduler is not fitted to the run it is judging

`expected_information_gain` is a hand-written heuristic over a few coarse
features. That is deliberate, for two reasons:

* A learned scheduler cannot be verified by the audit it schedules. It would be
  grading its own homework.
* A fitted scheduler is a Goodhart target. Optimise its information-gain
  function and you optimise for the appearance of progress. `arx.audit.goodhart`
  watches for exactly that divergence between prediction and outcome.

The constants are coarse and monotone, and the ledger records the score of
every action at every step so a third party can see the ranking rather than
take it on trust.

## What runs, and where

| state | calls out to | failure mode |
|---|---|---|
| `LITERATURE` | arXiv and Semantic Scholar HTTP, optional Zotero | source listed in `sources_failed`; the run continues |
| `HYPOTHESIZE` | an OpenAI-compatible chat endpoint | `NullLLM`; the run records that no candidate was produced |
| `EXPERIMENT` | `python3 <script>` in a subprocess, hard timeout | non-zero exit or timeout → `UNDECIDABLE`, never a result |
| `GATE` | nothing external | pure function over the script's JSON output |
| `WRITE` | nothing external | assembles the report from the ledger |

Every dependency has an offline default, which is why `arx doctor` reports what
this install can actually do before the run starts.

## The experiment contract

A script must print one JSON object on its last stdout line:

```json
{
  "votes": [[0, 1, null], [1, 1, 0]],
  "scores": {"system_a": 0.91, "system_b": 0.62},
  "judge_ids": ["judge_1", "judge_2"],
  "contestant_ids": ["system_a", "system_b"],
  "item_ids": [0, 1, 2],
  "per_item_scores": {"system_a": [1, 0, 1]},
  "se": 0.01
}
```

`-1`/`null` means abstention. `item_ids` is what enables the denominator check.
`per_item_scores` switches the MDE from the binomial approximation to the paired
cluster bootstrap — the single biggest change in resolution you can make, and
worth reading about in [evaluation-gate.md](evaluation-gate.md).
