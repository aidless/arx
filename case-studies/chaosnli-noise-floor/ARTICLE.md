# A noise floor belongs to the label producer, not to the leaderboard

When someone reports that model X beats model Y on benchmark Z, they usually
give you a delta and a confidence interval. Both numbers are computed from
labels. Neither comes with an answer to the question that decides whether the
delta means anything: **who produced the labels, and how much does that producer
agree with itself?**

That omission is not cosmetic. On real data, holding the board fixed and
changing only the label producer inverts the verdict. The same nine models, on
the same items, with the same score gap, are above the noise floor when the
labels come from 100 humans and below it when the labels come from an LLM panel.

Here is the whole experiment, three ways.

## The gate, in one paragraph

`arx` is a small open-source gate for exactly this question. It refuses to let a
ranking be described as readable unless the leader's margin over the noise floor
of its own labels exceeds the smallest difference the apparatus could detect. It
computes the floor by leave-one-out: for each label producer, how often does it
agree with the majority of the *other* producers. Then it checks four premises
and reports one of four verdicts, all of which are reasons not to believe
something — `READABLE`, `MARGINAL`, `UNREADABLE`, `UNDECIDABLE`.

The design point worth internalising: **`READABLE` is a statement about the
leader against the floor, and says nothing about the ranks underneath.** Most
leaderboard commentary treats "A is significantly better than B" as the whole
claim. The rest of this article is what happens when you check it.

## 1. Three boards, three verdicts, two refusals

ChaosNLI is a set of NLI benchmarks that were re-annotated by 100 people per
item, so the labels have a *measurable* floor instead of an assumed one. Nine
2019-era checkpoints publish per-item predictions on the same items. Feeding
both to the gate:

| component | n | floor | leader | leader acc | margin | MDE | verdict |
|---|---:|---:|---|---:|---:|---:|---|
| SNLI | 1,514 | 0.7484 | roberta-large | 0.7867 | **+0.0383** | 0.0203 | `READABLE` |
| MNLI-m | 1,599 | 0.6416 | roberta-large | 0.6354 | **−0.0062** | 0.0255 | `UNREADABLE` |
| αNLI | 1,532 | 0.8786 | albert-xxlarge | 0.8473 | **−0.0313** | 0.0238 | `UNREADABLE` |

Two of the three boards produce a refusal, and the refusal is not subtle: **the
leader itself scores below the floor.** On MNLI-m the best model is 0.62 points
below the level one human annotator reproduces from the other 99. On αNLI it is
3.13 points below. There is no model on those boards that is better than the
labels can resolve.

And where the gate does say `READABLE` — SNLI — the top four systems are
separated by **0.29x** of the minimum detectable effect. First to second is
**0.19x**. Four systems that a leaderboard renders as ordered ranks are, by that
measurement, indistinguishable. `READABLE` earned by the leader is not a
promotion of the ranking.

## 2. The same board, two floors

Now the part that I did not expect to be this clean. Take nine checkpoints, one
1,000-item sample, one set of per-item scores, one declared delta. Change only
who produced the labels used to build the floor:

| floor produced by | floor | leader | leader acc | margin | MDE | verdict |
|---|---:|---|---:|---:|---:|---|
| 100 human annotations | 0.7467 | albert-xxlarge | 0.7890 | **+0.0423** | 0.0321 | `READABLE` |
| 32 named LLM judges | 0.8660 | albert-xxlarge | 0.7890 | **−0.0770** | 0.0321 | `UNREADABLE` |

Nothing about the board moved. The verdict inverted. The reason is
arithmetically simple and rhetorically fatal: **an LLM panel agrees with itself
(0.8660) far more than any model in the board scores (0.7890).** A floor is not
a constant of nature. It is a property of the producer, and a panel of models
drawn from the same families as the models under test produces a floor that
almost nothing can clear.

Worth adding: the best individual judge in that panel reaches 0.8530 against the
human majority — *still below* the panel's collective 0.8660. Being the best
judge in the room does not put you above the room.

## 3. The judges move on formatting alone

One objection remains: surely the labels are at least *stable*. A judge asked
the same question twice, with nothing changed, will answer the same way. The
panel release ships an arm that tests a weaker version of that: 500 items
re-asked with the answer options permuted three ways, per judge.

| on those 500 items | value |
|---|---:|
| board top1–top2 gap | 0.0020 |
| **same judge, only the option order changed** — accuracy swing vs the human majority | median **0.0120**, max 0.0340 |
| label flip rate vs the unpermuted run | median 7.1%, max 15.8% |
| **swing / top1–top2 gap** | **6.00x** |

**All 32 judges** swing by more than the board's top gap; this is not two bad
apples. The ranking is inside its own judges' formatting noise before any
question of model quality is reached.

Two honest qualifications, because both could be overclaimed:

* Order is the *smaller* perturbation of the two. A judge reproduces itself
  across option orders with median agreement 0.9317, which is *better* than the
  panel's 0.8616 across judges. Both dwarf the top gap; only the second moves
  the floor.
* This is an artefact of **presentation**, not a repeated-call noise floor. The
  data release states explicitly that its archive establishes no same-prompt
  repeated-call floor, and an order flip is not that. Measuring repeat-call
  variance is a separate experiment nobody in this chain has run.

## What follows from this

If you take one thing: **a benchmark ranking reported without its floor's
producer is half a sentence.** "Model X leads with 78.9%" is a number. "Model X
leads with 78.9%, against a floor of 0.75 from 100 human annotators, with a
resolution bound of 3.2 points, and here is what happens to that verdict when the
floor instead comes from a 32-model judge panel" is a claim. The first travels
better; the second is the one that survives contact with a reviewer.

Three checks that cost almost nothing and catch most of this:

1. **Name the floor's producer, and check it is disjoint from the contestants.**
   A judge panel drawn from the same families as the ranked models is not an
   independent producer; a leaderboard whose judge is also a contestant has no
   floor at all. (I checked AlpacaEval 2.0 for this and it failed: of the five
   evaluators it publishes, **three are prompt variants of one GPT-4-turbo** and
   the other two are single Claude 3 Opus and Mistral-Large runs; two of those
   three variants return an identical verdict on 805 of 805 items — one judge
   run counted twice; and its judged set includes `gpt4`, `gpt4_1106_preview`
   and `gpt-3.5-turbo-0301`, the judge's own family. Its denominators were fine,
   805 aligned. It just had no independent producer to build a floor from. All
   four premises recompute: `audit_alpaca_eval.py` → `results_alpaca_eval.json`.)
2. **Compute the resolution bound on the board's own item set**, with per-item
   scores. Two estimators on the same board can differ by ~2.7x, and the
   friendlier one is the wrong one.
3. **Check that the ranking's top gaps exceed that bound before discussing which
   model is better.** On the one board here that passed, the top four were
   inside 0.29x of it.

## What this does not establish

* Not that these checkpoints are the better or worse models. Every verdict here
  is about whether the measurement could resolve the comparison.
* Not that the human majority is ground truth. ChaosNLI exists because it is
  not; 14 items have no majority at all, and the mean majority share is 0.755.
* Not a repeated-call noise floor, per the point above.
* The model predictions come from a vendored copy of the upstream release (the
  upstream Dropbox link returned an HTML interstitial when this ran); the human
  annotations are a mirror whose per-file SHA-256 is re-verified rather than
  trusted.
* The judge panel's *identities* are taken on trust. That is the release's own
  caveat: the model strings are what was requested, not what is authenticated to
  have served the request. What is measured here is the panel's *agreement
  level*, which does not depend on the model names being right.

Everything above is reproducible from four scripts and one JSON per audit. Each
script verifies the SHA-256 of every input before reporting a number, and
refuses to report a number from a file that fails:

```
case-studies/chaosnli-noise-floor/
  audit_chaosnli.py        # audit 1
  audit_judge_panel.py     # audit 2
  audit_position_bias.py   # audit 3
  audit_alpaca_eval.py     # audit 4 (AlpacaEval 2.0: premises fail)
  results.json  results_judge_panel.json
  results_position_bias.json  results_alpaca_eval.json
```

Repository: [aidless/arx](https://github.com/aidless/arx). The gate is
`pip install -e .`; the case study is the directory above.
