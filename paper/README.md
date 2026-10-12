# arx technical report

`main.tex` is a technical report about what one evaluation run of `arx`
actually found. It exists because the repository's own `README.md` makes
claims about what the gate does, and a claim should be traceable to a run.

## What the paper is not

It is not a benchmark submission and it reports no generalisable number. One
model, five topics, 19 gate decisions. The paper says this in its abstract,
its limitations section, and its headline table.

## Where the numbers come from

Every number in the paper is read back out of the ledgers produced by
`bash scripts/scale_run.sh`, not written by hand. The reproducing command is:

```bash
bash scripts/scale_run.sh 5        # five topics, ~20 min on this machine
PYTHONPATH=src python3 /tmp/tally.py    # prints Table 1 and Table 2
```

The ledgers are hash-chained, so the tally can be recomputed by anyone who has
them; `arx replay` checks that the recorded decisions match a re-run of the
scheduler.

## The number that matters

The paper's central table splits the 19 gate decisions into:

- 17 the gate merely relayed (13 experiment crashes, 4 outputs with no
  contract), and
- 2 the gate reached statistically (1 floor not estimable, 1 leader below
  the measured floor).

A loop that reports "19 hypotheses tested, all inconclusive" describes both as
the same thing. That conflation, not the headline 0 findings, is what the
report is about.

## Notes on building

No LaTeX is installed here, so `main.tex` is the deliverable rather than a
PDF. It uses only standard packages (geometry, booktabs, amsmath, hyperref,
listings) and should build with any TeX distribution:

```bash
pdflatex main.tex && pdflatex main.tex   # twice, for the cross-references
```
