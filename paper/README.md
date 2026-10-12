# arx technical report

`main.tex` is a technical report about what one evaluation run of `arx`
actually found. It exists because the repository's own `README.md` makes
claims about what the gate does, and a claim should be traceable to a run.

## What the paper is not

It is not a benchmark submission and it reports no generalisable number. Two
small models (3B and 8B), five topics each, and 37 gate decisions across the
two runs. The paper says this in its abstract, its limitations section, and
its headline tables.

## Where the numbers come from

Every number in the paper is read back out of the ledgers produced by
`bash scripts/scale_run.sh`, not written by hand. The reproducing command is:

```bash
bash scripts/scale_run.sh 5        # five topics, ~20 min on this machine
ARX_LLM_MODEL=llama3.1:8b ARX_SCALE_OUT=/tmp/arx_scale_b \
    bash scripts/scale_run.sh 5    # the 8B replication
PYTHONPATH=src python3 scripts/compare_models.py \
    --label "A_llama3.2-3b=/tmp/arx_scale" \
    --label "B_llama3.1-8b=/tmp/arx_scale_b"   # Tables 1-3
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

`main.pdf` is committed so the report is readable without a TeX toolchain. It
is built with `pdflatex` and the Latin Modern fonts, so building it again
needs `texlive-latex-base`, `texlive-latex-recommended`, and `lmodern` (the
LaTeX package, not the fontconfig one):

```bash
pdflatex main.tex && pdflatex main.tex   # twice, for the cross-references
```

The machine that produced the committed PDF had no root access and no TeX
installed, so it built one from debs extracted into a local prefix
(texlive-latex-base, texlive-latex-recommended, lmodern; formats via
`fmtutil-user`; font maps via `updmap-user` with a merged `updmap.cfg`). None
of that machinery is needed to rebuild the paper on a machine with a normal
TeX distribution — the commands above are enough.
