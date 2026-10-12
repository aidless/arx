"""Example 01 -- one complete run, offline, with nothing hidden.

This is the smallest thing that exercises the whole loop: literature,
hypotheses, a pre-registered experiment, the gate, the ledger and the report.
It needs no API key and no network, because both the model and the experiment
are scripted in ``_support.py``.

Three runs, three different refusals to be wrong:

1. a design that can answer the question -- three hypotheses tested, three
   findings written, each with its own dataset and its own p-value;
2. the same loop with two gold generators -- every hypothesis comes back
   ``UNDECIDABLE``, because two raters cannot define a reproducible floor;
3. a script that contradicts its own pre-registration -- refused before the
   number is even looked at.

Run:  PYTHONPATH=src python3 examples/01_quick_run.py
"""

from __future__ import annotations

import json

import _support as S

from arx.agent import Agent
from arx.experiments.ledger import Ledger

TOPIC = "whether a leaderboard separated by less than its noise floor means anything"
FALSIFIED_BY = (
    "a leaderboard whose leader sits inside the measured floor is still reported "
    "as readable"
)


def _agent(workdir, provider, **overrides):
    return Agent(
        S.offline_config(workdir, **overrides),
        topic=TOPIC,
        llm=S.make_llm(),
        searcher=S.make_searcher(),
        provider=provider,
        theory_root="noise-floor",
        theory_falsified_by=FALSIFIED_BY,
    )


def main() -> None:
    workdir = S.out_dir("01_quick_run")

    # ---------------------------------------------------------------- run 1
    S.rule("Run 1: five gold generators -- a design that can answer the question")
    agent = _agent(
        workdir / "healthy",
        S.FactoryProvider(lambda h: S.leaderboard_script(h.text, generators=5, agreement=0.93)),
        max_iterations=20,
        # The scripted proposer has only three hypotheses in it. Once they are
        # all tested it keeps offering the same ones, the deduplicator keeps
        # dropping them, and the queue stops moving -- which is exactly what the
        # stall counter is for. A real model would keep producing candidates.
        stall_limit=2,
    )
    result = agent.run()

    print(f"papers read          : {len(result.papers)}")
    print(f"steps                : {result.steps}")
    print(f"halt reason          : {result.halt_reason}")
    print(f"accepted findings    : {result.n_findings}")
    print(f"rejected / untested  : {len(result.rejected)} / {len(result.untested)}")
    print(f"merkle root          : {result.merkle_root[:16]}...")
    print(f"chain valid          : {result.chain_valid}")
    print(f"budget spent         : {result.budget}")

    S.rule("Each finding carries its own numbers")
    for accepted in result.accepted:
        report = accepted.report
        print(f"  ACCEPTED  {accepted.hypothesis.text[:62]}")
        print(f"            floor={report.floor_estimate:.4f}  margin={report.margin:.4f}  "
              f"mde={report.mde:.4f} ({report.extra.get('mde_method')})  "
              f"n={report.n_items}  m={report.n_generators}")

    print("\n  Note what the three tests share and what they do not. They share a")
    print("  floor -- five generators, one measurement apparatus. They do not")
    print("  share data: each hypothesis is a separate experiment with its own")
    print("  seed, so three findings are three tests, not one test repeated.")

    S.rule("The family-wide budget is spent, not reset")
    mult = result.multiplicity
    print(f"  mode                : {mult['mode']}")
    print(f"  hypotheses tested   : {mult['m']} ({mult['n_scored']} scored)")
    print(f"  alpha spent so far  : {mult['family_alpha_spent']:.5f} of {mult['alpha']}")
    print(f"  next allowance      : {mult['next_allowance']:.6f}")
    print("  The first test was allowed alpha*6/pi^2 = 0.0304, not alpha = 0.05.")
    print("  Summing the spending schedule over all tests converges to alpha, so")
    print("  the family-wise error rate stays bounded however many are run.")

    # ---------------------------------------------------------------- run 2
    S.rule("Run 2: two gold generators -- the design cannot answer the question")
    agent2 = _agent(
        workdir / "two_generators",
        S.FactoryProvider(lambda h: S.leaderboard_script(h.text, generators=2, agreement=0.93)),
        max_iterations=8,
    )
    result2 = agent2.run()
    print(f"accepted findings    : {result2.n_findings}")
    for rejected in result2.rejected:
        print(f"  {rejected.verdict:<12} {rejected.reason}")
    print("\n  No finding, no margin, no 'directionally promising'. The agent has")
    print("  nothing to write because the design cannot produce an answer, and it")
    print("  says so rather than substituting a smaller number.")

    # ---------------------------------------------------------------- run 3
    S.rule("Run 3: a script that contradicts its own pre-registration")
    agent3 = _agent(
        workdir / "laundered",
        S.FactoryProvider(lambda h: S.leaderboard_script(h.text, declared_k_max=1)),
        max_iterations=6,
    )
    result3 = agent3.run()
    print(f"accepted findings    : {result3.n_findings}")
    for rejected in result3.rejected:
        print(f"  {rejected.verdict:<12} {rejected.reason}")
    print("\n  The plan is decided before the run and written to the ledger; the")
    print("  script has no say in it. Claiming one look after taking seven is the")
    print("  'k_max laundering' hole, and it is closed by ordering, not policy.")

    # -------------------------------------------------------------- the log
    S.rule("The ledger is the artefact, not the summary")
    ledger = Ledger(result.ledger_path)
    kinds: dict[str, int] = {}
    for record in ledger.iter_records():
        kinds[record.kind] = kinds.get(record.kind, 0) + 1
    for kind, count in sorted(kinds.items()):
        print(f"  {kind:<14} {count}")

    print("\n  The order that matters -- pre-registration precedes the run, and")
    print("  the gate reads k_max from the ledger rather than from the script:")
    for record in ledger.iter_records():
        if record.kind == "pre_register":
            print(f"    seq={record.seq:<3} pre_register  k_max={record.payload['k_max']} "
                  f"sha={record.payload['script_sha256'][:12]}")
        elif record.kind == "experiment":
            print(f"    seq={record.seq:<3} experiment    rc={record.payload.get('returncode')} "
                  f"sha={str(record.payload.get('script_sha256'))[:12]}")
        elif record.kind == "gate":
            print(f"    seq={record.seq:<3} gate          {record.payload.get('verdict')}")

    print("\n  A third party can check all of this without trusting us:")
    print(f"    python3 -m arx.cli verify-ledger {result.ledger_path} --tail")
    print(f"    python3 -m arx.cli replay {result.ledger_path}")

    print("\n  Run 1, machine-readable:")
    print(json.dumps(
        {k: v for k, v in result.to_dict().items()
         if k in ("n_papers", "steps", "chain_valid", "merkle_root", "budget",
                  "multiplicity", "theory", "goodhart")},
        indent=2, default=str,
    ))


if __name__ == "__main__":
    main()
