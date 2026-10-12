#!/usr/bin/env python3
"""Compare gate behaviour across model runs (scale_run.sh output dirs).

Reads one or more scale-run output directories, tallies each the same way
scripts/scale_run.sh does, and prints a side-by-side table. The point of the
split is the same as in the report: "N gate decisions" conflates execution
failures the gate merely relayed with verdicts the gate reached statistically,
and a comparison that reports only the headline number repeats the conflation.

Usage:
    PYTHONPATH=src python3 scripts/compare_models.py \
        --label A=/tmp/arx_scale --label B=/tmp/arx_scale_b
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arx.experiments.ledger import Ledger  # noqa: E402


def tally(out_dir: Path) -> dict:
    runs = sorted(p for p in out_dir.glob("run_*") if (p / "ledger.jsonl").exists())
    papers = exps = crashed = no_json = 0
    verdicts: Counter[str] = Counter()
    statistical: Counter[str] = Counter()
    chains_ok = 0

    for run in runs:
        ledger = Ledger(run / "ledger.jsonl")
        recs = list(ledger.iter_records())
        if recs and ledger.verify_chain():
            chains_ok += 1
        for rec in recs:
            p = rec.payload
            if rec.kind == "literature":
                papers += int(p.get("n_papers") or 0)
            elif rec.kind == "experiment":
                if p.get("status") == "no_entrypoint":
                    continue
                exps += 1
                rc = p.get("returncode")
                if rc is not None and rc != 0:
                    crashed += 1
                elif not p.get("stdout_tail", "").strip():
                    no_json += 1
            elif rec.kind == "gate":
                verdict = str(p.get("verdict"))
                verdicts[verdict] += 1
                reasons = ((p.get("report") or {}).get("reasons")
                           or [p.get("reason") or ""])
                joined = " | ".join(str(r) for r in reasons)
                if any(k in joined for k in (
                        "exit", "no JSON", "no contract", "crash",
                        "non-zero", "unparseable", "did not produce output")):
                    pass  # relayed; counted implicitly by not being statistical
                else:
                    statistical[verdict] += 1

    decisions = sum(verdicts.values())
    relayed = decisions - sum(statistical.values())
    return {
        "runs": len(runs),
        "chains_ok": chains_ok,
        "papers": papers,
        "experiments": exps,
        "crashed": crashed,
        "no_json": no_json,
        "decisions": decisions,
        "relayed": relayed,
        "statistical": sum(statistical.values()),
        "verdicts": dict(verdicts),
        "statistical_by_verdict": dict(statistical),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", action="append", required=True,
                    help="NAME=/path/to/scale-run-output (repeatable)")
    args = ap.parse_args()

    labels: list[tuple[str, Path]] = []
    for spec in args.label:
        name, _, path = spec.partition("=")
        if not name or not path:
            ap.error(f"--label must be NAME=PATH, got {spec!r}")
        labels.append((name, Path(path)))

    keys = ["runs", "chains_ok", "papers", "experiments", "crashed",
            "no_json", "decisions", "relayed", "statistical"]
    header = f"{'metric':<14}" + "".join(f"{name:>10}" for name, _ in labels)
    print(header)
    print("-" * len(header))
    tallies = {name: tally(path) for name, path in labels}
    for k in keys:
        print(f"{k:<14}" + "".join(f"{tallies[n][k]:>10}" for n, _ in labels))
    print()
    for name, _ in labels:
        t = tallies[name]
        print(f"{name}: verdicts={t['verdicts']}")
        print(f"{name}: statistical={t['statistical_by_verdict']}")


if __name__ == "__main__":
    main()
