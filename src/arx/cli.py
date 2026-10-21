"""Command line interface.

Subcommands are the entry points a reader of the repository will actually use,
so each one either proves something or refuses to pretend:

``run``           run the loop and write a report
``resume``        continue a ledger, with counters restored
``audit``         audit a leaderboard JSON through the same gate the loop uses
``verify-ledger`` check the hash chain, print the Merkle root and tail
``replay``        recompute every scheduler decision from the ledger alone
``theory``        print theory coverage for a finished run
``doctor``        report what this install can and cannot do

``replay`` is the trust entry point: given only ``ledger.jsonl`` — no network,
no model — it recomputes what the scheduler should have done at each step and
compares. A divergence is either a code change or a tamper, and both are worth
knowing about.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .audit.auditor import audit_leaderboard
from .audit.verdict import Verdict
from .config import AgentConfig
from .experiments.ledger import Ledger
from .scheduler import choose
from .scheduler.context import SchedulerContext

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _votes_from_blob(blob: dict):
    import numpy as np

    raw = blob.get("votes")
    if raw is None:
        raise SystemExit("leaderboard file must contain a 'votes' key")
    arr = np.array(raw)
    if arr.ndim != 2:
        raise SystemExit(
            f"'votes' must be 2-D (generators x items); got shape {arr.shape}"
        )
    return arr.astype(int)


def _opt_set(raw, cast):
    if raw is None:
        return None
    try:
        out = {cast(x) for x in raw}
    except (TypeError, ValueError):
        return None
    return out or None


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    from .agent import Agent
    from .writer import write_report

    cfg = _config_from_args(args)
    provider = _build_provider(args, cfg)

    agent = Agent(
        cfg,
        topic=args.topic,
        provider=provider,
        n_hypotheses=args.hypotheses,
        theory_root=args.theory_root,
        theory_falsified_by=args.falsified_by,
    )
    result = agent.run()
    artifact = write_report(result, args.out, agent.library)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        _print_summary(result, artifact.path)
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    from .agent import Agent
    from .writer import write_report

    cfg = _config_from_args(args)
    provider = _build_provider(args, cfg)
    try:
        agent = Agent.resume(cfg, args.ledger, provider=provider)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.max_iterations is not None:
        agent.cfg.max_iterations = max(1, int(args.max_iterations))
    result = agent.run()
    artifact = write_report(result, args.out, agent.library)
    _print_summary(result, artifact.path)
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    blob = _load_json(args.leaderboard)
    report = audit_leaderboard(
        _votes_from_blob(blob),
        {str(k): float(v) for k, v in (blob.get("scores") or {}).items()},
        leaderboard_item_ids=_opt_set(blob.get("item_ids"), int),
        floor_item_ids=_opt_set(blob.get("floor_item_ids"), int),
        judge_ids=_opt_set(blob.get("judge_ids"), str),
        contestant_ids=_opt_set(blob.get("contestant_ids"), str),
        delta_declared=args.delta,
        alpha=args.alpha,
        power=args.power,
        seed=args.seed,
    )
    if args.html:
        from .audit.report_html import render_html

        Path(args.html).write_text(render_html(report), encoding="utf-8")
        print(f"html report: {args.html}")
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print(report.to_text())
    return 0 if report.verdict in (Verdict.READABLE, Verdict.STOP_EARLY) else 1


def cmd_verify_ledger(args: argparse.Namespace) -> int:
    ledger = Ledger(args.ledger)
    ok = ledger.verify_chain()
    print(f"chain valid: {ok}")
    print(f"records: {len(ledger)}")
    print(f"merkle root: {ledger.root()}")
    if args.tail:
        print(f"tail hash: {ledger.tail_hash()}")
    if args.proof is not None:
        try:
            proof = ledger.inclusion_proof(args.proof)
        except (IndexError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        record = ledger.records(offset=args.proof, limit=1)[0]
        print(f"proof #{args.proof}: {proof.to_text()}")
        print(f"  verifies: {proof.verify(record)}")
    if args.anchor is not None:
        verdict = ledger.verify_anchor(args.anchor)
        print(f"anchor check: {'ok' if verdict['ok'] else 'MISMATCH'}")
        for reason in verdict["reasons"]:
            print(f"  - {reason}")
        if not verdict["ok"]:
            return 2
    return 0 if ok else 2


def cmd_anchor(args: argparse.Namespace) -> int:
    """Publish a ledger's state so later edits become detectable.

    A hash chain cannot notice its own tail being removed; only an outside
    reference can. This writes one. Store it where the ledger's author does
    not control it.
    """
    ledger = Ledger(args.ledger)
    if not ledger.verify_chain():
        print("refusing to anchor: ledger chain is invalid", file=sys.stderr)
        return 2
    payload = ledger.anchor(path=args.out, note=args.note)
    target = args.out or ledger.anchor_path()
    print(f"anchor written: {target}")
    print(f"  records: {payload['n_records']}")
    print(f"  merkle root: {payload['root']}")
    print(f"  tail hash: {payload['tail_hash']}")
    print("  publish this file somewhere the ledger author does not control;")
    print("  a copy stored beside its own ledger only guards against accident.")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    ledger = Ledger(args.ledger)
    if not ledger.verify_chain():
        print("refusing to replay: ledger chain is invalid", file=sys.stderr)
        return 2

    state: dict = {}
    mismatches: list[dict] = []
    checked = 0
    records = ledger.as_dicts(limit=args.until)

    for index, rec in enumerate(records):
        kind, payload = rec["kind"], rec["payload"]
        if kind == "scheduler":
            context = payload.get("context")
            if not isinstance(context, dict):
                continue
            ctx = SchedulerContext.from_dict(context)
            expected = choose(ctx).value
            recorded = payload.get("action")
            checked += 1
            if args.check and expected != recorded:
                mismatches.append(
                    {"index": index, "expected": expected, "recorded": recorded,
                     "context": context}
                )
        elif kind == "experiment":
            state["last_verdict"] = "RAN" if payload.get("returncode") == 0 else "RAN"
        elif kind == "gate":
            state["last_verdict"] = str(payload.get("verdict", "GATED"))

    if mismatches:
        print(f"{len(mismatches)} decision mismatch(es) out of {checked}:", file=sys.stderr)
        for item in mismatches[:10]:
            print(
                f"  record {item['index']}: ledger={item['recorded']} "
                f"replay={item['expected']}",
                file=sys.stderr,
            )
        return 1

    print(f"replayed {checked} scheduler decisions; all match")
    print(f"merkle root: {ledger.root()}")
    print(f"tail hash: {ledger.tail_hash()}")
    return 0


def cmd_theory(args: argparse.Namespace) -> int:
    from .theory.theory_layer import TheoryLayer

    ledger = Ledger(args.ledger)
    layer = TheoryLayer.from_ledger(ledger.as_dicts())
    coverage = layer.coverage()
    if args.json:
        print(json.dumps({"coverage": coverage, "theories": layer.to_dict()},
                         indent=2, default=str))
        return 0
    print(f"theories registered    : {coverage['n_theories']} "
          f"({coverage['n_theories_falsifiable']} falsifiable, "
          f"{coverage['n_theories_falsified']} falsified, "
          f"{coverage['n_theories_unfalsifiable']} unfalsifiable)")
    print(f"hypotheses tested      : {coverage['hypotheses_attached']} "
          f"({coverage['hypotheses_tested_theory_driven']} theory-driven, "
          f"{coverage['hypotheses_exploratory']} exploratory)")
    print(f"hypothesis coverage    : {coverage['hypothesis_coverage']:.2f}")
    print(f"theory coverage        : {coverage['theory_coverage']:.2f}")
    for node in layer.theories.values():
        state = "FALSIFIED" if node.falsified else "surviving"
        print(f"  [{state}] {node.root}: {node.n_tested} tested, "
              f"{node.n_rejected} rejected")
    if coverage.get("note"):
        print(f"note: {coverage['note']}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from . import probe_capabilities

    caps = probe_capabilities()
    for key, value in caps.items():
        print(f"{key}: {value}")
    return 0


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------

def _config_from_args(args: argparse.Namespace) -> AgentConfig:
    cfg = AgentConfig()
    if getattr(args, "workspace", None):
        cfg.workspace = args.workspace
    if getattr(args, "max_iterations", None) is not None:
        cfg.max_iterations = max(1, int(args.max_iterations))
    if getattr(args, "no_gate", False):
        cfg.gate_enabled = False
    if getattr(args, "loose_gate", False):
        cfg.gate_strict = False
    if getattr(args, "delta", None) is not None:
        cfg.gate_delta_declared = float(args.delta)
    if getattr(args, "zotero", False):
        cfg.zotero_enabled = True
    if getattr(args, "no_search", False):
        cfg.search_max_results = 0
    return cfg


def _build_provider(args: argparse.Namespace, cfg: AgentConfig):
    from .experiments.provider import (
        LLMExperimentProvider,
        NullExperimentProvider,
        WorkspaceExperimentProvider,
    )
    from .llm import build_llm

    kind = getattr(args, "provider", "workspace")
    if kind == "none":
        return NullExperimentProvider()
    if kind == "llm":
        return LLMExperimentProvider(build_llm(cfg), workspace=Path(cfg.workspace))
    return WorkspaceExperimentProvider(cfg.workspace / "current_experiment.py")


def _print_summary(result, report_path: str | None) -> None:
    print(f"topic:            {result.topic}")
    print(f"papers reviewed:  {len(result.papers)}")
    print(f"verified:         {result.n_findings}")
    print(f"unresolved:       {len(result.rejected)}")
    print(f"untested:         {len(result.untested)}")
    print(f"steps:            {result.steps} ({result.halt_reason})")
    print(f"merkle root:      {result.merkle_root}")
    print(f"chain valid:      {result.chain_valid}")
    coverage = result.theory.get("theory_coverage", 0.0)
    print(f"theory coverage:  {coverage:.2f}")
    if result.oracle is not None and not result.oracle.get("ok", True):
        print(f"usage mismatch:   reported={result.oracle.get('reported')} "
              f"external={result.oracle.get('external')}")
    if report_path:
        print(f"report:           {report_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arx",
        description=(
            "Autonomous research loop with a statistical evaluation gate. "
            "Every claim in a report has cleared a noise-floor and power check, "
            "or is reported as unresolved."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the research loop on a topic")
    run.add_argument("topic", help="the research question")
    run.add_argument("--out", default="arx_report.md", help="report path")
    run.add_argument("--workspace", default=None, help="working directory")
    run.add_argument("--hypotheses", type=int, default=3)
    run.add_argument("--max-iterations", type=int, default=None)
    run.add_argument("--provider", choices=("workspace", "llm", "none"),
                     default="workspace",
                     help="where experiment scripts come from")
    run.add_argument("--theory-root", default=None)
    run.add_argument("--falsified-by", default=None)
    run.add_argument("--delta", type=float, default=None,
                     help="declared separation threshold")
    run.add_argument("--no-gate", action="store_true",
                     help="disable the gate (recorded in the ledger)")
    run.add_argument("--loose-gate", action="store_true",
                     help="accept MARGINAL results as findings")
    run.add_argument("--zotero", action="store_true", help="require Zotero")
    run.add_argument("--no-search", action="store_true", help="skip literature")
    run.add_argument("--json", action="store_true")
    run.set_defaults(func=cmd_run)

    resume = sub.add_parser("resume", help="continue a ledger with counters restored")
    resume.add_argument("ledger")
    resume.add_argument("--out", default="arx_report.md")
    resume.add_argument("--workspace", default=None)
    resume.add_argument("--max-iterations", type=int, default=None)
    resume.add_argument("--provider", choices=("workspace", "llm", "none"),
                        default="workspace")
    resume.add_argument("--delta", type=float, default=None)
    resume.add_argument("--no-gate", action="store_true")
    resume.add_argument("--loose-gate", action="store_true")
    resume.add_argument("--zotero", action="store_true")
    resume.add_argument("--no-search", action="store_true")
    resume.set_defaults(func=cmd_resume)

    audit = sub.add_parser("audit", help="audit a leaderboard JSON")
    audit.add_argument("leaderboard")
    audit.add_argument("--delta", type=float, default=0.02)
    audit.add_argument("--alpha", type=float, default=0.05)
    audit.add_argument("--power", type=float, default=0.8)
    audit.add_argument("--seed", type=int, default=0)
    audit.add_argument("--json", action="store_true")
    audit.add_argument("--html", default=None, help="write a single-file HTML report")
    audit.set_defaults(func=cmd_audit)

    verify = sub.add_parser("verify-ledger", help="verify a ledger hash chain")
    verify.add_argument("ledger")
    verify.add_argument("--tail", action="store_true")
    verify.add_argument("--proof", type=int, default=None,
                        help="also emit an inclusion proof for this record index")
    verify.add_argument("--anchor", default=None,
                        help="compare against a previously published anchor file")
    verify.set_defaults(func=cmd_verify_ledger)

    anchor = sub.add_parser(
        "anchor", help="publish a ledger's root so later edits are detectable")
    anchor.add_argument("ledger")
    anchor.add_argument("--out", default=None,
                        help="where to write the anchor (default: <ledger>.anchor.json)")
    anchor.add_argument("--note", default="", help="free-text note stored in the anchor")
    anchor.set_defaults(func=cmd_anchor)

    replay = sub.add_parser("replay", help="recompute scheduler decisions from a ledger")
    replay.add_argument("ledger")
    replay.add_argument("--until", type=int, default=None)
    replay.add_argument("--check", action="store_true", default=True)
    replay.add_argument("--no-check", dest="check", action="store_false")
    replay.set_defaults(func=cmd_replay)

    theory = sub.add_parser("theory", help="theory coverage for a finished run")
    theory.add_argument("ledger")
    theory.add_argument("--json", action="store_true")
    theory.set_defaults(func=cmd_theory)

    doctor = sub.add_parser("doctor", help="report what this install can do")
    doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
