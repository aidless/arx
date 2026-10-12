"""The autonomous research loop.

States run through a budget-aware scheduler rather than a fixed pipeline:

    LITERATURE -> HYPOTHESIZE -> EXPERIMENT -> GATE -> (loop | WRITE)

Four things about this loop are design commitments, not implementation details.

**The gate is not optional.** A failed gate returns the agent to HYPOTHESIZE.
It never writes a claim. Turning the gate off is possible (``gate_enabled``) and
is recorded in the ledger, so a gated result and an ungated one can never be
confused for each other afterwards.

**Pre-registration happens before the run, not after.** ``do_experiment``
decides ``k_max`` and hashes the script, writes both to the ledger, and only
then launches the subprocess. ``do_gate`` reads ``k_max`` from the ledger and
never from the experiment's own report. This is what closes the
"run a fixed-sample test, miss, then claim you only ever intended one look"
hole — an ordering, not a policy.

**Resuming restores the counters, not just the state.** A resumed run that
starts its budget at zero will spend it twice, and ``exhausted`` stops biting at
exactly the moment it matters. The scheduler context is therefore rebuilt from
the last ``scheduler`` record in the ledger.

**Untested is an outcome.** When no experiment script is available the
hypothesis is recorded as ``no_entrypoint`` and set aside. The agent does not
generate a placeholder result; a plausible number in a ledger is worse than a
gap, because the gap is visible.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .audit.auditor import audit_leaderboard
from .audit.goodhart import GoodhartMonitor
from .audit.prereg import (
    MultiplicityLedger,
    Preregistration,
    check_preregistration,
    plan_k_max,
    read_preregistration,
    sha256_text,
    two_sided_p_from_z,
)
from .audit.sequential import SequentialState, z_from_effect
from .audit.verdict import AuditReport, Verdict
from .config import AgentConfig
from .experiments.ledger import Ledger
from .experiments.provider import (
    ExperimentProvider,
    ExperimentScript,
    WorkspaceExperimentProvider,
)
from .experiments.runner import RunResult, run_experiment
from .gate import EvaluationGate
from .hypothesis import DedupResult, Hypothesis, deduplicate, generate
from .literature.search import Paper, SearchOutcome, search_all
from .literature.zotero import Library, build_library
from .llm import ChatModel, build_llm
from .longterm.global_ledger import GlobalLedger
from .oracles.auditor import UsageSink, reconcile
from .scheduler import Action, Budget, SchedulerContext, log_decision
from .theory.theory_layer import TheoryLayer

DEFAULT_TOPIC = "a falsifiable claim about the system under study"


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------

@dataclass
class AcceptedFinding:
    hypothesis: Hypothesis
    report: AuditReport

    def to_dict(self) -> dict:
        return {
            "hypothesis": self.hypothesis.text,
            "verdict": self.report.verdict.value,
            "leader": self.report.leader_system,
            "leader_score": self.report.leader_score,
            "floor": self.report.floor_estimate,
            "margin": self.report.margin,
            "mde": self.report.mde,
            "threshold": self.report.threshold,
            "mde_method": self.report.extra.get("mde_method"),
            "n_generators": self.report.n_generators,
        }


@dataclass
class RejectedFinding:
    hypothesis: Hypothesis
    reason: str
    verdict: str = ""

    def to_dict(self) -> dict:
        return {
            "hypothesis": self.hypothesis.text,
            "reason": self.reason,
            "verdict": self.verdict,
        }


@dataclass
class AgentResult:
    topic: str
    papers: list[Paper] = field(default_factory=list)
    accepted: list[AcceptedFinding] = field(default_factory=list)
    rejected: list[RejectedFinding] = field(default_factory=list)
    untested: list[Hypothesis] = field(default_factory=list)
    ledger_path: str = ""
    chain_valid: bool = False
    merkle_root: str = ""
    steps: int = 0
    halt_reason: str = ""
    budget: dict = field(default_factory=dict)
    theory: dict = field(default_factory=dict)
    multiplicity: dict = field(default_factory=dict)
    goodhart: dict = field(default_factory=dict)
    oracle: dict | None = None
    gate_enabled: bool = True
    llm_name: str = "null"
    llm_stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "n_papers": len(self.papers),
            "papers": [p.to_dict() for p in self.papers],
            "accepted": [a.to_dict() for a in self.accepted],
            "rejected": [r.to_dict() for r in self.rejected],
            "untested": [h.to_dict() for h in self.untested],
            "ledger_path": self.ledger_path,
            "chain_valid": self.chain_valid,
            "merkle_root": self.merkle_root,
            "steps": self.steps,
            "halt_reason": self.halt_reason,
            "budget": dict(self.budget),
            "theory": dict(self.theory),
            "multiplicity": dict(self.multiplicity),
            "goodhart": dict(self.goodhart),
            "oracle": self.oracle,
            "gate_enabled": self.gate_enabled,
            "llm": self.llm_name,
            "llm_stats": dict(self.llm_stats),
        }

    @property
    def n_findings(self) -> int:
        return len(self.accepted)


# ---------------------------------------------------------------------------
# the agent
# ---------------------------------------------------------------------------

class Agent:
    """One research run, driven by the scheduler, recorded in a ledger."""

    def __init__(
        self,
        cfg: AgentConfig | None = None,
        *,
        topic: str = DEFAULT_TOPIC,
        llm: ChatModel | None = None,
        library: Library | None = None,
        searcher: Callable[..., SearchOutcome] | None = None,
        provider: ExperimentProvider | None = None,
        global_ledger: GlobalLedger | None = None,
        ledger_path: str | Path | None = None,
        gate: EvaluationGate | None = None,
        theory_root: str | None = None,
        theory_falsified_by: str | None = None,
        n_hypotheses: int = 3,
    ):
        self.cfg = cfg or AgentConfig()
        self.topic = topic.strip() or DEFAULT_TOPIC
        self.workspace = Path(self.cfg.workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

        # -- dependencies, each with an offline default --------------------
        self._usage_sink: UsageSink | None = (
            UsageSink(self.cfg.oracle_log_path) if self.cfg.oracle_log_path else None
        )
        self.llm = llm if llm is not None else build_llm(
            self.cfg, usage_sink=self._usage_sink
        )
        self.library = library if library is not None else build_library(self.cfg)
        self.searcher = searcher or (
            lambda query, max_results=20: search_all(query, max_results=max_results)
        )
        self.provider = provider if provider is not None else WorkspaceExperimentProvider(
            self.workspace / "current_experiment.py"
        )
        self.gate = gate or EvaluationGate(strict=self.cfg.gate_strict)
        self.n_hypotheses = max(1, int(n_hypotheses))

        self.global_ledger = global_ledger
        if self.global_ledger is None and self.cfg.semantic_duplicate_threshold > 0:
            try:
                self.global_ledger = GlobalLedger(self.cfg.global_ledger_path)
            except Exception:
                self.global_ledger = None

        self.ledger = Ledger(ledger_path or (self.workspace / "ledger.jsonl"))

        # -- bookkeeping ---------------------------------------------------
        self.budget = Budget(
            max_llm_calls=self.cfg.max_llm_calls,
            max_wall_seconds=self.cfg.max_wall_seconds,
            max_experiment_runs=self.cfg.max_experiment_runs,
            max_literature_sweeps=self.cfg.max_literature_sweeps,
        )
        self.multiplicity = MultiplicityLedger(alpha=self.cfg.gate_alpha)
        self.goodhart = GoodhartMonitor(
            window=self.cfg.goodhart_window,
            threshold=self.cfg.goodhart_threshold,
            min_samples=self.cfg.goodhart_min_samples,
        )
        self.theory = TheoryLayer()
        self.theory_root = (theory_root if theory_root is not None
                            else self.cfg.theory_root).strip()
        self.theory_falsified_by = (
            theory_falsified_by if theory_falsified_by is not None
            else self.cfg.theory_falsified_by
        ).strip()
        if self.theory_root:
            self.theory.register(
                self.theory_root,
                statement=self.topic,
                falsified_by=self.theory_falsified_by,
            )

        # -- research state ------------------------------------------------
        self.papers: list[Paper] = []
        self.hypotheses: list[Hypothesis] = []
        self.accepted: list[AcceptedFinding] = []
        self.rejected: list[RejectedFinding] = []
        self.untested: list[Hypothesis] = []
        self.run_id = f"{int(time.time())}-{sha256_text(self.topic)[:8]}"

        # -- execution state ------------------------------------------------
        self._pending: tuple[Hypothesis, RunResult, Preregistration] | None = None
        self._seq: SequentialState | None = None
        self._last_outcome: str | None = None
        self._last_gate_score: float | None = None
        self._predicted_ig: float | None = None
        self.stalled_steps = 0
        self.steps = 0
        self.halt_reason = ""
        self._wrote_end = False
        self._progress_signature: tuple | None = None

    # ------------------------------------------------------------------
    # scheduler context
    # ------------------------------------------------------------------
    def context(self) -> SchedulerContext:
        margin = self.accepted[-1].report.margin if self.accepted else None
        mde = self.accepted[-1].report.mde if self.accepted else None
        mde_ratio = None
        if margin is not None and mde is not None and margin > 0:
            mde_ratio = float(mde) / float(margin)
        return SchedulerContext(
            n_papers=len(self.papers),
            n_hypotheses=len(self.hypotheses),
            n_accepted=len(self.accepted),
            n_rejected=len(self.rejected),
            last_verdict=self._last_outcome,
            judge_available=bool(getattr(self.llm, "available", False)),
            mde_ratio=mde_ratio,
            margin=margin,
            stalled_steps=self.stalled_steps,
            stall_limit=self.cfg.stall_limit,
            budget=self.budget,
        )

    # ------------------------------------------------------------------
    # actions
    # ------------------------------------------------------------------
    def do_literature(self) -> None:
        outcome = self.searcher(self.topic, self.cfg.search_max_results)
        self.papers = outcome.papers
        self.ledger.append(
            "literature",
            {
                "query": self.topic,
                "n_papers": len(outcome.papers),
                "papers": [p.to_dict() for p in outcome.papers[:50]],
                "sources_ok": outcome.sources_ok,
                "sources_failed": outcome.sources_failed,
            },
        )
        self.budget.charge("literature_sweeps")
        self._last_outcome = "LITERATURE"

    def do_hypothesize(self) -> None:
        context_lines: list[str] = []
        if self.papers:
            context_lines.append("Relevant literature:")
            for paper in self.papers[:8]:
                year = paper.year or "n.d."
                context_lines.append(f"  - {paper.title} ({year})")
        if self.accepted:
            context_lines.append("Already confirmed (do not re-propose):")
            for item in self.accepted[-5:]:
                context_lines.append(f"  - {item.hypothesis.text}")
        if self.rejected:
            context_lines.append("Already rejected (do not re-propose):")
            for item in self.rejected[-5:]:
                context_lines.append(f"  - {item.hypothesis.text} :: {item.reason[:120]}")
        if self.theory_root:
            context_lines.append(
                f"Theory under test: {self.theory_root}. "
                f"If it is wrong we will observe: {self.theory_falsified_by or '(not stated)'}"
            )

        proposals = generate(
            self.llm,
            self.topic,
            n=self.n_hypotheses,
            context="\n".join(context_lines),
        )
        if self.llm.available:
            self.budget.charge("llm_calls")

        # Everything already settled, either way. Leaving accepted findings out
        # of this list lets the proposer re-propose a hypothesis that has
        # already been written up, which costs an experiment and inflates the
        # family without testing anything new.
        known = (
            [h.text for h in self.untested]
            + [i.hypothesis.text for i in self.rejected]
            + [a.hypothesis.text for a in self.accepted]
        )
        if self.global_ledger is not None:
            for proposal in proposals:
                try:
                    priors = self.global_ledger.similar(
                        proposal.text,
                        threshold=self.cfg.semantic_duplicate_threshold,
                        top_k=3,
                    )
                except Exception:
                    priors = []
                if priors:
                    proposal.duplicates = [p.hypothesis for p in priors]
                    proposal.rationale = (
                        proposal.rationale
                        + f" [prior: {priors[0].verdict or 'untested'} "
                          f"sim={priors[0].similarity:.2f}]"
                    ).strip()

        deduped: DedupResult = deduplicate(
            [p for p in proposals if not p.duplicates], threshold=0.6, known=known
        )
        for proposal in proposals:
            if proposal.duplicates:
                deduped.dropped.append(proposal)

        self.hypotheses = deduped.kept
        for hypothesis in self.hypotheses:
            if self.theory_root and self.theory_root in self.theory.theories:
                self.theory.attach(hypothesis.text, self.theory_root)
                self.ledger.append(
                    "theory",
                    {
                        "event": "attach",
                        "root": self.theory_root,
                        "hypothesis": hypothesis.text,
                    },
                )
            else:
                self.theory.mark_exploratory(hypothesis.text)
                self.ledger.append(
                    "theory",
                    {"event": "exploratory", "hypothesis": hypothesis.text},
                )

        self.ledger.append(
            "hypothesis",
            {
                "n_proposed": len(proposals),
                "n_kept": len(deduped.kept),
                "candidates": [h.to_dict() for h in proposals],
                "dropped": [
                    {"text": h.text, "duplicates": h.duplicates} for h in deduped.dropped
                ],
                "duplicate_pairs": list(deduped.pairs),
            },
        )
        self._last_outcome = "HYPOTHESIZE"

    def do_experiment(self) -> None:
        if not self.hypotheses:
            self._last_outcome = "NO_HYPOTHESES"
            return
        hypothesis = self.hypotheses[0]

        script: ExperimentScript | None = self.provider.provide(hypothesis)
        if script is None:
            self.ledger.append(
                "experiment",
                {
                    "hypothesis": hypothesis.text,
                    "status": "no_entrypoint",
                    "provider": getattr(self.provider, "name", "unknown"),
                    "note": (
                        "nothing to run; the hypothesis is set aside as untested "
                        "rather than replaced with a plausible-looking result"
                    ),
                },
            )
            self.untested.append(hypothesis)
            self.hypotheses = self.hypotheses[1:]
            self._last_outcome = "NO_ENTRYPOINT"
            self.budget.charge("experiment_runs")
            return

        # --- pre-register BEFORE the run ---------------------------------
        k_plan = plan_k_max(
            prior_confidence=hypothesis.confidence,
            alpha=self.cfg.gate_alpha,
            target_power=self.cfg.gate_power,
            effect_se_ratio=hypothesis.expected_effect_se_ratio,
        )
        prereg = Preregistration.create(
            hypothesis_text=hypothesis.text,
            k_max=k_plan,
            alpha=self.cfg.gate_alpha,
            script_sha256=script.sha256,
        )
        self.ledger.append("pre_register", prereg.to_dict())

        path = script.path or script.materialise(self.workspace, "arx_experiment.py")
        result = run_experiment(
            path,
            timeout_s=self.cfg.sandbox_timeout_s,
            cwd=self.workspace,
        )
        self.budget.charge("experiment_runs")
        self.budget.charge("wall_seconds", result.duration_s)

        self.ledger.append(
            "experiment",
            {
                "hypothesis": hypothesis.text,
                "provider": script.source,
                **result.to_dict(),
                "output_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
            },
        )
        self._pending = (hypothesis, result, prereg)
        self._last_outcome = "RAN"

    def do_gate(self) -> None:
        if self._pending is None:
            self._last_outcome = "NOTHING_TO_GATE"
            return
        hypothesis, result, _planned = self._pending
        self._pending = None

        base = {"hypothesis": hypothesis.text}

        if not result.ok:
            reason = "experiment timed out" if result.timed_out else (
                f"experiment exited with code {result.returncode}"
            )
            self._record_rejection(hypothesis, reason, Verdict.UNDECIDABLE.value)
            self.ledger.append("gate", {**base, "verdict": Verdict.UNDECIDABLE.value,
                                        "reason": reason})
            self._last_outcome = Verdict.UNDECIDABLE.value
            return

        ok, why = result.validate()
        if not ok:
            self._record_rejection(hypothesis, why, Verdict.UNDECIDABLE.value)
            self.ledger.append("gate", {**base, "verdict": Verdict.UNDECIDABLE.value,
                                        "reason": why})
            self._last_outcome = Verdict.UNDECIDABLE.value
            return

        blob = result.payload() or {}

        # --- pre-registration is checked against the LEDGER, not the blob ---
        records = self.ledger.as_dicts()
        prereg = read_preregistration(records, hypothesis.text)

        # The authoritative script hash is the one measured off the file that
        # actually ran, not the one the script prints about itself: a script
        # cannot be required to know its own hash, and a claim it makes about
        # itself cannot be given more weight than the bytes on disk.
        measured_hash = result.script_sha256
        claimed_hash = blob.get("script_sha256")
        if claimed_hash is not None and str(claimed_hash) != str(measured_hash):
            reason = (
                f"the experiment reported script_sha256={claimed_hash} but the "
                f"file that ran hashes to {measured_hash}"
            )
            self._record_rejection(hypothesis, f"pre-registration violated: {reason}",
                                   Verdict.UNDECIDABLE.value)
            self.ledger.append("gate", {**base, "verdict": Verdict.UNDECIDABLE.value,
                                        "reason": reason, "prereg_ok": False})
            self._last_outcome = Verdict.UNDECIDABLE.value
            return

        prereg_ok, prereg_reason = check_preregistration(
            prereg,
            blob_k_max=_opt_int(blob.get("k_max")),
            blob_script_sha256=measured_hash,
        )
        if not prereg_ok:
            self._record_rejection(hypothesis, f"pre-registration violated: {prereg_reason}",
                                   Verdict.UNDECIDABLE.value)
            self.ledger.append("gate", {**base, "verdict": Verdict.UNDECIDABLE.value,
                                        "reason": prereg_reason,
                                        "prereg_ok": False})
            self._last_outcome = Verdict.UNDECIDABLE.value
            return

        assert prereg is not None  # guaranteed by prereg_ok

        if not self.cfg.gate_enabled:
            self.ledger.append("gate", {**base, "verdict": "GATE_DISABLED",
                                        "note": "gate_enabled=False: no audit was performed"})
            self.multiplicity.register(hypothesis.text, None)
            self.ledger.append("multiplicity",
                               {"event": "register", **self.multiplicity.records[-1]})
            self.accepted.append(
                AcceptedFinding(hypothesis, AuditReport(verdict=Verdict.MARGINAL,
                                                        reasons=["gate disabled by config"]))
            )
            self.hypotheses = self.hypotheses[1:]
            self._last_outcome = "GATE_DISABLED"
            return

        # --- the audit ----------------------------------------------------

        votes = _as_votes(blob.get("votes"))
        report = audit_leaderboard(
            votes,
            {str(k): float(v) for k, v in (blob.get("scores") or {}).items()},
            leaderboard_item_ids=_opt_index_set(blob.get("item_ids")),
            floor_item_ids=None,
            judge_ids=_opt_str_set(blob.get("judge_ids")),
            contestant_ids=_opt_str_set(blob.get("contestant_ids")),
            per_item_scores=None,
            delta_declared=self.cfg.gate_delta_declared,
            alpha=self.cfg.gate_alpha,
            power=self.cfg.gate_power,
            seed=self.cfg.seed,
        )

        # --- sequential layer, pinned to the pre-registered k_max ---------
        sequential = None
        z_value: float | None = None
        margin = report.margin
        se = blob.get("se")
        if se is not None and margin is not None:
            try:
                se_f = float(se)
            except (TypeError, ValueError):
                se_f = 0.0
            if se_f > 0:
                if self._seq is None or self._seq.k_max != prereg.k_max:
                    self._seq = SequentialState(k_max=prereg.k_max,
                                                alpha=self.cfg.gate_alpha,
                                                label=hypothesis.text[:60])
                z_value = z_from_effect(float(margin), se_f)
                verdict_from_seq = self._seq.observe(z_value)
                if verdict_from_seq is Verdict.READABLE and not self._seq.done:
                    report.verdict = Verdict.STOP_EARLY
                elif verdict_from_seq is Verdict.MARGINAL and self._seq.done:
                    report.verdict = Verdict.MARGINAL
                sequential = self._seq.summary()
                report.sequential = sequential

        # --- multiplicity --------------------------------------------------
        p_value = two_sided_p_from_z(z_value) if z_value is not None else None
        mult_rec = self.multiplicity.register(hypothesis.text, p_value)
        self.ledger.append("multiplicity", {"event": "register", **mult_rec})
        if (
            report.verdict in (Verdict.READABLE, Verdict.STOP_EARLY)
            and not mult_rec["passed"]
        ):
            # The margin cleared the noise floor, but not the family-wise
            # allowance: the finding is real per-comparison and not reportable
            # family-wise. Downgrading it is the correction, not a detail.
            report.verdict = Verdict.MARGINAL
            report.reasons.append(
                f"p={p_value:.4g} exceeds the family-wise allowance "
                f"{mult_rec['allowance']:.4g} at family size {self.multiplicity.m}"
            )

        # --- theory --------------------------------------------------------
        node = self.theory.record(
            hypothesis.text,
            "supported" if report.verdict in (Verdict.READABLE, Verdict.STOP_EARLY)
            else report.verdict.value.lower(),
            reason="; ".join(report.reasons[:2]),
        )
        self.ledger.append(
            "theory",
            {
                "event": "outcome",
                "hypothesis": hypothesis.text,
                "root": node.root if node else None,
                "verdict": report.verdict.value,
                "theory_falsified": bool(node and node.falsified),
            },
        )

        # --- Goodhart monitor ----------------------------------------------
        predicted = self._predicted_ig if self._predicted_ig is not None else 0.0
        self.goodhart.observe(predicted, report.verdict.value)
        gh_status = self.goodhart.status()
        self.ledger.append("goodhart", {"event": "status", **gh_status})
        if self.goodhart.degraded:
            self.ledger.append(
                "goodhart",
                {
                    "event": "recalibrate",
                    "correlation": gh_status["correlation"],
                    "n": gh_status["n"],
                    "note": (
                        "scheduler IG no longer tracks realised outcomes; the "
                        "scoring constants need revisiting before they are "
                        "trusted again"
                    ),
                },
            )

        # --- dispatch ------------------------------------------------------
        self.ledger.append(
            "gate",
            {
                **base,
                "verdict": report.verdict.value,
                "prereg_ok": True,
                "k_max": prereg.k_max,
                "report": report.to_dict(),
                "sequential": sequential,
                "multiplicity": mult_rec,
                "predicted_ig": predicted,
                "observed_utility": gh_status,
            },
        )

        if self.global_ledger is not None:
            try:
                self.global_ledger.record(
                    hypothesis.text,
                    verdict=report.verdict.value,
                    reason="; ".join(report.reasons[:2]),
                    project=self.topic[:60],
                    run_id=self.run_id,
                    meta={"margin": report.margin, "mde": report.mde},
                )
            except Exception:
                pass

        if report.verdict in (Verdict.READABLE, Verdict.STOP_EARLY):
            self.accepted.append(AcceptedFinding(hypothesis, report))
            self.hypotheses = self.hypotheses[1:]
        elif report.verdict is Verdict.MARGINAL and not self.cfg.gate_strict:
            self.accepted.append(AcceptedFinding(hypothesis, report))
            self.hypotheses = self.hypotheses[1:]
        else:
            self._record_rejection(
                hypothesis, "; ".join(report.reasons) or report.verdict.value,
                report.verdict.value,
            )
            self.hypotheses = self.hypotheses[1:]

        self._last_outcome = report.verdict.value

    def do_write(self) -> None:
        self.ledger.append(
            "run_end",
            {
                "topic": self.topic,
                "accepted": [a.to_dict() for a in self.accepted],
                "rejected": [r.to_dict() for r in self.rejected],
                "untested": [h.to_dict() for h in self.untested],
                "steps": self.steps,
                "halt_reason": self.halt_reason or "write",
                # Full form, caps included: this is the snapshot a resume reads,
                # and restoring the counters without the ceilings would hand the
                # continued run a budget it was never granted.
                "budget": self.budget.to_dict(),
                "theory_coverage": self.theory.coverage(),
                "multiplicity": self.multiplicity.summary(),
                "goodhart": self.goodhart.status(),
            },
        )
        self._last_outcome = "WRITTEN"
        self._wrote_end = True

    def _record_rejection(self, hypothesis: Hypothesis, reason: str, verdict: str) -> None:
        self.rejected.append(
            RejectedFinding(hypothesis=hypothesis, reason=reason, verdict=verdict)
        )

    # ------------------------------------------------------------------
    # driver
    # ------------------------------------------------------------------
    def _progress(self) -> tuple:
        return (
            len(self.papers),
            len(self.hypotheses),
            len(self.accepted),
            len(self.rejected),
            len(self.untested),
        )

    def step(self) -> Action:
        """Execute one scheduler decision. Returns the action taken."""
        self.steps += 1
        ctx = self.context()
        decision = log_decision(self.steps, ctx)
        self._predicted_ig = decision.scores.get(Action.GATE.value)
        self.ledger.append("scheduler", decision.to_dict())

        action = decision.action
        t0 = time.monotonic()
        try:
            if action is Action.LITERATURE:
                self.do_literature()
            elif action is Action.HYPOTHESIZE:
                # do_hypothesize charges for the call it actually makes; charging
                # again here would halve the run's real LLM budget.
                self.do_hypothesize()
            elif action is Action.EXPERIMENT:
                self.do_experiment()
            elif action is Action.GATE:
                self.do_gate()
            elif action is Action.WRITE:
                self.do_write()
        except Exception as exc:  # a failed step is recorded, never fatal
            self.ledger.append(
                "error",
                {
                    "step": self.steps,
                    "action": action.value,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            self.halt_reason = f"{type(exc).__name__} in {action.value}"
            return Action.WRITE
        finally:
            self.budget.charge("wall_seconds", time.monotonic() - t0)

        # stall bookkeeping happens after the action, on its effect
        signature = self._progress()
        if signature == self._progress_signature:
            self.stalled_steps += 1
        else:
            self.stalled_steps = 0
            self._progress_signature = signature
        return action

    def run(self) -> AgentResult:
        self.ledger.append(
            "run_start",
            {
                "topic": self.topic,
                "run_id": self.run_id,
                "config": self.cfg.ledger_flags(),
                "llm": getattr(self.llm, "name", "unknown"),
                "provider": getattr(self.provider, "name", "unknown"),
                "workspace": str(self.workspace),
                "theory_root": self.theory_root or None,
                "theory_falsified_by": self.theory_falsified_by or None,
            },
        )
        if self.theory_root:
            self.ledger.append(
                "theory",
                {
                    "event": "register",
                    "root": self.theory_root,
                    "statement": self.topic,
                    "falsified_by": self.theory_falsified_by,
                },
            )

        ceiling = max(1, int(self.cfg.max_iterations))
        while self.steps < ceiling:
            halting = self.budget.halting()
            if halting:
                self.halt_reason = f"halting budget: {', '.join(halting)}"
                self.ledger.append("budget", {"halting": halting, "step": self.steps})
                break
            action = self.step()
            if action is Action.WRITE:
                break
        else:
            self.halt_reason = f"iteration ceiling ({ceiling})"
            self.ledger.append("stall", {"reason": self.halt_reason, "step": self.steps})

        if not self.halt_reason:
            self.halt_reason = "write"

        # A run that stops on a ceiling or a halting budget never reaches the
        # WRITE action, and without this its summary record -- the post-action
        # budget snapshot a resume depends on -- would be the one thing missing
        # exactly when the run was cut short.
        if not self._wrote_end:
            self.do_write()

        oracle = self._reconcile_usage()
        return AgentResult(
            topic=self.topic,
            papers=list(self.papers),
            accepted=list(self.accepted),
            rejected=list(self.rejected),
            untested=list(self.untested),
            ledger_path=str(self.ledger.path),
            chain_valid=self.ledger.verify_chain(),
            merkle_root=self.ledger.root(),
            steps=self.steps,
            halt_reason=self.halt_reason,
            budget=self.budget.snapshot(),
            theory=self.theory.coverage(),
            multiplicity=self.multiplicity.summary(),
            goodhart=self.goodhart.status(),
            oracle=oracle,
            gate_enabled=self.cfg.gate_enabled,
            llm_name=getattr(self.llm, "name", "unknown"),
            llm_stats=self._llm_stats(),
        )

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    def _llm_stats(self) -> dict:
        stats = getattr(self.llm, "stats", None)
        return stats.to_dict() if hasattr(stats, "to_dict") else {}

    def _reconcile_usage(self) -> dict | None:
        path = self.cfg.oracle_log_path
        if path is None:
            return None
        from .oracles.auditor import read_usage

        reported = {
            "llm_call": getattr(getattr(self.llm, "stats", None), "calls", 0),
        }
        return reconcile(reported, read_usage(path), tolerance=0.0).to_dict()

    # ------------------------------------------------------------------
    # resume
    # ------------------------------------------------------------------
    @classmethod
    def resume(
        cls,
        cfg: AgentConfig,
        ledger_path: str | Path,
        *,
        topic: str | None = None,
        llm: ChatModel | None = None,
        library: Library | None = None,
        searcher: Callable[..., SearchOutcome] | None = None,
        provider: ExperimentProvider | None = None,
        gate: EvaluationGate | None = None,
    ) -> Agent:
        """Rebuild an agent from a ledger and continue it.

        The counters come back too. A resume that restarts the budget at zero
        would spend it twice, and the ceiling would stop biting exactly when it
        starts to matter.
        """
        source = Ledger(ledger_path)
        if not source.verify_chain():
            raise ValueError(f"refusing to resume from a broken ledger: {ledger_path}")
        records = source.as_dicts()

        start = next((r for r in records if r["kind"] == "run_start"), None)
        resolved_topic = topic or (start["payload"].get("topic") if start else None) or DEFAULT_TOPIC

        agent = cls(
            cfg,
            topic=resolved_topic,
            llm=llm,
            library=library,
            searcher=searcher,
            provider=provider,
            gate=gate,
            ledger_path=ledger_path,
        )
        agent.ledger = source
        agent.run_id = (start["payload"].get("run_id") if start else None) or agent.run_id

        # state
        for rec in records:
            kind, payload = rec["kind"], rec["payload"]
            if kind == "literature":
                agent.papers = [
                    Paper(
                        title=p.get("title", ""),
                        authors=list(p.get("authors") or []),
                        year=p.get("year"),
                        url=p.get("url", ""),
                        doi=p.get("doi"),
                        source=p.get("source", ""),
                    )
                    for p in payload.get("papers", [])
                ]
            elif kind == "gate":
                hypothesis = Hypothesis(text=str(payload.get("hypothesis", "")))
                verdict = str(payload.get("verdict", ""))
                if verdict in ("READABLE", "STOP_EARLY", "MARGINAL"):
                    report = _report_from_payload(payload.get("report")) or AuditReport(
                        verdict=Verdict(verdict) if verdict in Verdict._value2member_map_
                        else Verdict.MARGINAL
                    )
                    agent.accepted.append(AcceptedFinding(hypothesis, report))
                else:
                    agent.rejected.append(
                        RejectedFinding(hypothesis, str(payload.get("reason", "")), verdict)
                    )
            elif kind == "experiment" and payload.get("status") == "no_entrypoint":
                agent.untested.append(Hypothesis(text=str(payload.get("hypothesis", ""))))

        # counters
        agent.multiplicity = MultiplicityLedger.from_ledger(
            records, alpha=cfg.gate_alpha
        )
        agent.theory = TheoryLayer.from_ledger(records)
        agent.goodhart = GoodhartMonitor.from_ledger(
            records,
            window=cfg.goodhart_window,
            threshold=cfg.goodhart_threshold,
            min_samples=cfg.goodhart_min_samples,
        )

        last_ctx = None
        last_end_budget = None
        for rec in records:
            kind, payload = rec["kind"], rec["payload"]
            if kind == "scheduler" and isinstance(payload.get("context"), dict):
                last_ctx = payload["context"]
            elif kind == "run_end" and isinstance(payload.get("budget"), dict):
                last_end_budget = payload["budget"]
        if last_ctx:
            restored = SchedulerContext.from_dict(last_ctx)
            agent.budget = restored.budget
            agent.stalled_steps = restored.stalled_steps
        if last_end_budget:
            # A run_end snapshot is taken *after* the final action, whereas a
            # scheduler context is captured before its action. Preferring the
            # former is what stops the last action being re-spent on resume.
            agent.budget = Budget.from_dict(last_end_budget)
        agent.steps = sum(1 for r in records if r["kind"] == "scheduler")
        agent._progress_signature = agent._progress()
        return agent


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _as_votes(raw) -> object:
    import numpy as np

    if raw is None:
        raise ValueError("experiment output has no 'votes'")
    arr = np.array(raw)
    if arr.ndim != 2:
        raise ValueError(f"'votes' must be 2-D (generators x items), got shape {arr.shape}")
    try:
        return arr.astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError("'votes' must contain integers (or null for abstention)") from exc


def _opt_str_set(raw) -> set[str] | None:
    if raw is None:
        return None
    if isinstance(raw, (str, bytes)):
        return None
    try:
        values = {str(x) for x in raw}
    except TypeError:
        return None
    return values or None


def _opt_index_set(raw) -> set[int] | None:
    if raw is None:
        return None
    out: set[int] = set()
    for value in raw:
        try:
            out.add(int(value))
        except (TypeError, ValueError):
            continue
    return out or None


def _opt_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _report_from_payload(payload) -> AuditReport | None:
    if not isinstance(payload, dict):
        return None
    verdict_raw = str(payload.get("verdict", ""))
    verdict = Verdict(verdict_raw) if verdict_raw in Verdict._value2member_map_ else None
    if verdict is None:
        return None
    return AuditReport(
        verdict=verdict,
        reasons=list(payload.get("reasons") or []),
        leader_system=payload.get("leader_system"),
        leader_score=payload.get("leader_score"),
        floor_estimate=payload.get("floor_estimate"),
        margin=payload.get("margin"),
        delta_declared=payload.get("delta_declared"),
        mde=payload.get("mde"),
        threshold=payload.get("threshold"),
        n_items=payload.get("n_items"),
        n_floor_items=payload.get("n_floor_items"),
        n_generators=payload.get("n_generators"),
        n_contestants=payload.get("n_contestants"),
        floor_method=payload.get("floor_method", ""),
        sequential=payload.get("sequential"),
        extra=dict(payload.get("extra") or {}),
    )


__all__ = [
    "AcceptedFinding",
    "Agent",
    "AgentResult",
    "DEFAULT_TOPIC",
    "RejectedFinding",
]
