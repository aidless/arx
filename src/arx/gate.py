"""The evaluation gate: a hard stop between a result and a claim.

Every experiment result that would become a finding passes through here. The
gate is not advisory. If the margin between the leader and the measured noise
floor is inside the resolution bound, the result is not "weak evidence" — it is
not a separation at all, and writing it up as one is the exact failure the
framework exists to interrupt.

The gate has three outcomes that matter operationally:

* **READABLE** — the claim may be written, with its margin and its floor.
* **MARGINAL** — the claim may be written only if ``strict=False``, and always
  labelled. Turning off strict mode is recorded in the ledger.
* **UNREADABLE / UNDECIDABLE** — ``GateRejection``. The result is not evidence
  against the hypothesis; it is evidence that the design could not have
  answered the question.

That last distinction is load-bearing. An underpowered study does not
disconfirm; it fails to inform, and the next step is to fix the design rather
than to weaken the claim. ``GateRejection`` carries the audit report so the
caller can tell the two apart.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from .audit.auditor import AuditConfig, audit_leaderboard
from .audit.verdict import AuditReport, Verdict

AuditorFn = Callable[..., AuditReport]

#: Verdicts that permit a claim to be written.
WRITABLE = frozenset({Verdict.READABLE})
#: Permitted only when strict mode is off.
TOLERATED = frozenset({Verdict.MARGINAL})


class GateRejection(RuntimeError):
    """Raised when a result must not be written into a report."""

    def __init__(self, report: AuditReport, message: str | None = None):
        self.report = report
        super().__init__(message or self._default_message())

    def _default_message(self) -> str:
        report = self.report
        if report.verdict is Verdict.UNDECIDABLE:
            return (
                "Experiment blocked: the design cannot answer the question, so "
                f"the result is not evidence either way.\n{report.to_text()}\n"
                "Fix the design (at least three independent generators, "
                "disjoint judges and contestants, one denominator) and rerun."
            )
        return (
            "Experiment blocked: the claimed margin is inside the noise-floor "
            f"resolution bound.\n{report.to_text()}\n"
            "Either collect enough data to resolve the margin, or restate the "
            "claim as 'not statistically separable'."
        )


@dataclass
class EvaluationGate:
    """Wrap an auditor so that a failed audit cannot be ignored."""

    auditor_fn: AuditorFn = audit_leaderboard
    config: AuditConfig = field(default_factory=AuditConfig)
    strict: bool = True
    #: When strict, MARGINAL is rejected; when not, it passes with a label.
    allow_marginal_with_note: bool = False

    def decision(self, *args, **kwargs) -> tuple[bool, AuditReport]:
        """Audit without raising. ``(allowed, report)``."""
        merged = {**self.config.as_kwargs(), **kwargs}
        report = self.auditor_fn(*args, **merged)
        if report.verdict in WRITABLE:
            return True, report
        if report.verdict in TOLERATED and (
            self.allow_marginal_with_note or not self.strict
        ):
            return True, report
        return False, report

    @contextmanager
    def check(self, *args, **kwargs) -> Iterator[AuditReport]:
        """Yield the audit report, or raise :class:`GateRejection`.

        Usage::

            gate = EvaluationGate()
            with gate.check(votes, scores, ...) as report:
                write_finding(report)
        """
        allowed, report = self.decision(*args, **kwargs)
        if not allowed:
            raise GateRejection(report)
        yield report

    def labelled_verdict(self, report: AuditReport) -> str:
        """A verdict string that carries its own caveat.

        MARGINAL results are never reported as plain findings; the label makes
        the weakness travel with the number, so it cannot be dropped in the
        next summary that quotes it.
        """
        if report.verdict is Verdict.MARGINAL:
            return "MARGINAL (not separable at the declared threshold)"
        return report.verdict.value


def gate_report(
    votes,
    leaderboard_scores: dict,
    *,
    strict: bool = True,
    **kwargs,
) -> tuple[bool, AuditReport]:
    """One-shot convenience used by the CLI and by examples."""
    gate = EvaluationGate(strict=strict)
    return gate.decision(votes, leaderboard_scores, **kwargs)


__all__ = [
    "AuditorFn",
    "EvaluationGate",
    "GateRejection",
    "TOLERATED",
    "WRITABLE",
    "gate_report",
]
