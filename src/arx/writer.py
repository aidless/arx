"""Report assembly.

The report's job is to make the difference between the three outcomes legible,
because the failure mode this whole project exists to avoid is a report where
they look the same.

  * **Verified findings** cleared the gate. They carry the number.
  * **Unresolved claims** were tested and did not clear it. They are printed
    verbatim, with the reason. They are not failures to be swept up; a
    hypothesis that could not be separated from the noise floor is a result.
  * **Untested hypotheses** were never run. They are listed as gaps.

Every section that could be empty prints "None." rather than being omitted, so
a reader can tell "we looked and found nothing" from "we did not look".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .agent import AgentResult
from .literature.citation_check import audit_references
from .literature.zotero import Library


@dataclass
class ReportArtifact:
    markdown: str
    path: str | None = None

    def to_text(self) -> str:
        return self.markdown


def render_markdown(result: AgentResult, library: Library | None = None) -> str:
    lines: list[str] = []

    # -- header ------------------------------------------------------------
    lines.append(f"# Research report: {result.topic}")
    lines.append("")
    lines.append(
        f"_Papers reviewed: **{len(result.papers)}** · "
        f"Verified findings: **{result.n_findings}** · "
        f"Unresolved: **{len(result.rejected)}** · "
        f"Untested: **{len(result.untested)}**_"
    )
    lines.append("")
    if not result.gate_enabled:
        lines.append(
            "> **The evaluation gate was disabled for this run.** Every result "
            "below is ungated: no noise-floor audit, no power check, no family "
            "correction was applied. Do not read these numbers as findings."
        )
        lines.append("")

    # -- hypotheses under test --------------------------------------------
    if result.papers:
        lines.append("## Literature reviewed")
        lines.append("")
        for paper in result.papers[:20]:
            year = paper.year or "n.d."
            link = f" [link]({paper.url})" if paper.url else ""
            lines.append(f"- {paper.title} ({year}){link}")
        if len(result.papers) > 20:
            lines.append(f"- _… and {len(result.papers) - 20} more_")
        lines.append("")

    # -- verified ---------------------------------------------------------
    lines.append("## Verified findings")
    lines.append("")
    if result.accepted:
        for finding in result.accepted:
            report = finding.report
            lines.append(f"### {finding.hypothesis.text}")
            lines.append("")
            lines.append(f"- Verdict: `{report.verdict.value}`")
            if report.leader_system is not None and report.leader_score is not None:
                lines.append(
                    f"- Leader: `{report.leader_system}` at `{report.leader_score:.4f}`"
                )
            if report.floor_estimate is not None:
                lines.append(f"- Noise floor: `{report.floor_estimate:.4f}` "
                             f"({report.floor_method or 'n/a'})")
            if report.margin is not None:
                lines.append(f"- Margin over floor: `{report.margin:+.4f}`")
            if report.mde is not None:
                lines.append(
                    f"- Minimum detectable effect: `{report.mde:.4f}` "
                    f"(`{report.extra.get('mde_method', 'n/a')}`)"
                )
            if report.threshold is not None:
                lines.append(f"- Threshold applied: `{report.threshold:.4f}`")
            if report.n_generators is not None:
                lines.append(
                    f"- Generators: `{report.n_generators}` over "
                    f"`{report.n_items}` items"
                )
            if report.reasons:
                for reason in report.reasons:
                    lines.append(f"- Note: {reason}")
            if finding.hypothesis.rationale:
                lines.append(f"- Rationale: {finding.hypothesis.rationale}")
            lines.append("")
    else:
        lines.append("_None. No hypothesis cleared the evaluation gate._")
        lines.append("")

    # -- unresolved (first-class output) ----------------------------------
    lines.append("## Unresolved claims (tested, not separable)")
    lines.append("")
    lines.append(
        "_These were run and did not clear the gate. A claim that cannot be "
        "separated from the noise floor is a result, not a failure, and is "
        "reproduced here rather than omitted._"
    )
    lines.append("")
    if result.rejected:
        for item in result.rejected:
            verdict = f" `{item.verdict}`" if item.verdict else ""
            lines.append(f"- **{item.hypothesis.text}**{verdict}  ")
            lines.append(f"  {item.reason}")
            if item.hypothesis.rationale:
                lines.append(f"  _Rationale: {item.hypothesis.rationale}_")
    else:
        lines.append("_None._")
    lines.append("")

    # -- untested ---------------------------------------------------------
    lines.append("## Untested hypotheses")
    lines.append("")
    lines.append(
        "_Proposed, never run. These are gaps in the evidence, recorded so "
        "they are not mistaken for negative results._"
    )
    lines.append("")
    if result.untested:
        for hypothesis in result.untested:
            lines.append(f"- {hypothesis.text}")
            if hypothesis.rationale:
                lines.append(f"  _Rationale: {hypothesis.rationale}_")
    else:
        lines.append("_None._")
    lines.append("")

    # -- citation audit ---------------------------------------------------
    if result.papers:
        audit = audit_references([p.title for p in result.papers], library)
        lines.append("## Citation audit")
        lines.append("")
        lines.append(f"- Sources responding: `{', '.join(audit.sources_ok) or 'none'}`")
        if audit.sources_failed:
            lines.append(f"- Sources failed: `{', '.join(audit.sources_failed)}`")
        lines.append(f"- Verified: `{audit.n_verified}` of `{len(audit.verdicts)}`")
        if audit.problems:
            lines.append("")
            lines.append("| reference | status | reason |")
            lines.append("|---|---|---|")
            for verdict in audit.problems:
                title = verdict.title.replace("|", "\\|")[:70]
                lines.append(f"| {title} | `{verdict.status.value}` | {verdict.reason} |")
        else:
            lines.append("- No problems detected.")
        lines.append("")

    # -- theory ------------------------------------------------------------
    theory = result.theory or {}
    lines.append("## Theory coverage")
    lines.append("")
    lines.append(
        f"- Theory-driven tests: `{theory.get('hypotheses_tested_theory_driven', 0)}` "
        f"· exploratory tests: `{theory.get('hypotheses_exploratory', 0)}`"
    )
    lines.append(
        f"- Theory-driven share: `{theory.get('theory_coverage', 0.0):.2f}` · "
        f"hypotheses attached to a theory: "
        f"`{theory.get('hypothesis_coverage', 0.0):.2f}`"
    )
    lines.append(
        f"- Theories: `{theory.get('n_theories', 0)}` "
        f"(`{theory.get('n_theories_falsifiable', 0)}` falsifiable, "
        f"`{theory.get('n_theories_unfalsifiable', 0)}` not)"
    )
    lines.append(f"- Falsified: `{theory.get('n_theories_falsified', 0)}`")
    if theory.get("note"):
        lines.append(f"- {theory['note']}")
    lines.append("")

    # -- multiplicity / sequential ----------------------------------------
    mult = result.multiplicity or {}
    lines.append("## Multiplicity control")
    lines.append("")
    lines.append(
        f"- Family size: `{mult.get('m', 0)}` tests "
        f"(`{mult.get('n_scored', 0)}` scored, "
        f"`{mult.get('n_unscored', 0)}` without a p-value)"
    )
    lines.append(f"- Correction: `{mult.get('mode', 'n/a')}`")
    if "family_alpha_spent" in mult:
        lines.append(
            f"- Alpha spent: `{mult['family_alpha_spent']:.4f}` of "
            f"`{mult.get('alpha', 0.0)}`"
        )
    lines.append(f"- Passing the allowance: `{len(mult.get('accepted', []))}`")
    lines.append(f"- Next allowance: `{mult.get('next_allowance', 0.0):.5f}`")
    if result.goodhart:
        status = result.goodhart
        lines.append(
            f"- Scheduler/outcome rank correlation: "
            f"`{status.get('correlation', 0.0):+.3f}` "
            f"(n={status.get('n', 0)}, threshold={status.get('threshold')})"
        )
        if status.get("degraded"):
            lines.append(
                "- **Recalibration flagged**: the scheduler's expected-"
                "information-gain model no longer tracks realised outcomes."
            )
    lines.append("")

    # -- provenance --------------------------------------------------------
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- Ledger: `{result.ledger_path}`")
    lines.append(f"- Merkle root: `{result.merkle_root}`")
    lines.append(f"- Hash chain valid: `{result.chain_valid}`")
    lines.append(f"- Steps: `{result.steps}` · halted: `{result.halt_reason}`")
    budget = result.budget or {}
    lines.append(
        f"- Budget used: `{budget.get('llm_calls', 0)}` LLM calls, "
        f"`{budget.get('experiment_runs', 0)}` experiments, "
        f"`{budget.get('literature_sweeps', 0)}` literature sweeps, "
        f"`{budget.get('wall_seconds', 0.0):.1f}s`"
    )
    lines.append(f"- Model: `{result.llm_name}`")
    lines.append("")

    if result.oracle is not None:
        oracle = result.oracle
        lines.append("### External usage reconciliation")
        lines.append("")
        lines.append(f"- Reported: `{oracle.get('reported')}`")
        lines.append(f"- Externally observed: `{oracle.get('external')}`")
        lines.append(f"- Consistent: `{oracle.get('ok')}`")
        for note in oracle.get("notes") or []:
            lines.append(f"- {note}")
        lines.append("")

    # -- the honest footer -------------------------------------------------
    lines.append("## What this report does and does not claim")
    lines.append("")
    lines.append(
        "- A `READABLE` verdict means the margin over the measured noise floor "
        "exceeded both the declared delta and the minimum detectable effect at "
        "this sample size, on a shared denominator, with disjoint actors."
    )
    lines.append(
        "- It does **not** mean the hypothesis is true. It means the "
        "measurement is capable of distinguishing it from noise."
    )
    lines.append(
        "- `UNDECIDABLE` means the benchmark design cannot answer the question "
        "at all. More data will not change it; the design has to change."
    )
    lines.append(
        "- Theory coverage bounds how much any of this means. Tests with no "
        "attached theory cannot constrain anything, whatever their p-value."
    )
    lines.append(
        "- The noise-floor audit only addresses procedural self-deception "
        "(p-hacking, denominator drift, actor overlap, underpowered margins). "
        "It does not address whether the hypothesis was worth testing."
    )
    return "\n".join(lines)


def write_report(
    result: AgentResult,
    path: str | Path,
    library: Library | None = None,
) -> ReportArtifact:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = render_markdown(result, library)
    target.write_text(text, encoding="utf-8")
    return ReportArtifact(markdown=text, path=str(target))


__all__ = ["ReportArtifact", "render_markdown", "write_report"]
