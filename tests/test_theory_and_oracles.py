"""Tests for the theory layer and the external usage oracle.

Both modules exist to make a failure mode *visible* rather than to prevent it,
and the tests are written to keep that honesty:

* The theory layer must not let a single rejected hypothesis count as a
  refutation, and it must count exploratory work separately rather than
  silently folding it in as theory-driven.
* The oracle must not report "reconciled" when there was nothing to compare.
  A missing gateway log is a gap in the evidence, not a clean bill of health.
"""

from __future__ import annotations

import json

import pytest

from arx.oracles.auditor import (
    OracleEntry,
    Reconciliation,
    UsageSink,
    count_events,
    read_usage,
    reconcile,
    record_usage,
)
from arx.theory.theory_layer import (
    MIN_TESTS_TO_FALSIFY,
    TheoryLayer,
)

# ---------------------------------------------------------------------------
# theory: registering and attaching
# ---------------------------------------------------------------------------

def test_a_theory_needs_a_root_and_a_falsifier_to_count():
    layer = TheoryLayer()
    node = layer.register("retrieval-goodhart",
                          statement="reranking gains are judge artefacts")
    assert node.root == "retrieval-goodhart"
    assert node.falsifiable is False  # no falsifier stated yet

    node = layer.register("retrieval-goodhart",
                          falsified_by="a reranker beats the floor on held-out gold")
    assert node.falsifiable is True


def test_registering_an_empty_root_is_refused():
    with pytest.raises(ValueError):
        TheoryLayer().register("   ")


def test_registering_twice_updates_rather_than_replaces():
    layer = TheoryLayer()
    layer.register("t", statement="first")
    node = layer.register("t", falsified_by="X")
    assert node.statement == "first"     # not wiped
    assert node.falsified_by == "X"
    assert len(layer.theories) == 1


def test_attaching_to_an_unknown_theory_is_refused():
    with pytest.raises(KeyError):
        TheoryLayer().attach("H1", "nonexistent")


def test_an_attached_hypothesis_resolves_to_its_theory():
    layer = TheoryLayer()
    layer.register("t", falsified_by="X")
    layer.attach("H1", "t")
    assert layer.theory_of("H1").root == "t"
    assert layer.theory_of("H2") is None


def test_exploratory_hypotheses_are_counted_separately():
    """Legitimate work, but not a test of anything — and it must not be
    able to inflate the theory-driven share."""
    layer = TheoryLayer()
    layer.mark_exploratory("H1")
    layer.mark_exploratory("H1")  # idempotent
    assert layer.exploratory == ["H1"]
    assert layer.coverage()["hypotheses_exploratory"] == 1


def test_recording_an_unattached_outcome_marks_it_exploratory():
    layer = TheoryLayer()
    assert layer.record("H1", "rejected") is None
    assert "H1" in layer.exploratory


# ---------------------------------------------------------------------------
# theory: falsification requires more than one bad day
# ---------------------------------------------------------------------------

def test_one_rejected_hypothesis_does_not_falsify_a_theory():
    layer = TheoryLayer()
    layer.register("t", falsified_by="X")
    layer.attach("H1", "t")
    layer.record("H1", "rejected")
    assert layer.falsified_theories() == []


def test_enough_uniform_rejections_do_falsify():
    layer = TheoryLayer()
    layer.register("t", falsified_by="X")
    for i in range(MIN_TESTS_TO_FALSIFY):
        layer.attach(f"H{i}", "t")
        layer.record(f"H{i}", "rejected")
    falsified = layer.falsified_theories()
    assert [t.root for t in falsified] == ["t"]


def test_a_single_supported_result_blocks_falsification():
    """Refutation needs every test to fail. One survivor means the theory is
    still standing, whatever the others did."""
    layer = TheoryLayer()
    layer.register("t", falsified_by="X")
    for i in range(3):
        layer.attach(f"H{i}", "t")
        layer.record(f"H{i}", "rejected")
    layer.attach("H9", "t")
    layer.record("H9", "readable")
    assert layer.falsified_theories() == []
    assert layer.surviving_theories()[0].root == "t"


def test_a_theory_without_a_falsifier_can_never_be_falsified():
    """This is the point of requiring one: a theory that cannot say what
    would refute it is not being tested, only decorated."""
    layer = TheoryLayer()
    layer.register("vague")
    for i in range(5):
        layer.attach(f"H{i}", "vague")
        layer.record(f"H{i}", "rejected")
    assert layer.falsified_theories() == []
    assert [t.root for t in layer.unfalsifiable_theories()] == ["vague"]


def test_the_negative_outcomes_are_recognised():
    """`undecidable` and `unreadable` are failures to learn, and counting
    them as support would be the exact self-deception being guarded against."""
    layer = TheoryLayer()
    layer.register("t", falsified_by="X")
    for i, verdict in enumerate(["rejected", "undecidable", "unreadable"]):
        layer.attach(f"H{i}", "t")
        layer.record(f"H{i}", verdict)
    assert layer.falsified_theories()[0].n_rejected == 3


def test_a_raised_minimum_defers_falsification():
    layer = TheoryLayer(min_tests_to_falsify=3)
    layer.register("t", falsified_by="X")
    for i in range(2):
        layer.attach(f"H{i}", "t")
        layer.record(f"H{i}", "rejected")
    assert layer.falsified_theories() == []


def test_reasons_are_kept_with_the_outcome():
    layer = TheoryLayer()
    layer.register("t", falsified_by="X")
    layer.attach("H1", "t")
    layer.record("H1", "undecidable", reason="only two generators")
    assert layer.theories["t"].reasons["H1"] == "only two generators"


# ---------------------------------------------------------------------------
# theory: coverage is reported, never optimised
# ---------------------------------------------------------------------------

def test_coverage_reports_the_shape_of_the_work():
    layer = TheoryLayer()
    layer.register("t", falsified_by="X")
    layer.attach("H1", "t")
    layer.record("H1", "readable")
    layer.mark_exploratory("H2")
    layer.record("H2", "readable")

    coverage = layer.coverage()
    assert coverage["hypotheses_attached"] == 1
    assert coverage["hypotheses_exploratory"] == 1
    assert coverage["theory_coverage"] == pytest.approx(0.5)
    assert coverage["hypothesis_coverage"] == pytest.approx(0.5)
    assert "not all" in coverage["note"] or "bounds one failure mode" in coverage["note"]


def test_coverage_of_an_empty_run_is_zero_not_undefined():
    coverage = TheoryLayer().coverage()
    assert coverage["theory_coverage"] == 0.0
    assert coverage["hypothesis_coverage"] == 0.0
    assert coverage["n_theories"] == 0


def test_coverage_is_json_safe_and_round_trips():
    layer = TheoryLayer()
    layer.register("t", falsified_by="X")
    layer.attach("H1", "t")
    layer.record("H1", "rejected")
    dumped = json.dumps(layer.to_dict())
    assert "t" in dumped


def test_rebuilding_from_ledger_records_restores_the_theories():
    records = [
        {"kind": "theory", "payload": {"event": "register", "root": "t",
                                       "statement": "s", "falsified_by": "X"}},
        {"kind": "theory", "payload": {"event": "attach", "hypothesis": "H1",
                                       "root": "t"}},
        {"kind": "theory", "payload": {"event": "outcome", "hypothesis": "H1",
                                       "verdict": "rejected", "reason": ""}},
    ]
    layer = TheoryLayer.from_ledger(records)
    assert layer.theory_of("H1").root == "t"
    assert layer.theories["t"].outcomes["H1"] == "rejected"


def test_rebuilding_ignores_unrelated_records():
    layer = TheoryLayer.from_ledger([{"kind": "gate", "payload": {}}])
    assert layer.theories == {}


# ---------------------------------------------------------------------------
# oracles: the external log
# ---------------------------------------------------------------------------

def test_usage_events_round_trip_through_the_log(tmp_path):
    path = tmp_path / "oracle.jsonl"
    record_usage(path, "llm_call", {"model": "m"})
    record_usage(path, "llm_call", {"model": "m"})
    entries = read_usage(path)
    assert len(entries) == 2
    assert entries[0].event == "llm_call"
    assert entries[0].payload == {"model": "m"}


def test_a_missing_log_reads_as_empty_not_as_an_error(tmp_path):
    assert read_usage(tmp_path / "nope.jsonl") == []


def test_a_truncated_line_does_not_lose_the_good_ones(tmp_path):
    path = tmp_path / "oracle.jsonl"
    record_usage(path, "llm_call", {})
    with path.open("a") as fh:
        fh.write('{"ts": 1, "event": "llm_ca')
    assert len(read_usage(path)) == 1


def test_counting_groups_by_event():
    entries = [OracleEntry(ts=1.0, event="llm_call", payload={}),
               OracleEntry(ts=2.0, event="llm_call", payload={}),
               OracleEntry(ts=3.0, event="llm_tokens", payload={})]
    assert count_events(entries)["llm_call"] == 2
    assert count_events(entries)["llm_tokens"] == 1


def test_the_usage_sink_writes_records(tmp_path):
    sink = UsageSink(tmp_path / "oracle.jsonl")
    sink("llm_call", {"model": "m"})
    assert read_usage(tmp_path / "oracle.jsonl")[0].payload["model"] == "m"


# ---------------------------------------------------------------------------
# oracles: reconciliation
# ---------------------------------------------------------------------------

def test_a_matching_count_reconciles(tmp_path):
    entries = [OracleEntry(ts=1.0, event="llm_call", payload={}) for _ in range(3)]
    verdict = reconcile({"llm_call": 3}, entries)
    assert verdict.ok is True
    assert verdict.gaps["llm_call"] == 0


def test_under_reporting_is_caught():
    entries = [OracleEntry(ts=1.0, event="llm_call", payload={}) for _ in range(5)]
    verdict = reconcile({"llm_call": 2}, entries)
    assert verdict.ok is False
    assert verdict.gaps["llm_call"] == -3


def test_over_reporting_is_also_caught():
    """The transcribed design assumed the ledger under-counts. It can also
    over-count, and that needs explaining too."""
    entries = [OracleEntry(ts=1.0, event="llm_call", payload={})]
    verdict = reconcile({"llm_call": 4}, entries)
    assert verdict.ok is False
    assert verdict.gaps["llm_call"] == 3


def test_a_run_that_reported_nothing_against_a_busy_log_is_called_out():
    entries = [OracleEntry(ts=1.0, event="llm_call", payload={}) for _ in range(4)]
    verdict = reconcile({}, entries)
    assert verdict.ok is False
    assert any("reported none" in n for n in verdict.notes)


def test_nothing_to_compare_is_not_a_pass():
    """A missing gateway log means the evidence is absent. Reporting 'ok'
    here would turn 'we could not check' into 'we checked and it was fine'."""
    verdict = reconcile({}, [])
    assert verdict.ok is False
    assert any("nothing was logged" in n for n in verdict.notes)


def test_tolerance_is_zero_by_default_and_can_be_widened():
    entries = [OracleEntry(ts=1.0, event="llm_call", payload={}) for _ in range(3)]
    assert reconcile({"llm_call": 4}, entries).ok is False
    assert reconcile({"llm_call": 4}, entries, tolerance=1.0).ok is True


def test_token_counts_are_compared_when_both_sides_have_them():
    entries = [OracleEntry(ts=1.0, event="llm_tokens",
                           payload={"prompt_tokens": 100, "completion_tokens": 20})]
    verdict = reconcile({"llm_call": 1, "prompt_tokens": 130, "completion_tokens": 20},
                        entries)
    assert verdict.gaps["prompt_tokens"] == 30
    assert verdict.ok is False


def test_reconciliation_serialises_and_renders():
    verdict = reconcile({"llm_call": 1}, [])
    json.dumps(verdict.to_dict())
    assert "llm_call" in verdict.to_text()


def test_the_reconciliation_dataclass_is_constructible_directly():
    verdict = Reconciliation(reported={"a": 1}, external={"a": 1}, gaps={"a": 0},
                             tolerance=0.0, ok=True, notes=[])
    assert verdict.ok


def test_a_recovered_theory_stops_being_reported_as_falsified():
    """The label is the current state of the evidence, not a verdict for life.

    Latching it would leave a stale label on the record; silently clearing it
    would hide the reversal. So the theory is no longer *falsified* and is
    flagged *revived*, which is the fact a reader actually needs.
    """
    layer = TheoryLayer()
    layer.register("t", falsified_by="X")
    for i in range(2):
        layer.attach(f"H{i}", "t")
        layer.record(f"H{i}", "rejected")
    assert layer.falsified_theories()[0].root == "t"

    layer.attach("H9", "t")
    node = layer.record("H9", "readable")
    assert node.falsified is False
    assert node.revived is True
    assert node.falsified_at is None
    assert layer.falsified_theories() == []
    assert layer.surviving_theories()[0].root == "t"
    assert json.loads(json.dumps(node.to_dict()))["revived"] is True


def test_token_totals_are_summed_across_external_events():
    entries = [
        OracleEntry(ts=1.0, event="llm_tokens", payload={"prompt_tokens": 60}),
        OracleEntry(ts=2.0, event="llm_tokens", payload={"prompt_tokens": 40}),
    ]
    assert count_events(entries)["prompt_tokens"] == 100
