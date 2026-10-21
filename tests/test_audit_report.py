"""Tests for the single-file HTML report.

The renderer is the part a reviewer actually opens, so it gets the same
treatment as the gate: a sound board must show its verdict and numbers, a
broken board must show the refusal, and a hostile leaderboard must not be able
to inject markup.
"""

from __future__ import annotations

import json

import numpy as np

from arx.audit.auditor import audit_leaderboard
from arx.audit.report_html import render_html
from arx.audit.verdict import AuditReport, Verdict
from arx.cli import main


def _votes(m: int, n: int = 400, agreement: float = 0.93, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 2, size=n)
    return np.array(
        [np.where(rng.random(n) > agreement, 1 - truth, truth) for _ in range(m)],
        dtype=int,
    )


def _sound_report() -> AuditReport:
    return audit_leaderboard(
        _votes(5),
        {"padded": 0.965, "plain": 0.70},
        judge_ids={"j1", "j2", "j3", "j4", "j5"},
        contestant_ids={"padded", "plain"},
        delta_declared=0.02,
    )


def test_readable_report_shows_verdict_and_numbers():
    report = _sound_report()
    page = render_html(report, generated_at="2026-10-20T00:00:00+00:00")
    assert report.verdict is Verdict.READABLE
    assert "READABLE" in page
    assert "Noise floor" in page
    assert "Margin over floor" in page
    assert "arx audit" in page
    assert "The verdict is computed from the input JSON only" in page


def test_hostile_leader_name_is_escaped():
    report = AuditReport(
        verdict=Verdict.READABLE,
        leader_system="<script>alert(1)</script>",
        leader_score=0.9,
        floor_estimate=0.8,
        margin=0.1,
    )
    page = render_html(report, generated_at="2026-10-20T00:00:00+00:00")
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_undecidable_report_explains_the_refusal():
    report = AuditReport(
        verdict=Verdict.UNDECIDABLE,
        reasons=["single_generator"],
    )
    page = render_html(report, generated_at="2026-10-20T00:00:00+00:00")
    assert "UNDECIDABLE" in page
    assert "no floor is estimable" in page
    assert "single_generator" in page


def test_cli_writes_the_html_report(tmp_path):
    board = {
        "votes": _votes(5).tolist(),
        "scores": {"padded": 0.965, "plain": 0.70},
        "judge_ids": ["j1", "j2", "j3", "j4", "j5"],
        "contestant_ids": ["padded", "plain"],
    }
    src = tmp_path / "board.json"
    src.write_text(json.dumps(board), encoding="utf-8")
    out = tmp_path / "report.html"
    assert main(["audit", str(src), "--html", str(out)]) == 0
    assert out.exists()
    page = out.read_text(encoding="utf-8")
    assert "READABLE" in page
    assert "Noise floor" in page
