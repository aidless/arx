"""Example 06 -- a single-file HTML audit report.

The audit is only useful if someone opens it. This example builds a sound board,
renders the report, and prints the path. No network, no model.

Run:  PYTHONPATH=src python3 examples/06_audit_report.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from arx.audit import audit_leaderboard
from arx.audit.report_html import render_html


def votes(n_generators: int = 5, n_items: int = 400, agreement: float = 0.93, seed: int = 0):
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 2, size=n_items)
    return np.array(
        [
            np.where(rng.random(n_items) > agreement, 1 - truth, truth)
            for _ in range(n_generators)
        ],
        dtype=int,
    )


def main() -> None:
    report = audit_leaderboard(
        votes(),
        {"strong": 0.965, "weak": 0.70},
        judge_ids={"g1", "g2", "g3", "g4", "g5"},
        contestant_ids={"strong", "weak"},
        delta_declared=0.02,
    )
    out = Path(__file__).resolve().parent / "_out" / "audit_report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(report), encoding="utf-8")

    print(report.to_text())
    print(f"\nhtml report: {out}")
    print("open it in a browser; it is a single file with no external assets.")


if __name__ == "__main__":
    main()
