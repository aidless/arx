#!/usr/bin/env python3
"""Decision-level calibration: does the verdict match known ground truth?

`measure_floor_calibration.py` checks the floor estimate. This script checks
the thing a user actually consumes: the **verdict**. Boards are simulated with
a *known* true gap between two systems (paired per-item outcomes drawn from a
latent with a controlled difference), so "was the gate right?" is answerable
instead of assumed.

Design
------
* n_items = 400, two systems, SEEDS seeds per cell.
* Cell = true accuracy gap ``d`` in {0.000, 0.005, 0.010, 0.020, 0.050, 0.100}.
* The reported statistic is the **READABLE rate** at each true gap -- a power
  curve. There is no invented "correctness" threshold: a refusal at a small
  gap is the documented behaviour, not an error, and pretending otherwise
  would manufacture a number the tool never claimed.
* Two judge pools are compared: 5 clean generators (0.93 each) and the same
  pool with one 0.85-accuracy ringer. The second arm exists because a weak
  generator lowers the leave-one-out floor, and the floor is what the margin
  is measured against.

Outputs: docs/measurements/verdict_calibration.json and a power curve PNG.
No network, no model, no credentials.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arx.audit import audit_leaderboard  # noqa: E402

SEEDS = 80
N_ITEMS = 400
GAPS = (0.0, 0.005, 0.010, 0.020, 0.050, 0.100)
DELTA = 0.02


def generators(seed: int, n_items: int, accuracies: list[float]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 2, size=n_items)
    return np.array([np.where(rng.random(n_items) < a, truth, 1 - truth) for a in accuracies])


def paired_systems(seed: int, n_items: int, gap: float, base: float = 0.90):
    """Two systems whose true accuracies differ by exactly ``gap``.

    Items are drawn once; the weak system is correct with probability ``base``
    and the strong system with ``base + gap``. The per-item outcomes are kept
    paired, which is what the cluster bootstrap MDE is designed for.
    """
    rng = np.random.default_rng(10_000 + seed)
    strong = rng.random(n_items) < (base + gap)
    weak = rng.random(n_items) < base
    return strong.astype(int), weak.astype(int)


def run_cell(gap: float, ringer: bool) -> dict:
    accs = [0.93, 0.93, 0.93, 0.93, 0.85] if ringer else [0.93] * 5
    verdicts, margins, mdes = [], [], []
    for seed in range(SEEDS):
        g = generators(seed, N_ITEMS, accs)
        strong, weak = paired_systems(seed, N_ITEMS, gap)
        scores = {"strong": float(strong.mean()), "weak": float(weak.mean())}
        rep = audit_leaderboard(
            g, scores,
            judge_ids=[f"g{i}" for i in range(5)],
            contestant_ids=["strong", "weak"],
            per_item_scores={"strong": strong, "weak": weak},
            delta_declared=DELTA, n_boot=300, seed=seed,
        )
        verdicts.append(rep.verdict)
        margins.append(float(rep.margin))
        mdes.append(float(rep.mde) if rep.mde is not None else float("nan"))

    claimed = sum(v == "READABLE" for v in verdicts) / SEEDS
    return {
        "true_gap": gap,
        "readable_rate": round(claimed, 4),
        "mean_margin": round(float(np.mean(margins)), 4),
        "mean_mde": round(float(np.nanmean(mdes)), 4),
        "verdict_mix": {v: verdicts.count(v) for v in sorted(set(verdicts))},
    }


def main() -> int:
    out = {"settings": {"seeds": SEEDS, "n_items": N_ITEMS, "delta_declared": DELTA},
           "cells": {"clean_floor": [], "degraded_floor": []}}
    for label, ringer in (("clean_floor", False), ("degraded_floor", True)):
        for gap in GAPS:
            out["cells"][label].append(run_cell(gap, ringer))
            c = out["cells"][label][-1]
            print(f"{label:14} gap={gap:<6} readable={c['readable_rate']:<6} "
                  f"margin={c['mean_margin']:<7} mde={c['mean_mde']:<7} "
                  f"verdicts={c['verdict_mix']}")

    dest = ROOT / "docs" / "measurements"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "verdict_calibration.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; JSON written only")
        return 0

    fig, ax = plt.subplots(figsize=(7.6, 4), dpi=160)
    for label, colour, name in (("clean_floor", "#1f6f5f", "5 clean generators"),
                                ("degraded_floor", "#c4552d", "one 0.85 ringer")):
        xs = [c["true_gap"] for c in out["cells"][label]]
        ys = [c["readable_rate"] for c in out["cells"][label]]
        ax.plot(xs, ys, marker="o", color=colour, linewidth=1.4, label=name)
    ax.axvline(DELTA, color="#555", linestyle="--", linewidth=1.0)
    ax.text(DELTA, 0.04, " delta_declared = 0.02", fontsize=8, color="#555")
    ax.set_xlabel("true accuracy gap between the two systems")
    ax.set_ylabel("rate of READABLE verdict")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Verdict calibration against known ground truth "
                 f"({SEEDS} seeds per point, n={N_ITEMS} items)", fontsize=9.5)
    ax.grid(color="#e6e9e8", linewidth=0.7)
    ax.legend(fontsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    (ROOT / "docs" / "assets").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "docs" / "assets" / "verdict_calibration.png")
    print("wrote docs/assets/verdict_calibration.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
