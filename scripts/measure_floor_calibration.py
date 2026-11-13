#!/usr/bin/env python3
"""Measure the noise-floor estimator against two baselines (simulation).

The gate's first job is to estimate how much of a leaderboard gap is
measurement noise. This script asks whether that estimate is *calibrated*,
using boards whose ground truth is known by construction: 5 gold generators
each with true accuracy ``a`` on 400 items, so the true floor of any held-out
generator is exactly ``a``.

Baselines compared (all computed from the same votes matrix):
  * ``loo``      -- arx's leave-one-out floor (what the package ships)
  * ``optimistic`` -- the best single generator's accuracy (an oracle that
    cannot be observed in a real leaderboard; it is the upper bound any
    "take the strongest judge as the floor" heuristic converges to)
  * ``mean``     -- the mean generator accuracy (a naive lower bound)

Reported: bias (mean estimate - true floor), sd across seeds, and 95%
coverage of the true floor. Every number below is produced by this script;
run it with ``python3 scripts/measure_floor_calibration.py``. No network, no
model, no credentials.

Outputs:
  docs/measurements/floor_calibration.json
  docs/assets/floor_calibration.png
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arx.audit.floor import column_majority, loo_floor  # noqa: E402

SEEDS = 200
N_ITEMS = 400
N_GEN = 5
AGREEMENT = 0.93


def make_votes(seed: int, agreement: float = AGREEMENT) -> np.ndarray:
    """5 generators with a known accuracy on N_ITEMS items."""
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 2, size=N_ITEMS)
    return np.array([np.where(rng.random(N_ITEMS) < agreement, truth, 1 - truth)
                     for _ in range(N_GEN)])


def consensus_agreement(votes: np.ndarray) -> np.ndarray:
    """Per-generator agreement with the majority of *all* generators.

    This is the observable substitute for accuracy that the naive baselines use.
    The generator is inside the consensus it is graded against, so the number is
    inflated -- that inflation is the thing the leave-one-out floor removes.
    """
    consensus = column_majority(np.asarray(votes))
    return np.array([float(np.mean(votes[g] == consensus)) for g in range(votes.shape[0])])


def main() -> int:
    est: dict[str, list[float]] = {"loo": [], "optimistic": [], "mean": []}
    for seed in range(SEEDS):
        v = make_votes(seed)
        per_gen = [float(x) for x in loo_floor(v) if not np.isnan(x)]
        est["loo"].append(statistics.mean(per_gen))
        est["optimistic"].append(float(consensus_agreement(v).max()))
        est["mean"].append(float(consensus_agreement(v).mean()))

    true_floor = AGREEMENT
    out = {"settings": {"seeds": SEEDS, "n_items": N_ITEMS, "n_generators": N_GEN,
                        "generator_accuracy": AGREEMENT},
           "true_floor": true_floor, "estimators": {}}
    for name, vals in est.items():
        bias = statistics.mean(vals) - true_floor
        sd = statistics.pstdev(vals)
        lo, hi = statistics.mean(vals) - 1.96 * sd, statistics.mean(vals) + 1.96 * sd
        out["estimators"][name] = {
            "mean": round(statistics.mean(vals), 4),
            "bias": round(bias, 4),
            "sd": round(sd, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "covers_true_floor": bool(lo <= true_floor <= hi),
            "mean_abs_error": round(statistics.mean(abs(x - true_floor) for x in vals), 4),
        }

    dest = ROOT / "docs" / "measurements"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "floor_calibration.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; JSON written only")
        return 0

    fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
    for name, colour in (("loo", "#1f6f5f"), ("mean", "#6b8f89"), ("optimistic", "#c4552d")):
        ax.hist(est[name], bins=24, alpha=0.65, label=name, color=colour,
                edgecolor="white", linewidth=0.6)
    ax.axvline(true_floor, color="#222", linestyle="--", linewidth=1.2,
               label=f"true floor = {true_floor}")
    ax.set_xlabel("floor estimate")
    ax.set_ylabel(f"seeds (n={SEEDS})")
    ax.set_title("Is the noise floor calibrated? 5 generators, 400 items, known ground truth",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", color="#e6e9e8", linewidth=0.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    (ROOT / "docs" / "assets").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "docs" / "assets" / "floor_calibration.png")
    print("wrote docs/assets/floor_calibration.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
