#!/usr/bin/env python3
"""Segment 3: how far do a judge's labels move when only the option order changes?

The question this answers
-------------------------
The board's top ranks are separated by a fraction of the resolution bound. But
those gaps come from labels, and the labels were produced by a prompt. Before
concluding that a ranking is too tight to read, it is worth asking how much the
*same judge on the same item* moves when nothing about the item changes and
only the order in which the answer options are presented is permuted.

The panel release ships exactly that arm: 500 of the 1,000 items, re-asked
under three option permutations, for all 32 judges. The release's own analysis
guide lists presentation-order analyses under **"Not covered"**, so this
segment is filling a gap the release names rather than re-deriving a published
number.

Measured in the same units as the ranking gap
---------------------------------------------
A flip rate is not a score change: flips can cancel. So the headline number
here is the **accuracy swing across option orders** -- the same judge, scored
against the human majority on the same 500 items, under each of the three
permutations. That is directly comparable to a top1-top2 gap.

Two things stay diagnostics, not verdicts
-----------------------------------------
* The three permutations are one judge, not three producers, so a floor over
  them measures that judge's self-reproducibility across formatting, not an
  inter-annotator floor. It is reported as a diagnostic.
* The release states that **no same-prompt repeated-call noise floor is
  established by this archive**. An order flip is an artefact of presentation,
  not sampling noise, and the two are not interchangeable. Nothing here is
  offered as a repeated-call floor.

Also checked here: the release publishes a SHA-256 per vote file. Those hashes
are verified against what is on disk, which is stronger than the first-seen
recording used before this file existed.

Usage
-----
    python3 audit_position_bias.py
    python3 audit_position_bias.py --offline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from audit_chaosnli import CONFIGS, N_PSEUDO_ANNOTATORS, build_votes, fetch, load_annotations
from audit_judge_panel import (
    JUDGE_LABEL_TO_INT,
    JUDGES,
    manifest_file,
    panel_file,
)

from arx.audit import audit_leaderboard
from arx.audit.floor import loo_floor

HERE = Path(__file__).resolve().parent

# The release excludes this judge from its own panel analysis, for a reason it
# states: "complete here, but excluded so that one panel is used across
# datasets". The headline panel floor is checked both ways.
RELEASE_EXCLUDED = {"dsv32"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_published_hashes(manifest: dict, data_dir: Path) -> dict:
    """Check every vote file against the SHA-256 the release publishes."""
    report = {"baseline": {}, "swap": {}}
    for arm in ("baseline", "swap"):
        files = manifest["arms"][arm].get("files", {})
        for judge, meta in files.items():
            path = panel_file(data_dir, arm, judge)
            if not path.exists():
                report[arm][judge] = "missing"
                continue
            got = _sha256(path)
            report[arm][judge] = "ok" if got == meta.get("sha256") else f"mismatch:{got[:16]}"
    return report


def load_swap(data_dir: Path, uids: list[str]) -> dict[str, dict[int, np.ndarray]]:
    """judge -> variant -> per-item label (-1 where unparseable)."""
    index = {uid: i for i, uid in enumerate(uids)}
    out: dict[str, dict[int, np.ndarray]] = {}
    for judge in JUDGES:
        per_variant = {v: np.full(len(uids), -1, dtype=int) for v in (0, 1, 2)}
        path = panel_file(data_dir, "swap", judge)
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                record = json.loads(line)
                col = index.get(record["uid"])
                if col is None or record.get("parse_fail"):
                    continue
                per_variant[record["variant"]][col] = JUDGE_LABEL_TO_INT[record["label"]]
        out[judge] = per_variant
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=HERE / "_data")
    parser.add_argument("--out", type=Path, default=HERE / "results_position_bias.json")
    parser.add_argument("--delta", type=float, default=0.02)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)

    spec = CONFIGS["snli"]
    ann = load_annotations(fetch(spec["annotation_urls"],
                                 args.data_dir / spec["annotation_file"],
                                 spec["annotation_sha256"], offline=args.offline))
    pred = json.loads(fetch(spec["prediction_urls"],
                            args.data_dir / spec["prediction_file"],
                            spec["prediction_sha256"],
                            offline=args.offline).read_text(encoding="utf-8"))
    manifest = json.loads(manifest_file(args.data_dir, offline=args.offline).read_text("utf-8"))

    # Item set: the swap arm's 500 items, intersected with everything else.
    swap_uids: set[str] = set()
    for judge in JUDGES:
        path = panel_file(args.data_dir, "swap", judge)
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        swap_uids.add(json.loads(line)["uid"])
    conds = sorted(swap_uids & set(ann) & set.intersection(*[set(s) for s in pred.values()]))

    truth = np.array([{"e": 0, "n": 1, "c": 2}[ann[u]["majority_label"]] for u in conds], dtype=int)
    counts = np.array([ann[u]["label_count"] for u in conds], dtype=int)

    # The board, recomputed on the swap arm's own item set so the comparison is
    # like-for-like rather than across two different denominators.
    model_scores: dict[str, float] = {}
    model_items: dict[str, np.ndarray] = {}
    for name, sub in pred.items():
        got = np.array([{"entailment": 0, "neutral": 1, "contradiction": 2}[sub[u]["predicted_label"]]
                        for u in conds], dtype=int)
        model_scores[name] = float((got == truth).mean())
        model_items[name] = (got == truth).astype(int)
    ranked_models = sorted(model_scores.items(), key=lambda kv: -kv[1])
    top_gap_1_2 = ranked_models[0][1] - ranked_models[1][1]
    top_gap_1_4 = ranked_models[0][1] - ranked_models[3][1]

    swaps = load_swap(args.data_dir, conds)
    per_judge = []
    for judge, per_variant in swaps.items():
        accs = {}
        for variant, labels in per_variant.items():
            scored = labels >= 0
            accs[variant] = float((labels[scored] == truth[scored]).mean()) if scored.any() else None
        valid = [a for a in accs.values() if a is not None]
        swing = max(valid) - min(valid)
        base = per_variant[0]
        flips = 0
        cells = 0
        for variant in (1, 2):
            other = per_variant[variant]
            both = (base >= 0) & (other >= 0)
            cells += int(both.sum())
            flips += int((base[both] != other[both]).sum())
        flip_rate = flips / cells if cells else None
        stack = np.vstack([per_variant[0], per_variant[1], per_variant[2]])
        loo = loo_floor(stack)
        self_repro = float(np.nanmean(loo)) if not np.all(np.isnan(loo)) else None
        per_judge.append({
            "judge": judge,
            "accuracy_by_variant": {str(k): v for k, v in accs.items()},
            "accuracy_swing": swing,
            "flip_rate_vs_variant_0": flip_rate,
            "self_reproducibility_across_orders": self_repro,
        })
    per_judge.sort(key=lambda row: -(row["accuracy_swing"] or 0))
    swings = [row["accuracy_swing"] for row in per_judge if row["accuracy_swing"] is not None]
    flip_rates = [row["flip_rate_vs_variant_0"] for row in per_judge
                  if row["flip_rate_vs_variant_0"] is not None]

    # Sensitivity: the release excludes one judge from its panel analysis.
    human_votes = build_votes(counts, 0)
    judge_stack = np.vstack([swaps[j][0] for j in swaps])
    roster = [j for j in swaps]
    variants = {}
    for label, roster_used in (("all 32 judges", roster),
                               ("31 judges (release excludes dsv32)",
                                [j for j in roster if j not in RELEASE_EXCLUDED])):
        rows = [roster.index(j) for j in roster_used]
        votes = judge_stack[rows]
        variants[label] = {
            "panel_loo_floor": float(np.nanmean(loo_floor(votes))),
            "n_judges": len(rows),
        }
        for floor_label, floor_votes in (("human", human_votes), ("panel", votes)):
            report = audit_leaderboard(
                floor_votes, model_scores,
                leaderboard_item_ids=set(conds), floor_item_ids=set(conds),
                judge_ids=({f"human_{i:03d}" for i in range(N_PSEUDO_ANNOTATORS)}
                           if floor_label == "human" else set(roster_used)),
                contestant_ids=set(pred),
                per_item_scores={n: model_items[n]
                                 for n, _ in sorted(model_scores.items(),
                                                    key=lambda kv: -kv[1])[:2]},
                delta_declared=args.delta,
            )
            variants[label][f"verdict_floor_{floor_label}"] = report.verdict
            variants[label][f"floor_{floor_label}"] = float(report.floor_estimate)

    payload = {
        "generated_by": "case-studies/chaosnli-noise-floor/audit_position_bias.py",
        "segment": "presentation-order arm of chaosnli-snli-1000",
        "note_from_release": ("the release lists presentation-order analyses under "
                              "'Not covered', and states it establishes no same-prompt "
                              "repeated-call noise floor; this is an order artefact, "
                              "not a sampling floor"),
        "n_items": len(conds),
        "n_variants": 3,
        "release_excluded_judges": sorted(RELEASE_EXCLUDED),
        "published_hash_check": verify_published_hashes(manifest, args.data_dir),
        "board_on_swap_items": [{"system": k, "accuracy": v} for k, v in ranked_models],
        "top1_top2_gap": top_gap_1_2,
        "top1_top4_spread": top_gap_1_4,
        "per_judge": per_judge,
        "order_swing_median": float(np.median(swings)),
        "order_swing_mean": float(np.mean(swings)),
        "order_swing_max": float(np.max(swings)),
        "flip_rate_median": float(np.median(flip_rates)),
        "flip_rate_max": float(np.max(flip_rates)),
        "panel_sensitivity": variants,
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    bad = [f"{arm}:{j}={v}" for arm, per in payload["published_hash_check"].items()
           for j, v in per.items() if v != "ok"]
    print(f"items={len(conds)}  judges={len(per_judge)}  variants=3")
    print(f"published per-file hashes: {'all 64 files verified' if not bad else bad}")
    print(f"board on these items: top1-top2={top_gap_1_2:.4f}  top1-top4={top_gap_1_4:.4f}")
    print(f"order-induced accuracy swing: median={np.median(swings):.4f} "
          f"mean={np.mean(swings):.4f} max={np.max(swings):.4f}")
    print(f"label flip rate vs variant 0: median={np.median(flip_rates):.4f} "
          f"max={np.max(flip_rates):.4f}")
    print(f"swing / top1-top2 gap = {np.median(swings) / top_gap_1_2:.2f}x")
    for label, entry in variants.items():
        print(f"panel sensitivity, {label}: loo={entry['panel_loo_floor']:.4f}  "
              f"floor_human={entry['floor_human']:.4f}->{entry['verdict_floor_human']}  "
              f"floor_panel={entry['floor_panel']:.4f}->{entry['verdict_floor_panel']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
