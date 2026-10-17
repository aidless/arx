#!/usr/bin/env python3
"""audit_alpaca_eval.py -- why AlpacaEval 2.0 fails the gate's premises.

This is the artifact behind the "Why not AlpacaEval" section of this
case study and the corresponding sentence in ARTICLE.md /
ARTICLE.zh-CN.md. It exists because that claim used to live only in
prose: a reader who tried to check it found no script and no result file.

The gate needs per-item labels produced by annotators independent of the
judged set (see docs/evaluation-gate.md). This script records that
AlpacaEval 2.0 cannot supply that, from the release's own published
files:

  P1  what is published about human labels is one aggregate
      `Human agreement` column per evaluator -- nothing per item;
  P2  of the five evaluators AlpacaEval 2.0 publishes, three are prompt
      variants of the same GPT-4-turbo and two are single runs of
      different models (Claude 3 Opus, Mistral Large), so a leave-one-out
      floor over them would measure one judge's self-consistency rather
      than an inter-annotator noise floor;
  P3  two of those GPT-4-turbo variants return an identical verdict on
      every one of the 805 items -- one judge run, not two producers;
  P4  the judged contestant set contains GPT-4-family models while the
      judges are GPT-4-family -- an actor-disjointness violation.

Input provenance and honesty notes:

  * The inputs are the release's published per-evaluator files. They are
    NOT redistributed here (and `_data/` is git-ignored); point
    --data-dir at your own copy. Every input is verified against the
    SHA-256 pinned in INPUTS below, and the script refuses to run on a
    mismatch -- a hash check is the only thing that makes a third-party
    copy comparable to the numbers recorded here.
  * The five-evaluator list is pinned from AlpacaEval's own repository
    listing (`docs/data_AlpacaEval_2/`), not inferred at run time.
  * P1 is an observation about what the release publishes. It is not
    proof that no per-item human labels exist anywhere in the world.

Usage:
    python3 audit_alpaca_eval.py [--data-dir DIR] [--out PATH]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE / "_data" / "alpaca"
DEFAULT_OUT = HERE / "results_alpaca_eval.json"

# SHA-256 of every input this script reads. Verify, never trust:
# a copy that fails these hashes is not the file the numbers came from.
INPUTS = {
    "ann_gpt41106__alpaca_eval_gpt4.json":
        "384930aa5e497028df2d23ea7b57db6ffdc2635c7af1a8dbd1ea8305e910773e",
    "ann_gpt41106__alpaca_eval_cot_gpt4_turbo_fn.json":
        "73262f232ca45b3d6cc67905d71dd14873bd59c444acf4da65f63314fc2a4cb4",
    "ann_gpt41106__weighted_alpaca_eval_gpt4_turbo.json":
        "2a05177cffbab8fa409adf7ee1f0f90414cab169afe391b861489a5734c79363",
    "evaluators_lb.csv":
        "db679d719e2a8705333ea2adbc184e77958226de58fe3146a815763e18ec5a3d",
    "board_wgpt4.csv":
        "fe32bf85c2f7429038faa4c9a884d1d131726454ff84a4b810a91c1c7a2aa0e9",
    "board_cot_gpt4turbo.csv":
        "ebb9492f5650edb0ee8f0bb6ab0491e1373dbdf9bd27887c9ba9c4fa177754d5",
    "board_gpt4turbo_fn.csv":
        "a2c393aae1652e883a75eb3ca2a9838c726ce9c3b55385f0748015f0fbf7675d",
    "board_claude3opus.csv":
        "1a50d5ab466efad5e8b76a8625b92943ce5271d4776b224401c0ce360b1dcec8",
    "board_mistrallarge.csv":
        "80da67bb9f6764eded69c43150311315a20d707f71b88a655de743a6328bf0a4",
}

# The five evaluators AlpacaEval 2.0 publishes, pinned from its own
# repository listing (docs/data_AlpacaEval_2/*_leaderboard.csv).
# Classification is stated explicitly rather than guessed from the name,
# so a reader can check it against the names in the release.
PUBLISHED_EVALUATORS = {
    "alpaca_eval_gpt4_turbo_fn": ("gpt4_turbo_variant", "board_gpt4turbo_fn.csv"),
    "alpaca_eval_cot_gpt4_turbo_fn": ("gpt4_turbo_variant", "board_cot_gpt4turbo.csv"),
    "weighted_alpaca_eval_gpt4_turbo": ("gpt4_turbo_variant", "board_wgpt4.csv"),
    "claude_3_opus_ranking": ("other_model_single_run", "board_claude3opus.csv"),
    "mistral-large-2402_ranking": ("other_model_single_run", "board_mistrallarge.csv"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_verified(data_dir: Path) -> dict:
    """Read every pinned input, verifying its SHA-256 first."""
    missing, mismatched, blobs = [], [], {}
    for name, want in INPUTS.items():
        p = data_dir / name
        if not p.exists():
            missing.append(name)
            continue
        got = sha256(p)
        if got != want:
            mismatched.append((name, want, got))
            continue
        blobs[name] = p

    if missing or mismatched:
        print("INPUT VERIFICATION FAILED", file=sys.stderr)
        for n in missing:
            print(f"  missing:    {n}", file=sys.stderr)
        for n, want, got in mismatched:
            print(f"  hash differs: {n}\n    want {want}\n    got  {got}",
                  file=sys.stderr)
        print(
            f"\nPut the release's published files in {data_dir} "
            f"(see README 'Sources, with hashes'). A copy that fails these "
            f"hashes is not the file the numbers below came from -- the "
            f"script refuses to report numbers from unverified inputs.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return blobs


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def check_p1_human_labels(blobs: dict) -> dict:
    """What the release publishes about human labels."""
    rows = read_csv_rows(blobs["evaluators_lb.csv"])
    header, body = rows[0], rows[1:]
    human_cols = [c for c in header if "human" in c.lower()]
    agreement = {
        r[0]: float(r[header.index(human_cols[0])])
        for r in body
        if r and r[0] in PUBLISHED_EVALUATORS and human_cols
    }
    return {
        "columns_exposing_human_signal": human_cols,
        "per_item_human_columns": [
            c for c in header if "human" in c.lower() and "item" in c.lower()
        ],
        "aggregate_human_agreement_for_published_evaluators": agreement,
        "note": (
            "The human signal is one aggregate percentage per evaluator. "
            "No per-item human vote is published, so no floor can be built "
            "from an independent human producer on this board."
        ),
    }


def check_p2_composition(board_dirs: dict) -> dict:
    """Of the five published evaluators, how many are the same GPT-4?"""
    gpt4 = [n for n, (k, _) in PUBLISHED_EVALUATORS.items()
            if k == "gpt4_turbo_variant"]
    other = [n for n, (k, _) in PUBLISHED_EVALUATORS.items()
             if k != "gpt4_turbo_variant"]
    return {
        "published_evaluators": sorted(PUBLISHED_EVALUATORS),
        "gpt4_turbo_variants": sorted(gpt4),
        "other_single_runs": sorted(other),
        "n_published": len(PUBLISHED_EVALUATORS),
        "note": (
            "Three of the five are prompt variants of one GPT-4-turbo, so a "
            "leave-one-out floor over them measures that judge's "
            "self-consistency, not an inter-annotator noise floor."
        ),
    }


def check_p3_identical_variants(blobs: dict) -> dict:
    """Two GPT-4-turbo variants identical on all 805 items?"""
    names = {
        "alpaca_eval_gpt4": "ann_gpt41106__alpaca_eval_gpt4.json",
        "alpaca_eval_cot_gpt4_turbo_fn":
            "ann_gpt41106__alpaca_eval_cot_gpt4_turbo_fn.json",
        "weighted_alpaca_eval_gpt4_turbo":
            "ann_gpt41106__weighted_alpaca_eval_gpt4_turbo.json",
    }
    data = {k: read_json(blobs[v]) for k, v in names.items()}
    lengths = {k: len(v) for k, v in data.items()}

    def same_sequence(x, y):
        return ([r["instruction"] for r in x] == [r["instruction"] for r in y])

    pairs = {}
    keys = sorted(data)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            identical = sum(
                1 for j in range(min(len(data[a]), len(data[b])))
                if data[a][j]["preference"] == data[b][j]["preference"]
            )
            pairs[f"{a} vs {b}"] = {
                "identical_verdicts": identical,
                "compared": min(len(data[a]), len(data[b])),
            }

    identical_pairs = [k for k, v in pairs.items()
                       if v["identical_verdicts"] == v["compared"]
                       and v["compared"] > 0]
    return {
        "annotation_files": {k: v for k, v in names.items()},
        "items_per_file": lengths,
        "same_item_sequence": same_sequence(
            data["alpaca_eval_cot_gpt4_turbo_fn"],
            data["weighted_alpaca_eval_gpt4_turbo"],
        ),
        "pairwise": pairs,
        "pairs_identical_on_every_item": identical_pairs,
        "note": (
            "An identical verdict on every item means one judge run counted "
            "twice, not two independent producers."
            if identical_pairs else
            "No pair was identical on every item; the prose claim does not "
            "hold on this data."),
    }


def _gpt_family(name: str) -> bool:
    n = name.lower()
    return "gpt4" in n or "gpt-4" in n or "gpt-3.5" in n or "gpt35" in n


def check_p4_actor_disjointness(blobs: dict) -> dict:
    """Are GPT-4-family models judged by GPT-4-family judges?"""
    per_board = {}
    for evaluator, (_kind, fname) in PUBLISHED_EVALUATORS.items():
        if fname not in blobs:
            continue
        rows = read_csv_rows(blobs[fname])
        contestants = [r[0] for r in rows[1:] if r and r[0] and r[0] != "NullModel"]
        judge_is_gpt = _gpt_family(evaluator)
        overlap = sorted(c for c in contestants if _gpt_family(c))
        per_board[evaluator] = {
            "judge_gpt_family": judge_is_gpt,
            "n_contestants": len(contestants),
            "n_total": (
                rows[1][6] if len(rows) > 1 and len(rows[1]) > 6 else None
            ),
            "gpt_family_contestants": overlap,
        }
    violations = sorted(e for e, v in per_board.items()
                        if v["judge_gpt_family"] and v["gpt_family_contestants"])
    return {
        "per_board": per_board,
        "boards_with_actor_overlap": violations,
        "note": (
            "On these boards a GPT-4-family judge ranks GPT-4-family "
            "contestants, so the judge is not an independent producer of "
            "the labels it is ranked by."
        ),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    blobs = load_verified(args.data_dir)

    result = {
        "board": "AlpacaEval 2.0",
        "gate_verdict": "PREMISES_FAIL",
        "inputs_verified": {n: INPUTS[n] for n in sorted(blobs)},
        "p1_human_labels": check_p1_human_labels(blobs),
        "p2_evaluator_composition": check_p2_composition(blobs),
        "p3_identical_variants": check_p3_identical_variants(blobs),
        "p4_actor_disjointness": check_p4_actor_disjointness(blobs),
    }

    args.out.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n",
                        encoding="utf-8")

    p3 = result["p3_identical_variants"]
    print(f"AlpacaEval 2.0 gate verdict: {result['gate_verdict']}")
    print(f"  P1 aggregate-only human signal: "
          f"{result['p1_human_labels']['columns_exposing_human_signal']}")
    print(f"  P2 published evaluators: "
          f"{result['p2_evaluator_composition']['n_published']} "
          f"({len(result['p2_evaluator_composition']['gpt4_turbo_variants'])} "
          f"GPT-4-turbo variants + "
          f"{len(result['p2_evaluator_composition']['other_single_runs'])} "
          f"other single runs)")
    print(f"  P3 identical on every item: "
          f"{p3['pairs_identical_on_every_item']} "
          f"(items: {p3['items_per_file']})")
    print(f"  P4 boards with judge/contestant overlap: "
          f"{result['p4_actor_disjointness']['boards_with_actor_overlap']}")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
