#!/usr/bin/env python3
"""Audit the ChaosNLI-S model ranking with arx's evaluation gate.

What this asks
--------------
ChaosNLI-S carries 100 human annotations for each of 1,514 SNLI items, and
nine 2019-era transformer checkpoints have published per-item predictions on
exactly those items. So the question arx exists to ask can be asked on real
data instead of on a worked example:

    a leaderboard's rank difference is readable only if it exceeds the noise
    floor of the labels it was measured against -- and most of them do not.

What this does *not* do
-----------------------
* It does not certify any model. Every verdict it can print is a statement
  about the *measurement*: whether this apparatus could resolve this
  comparison at all.
* It does not redistribute the raw data. ChaosNLI is CC-BY-NC-4.0, so the
  script downloads it at run time into ``_data/`` (git-ignored) and verifies
  the published SHA-256 before using it.

The floor, and the one modelling step
-------------------------------------
The public release publishes each item's vote *counts* -- 100 votes per item
as a 3-vector -- not worker identities, so a ``(generators x items)`` matrix
cannot be read off the file directly. This script expands each item's counts
into 100 exchangeable pseudo-annotators using a seeded shuffle.

Every assignment consistent with the released counts is equally likely given
those counts, so the resulting leave-one-out floor is a draw of "how often
one annotator reproduces the majority of the other 99". It is a derived
quantity, not a measured one, and it is reported as such: ``--seeds`` runs
the expansion more than once so the reader can see the spread instead of
having to assume it is small.

Usage
-----
    python3 audit_chaosnli.py                    # snli, 3 seeds
    python3 audit_chaosnli.py --config mnli_m
    python3 audit_chaosnli.py --offline          # use whatever is in _data/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np

from arx.audit import audit_leaderboard

HERE = Path(__file__).resolve().parent

# --- Sources, with the hashes we verified ---------------------------------
#
# Annotations: the ChaosNLI v1.0 release as redistributed by the HF mirror
# ``earino/chaosnli``, whose own audit (audit/AUDIT.md) records these exact
# hashes. We check them again rather than trusting the mirror's word.
ANNOTATION_URL = {
    "snli": "https://huggingface.co/datasets/earino/chaosnli/resolve/main/raw/chaosNLI_snli.jsonl",
    "mnli_m": "https://huggingface.co/datasets/earino/chaosnli/resolve/main/raw/chaosNLI_mnli_m.jsonl",
}
ANNOTATION_FILE = {"snli": "chaosNLI_snli.jsonl", "mnli_m": "chaosNLI_mnli_m.jsonl"}
ANNOTATION_SHA256 = {
    "snli": "99f9015ddda7d85f66a087452bc30d53974314fe27e7d589e2f41ad44bd509c1",
    "mnli_m": "8eb49b589488e7b1a9ec95fb8a864bd50f0509b4f808eed0aed3f8167617664d",
}

# Model predictions: the upstream release points at a Dropbox zip
# (``scripts/download_data.sh`` -> model_predictions.zip). We could not fetch
# it -- Dropbox answered with an HTML interstitial -- so the file below is the
# copy of that release vendored in a third-party repository. It is a mirror,
# not the upstream URL, and it is labelled as such in the report.
MODEL_PREDS_URL = (
    "https://raw.githubusercontent.com/TheGuy-26/chaosnli-deberta-modernbert/"
    "main/ChaosNLI/data/model_predictions/model_predictions_for_snli_mnli.json"
)
MODEL_PREDS_FILE = "model_predictions_for_snli_mnli.json"
MODEL_PREDS_SHA256 = "7c788bb9df6917f56a99a3bbf19c307bf2d02ad14d8f02537338b7bfb629bb53"

# ChaosNLI's own vector order for the 3-way NLI components is (e, n, c).
LABEL_TO_INT = {"e": 0, "n": 1, "c": 2}
# The checkpoints emit SNLI's label names for the same three classes.
MODEL_LABEL_TO_INT = {"entailment": 0, "neutral": 1, "contradiction": 2}

N_PSEUDO_ANNOTATORS = 100


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, path: Path, sha256: str | None, *, offline: bool) -> Path:
    """Return ``path``, downloading it first if needed, and verify the hash."""
    if path.exists():
        got = _sha256(path)
        if sha256 is None or got == sha256:
            print(f"[cache] {path.name}  sha256={got[:16]}...")
            return path
        print(f"[warn ] {path.name} cached hash {got[:16]}... != {sha256[:16]}..., refetching")
    if offline:
        raise SystemExit(f"{path} missing or stale, and --offline was given")

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] {url}")
    part = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=300) as response, part.open("wb") as fh:
        shutil.copyfileobj(response, fh)
    part.replace(path)

    got = _sha256(path)
    if sha256 is not None and got != sha256:
        path.unlink(missing_ok=True)
        raise SystemExit(f"SHA-256 mismatch for {path.name}: {got} != {sha256}")
    print(f"[ok   ] {path.name}  sha256={got[:16]}...  ({path.stat().st_size} bytes)")
    return path


def load_annotations(path: Path) -> dict[str, dict]:
    items: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                items[record["uid"]] = record
    return items


def build_votes(counts: np.ndarray, seed: int, m: int = N_PSEUDO_ANNOTATORS) -> np.ndarray:
    """Expand per-item vote counts into ``(m, n_items)`` pseudo-annotator votes.

    Each item's counts describe a multiset of ``m`` labels; the shuffle picks
    one of the ``m!`` orderings consistent with it. See the module docstring
    for why this is the honest construction given a counts-only release.
    """
    rng = np.random.default_rng(seed)
    votes = np.empty((m, counts.shape[0]), dtype=int)
    for j, row in enumerate(counts):
        labels = np.repeat(np.arange(len(row)), row)
        if labels.size != m:
            raise SystemExit(f"item {j}: counts sum to {labels.size}, expected {m}")
        rng.shuffle(labels)
        votes[:, j] = labels
    return votes


def run_audit(config: str, data_dir: Path, seeds: int, delta: float, offline: bool) -> dict:
    ann_path = fetch(
        ANNOTATION_URL[config], data_dir / ANNOTATION_FILE[config], ANNOTATION_SHA256[config],
        offline=offline,
    )
    pred_path = fetch(
        MODEL_PREDS_URL, data_dir / MODEL_PREDS_FILE, MODEL_PREDS_SHA256, offline=offline,
    )

    ann = load_annotations(ann_path)
    pred = json.loads(pred_path.read_text(encoding="utf-8"))

    # Common denominator: the item set every contestant and the floor share.
    per_model_items = {name: set(sub) & set(ann) for name, sub in pred.items()}
    conds = sorted(set.intersection(*per_model_items.values()))

    counts = np.array([ann[u]["label_count"] for u in conds], dtype=int)
    truth = np.array([LABEL_TO_INT[ann[u]["majority_label"]] for u in conds], dtype=int)

    scores: dict[str, float] = {}
    per_item: dict[str, np.ndarray] = {}
    for name, sub in pred.items():
        got = np.array([MODEL_LABEL_TO_INT[sub[u]["predicted_label"]] for u in conds], dtype=int)
        correct = got == truth
        scores[name] = float(correct.mean())
        per_item[name] = correct.astype(int)

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top_two = [name for name, _ in ranked[:2]]

    reports = []
    floors = []
    for seed in range(seeds):
        votes = build_votes(counts, seed)
        report = audit_leaderboard(
            votes,
            scores,
            leaderboard_item_ids=set(conds),
            floor_item_ids=set(conds),
            judge_ids={f"human_{i:03d}" for i in range(N_PSEUDO_ANNOTATORS)},
            contestant_ids=set(pred),
            per_item_scores={name: per_item[name] for name in top_two},
            delta_declared=delta,
        )
        floors.append(float(report.floor_estimate))
        if seed == 0:
            reports.append(report)

    report = reports[0]
    return {
        "config": config,
        "n_items": len(conds),
        "n_generators": int(report.n_generators),
        "n_contestants": int(report.n_contestants),
        "floor_generators": "100 pseudo-annotators expanded from per-item vote counts",
        "verdict": report.verdict,
        "floor_estimate": float(report.floor_estimate),
        "floor_method": report.floor_method,
        "floor_across_seeds": floors,
        "floor_seed_spread": max(floors) - min(floors),
        "chance_level": report.extra.get("chance_level"),
        "floor_vs_chance": report.extra.get("floor_vs_chance"),
        "leader_system": report.leader_system,
        "leader_score": float(report.leader_score),
        "margin": float(report.margin),
        "mde": float(report.mde),
        "mde_method": report.extra.get("mde_method"),
        "delta_declared": float(report.delta_declared),
        "threshold": float(report.threshold),
        "reasons": list(report.reasons),
        "board": [{"system": name, "accuracy": score} for name, score in ranked],
        "top1_top2_gap": ranked[0][1] - ranked[1][1],
        "top1_top4_spread": ranked[0][1] - ranked[3][1],
        "sources": {
            "annotations": {
                "url": ANNOTATION_URL[config],
                "sha256": ANNOTATION_SHA256[config],
                "license": "CC-BY-NC-4.0 (ChaosNLI v1.0, Nie et al. 2020, arXiv:2010.03532)",
            },
            "model_predictions": {
                "url": MODEL_PREDS_URL,
                "sha256": MODEL_PREDS_SHA256,
                "note": "third-party vendored copy; the upstream Dropbox link was unreachable",
            },
        },
    }


def print_summary(result: dict) -> None:
    print()
    print(f"config           : {result['config']}  (n={result['n_items']} items, "
          f"m={result['n_generators']} generators, "
          f"{result['n_contestants']} contestants)")
    print(f"verdict          : {result['verdict']}")
    print(f"reasons          : {result['reasons'] or '(none)'}")
    print(f"floor            : {result['floor_estimate']:.4f}  "
          f"({result['floor_method']}, spread over seeds "
          f"{result['floor_seed_spread']:.2e})")
    print(f"chance level     : {result['chance_level']!r}")
    print(f"leader           : {result['leader_system']}  "
          f"accuracy={result['leader_score']:.4f}")
    print(f"margin over floor: {result['margin']:.4f}")
    print(f"declared delta   : {result['delta_declared']:.4f}")
    print(f"MDE              : {result['mde']:.4f}  ({result['mde_method']})")
    print()
    print("board:")
    for row in result["board"]:
        print(f"  {row['accuracy']:.4f}  {row['system']}")
    print()
    print(f"top1-top2 gap    : {result['top1_top2_gap']:.4f}  "
          f"({result['top1_top2_gap'] / result['mde']:.1f}x the MDE)")
    print(f"top1-top4 spread : {result['top1_top4_spread']:.4f}  "
          f"({result['top1_top4_spread'] / result['mde']:.1f}x the MDE)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", choices=sorted(ANNOTATION_URL), default="snli")
    parser.add_argument("--data-dir", type=Path, default=HERE / "_data")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: results.json next to this script")
    parser.add_argument("--seeds", type=int, default=3,
                        help="how many expansions of the counts to repeat")
    parser.add_argument("--delta", type=float, default=0.02,
                        help="the margin the audit declares it cares about")
    parser.add_argument("--offline", action="store_true",
                        help="never download; fail if _data/ is incomplete")
    args = parser.parse_args(argv)

    result = run_audit(args.config, args.data_dir, args.seeds, args.delta, args.offline)
    print_summary(result)

    out = args.out or (HERE / "results.json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
