#!/usr/bin/env python3
"""Audit the ChaosNLI model rankings with arx's evaluation gate.

What this asks
--------------
Every ChaosNLI component carries 100 human annotations per item, and nine
2019-era transformer checkpoints have published per-item predictions on the
same items. So the question arx exists to ask can be asked on real data
instead of on a worked example:

    a leaderboard's rank difference is readable only if it exceeds the noise
    floor of the labels it was measured against -- and most of them do not.

All three components are audited here, in one run, because a single dataset
that happens to agree with the tool proves much less than three that were not
selected for it:

    snli     1,514 items, 3-way (e / n / c)
    mnli_m   1,599 items, 3-way (e / n / c)
    alphanli 1,532 items, 2-way (1 / 2)

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
The public release publishes each item's vote *counts* -- 100 votes per item as
a 2- or 3-vector -- not worker identities, so a ``(generators x items)`` matrix
cannot be read off the file directly. This script expands each item's counts
into 100 exchangeable pseudo-annotators using a seeded shuffle.

Every assignment consistent with the released counts is equally likely given
those counts, so the resulting leave-one-out floor is a draw of "how often one
annotator reproduces the majority of the other 99". It is a derived quantity,
not a measured one, and it is reported as such: ``--seeds`` runs the expansion
more than once so the reader can see the spread instead of having to assume it
is small.

Usage
-----
    python3 audit_chaosnli.py                     # all three components
    python3 audit_chaosnli.py --config snli       # just one
    python3 audit_chaosnli.py --offline           # reuse whatever is in _data/
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

HF_RAW = "https://huggingface.co/datasets/earino/chaosnli/resolve/main/raw"
# The upstream release points at a Dropbox zip (scripts/download_data.sh ->
# model_predictions.zip) and at the ChaosNLI v1.0 archive. Neither could be
# fetched -- Dropbox answered with an HTML interstitial -- so both the
# annotations and the predictions fall back to copies vendored in a
# third-party repository. Where a mirror is used it is named in the report
# rather than passed off as upstream.
MIRROR = "repos/TheGuy-26/chaosnli-deberta-modernbert/contents"
MIRROR_RAW = "raw.githubusercontent.com/TheGuy-26/chaosnli-deberta-modernbert/main"


def _truth_3way(record: dict) -> int:
    return {"e": 0, "n": 1, "c": 2}[record["majority_label"]]


def _truth_2way(record: dict) -> int:
    return int(record["majority_label"]) - 1


def _model_3way(label) -> int:
    return {"entailment": 0, "neutral": 1, "contradiction": 2}[label]


def _model_2way(label) -> int:
    return int(label) - 1


# Annotation hashes are the ones recorded in the mirror's own audit file
# (audit/AUDIT.md); we recompute them rather than trusting that file.
CONFIGS: dict[str, dict] = {
    "snli": {
        "annotation_urls": [f"{HF_RAW}/chaosNLI_snli.jsonl"],
        "annotation_file": "chaosNLI_snli.jsonl",
        "annotation_sha256": "99f9015ddda7d85f66a087452bc30d53974314fe27e7d589e2f41ad44bd509c1",
        "prediction_urls": [f"https://{MIRROR_RAW}/ChaosNLI/data/model_predictions/"
                            "model_predictions_for_snli_mnli.json"],
        "prediction_file": "model_predictions_for_snli_mnli.json",
        "prediction_sha256": "7c788bb9df6917f56a99a3bbf19c307bf2d02ad14d8f02537338b7bfb629bb53",
        "truth": _truth_3way,
        "model_label": _model_3way,
        "labels": 3,
    },
    "mnli_m": {
        "annotation_urls": [f"{HF_RAW}/chaosNLI_mnli_m.jsonl"],
        "annotation_file": "chaosNLI_mnli_m.jsonl",
        "annotation_sha256": "8eb49b589488e7b1a9ec95fb8a864bd50f0509b4f808eed0aed3f8167617664d",
        "prediction_urls": [f"https://{MIRROR_RAW}/ChaosNLI/data/model_predictions/"
                            "model_predictions_for_snli_mnli.json"],
        "prediction_file": "model_predictions_for_snli_mnli.json",
        "prediction_sha256": "7c788bb9df6917f56a99a3bbf19c307bf2d02ad14d8f02537338b7bfb629bb53",
        "truth": _truth_3way,
        "model_label": _model_3way,
        "labels": 3,
    },
    "alphanli": {
        "annotation_urls": [f"{HF_RAW}/chaosNLI_alphanli.jsonl"],
        "annotation_file": "chaosNLI_alphanli.jsonl",
        "annotation_sha256": "f55817ecc11a02ba3030cd7d2c567d23843456809020723abad53f4fde024c9e",
        "prediction_urls": [f"https://{MIRROR_RAW}/ChaosNLI/data/model_predictions/"
                            "model_predictions_for_abdnli.json"],
        "prediction_file": "model_predictions_for_abdnli.json",
        "prediction_sha256": "58ced70a3b2eb9b9ad4e0573e1a5b5096d76da1f2f9836dda14754c43726f938",
        "truth": _truth_2way,
        "model_label": _model_2way,
        "labels": 2,
    },
}

N_PSEUDO_ANNOTATORS = 100


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, part: Path) -> None:
    with urllib.request.urlopen(url, timeout=300) as response, part.open("wb") as fh:
        shutil.copyfileobj(response, fh)


def fetch(urls: list[str], path: Path, sha256: str, *, offline: bool) -> Path:
    """Return ``path``, downloading it if needed, and verify the hash."""
    if path.exists():
        got = _sha256(path)
        if got == sha256:
            print(f"[cache] {path.name}  sha256={got[:16]}...")
            return path
        print(f"[warn ] {path.name} cached hash {got[:16]}... != {sha256[:16]}..., refetching")
    if offline:
        raise SystemExit(f"{path} missing or stale, and --offline was given")

    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    for url in urls:
        print(f"[fetch] {url}")
        try:
            _download(url, part)
        except Exception as exc:  # noqa: BLE001 - any transport error falls through
            print(f"[fail ] {type(exc).__name__}: {exc}")
            continue
        if _sha256(part) == sha256:
            part.replace(path)
            print(f"[ok   ] {path.name}  sha256={sha256[:16]}...  "
                  f"({path.stat().st_size} bytes)")
            return path
        print(f"[fail ] {path.name}: downloaded content hashes "
              f"{_sha256(part)[:16]}..., expected {sha256[:16]}...")
        part.unlink(missing_ok=True)

    raise SystemExit(f"could not fetch {path.name} with the expected hash from any source")


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
    spec = CONFIGS[config]
    ann_path = fetch(spec["annotation_urls"], data_dir / spec["annotation_file"],
                     spec["annotation_sha256"], offline=offline)
    pred_path = fetch(spec["prediction_urls"], data_dir / spec["prediction_file"],
                      spec["prediction_sha256"], offline=offline)

    ann = load_annotations(ann_path)
    pred = json.loads(pred_path.read_text(encoding="utf-8"))

    # Common denominator: the item set every contestant and the floor share.
    conds = sorted(set.intersection(*[set(sub) & set(ann) for sub in pred.values()]))

    counts = np.array([ann[u]["label_count"] for u in conds], dtype=int)
    truth = np.array([spec["truth"](ann[u]) for u in conds], dtype=int)

    scores: dict[str, float] = {}
    per_item: dict[str, np.ndarray] = {}
    for name, sub in pred.items():
        got = np.array([spec["model_label"](sub[u]["predicted_label"]) for u in conds], dtype=int)
        correct = got == truth
        scores[name] = float(correct.mean())
        per_item[name] = correct.astype(int)

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top_two = [name for name, _ in ranked[:2]]

    floors = []
    report = None
    for seed in range(seeds):
        votes = build_votes(counts, seed)
        current = audit_leaderboard(
            votes,
            scores,
            leaderboard_item_ids=set(conds),
            floor_item_ids=set(conds),
            judge_ids={f"human_{i:03d}" for i in range(N_PSEUDO_ANNOTATORS)},
            contestant_ids=set(pred),
            per_item_scores={name: per_item[name] for name in top_two},
            delta_declared=delta,
        )
        floors.append(float(current.floor_estimate))
        report = report or current

    return {
        "config": config,
        "task": f"{spec['labels']}-way",
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
                "urls_tried": spec["annotation_urls"],
                "sha256": spec["annotation_sha256"],
                "license": "CC-BY-NC-4.0 (ChaosNLI v1.0, Nie et al. 2020, arXiv:2010.03532)",
            },
            "model_predictions": {
                "urls_tried": spec["prediction_urls"],
                "sha256": spec["prediction_sha256"],
                "note": "third-party vendored copy; the upstream Dropbox link was unreachable",
            },
        },
    }


def print_summary(result: dict) -> None:
    mde = result["mde"]
    print()
    print(f"--- {result['config']} ({result['task']}, n={result['n_items']} items, "
          f"m={result['n_generators']}, contestants={result['n_contestants']}) ---")
    print(f"verdict          : {result['verdict']}   reasons: {result['reasons'] or '(none)'}")
    print(f"floor            : {result['floor_estimate']:.4f}  ({result['floor_method']}, "
          f"seed spread {result['floor_seed_spread']:.2e}, chance {result['chance_level']:.4f})")
    print(f"leader           : {result['leader_system']} = {result['leader_score']:.4f}")
    print(f"margin over floor: {result['margin']:.4f}   MDE = {result['mde']:.4f} "
          f"({result['mde_method']})")
    print(f"top1-top2 gap    : {result['top1_top2_gap']:.4f} "
          f"({result['top1_top2_gap'] / mde:.2f}x MDE)")
    print(f"top1-top4 spread : {result['top1_top4_spread']:.4f} "
          f"({result['top1_top4_spread'] / mde:.2f}x MDE)")
    for row in result["board"]:
        print(f"    {row['accuracy']:.4f}  {row['system']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", choices=[*sorted(CONFIGS), "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=HERE / "_data")
    parser.add_argument("--out", type=Path, default=HERE / "results.json")
    parser.add_argument("--seeds", type=int, default=3,
                        help="how many expansions of the counts to repeat")
    parser.add_argument("--delta", type=float, default=0.02,
                        help="the margin the audit declares it cares about")
    parser.add_argument("--offline", action="store_true",
                        help="never download; fail if _data/ is incomplete")
    args = parser.parse_args(argv)

    names = sorted(CONFIGS) if args.config == "all" else [args.config]
    results: dict[str, dict] = {}
    for name in names:
        results[name] = run_audit(name, args.data_dir, args.seeds, args.delta, args.offline)
        print_summary(results[name])

    payload = {
        "generated_by": "case-studies/chaosnli-noise-floor/audit_chaosnli.py",
        "configs": results,
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
