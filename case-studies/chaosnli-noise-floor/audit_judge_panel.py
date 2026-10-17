#!/usr/bin/env python3
"""Audit one nine-model ranking twice: against a human floor, then an LLM-judge floor.

Why this is the interesting pair
--------------------------------
The companion script (``audit_chaosnli.py``) measures a board against one floor:
100 human annotations per item. This script holds the board fixed -- the same
nine checkpoints, the same 1,000 ChaosNLI-S items -- and swaps only the thing
that produced the labels:

    floor A  100 human annotations per item, expanded into pseudo-annotators
    floor B  32 named LLM judges, one vote per item, real identities

Both floors satisfy arx's premises. The judges are a different actor set from
the nine checkpoints, so actor-disjointness holds either way, and both floors
are computed on the board's own item set.

What that buys is a direct answer to a question LLM evaluation usually assumes
away: **how much of a ranking's readability is a property of the board, and how
much of it is a property of who produced the labels?** A floor is not a
constant of nature. An LLM panel agrees with itself differently than 100 humans
do, and the verdict moves with it.

What this does *not* do
-----------------------
* It does not rank the judges as a board. Passing the panel as both the judge
  set and the contestant set is an actor-disjointness violation, and the gate
  refuses it -- correctly. The panel's own agreement is reported here as a
  diagnostic, computed with ``arx.audit.loo_floor`` directly, not smuggled in
  as a verdict.
* It does not redistribute the votes. The panel release declares ChaosNLI as a
  non-redistributed human reference; both are downloaded at run time.

Usage
-----
    python3 audit_judge_panel.py
    python3 audit_judge_panel.py --offline
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
from audit_chaosnli import CONFIGS, N_PSEUDO_ANNOTATORS, build_votes, fetch, load_annotations

from arx.audit import audit_leaderboard
from arx.audit.floor import loo_floor

HERE = Path(__file__).resolve().parent

PANEL_REPO = "Chao1208/32judges-votes"
PANEL_RAW = f"https://raw.githubusercontent.com/{PANEL_REPO}/main"
PANEL_DATASET = "datasets/chaosnli-snli-1000"

# The panel manifest publishes the item-list hash; the vote files do not carry
# published hashes, so their hashes are recorded here as first-seen values
# rather than verified against anything.
UIDS_URLS = [f"{PANEL_RAW}/{PANEL_DATASET}/items/uids.txt"]
UIDS_SHA256 = "f0e5bea82f6d674d7e689488bb732f3a010e5c9b7176fdbe0093ffcd1928c209"
MANIFEST_URLS = [f"{PANEL_RAW}/{PANEL_DATASET}/manifest.json"]

JUDGE_LABEL_TO_INT = {"e": 0, "n": 1, "c": 2}
JUDGES = [
    "doubao18", "doubao20pro", "dsv32", "dsv4flash", "dsv4pro", "gem25pro", "gem31pro",
    "gem36flash", "gem37flash", "glm5", "glm51", "glm52", "glm53", "gpt41", "gpt54",
    "gpt56sol", "gpt56terra", "grok45", "grok46", "haiku45", "kimik25", "kimik26",
    "kimik27code", "kimik3", "minimaxm3", "o4mini", "opus5", "qwen35plus", "qwen37plus",
    "qwen38max", "qwen3max", "sonnet46",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_unhashed(urls: list[str], path: Path, *, offline: bool) -> Path:
    """Fetch a file that has no published hash, and record what we got."""
    if path.exists():
        return path
    if offline:
        raise SystemExit(f"{path} missing, and --offline was given")
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    for url in urls:
        print(f"[fetch] {url}")
        try:
            with urllib.request.urlopen(url, timeout=300) as response, part.open("wb") as fh:
                shutil.copyfileobj(response, fh)
        except Exception as exc:  # noqa: BLE001
            print(f"[fail ] {type(exc).__name__}: {exc}")
            continue
        part.replace(path)
        return path
    raise SystemExit(f"could not fetch {path.name} from any source")


def panel_file(data_dir: Path, arm: str, judge: str) -> Path:
    """Where a panel vote file for one arm and one judge lives once cached."""
    return data_dir / "panel" / f"{arm}__{judge}.jsonl"


def manifest_file(data_dir: Path, *, offline: bool) -> Path:
    return fetch_unhashed(MANIFEST_URLS, data_dir / "panel" / "manifest.json", offline=offline)


def load_panel_votes(data_dir: Path, uids: list[str], offline: bool) -> tuple[np.ndarray, dict]:
    index = {uid: i for i, uid in enumerate(uids)}
    votes = np.full((len(JUDGES), len(uids)), -1, dtype=int)
    provenance: dict[str, dict] = {}
    for row, judge in enumerate(JUDGES):
        path = fetch_unhashed(
            [f"{PANEL_RAW}/{PANEL_DATASET}/votes/baseline/{judge}.jsonl"],
            data_dir / "panel" / f"baseline__{judge}.jsonl",
            offline=offline,
        )
        served = 0
        failed = 0
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                record = json.loads(line)
                col = index.get(record["uid"])
                if col is None:
                    continue
                if record.get("parse_fail"):
                    failed += 1
                    continue
                votes[row, col] = JUDGE_LABEL_TO_INT[record["label"]]
                served += 1
        provenance[judge] = {
            "sha256": _sha256(path),
            "votes_used": served,
            "parse_fail": failed,
            "abstentions": int((votes[row] < 0).sum()),
        }
    return votes, provenance


def audit_with_floor(
    label: str, votes: np.ndarray, model_scores: dict[str, float],
    model_items: dict[str, np.ndarray], condo_ids: set[str], contestants: set[str],
    delta: float,
) -> dict:
    judge_ids = ({f"human_{i:03d}" for i in range(N_PSEUDO_ANNOTATORS)}
                 if label == "human" else set(JUDGES))
    top_two = [name for name, _ in sorted(model_scores.items(), key=lambda kv: -kv[1])[:2]]
    report = audit_leaderboard(
        votes,
        model_scores,
        leaderboard_item_ids=condo_ids,
        floor_item_ids=condo_ids,
        judge_ids=judge_ids,
        contestant_ids=contestants,
        per_item_scores={name: model_items[name] for name in top_two},
        delta_declared=delta,
    )
    return {
        "floor_source": label,
        "n_generators": int(report.n_generators),
        "verdict": report.verdict,
        "reasons": list(report.reasons),
        "floor_estimate": float(report.floor_estimate),
        "chance_level": report.extra.get("chance_level"),
        "floor_vs_chance": report.extra.get("floor_vs_chance"),
        "leader_system": report.leader_system,
        "leader_score": float(report.leader_score),
        "margin": float(report.margin),
        "mde": float(report.mde),
        "mde_method": report.extra.get("mde_method"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=HERE / "_data")
    parser.add_argument("--out", type=Path, default=HERE / "results_judge_panel.json")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--delta", type=float, default=0.02)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)

    spec = CONFIGS["snli"]
    ann_path = fetch(spec["annotation_urls"], args.data_dir / spec["annotation_file"],
                     spec["annotation_sha256"], offline=args.offline)
    pred_path = fetch(spec["prediction_urls"], args.data_dir / spec["prediction_file"],
                      spec["prediction_sha256"], offline=args.offline)
    uids_path = fetch(UIDS_URLS, args.data_dir / "panel" / "uids.txt", UIDS_SHA256,
                      offline=args.offline)
    manifest_path = fetch_unhashed(MANIFEST_URLS, args.data_dir / "panel" / "manifest.json",
                                  offline=args.offline)

    ann = load_annotations(ann_path)
    pred = json.loads(pred_path.read_text(encoding="utf-8"))
    uids_all = [line.strip() for line in uids_path.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    # Same denominator for both floors and the board.
    conds = sorted(set(uids_all) & set(ann) & set.intersection(*[set(sub) for sub in pred.values()]))
    counts = np.array([ann[u]["label_count"] for u in conds], dtype=int)
    truth = np.array([{"e": 0, "n": 1, "c": 2}[ann[u]["majority_label"]] for u in conds], dtype=int)

    model_scores: dict[str, float] = {}
    model_items: dict[str, np.ndarray] = {}
    for name, sub in pred.items():
        got = np.array([{"entailment": 0, "neutral": 1, "contradiction": 2}[sub[u]["predicted_label"]]
                        for u in conds], dtype=int)
        model_scores[name] = float((got == truth).mean())
        model_items[name] = (got == truth).astype(int)

    panel_votes, provenance = load_panel_votes(args.data_dir, conds, args.offline)

    human_floors = []
    panel_floors = []
    audits = {"human": None, "llm_judge_panel": None}
    for seed in range(args.seeds):
        human_votes = build_votes(counts, seed)
        report = audit_with_floor("human", human_votes, model_scores, model_items,
                                  set(conds), set(pred), args.delta)
        human_floors.append(report["floor_estimate"])
        audits["human"] = report
        if seed == 0:
            panel_floors.append(float(loo_floor(panel_votes).mean()))

    panel_audit = audit_with_floor("llm_judge_panel", panel_votes, model_scores, model_items,
                                   set(conds), set(pred), args.delta)
    audits["llm_judge_panel"] = panel_audit

    # Judge accuracy against the human majority, for context on how good the
    # panel is; not a verdict, since the panel cannot judge itself.
    judge_accuracy = {}
    for row, judge in enumerate(JUDGES):
        voted = panel_votes[row] >= 0
        if voted.any():
            judge_accuracy[judge] = float((panel_votes[row][voted] == truth[voted]).mean())

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = {
        "generated_by": "case-studies/chaosnli-noise-floor/audit_judge_panel.py",
        "n_items": len(conds),
        "panel_size": len(JUDGES),
        "panel_release": {
            "repo": PANEL_REPO,
            "manifest_dataset_id": manifest.get("dataset_id"),
            "panel_id": manifest.get("panel", {}).get("panel_id"),
            "caveat": ("backend identity/version is not independently authenticated "
                       "(quoted from the release's own panel manifest)"),
            "uids_sha256": UIDS_SHA256,
        },
        "board": [{"system": k, "accuracy": v}
                  for k, v in sorted(model_scores.items(), key=lambda kv: -kv[1])],
        "audits": audits,
        "human_floor_across_seeds": human_floors,
        "human_floor_seed_spread": max(human_floors) - min(human_floors),
        "panel_loo_floor_diagnostic": panel_floors[0] if panel_floors else None,
        "judge_accuracy_vs_human_majority": dict(
            sorted(judge_accuracy.items(), key=lambda kv: -kv[1])),
        "judge_file_provenance": provenance,
    }

    out = args.out
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print()
    print(f"items={len(conds)}  board={len(model_scores)} contestants  panel={len(JUDGES)} judges")
    for label in ("human", "llm_judge_panel"):
        entry = audits[label]
        print(f"--- floor: {label}")
        print(f"    verdict={entry['verdict']}  floor={entry['floor_estimate']:.4f}  "
              f"chance={entry['chance_level']:.4f}")
        print(f"    leader={entry['leader_system']}={entry['leader_score']:.4f}  "
              f"margin={entry['margin']:+.4f}  mde={entry['mde']:.4f} ({entry['mde_method']})")
        if entry["reasons"]:
            print(f"    reasons={entry['reasons']}")
    ranked_judges = sorted(judge_accuracy.items(), key=lambda kv: -kv[1])
    print(f"--- panel diagnostic: LOO floor among the 32 judges = "
          f"{payload['panel_loo_floor_diagnostic']:.4f}")
    print(f"    judges' accuracy vs human majority: best={ranked_judges[0][1]:.4f} "
          f"({ranked_judges[0][0]}), worst={ranked_judges[-1][1]:.4f} ({ranked_judges[-1][0]}), "
          f"median={np.median([v for _, v in ranked_judges]):.4f}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
