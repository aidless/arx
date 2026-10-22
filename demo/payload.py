"""Build the arx demo payload from real ``audit_leaderboard`` calls.

The gallery does not re-implement the verdict logic. Each synthetic case is
run through ``arx.audit.audit_leaderboard`` at build time, and the report HTML
is produced by ``arx.audit.report_html.render_html``. The payload keeps the
AuditReport fields so every displayed number has a named source.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arx.audit import audit_leaderboard  # noqa: E402
from arx.audit.report_html import render_html  # noqa: E402

VERDICTS = ("READABLE", "MARGINAL", "UNREADABLE", "UNDECIDABLE")

VERDICT_MEANING = {
    "READABLE": "差距大于测量误差。它不表示这个指标本身是对的。",
    "MARGINAL": "差距为正，但落在当前样本能分辨的范围内；需要更多 item。",
    "UNREADABLE": "领先者没有清过噪声地板；这份测量不支持这个排名。",
    "UNDECIDABLE": "前提不成立，无法估计地板。这是一个结论，不是分析失败。",
}

REASON_TEXT = {
    "floor_degenerate": "地板退化：没有可估计的独立复现地板，或地板未能高于随机水平。",
    "actors_overlap": "评审者与参赛者有交集；地板测到的是同伴分布，不是测量误差。",
    "domain_mismatch": "地板与榜面用了不同的 item 集；两个统计量不可相减。",
    "single_generator": "只有一名生成者，无法做 leave-one-out。",
}

CASE_STUDY = {
    "title": "ChaosNLI noise floor（真实数据案例）",
    "url": "https://github.com/aidless/arx/blob/main/case-studies/chaosnli-noise-floor/README.md",
    "note": "真实数据上的三条审计；本页不重算它，只链接到一手 case study。",
}


def votes(n_generators: int, n_items: int = 400, agreement: float = 0.93,
          seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 2, size=n_items)
    return np.array([
        np.where(rng.random(n_items) < agreement, truth, 1 - truth)
        for _ in range(n_generators)
    ])


def _case_specs() -> list[dict]:
    base_judges = ["g1", "g2", "g3", "g4", "g5"]
    base_contestants = ["strong", "weak"]
    base_scores = {"strong": 0.96, "weak": 0.60}
    base_kwargs = {
        "judge_ids": base_judges,
        "contestant_ids": base_contestants,
        "delta_declared": 0.02,
    }
    return [
        {
            "id": "readable",
            "title": "五名生成者，领先者高出噪声地板",
            "question": "这份榜面能分辨 strong 与 weak 的差距吗？",
            "expected": "READABLE",
            "votes": votes(5, 400, 0.93, 0),
            "scores": base_scores,
            "kwargs": base_kwargs,
            "board": {
                "n_generators": 5, "n_items": 400, "agreement": 0.93,
                "seed": 0, "scores": base_scores, "delta_declared": 0.02,
                "judge_ids": base_judges, "contestant_ids": base_contestants,
            },
        },
        {
            "id": "marginal",
            "title": "同样的榜面，只有 120 个 item",
            "question": "差距没变，但样本分辨率变了吗？",
            "expected": "MARGINAL",
            "votes": votes(5, 120, 0.93, 3),
            "scores": base_scores,
            "kwargs": base_kwargs,
            "board": {
                "n_generators": 5, "n_items": 120, "agreement": 0.93,
                "seed": 3, "scores": base_scores, "delta_declared": 0.02,
                "judge_ids": base_judges, "contestant_ids": base_contestants,
            },
        },
        {
            "id": "unreadable",
            "title": "领先者没有清过噪声地板",
            "question": "strong=0.86 真的高于这些标签自身的复现率吗？",
            "expected": "UNREADABLE",
            "votes": votes(5, 400, 0.93, 0),
            "scores": {"strong": 0.86, "weak": 0.82},
            "kwargs": base_kwargs,
            "board": {
                "n_generators": 5, "n_items": 400, "agreement": 0.93,
                "seed": 0, "scores": {"strong": 0.86, "weak": 0.82},
                "delta_declared": 0.02,
                "judge_ids": base_judges, "contestant_ids": base_contestants,
            },
        },
        {
            "id": "two_generators",
            "title": "只有两名生成者",
            "question": "去掉一个评审者后，地板还能估计吗？",
            "expected": "UNDECIDABLE",
            "votes": votes(2, 400, 0.93, 0),
            "scores": base_scores,
            "kwargs": {
                "judge_ids": ["g1", "g2"],
                "contestant_ids": base_contestants,
                "delta_declared": 0.02,
            },
            "board": {
                "n_generators": 2, "n_items": 400, "agreement": 0.93,
                "seed": 0, "scores": base_scores, "delta_declared": 0.02,
                "judge_ids": ["g1", "g2"],
                "contestant_ids": base_contestants,
            },
        },
        {
            "id": "actors_overlap",
            "title": "评审者同时是参赛者",
            "question": "如果 strong 也参与了 gold，它是在和自己比吗？",
            "expected": "UNDECIDABLE",
            "votes": votes(5, 400, 0.93, 0),
            "scores": base_scores,
            "kwargs": {
                "judge_ids": ["strong", "g2", "g3", "g4", "g5"],
                "contestant_ids": base_contestants,
                "delta_declared": 0.02,
            },
            "board": {
                "n_generators": 5, "n_items": 400, "agreement": 0.93,
                "seed": 0, "scores": base_scores, "delta_declared": 0.02,
                "judge_ids": ["strong", "g2", "g3", "g4", "g5"],
                "contestant_ids": base_contestants,
            },
        },
        {
            "id": "noise",
            "title": "五名生成者只在掷硬币",
            "question": "地板接近随机时，领先者的大 margin 还意味着什么？",
            "expected": "UNDECIDABLE",
            "votes": np.random.default_rng(7).integers(0, 2, size=(5, 400)),
            "scores": {"padded": 0.965, "plain": 0.70},
            "kwargs": {
                "judge_ids": base_judges,
                "contestant_ids": ["padded", "plain"],
                "delta_declared": 0.02,
            },
            "board": {
                "n_generators": 5, "n_items": 400, "agreement": None,
                "seed": 7, "scores": {"padded": 0.965, "plain": 0.70},
                "delta_declared": 0.02,
                "judge_ids": base_judges,
                "contestant_ids": ["padded", "plain"],
                "note": "pure coin-flip generators; floor is not above chance",
            },
        },
    ]


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _metrics(report) -> list[dict]:
    leader = "—"
    if report.leader_system is not None and report.leader_score is not None:
        leader = f"{report.leader_system} = {report.leader_score:.4f}"
    return [
        {"label": "噪声地板", "value": _fmt(report.floor_estimate),
         "source": "AuditReport.floor_estimate"},
        {"label": "领先者", "value": leader,
         "source": "AuditReport.leader_system / leader_score"},
        {"label": "高出地板", "value": (f"{report.margin:+.4f}"
                                      if report.margin is not None else "—"),
         "source": "AuditReport.margin"},
        {"label": "分辨率下界", "value": _fmt(report.threshold),
         "source": "AuditReport.threshold"},
        {"label": "MDE", "value": _fmt(report.mde),
         "source": "AuditReport.mde"},
        {"label": "items", "value": report.n_items,
         "source": "AuditReport.n_items"},
        {"label": "generators", "value": report.n_generators,
         "source": "AuditReport.n_generators"},
    ]


def _case_payload(spec: dict, generated_at: str) -> dict:
    kwargs = dict(spec["kwargs"])
    report = audit_leaderboard(spec["votes"], spec["scores"], **kwargs)
    report_html = render_html(report, title=spec["title"],
                              generated_at=generated_at)
    chance = report.extra.get("floor_vs_chance")
    chance_payload = None
    if isinstance(chance, dict):
        chance_payload = {
            "above_chance": chance.get("above_chance"),
            "z": chance.get("z"),
            "critical_z": chance.get("critical_z"),
            "n_items_needed": chance.get("n_items_needed"),
            "source": "AuditReport.extra.floor_vs_chance",
        }
    notes = []
    if report.extra.get("floor_error"):
        notes.append(str(report.extra["floor_error"]))
    if report.extra.get("overlapping_actors"):
        notes.append(
            "重叠的 actor：" + ", ".join(map(str, report.extra["overlapping_actors"]))
        )
    return {
        "id": spec["id"],
        "title": spec["title"],
        "question": spec["question"],
        "verdict": report.verdict.value,
        "meaning": VERDICT_MEANING.get(report.verdict.value, ""),
        "reasons": [
            REASON_TEXT.get(reason, reason) for reason in report.reasons
        ],
        "reason_codes": list(report.reasons),
        "metrics": _metrics(report),
        "chance": chance_payload,
        "notes": notes,
        "board": {
            **spec["board"],
            "votes_sha256_16": hashlib.sha256(
                np.asarray(spec["votes"], dtype=np.int8).tobytes()
            ).hexdigest()[:16],
        },
        "report": _json_safe(report.to_dict()),
        "report_html": report_html,
        "source": {
            "kind": "recomputed",
            "audit": "arx.audit.audit_leaderboard",
            "renderer": "arx.audit.report_html.render_html",
        },
    }


def _git_snapshot() -> tuple[str, bool, list[str]]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        raw = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        ).splitlines()
        ignored = ("demo/public/cases.json",)
        dirty_paths = [
            line[3:] for line in raw
            if len(line) > 3 and not line[3:].startswith(ignored)
        ]
        return commit, bool(dirty_paths), dirty_paths[:12]
    except Exception:
        return "unknown", False, []


def build_payload(generated_at: str | None = None) -> dict:
    generated_at = generated_at or datetime.now(timezone.utc).isoformat(
        timespec="seconds")
    specs = _case_specs()
    cases = [_case_payload(spec, generated_at) for spec in specs]
    verdicts = {
        verdict: sum(1 for case in cases if case["verdict"] == verdict)
        for verdict in VERDICTS
    }
    expected = {spec["id"]: spec["expected"] for spec in specs}
    mismatches = [
        case["id"] for case in cases
        if case["verdict"] != expected.get(case["id"])
    ]
    commit, dirty, dirty_paths = _git_snapshot()
    return {
        "meta": {
            "repo": "arx",
            "demo": "leaderboard-audit",
            "commit": commit,
            "worktree_dirty": dirty,
            "dirty_paths": dirty_paths,
            "generated_at": generated_at,
            "mode": "offline-numpy",
            "seed": 0,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "note": (
                "六个合成案例在构建时由 arx.audit 重算；"
                "报告 HTML 由 arx.audit.report_html 生成。"
            ),
        },
        "summary": {
            "cases": len(cases),
            "verdicts": verdicts,
        },
        "cases": cases,
        "case_study": CASE_STUDY,
        "boundaries": [
            "这六个案例是合成投票矩阵，用来行使闸的四种判决；不是真实模型排名。",
            "READABLE 只说明差距超过测量误差，不说明指标本身是对的。",
            "闸审计的是榜面的可读性，不审计模型输出是否忠实、任务是否有意义。",
            "真实数据案例见 case-studies/chaosnli-noise-floor/；本页不重算那份案例。",
            "报告 HTML 由 arx.audit.report_html 生成；页面不重新实现判决逻辑。",
        ],
        "verification": {
            "status": "MATCH" if not mismatches else "MISMATCH",
            "cases": len(cases),
            "mismatches": mismatches,
            "expected": expected,
        },
    }


def write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
