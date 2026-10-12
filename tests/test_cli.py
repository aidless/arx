"""CLI tests.

These are not smoke tests. Each one drives a subcommand that the README tells
a reader to run, because a CLI that raises on the documented path is worse than
no CLI — the reader concludes the tool is broken rather than the docs are.
"""

from __future__ import annotations

import json
from pathlib import Path

import _support as S
import pytest

from arx.agent import Agent
from arx.cli import main
from arx.experiments.ledger import Ledger


@pytest.fixture
def ledger_path(tmp_path: Path) -> Path:
    """A real ledger from a real run, not a hand-built one."""
    agent = Agent(
        S.offline_config(tmp_path / "ws"),
        topic="whether a margin inside the noise floor means anything",
        llm=S.make_llm(),
        searcher=S.make_searcher(),
        provider=S.StaticProvider(S.leaderboard_script("cli")),
        theory_root="noise-floor",
        theory_falsified_by="a leader inside the floor is reported as readable",
    )
    result = agent.run()
    assert result.chain_valid, "the fixture run must produce a valid ledger"
    return Path(result.ledger_path)


# ---------------------------------------------------------------------------
# verify-ledger
# ---------------------------------------------------------------------------

def test_verify_ledger_reports_the_chain(ledger_path, capsys):
    assert main(["verify-ledger", str(ledger_path)]) == 0
    out = capsys.readouterr().out
    assert "chain valid: True" in out
    assert "merkle root:" in out
    assert "tail hash:" in out or "records:" in out


def test_verify_ledger_refuses_a_tampered_ledger(tmp_path, capsys):
    path = tmp_path / "l.jsonl"
    led = Ledger(path)
    led.append("step", {"i": 0})
    lines = path.read_text().splitlines()
    rec = json.loads(lines[0])
    rec["payload"]["i"] = 99
    lines[0] = json.dumps(rec, sort_keys=True)
    path.write_text("\n".join(lines) + "\n")

    rc = main(["verify-ledger", str(path)])
    assert rc != 0
    assert "chain valid: False" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# anchor / verify against it
# ---------------------------------------------------------------------------

def test_anchor_then_verify_against_it(ledger_path, tmp_path, capsys):
    anchor = tmp_path / "pinned.json"
    assert main(["anchor", str(ledger_path), "--out", str(anchor)]) == 0
    assert anchor.exists()

    capsys.readouterr()
    assert main(["verify-ledger", str(ledger_path), "--anchor", str(anchor)]) == 0
    assert "anchor check: ok" in capsys.readouterr().out


def test_an_anchor_catches_a_trimmed_tail(ledger_path, tmp_path, capsys):
    """The one tamper the in-file chain cannot see, which is why anchor exists."""
    anchor = tmp_path / "pinned.json"
    assert main(["anchor", str(ledger_path), "--out", str(anchor)]) == 0

    trimmed = tmp_path / "trimmed.jsonl"
    lines = ledger_path.read_text().splitlines()
    trimmed.write_text("\n".join(lines[:-1]) + "\n")

    capsys.readouterr()
    rc = main(["verify-ledger", str(trimmed), "--anchor", str(anchor)])
    assert rc != 0
    assert "MISMATCH" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------

def test_replay_reproduces_every_decision(ledger_path, capsys):
    capsys.readouterr()
    assert main(["replay", str(ledger_path)]) == 0
    out = capsys.readouterr().out
    assert "all match" in out
    assert "merkle root:" in out


# ---------------------------------------------------------------------------
# theory -- the subcommand that regressed
# ---------------------------------------------------------------------------

def test_theory_prints_coverage_without_raising(ledger_path, capsys):
    """A KeyError on the documented path is the failure this test pins.

    ``cmd_theory`` used to read ``coverage['theory_driven_tests']``, a key the
    coverage dict never had, so the subcommand raised on every ledger.
    """
    assert main(["theory", str(ledger_path)]) == 0
    out = capsys.readouterr().out
    assert "theories registered" in out
    assert "hypotheses tested" in out
    assert "hypothesis coverage" in out
    assert "note:" in out


def test_theory_json_is_valid_json(ledger_path, capsys):
    assert main(["theory", str(ledger_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) >= {"coverage", "theories"}
    assert payload["coverage"]["n_theories"] == 1


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

@pytest.fixture
def board_json(tmp_path) -> Path:
    path = tmp_path / "board.json"
    path.write_text(json.dumps({
        "votes": [[0, 1], [0, 1], [0, 1]],
        "scores": {"strong": 0.96, "weak": 0.60},
        "judge_ids": ["j1", "j2", "j3"],
        "contestant_ids": ["strong", "weak"],
    }))
    return path


def test_audit_reports_a_verdict_with_its_reason(board_json, capsys):
    # 0 for READABLE, 1 for a refusal: a gate whose exit code cannot tell
    # those apart is a gate a caller has to parse text to use.
    rc = main(["audit", str(board_json), "--delta", "0.02"])
    assert rc in (0, 1)
    out = capsys.readouterr().out
    assert "Verdict:" in out
    # Three generators pass the multi-generator check; the board is small
    # enough that the verdict could be any of them, but it must be one of them
    # and it must name its method.
    assert any(v in out for v in ("READABLE", "MARGINAL", "UNREADABLE",
                                  "UNDECIDABLE"))
    assert "mde_method" in out or "MDE" in out or "floor" in out


def test_audit_refuses_a_two_generator_board(tmp_path, capsys):
    path = tmp_path / "two.json"
    path.write_text(json.dumps({
        "votes": [[0, 1], [0, 1]],
        "scores": {"strong": 0.96, "weak": 0.60},
        "judge_ids": ["j1", "j2"],
        "contestant_ids": ["strong", "weak"],
    }))
    assert main(["audit", str(path)]) == 1       # refused, not crashed
    out = capsys.readouterr().out
    assert "UNDECIDABLE" in out
    assert "floor_degenerate" in out


def test_audit_refuses_an_overlapping_actor(tmp_path, capsys):
    path = tmp_path / "overlap.json"
    path.write_text(json.dumps({
        "votes": [[0, 1, 0], [0, 1, 0], [0, 1, 1]],
        "scores": {"strong": 0.96, "weak": 0.60},
        # 'strong' is both a gold generator and a contestant.
        "judge_ids": ["strong", "j2", "j3"],
        "contestant_ids": ["strong", "weak"],
    }))
    assert main(["audit", str(path)]) == 1       # refused, not crashed
    out = capsys.readouterr().out
    assert "UNDECIDABLE" in out
    assert "actors_overlap" in out


def test_audit_refuses_mismatched_denominators(tmp_path, capsys):
    path = tmp_path / "mismatch.json"
    path.write_text(json.dumps({
        "votes": [[0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1]],
        "scores": {"strong": 0.96, "weak": 0.60},
        "judge_ids": ["j1", "j2", "j3"],
        "contestant_ids": ["strong", "weak"],
        "item_ids": [0, 1],
        "floor_item_ids": [0, 1, 2, 3],
    }))
    assert main(["audit", str(path)]) == 1       # refused, not crashed
    out = capsys.readouterr().out
    assert "UNDECIDABLE" in out
    assert "domain_mismatch" in out


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

def test_doctor_reports_capabilities(capsys):
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "version:" in out
    assert "python:" in out
    assert "llm_configured:" in out


def test_no_command_prints_help_and_exits_nonzero(capsys):
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code != 0
