"""End-to-end tests for the research loop.

These drive a whole run with every dependency stubbed, because the properties
worth pinning are about *ordering and refusal*, not about model quality:

* the pre-registration record must hit the ledger **before** the experiment
  subprocess starts, otherwise ``k_max`` can be retro-fitted to whatever the run
  produced;
* a gate that refuses must leave no finding behind, and must come back with a
  hypothesis rather than a number;
* the same hypothesis must not be re-run after it has been accepted;
* ``resume`` must continue from the ledger rather than start over.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arx.agent import Agent
from arx.config import AgentConfig
from arx.experiments.ledger import Ledger
from arx.experiments.provider import ExperimentScript, NullExperimentProvider
from arx.hypothesis import Hypothesis
from arx.literature.search import Paper, SearchOutcome
from arx.literature.zotero import NullLibrary
from arx.llm import ScriptedLLM

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

READABLE_SCRIPT = '''
import json, random
rng = random.Random(7)
n = 400
truth = [rng.randint(0, 1) for _ in range(n)]
votes = [[t if rng.random() < 0.96 else 1 - t for t in truth] for _ in range(5)]
print(json.dumps({
    "votes": votes,
    "scores": {"strong": 0.99, "weak": 0.60},
    "judge_ids": ["j1", "j2", "j3", "j4", "j5"],
    "contestant_ids": ["strong", "weak"],
    "item_ids": list(range(n)),
    "se": 0.01,
    "n_looks": 1,
}))
'''

# Only two generators: the floor is not estimable, whatever the numbers say.
UNDECIDABLE_SCRIPT = '''
import json, random
rng = random.Random(3)
n = 200
truth = [rng.randint(0, 1) for _ in range(n)]
votes = [[t if rng.random() < 0.95 else 1 - t for t in truth] for _ in range(2)]
print(json.dumps({
    "votes": votes,
    "scores": {"strong": 0.99, "weak": 0.60},
    "judge_ids": ["j1", "j2"],
    "contestant_ids": ["strong", "weak"],
}))
'''

# Claims a different number of looks than the ledger pre-registered.
LAUNDERED_SCRIPT = READABLE_SCRIPT.replace(
    '"n_looks": 1,', '"n_looks": 1, "k_max": 99,'
)


class StaticProvider:
    """Serves one script, materialised in the workspace like a real provider."""

    name = "static"

    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    def provide(self, hypothesis: Hypothesis) -> ExperimentScript:
        self.calls += 1
        return ExperimentScript(text=self.text, source="static")


def _papers() -> SearchOutcome:
    return SearchOutcome(
        papers=[
            Paper(title="A noise floor for language model evaluation", year=2024),
            Paper(title="Reading a leaderboard before citing it", year=2024),
            Paper(title="Judge reliability under correlated raters", year=2023),
        ],
        sources_ok=["stub"],
    )


HYPOTHESES_REPLY = json.dumps(
    {
        "hypotheses": [
            {
                "text": f"A reranker improves retrieval on long documents v{i}",
                "rationale": "because the baseline truncates",
                "confidence": 0.5,
            }
            for i in range(3)
        ]
    }
)


def _cfg(tmp_path: Path, **overrides) -> AgentConfig:
    base = dict(
        workspace=str(tmp_path / "ws"),
        semantic_duplicate_threshold=0.0,  # do not touch ~/.arx in tests
        max_iterations=14,
        max_experiment_runs=6,
        max_literature_sweeps=2,
        max_llm_calls=50,
        max_wall_seconds=600.0,
        sandbox_timeout_s=60,
        gate_strict=True,
        verbose=False,
    )
    base.update(overrides)
    return AgentConfig(**base)


def _agent(
    tmp_path: Path,
    script: str | None = READABLE_SCRIPT,
    *,
    theory_root: str = "retrieval-goodhart",
    **cfg_kwargs,
) -> Agent:
    provider = StaticProvider(script) if script is not None else NullExperimentProvider()
    return Agent(
        _cfg(tmp_path, **cfg_kwargs),
        topic="does a reranker improve retrieval on long documents",
        llm=ScriptedLLM({"hypotheses": HYPOTHESES_REPLY}),
        library=NullLibrary(),
        searcher=lambda query, max_results=20: _papers(),
        provider=provider,
        theory_root=theory_root,
        theory_falsified_by="reranking raises retrieval scores without raising answer accuracy",
    )


# ---------------------------------------------------------------------------
# ordering: pre-registration before the run
# ---------------------------------------------------------------------------

def test_the_pre_registration_lands_before_the_experiment_runs(tmp_path):
    """The whole point of pre-registration is that it is *earlier* in the
    ledger than the result. If it were written afterwards the ordering could be
    retro-fitted, which is the self-deception it exists to prevent."""
    agent = _agent(tmp_path)
    result = agent.run()

    kinds = [r.kind for r in agent.ledger.records()]
    assert kinds.count("pre_register") >= 1
    assert kinds.count("experiment") >= 1
    assert kinds.index("pre_register") < kinds.index("experiment")
    assert result.chain_valid


def test_the_pre_registration_pins_the_script_hash(tmp_path):
    agent = _agent(tmp_path)
    agent.run()
    prereg = [r for r in agent.ledger.records() if r.kind == "pre_register"][0]
    assert prereg.payload["script_sha256"]
    experiment = [r for r in agent.ledger.records() if r.kind == "experiment"][0]
    assert experiment.payload["script_sha256"] == prereg.payload["script_sha256"]


# ---------------------------------------------------------------------------
# the happy path
# ---------------------------------------------------------------------------

def test_a_readable_board_produces_a_finding_and_a_report(tmp_path):
    agent = _agent(tmp_path)
    result = agent.run()

    assert result.n_findings >= 1
    finding = result.accepted[0]
    assert finding.report.verdict.value == "READABLE"
    assert finding.report.margin > 0
    assert finding.report.floor_estimate is not None
    assert result.chain_valid
    json.dumps(result.to_dict())


def test_every_gate_event_is_recorded_with_its_reasoning(tmp_path):
    agent = _agent(tmp_path)
    agent.run()
    gates = [r for r in agent.ledger.records() if r.kind == "gate"]
    assert gates
    for gate in gates:
        assert gate.payload["verdict"]
        assert gate.payload["hypothesis"]


def test_the_run_records_its_own_configuration(tmp_path):
    """Without the gate settings, a number in the ledger is not interpretable:
    the same board under a different delta or alpha means something else."""
    agent = _agent(tmp_path)
    agent.run()
    start = [r for r in agent.ledger.records() if r.kind == "run_start"][0]
    config = start.payload["config"]
    for key in ("gate_enabled", "gate_strict", "gate_delta_declared",
                "gate_alpha", "gate_power", "stall_limit", "seed"):
        assert key in config, key


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

def test_a_board_with_two_generators_is_refused_and_leaves_no_finding(tmp_path):
    agent = _agent(tmp_path, UNDECIDABLE_SCRIPT)
    result = agent.run()

    assert result.n_findings == 0
    assert result.rejected
    assert all(r.verdict == "UNDECIDABLE" for r in result.rejected)
    assert any("floor_degenerate" in r.reason or "degenerate" in r.reason.lower()
               for r in result.rejected)


def test_a_laundered_k_max_is_caught_rather_than_read(tmp_path):
    """The failure this exists to stop: run a fixed-sample test, miss, then
    claim you only ever intended one look."""
    agent = _agent(tmp_path, LAUNDERED_SCRIPT)
    result = agent.run()

    assert result.n_findings == 0
    assert result.rejected
    assert any("k_max" in r.reason for r in result.rejected), \
        [r.reason for r in result.rejected]


def test_a_missing_experiment_is_recorded_as_untested_not_as_a_result(tmp_path):
    """A provider that supplies nothing must produce ``untested``. Inventing a
    plausible result here is the failure the framework exists to prevent."""
    agent = _agent(tmp_path, script=None)
    result = agent.run()

    assert result.n_findings == 0
    assert result.untested
    kinds = [r.kind for r in agent.ledger.records()]
    assert "experiment" in kinds
    statuses = [r.payload.get("status") for r in agent.ledger.records()
                if r.kind == "experiment"]
    assert "no_entrypoint" in statuses
    assert "pre_register" not in kinds


def test_a_hypothesis_is_not_re_run_after_being_accepted(tmp_path):
    """The proposer returns the same three hypotheses every time. Each must be
    tested once: re-running an accepted hypothesis burns an experiment and
    widens the family without asking anything new."""
    agent = _agent(tmp_path)
    result = agent.run()

    tested = [r.payload["hypothesis"] for r in agent.ledger.records()
              if r.kind == "experiment"]
    assert len(tested) == len(set(tested)), tested
    assert len(tested) == result.n_findings


def test_the_llm_budget_is_charged_once_per_hypothesis_round(tmp_path):
    """Two charges for one call would halve the run's real budget; the reported
    count must match the number of requests the backend actually saw."""
    agent = _agent(tmp_path)
    result = agent.run()
    assert result.budget["llm_calls"] == agent.llm.stats.calls


# ---------------------------------------------------------------------------
# termination
# ---------------------------------------------------------------------------

def test_the_loop_terminates_and_says_why(tmp_path):
    agent = _agent(tmp_path)
    result = agent.run()
    assert result.halt_reason
    assert result.steps <= _cfg(tmp_path).max_iterations


def test_the_iteration_ceiling_is_recorded_as_such(tmp_path):
    agent = _agent(tmp_path, max_iterations=2)
    result = agent.run()
    assert "ceiling" in result.halt_reason
    kinds = [r.kind for r in agent.ledger.records()]
    assert "stall" in kinds


def test_an_experiment_run_quota_exhausts_without_halting_the_loop(tmp_path):
    """A per-action quota disqualifies that action, it does not end the run.
    Conflating the two is what killed the loop after one literature sweep."""
    agent = _agent(tmp_path, max_experiment_runs=1)
    result = agent.run()
    assert result.budget["experiment_runs"] == 1
    assert result.halt_reason


# ---------------------------------------------------------------------------
# theory layer
# ---------------------------------------------------------------------------

def test_hypotheses_are_attached_to_the_registered_theory(tmp_path):
    agent = _agent(tmp_path)
    agent.run()
    coverage = agent.theory.coverage()
    assert coverage["n_theories"] == 1
    assert coverage["hypotheses_attached"] >= 1
    assert coverage["theory_coverage"] > 0


def test_without_a_theory_root_every_test_is_labelled_exploratory(tmp_path):
    agent = _agent(tmp_path, theory_root="")
    agent.run()
    coverage = agent.theory.coverage()
    assert coverage["n_theories"] == 0
    assert coverage["hypotheses_exploratory"] >= 1
    assert coverage["theory_coverage"] == 0.0


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------

def test_resume_continues_from_the_ledger_rather_than_restarting(tmp_path):
    cfg = _cfg(tmp_path, max_iterations=3)
    agent = Agent(
        cfg,
        topic="does a reranker improve retrieval on long documents",
        llm=ScriptedLLM({"hypotheses": HYPOTHESES_REPLY}),
        library=NullLibrary(),
        searcher=lambda query, max_results=20: _papers(),
        provider=StaticProvider(READABLE_SCRIPT),
    )
    first = agent.run()
    ledger_path = Path(first.ledger_path)
    assert ledger_path.exists()
    n_before = len(list(ledger_path.read_text().splitlines()))

    resumed = Agent.resume(
        _cfg(tmp_path, max_iterations=12),
        ledger_path,
        llm=ScriptedLLM({"hypotheses": HYPOTHESES_REPLY}),
        library=NullLibrary(),
        searcher=lambda query, max_results=20: _papers(),
        provider=StaticProvider(READABLE_SCRIPT),
    )
    # Budget counters come off the last scheduler record, not from zero.
    assert resumed.budget.llm_calls == agent.budget.llm_calls
    assert resumed.budget.experiment_runs == agent.budget.experiment_runs

    second = resumed.run()
    assert len(list(ledger_path.read_text().splitlines())) > n_before
    assert second.chain_valid


def test_resume_refuses_a_tampered_ledger(tmp_path):
    agent = _agent(tmp_path, max_iterations=3)
    result = agent.run()
    ledger_path = Path(result.ledger_path)

    lines = ledger_path.read_text().splitlines()
    record = json.loads(lines[0])
    record["payload"]["topic"] = "a different topic entirely"
    lines[0] = json.dumps(record, sort_keys=True)
    ledger_path.write_text("\n".join(lines) + "\n")

    assert not Ledger(ledger_path).verify_chain()
    with pytest.raises(ValueError, match="broken ledger"):
        Agent.resume(_cfg(tmp_path), ledger_path)
