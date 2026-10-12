"""Tests for experiment execution and experiment supply.

Two boundaries are under test, and they are different boundaries:

* **Execution** — the runner starts a real subprocess and reads a documented
  JSON contract off its stdout. It is deliberately *not* a security sandbox,
  and the code says so; what is tested here is that a crash, a timeout, a
  missing contract key, and a chatty script are each handled the way the
  ledger downstream expects.
* **Supply** — where the experiment script comes from. The load-bearing rule
  is that a provider with nothing to give returns ``None`` rather than a
  plausible-looking default. A synthesised template would produce output that
  is indistinguishable from a real result once it reaches the ledger.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arx.experiments.provider import (
    ExperimentScript,
    LLMExperimentProvider,
    NullExperimentProvider,
    WorkspaceExperimentProvider,
    _looks_like_script,
    _strip_fences,
    build_provider,
)
from arx.experiments.runner import REQUIRED_KEYS, RunResult, run_experiment
from arx.hypothesis import Hypothesis
from arx.llm import NullLLM, ScriptedLLM


def _write(tmp_path: Path, body: str, name: str = "exp.py") -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


GOOD_SCRIPT = """\
import json
print(json.dumps({
    "votes": [[0, 1, 0, 1], [0, 1, 1, 1], [1, 1, 0, 1]],
    "scores": {"a": 0.75, "b": 0.50},
    "judge_ids": ["j1", "j2", "j3"],
    "contestant_ids": ["a", "b"],
}))
"""


# ---------------------------------------------------------------------------
# running a real subprocess
# ---------------------------------------------------------------------------

def test_a_good_experiment_runs_and_reports_ok(tmp_path):
    result = run_experiment(_write(tmp_path, GOOD_SCRIPT))
    assert result.ok
    assert result.returncode == 0
    assert result.script_sha256
    ok, reason = result.validate()
    assert ok, reason


def test_the_payload_is_the_last_json_object_on_stdout(tmp_path):
    """Scripts print progress; the contract is the final object."""
    body = """\
import json
print("loading data...")
print(json.dumps({"votes": [[0, 1]], "scores": {"a": 1.0}}))
print("done")
"""
    result = run_experiment(_write(tmp_path, body))
    assert result.payload() == {"votes": [[0, 1]], "scores": {"a": 1.0}}


def test_non_json_stdout_yields_no_payload(tmp_path):
    result = run_experiment(_write(tmp_path, 'print("just a message")\n'))
    assert result.payload() is None
    ok, reason = result.validate()
    assert not ok
    assert "no JSON object" in reason


def test_a_crashing_experiment_is_not_ok_but_does_not_raise(tmp_path):
    result = run_experiment(_write(tmp_path, "raise SystemExit(3)\n"))
    assert result.returncode == 3
    assert not result.ok


def test_stderr_is_captured(tmp_path):
    body = "import sys\nsys.stderr.write('boom\\n')\n"
    result = run_experiment(_write(tmp_path, body))
    assert "boom" in result.stderr


def test_a_timeout_is_reported_as_a_timeout(tmp_path):
    """A hung experiment must not hang the loop; it must come back labelled."""
    result = run_experiment(_write(tmp_path, "import time\ntime.sleep(30)\n"),
                            timeout_s=1.0)
    assert result.timed_out is True
    assert not result.ok


def test_the_documented_hash_matches_the_file_bytes(tmp_path):
    import hashlib

    path = _write(tmp_path, GOOD_SCRIPT)
    result = run_experiment(path)
    assert result.script_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_same_script_hashes_the_same_way(tmp_path):
    a = run_experiment(_write(tmp_path, GOOD_SCRIPT, "a.py"))
    b = run_experiment(_write(tmp_path, GOOD_SCRIPT, "b.py"))
    assert a.script_sha256 == b.script_sha256


# ---------------------------------------------------------------------------
# contract validation
# ---------------------------------------------------------------------------

def _result(stdout: str) -> RunResult:
    return RunResult(entrypoint="x", returncode=0, stdout=stdout, stderr="",
                     timed_out=False, duration_s=0.0, script_sha256="0" * 64)


def test_missing_required_keys_are_named():
    ok, reason = _result(json.dumps({"votes": [[0, 1]]})).validate()
    assert not ok
    assert "scores" in reason


def test_an_empty_leaderboard_is_refused():
    ok, reason = _result(json.dumps({"votes": [[0, 1]], "scores": {}})).validate()
    assert not ok
    assert "non-empty object" in reason


def test_an_empty_vote_matrix_is_refused():
    ok, reason = _result(json.dumps({"votes": [], "scores": {"a": 1.0}})).validate()
    assert not ok
    assert "non-empty list" in reason


def test_the_required_keys_are_the_documented_ones():
    assert REQUIRED_KEYS == ("votes", "scores")


def test_run_result_serialises_without_the_full_stdout():
    """The ledger must stay readable; tails are enough."""
    long_stdout = "\n".join(f"line {i}" for i in range(5000))
    dumped = _result(long_stdout).to_dict()
    assert len(dumped["stdout_tail"]) <= 4000
    json.dumps(dumped)


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------

def test_the_null_provider_has_nothing_to_give():
    """Returning None is the contract: the agent records 'untested'."""
    assert NullExperimentProvider().provide(Hypothesis(text="h")) is None


def test_the_workspace_provider_reads_a_fixed_path(tmp_path):
    path = tmp_path / "current_experiment.py"
    path.write_text(GOOD_SCRIPT, encoding="utf-8")
    script = WorkspaceExperimentProvider(path).provide(Hypothesis(text="h"))
    assert script is not None
    assert script.text == GOOD_SCRIPT
    assert script.source == "workspace"


def test_the_workspace_provider_returns_none_for_a_missing_file(tmp_path):
    provider = WorkspaceExperimentProvider(tmp_path / "nope.py")
    assert provider.provide(Hypothesis(text="h")) is None


def test_the_workspace_provider_returns_none_for_an_empty_file(tmp_path):
    path = tmp_path / "empty.py"
    path.write_text("   \n", encoding="utf-8")
    assert WorkspaceExperimentProvider(path).provide(Hypothesis(text="h")) is None


def test_the_workspace_provider_does_not_glob(tmp_path):
    """A 'newest .py in the workspace' heuristic picks up scratch buffers."""
    (tmp_path / "scratch.py").write_text(GOOD_SCRIPT, encoding="utf-8")
    provider = WorkspaceExperimentProvider(tmp_path / "current_experiment.py")
    assert provider.provide(Hypothesis(text="h")) is None


def test_the_llm_provider_returns_none_without_a_model():
    provider = LLMExperimentProvider(NullLLM())
    assert provider.provide(Hypothesis(text="h")) is None


def test_the_llm_provider_returns_none_when_the_reply_is_not_a_script():
    """Never fall back to a template: a template runs and reports something,
    and 'something' cannot be told apart from a real result downstream."""
    provider = LLMExperimentProvider(ScriptedLLM(default="I cannot write that."))
    assert provider.provide(Hypothesis(text="h")) is None


def test_the_llm_provider_accepts_a_real_script():
    provider = LLMExperimentProvider(ScriptedLLM(default=GOOD_SCRIPT))
    script = provider.provide(Hypothesis(text="h"))
    assert script is not None
    assert script.source == "llm"
    assert script.path is None
    ok, _ = run_experiment_text(script.text)
    assert ok


def run_experiment_text(text: str) -> tuple[bool, str]:
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "exp.py"
        path.write_text(text, encoding="utf-8")
        result = run_experiment(path)
        return result.validate()


def test_the_llm_provider_strips_code_fences():
    fenced = "```python\n" + GOOD_SCRIPT + "```"
    provider = LLMExperimentProvider(ScriptedLLM(default=fenced))
    script = provider.provide(Hypothesis(text="h"))
    assert script is not None
    assert script.text.startswith("import json")
    assert "```" not in script.text


def test_script_detection_needs_both_code_and_the_contract():
    assert _looks_like_script(GOOD_SCRIPT)
    assert not _looks_like_script("This is prose about an experiment.")
    assert not _looks_like_script("import json\nprint('hi')\n" * 5)
    assert not _looks_like_script("")


def test_fence_stripping_is_forgiving():
    assert _strip_fences("plain") == "plain"
    assert _strip_fences("```\nx=1\n```") == "x=1"
    assert _strip_fences("") == ""


# ---------------------------------------------------------------------------
# script materialisation and hashing
# ---------------------------------------------------------------------------

def test_the_script_hash_is_stable_across_materialisation(tmp_path):
    script = ExperimentScript(text=GOOD_SCRIPT)
    before = script.sha256
    script.materialise(tmp_path)
    assert script.sha256 == before
    assert script.path is not None and script.path.exists()


def test_materialising_twice_overwrites_rather_than_appends(tmp_path):
    script = ExperimentScript(text=GOOD_SCRIPT)
    script.materialise(tmp_path)
    script.materialise(tmp_path)
    assert script.path.read_text(encoding="utf-8") == GOOD_SCRIPT


def test_build_provider_dispatches_and_refuses_unknown_kinds(tmp_path):
    assert isinstance(build_provider("null"), NullExperimentProvider)
    assert isinstance(
        build_provider("workspace", path=tmp_path / "e.py"),
        WorkspaceExperimentProvider)
    with pytest.raises(ValueError):
        build_provider("workspace")  # no path
    with pytest.raises(ValueError):
        build_provider("llm")  # no llm
    with pytest.raises(ValueError):
        build_provider("telepathy")
