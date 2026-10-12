"""Where experiment scripts come from.

The agent does not invent experiment code. A *provider* supplies the script that
tests a hypothesis, and the agent's job is to run it honestly: hash it, pin the
hash before the run, and refuse to interpret results from a script that changed
afterwards.

Three providers ship:

* :class:`WorkspaceExperimentProvider` — use ``<workspace>/current_experiment.py``
  if it exists. This is the default, and it is what an interactive user wants:
  write the experiment, point arx at it, let the gate judge.
* :class:`LLMExperimentProvider` — ask a model to write the script. Opt-in,
  because executing model-generated code deserves a container, and because a
  provider that fabricates a script when the model is unavailable is worse than
  one that says it cannot.
* :class:`NullExperimentProvider` — never supplies anything. The agent then
  records ``no_entrypoint`` for the hypothesis and moves on. Recording that a
  hypothesis was never tested is the honest outcome; inventing a plausible
  result is the failure mode this whole framework exists to prevent.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..hypothesis import Hypothesis
from ..llm import ChatModel

EXPERIMENT_SYSTEM = (
    "You write short, self-contained Python experiments. You print exactly one "
    "JSON object on the last line of stdout, and nothing after it. You never "
    "invent results; the script computes them."
)

EXPERIMENT_TEMPLATE = """Write a Python 3.12 script that tests this hypothesis:

  {hypothesis}

It must print one JSON object on its last stdout line, shaped exactly like:

{{"votes": [[0,1,...], ...],
  "scores": {{"system-a": 0.91, "system-b": 0.88}},
  "judge_ids": ["g1","g2","g3"],
  "contestant_ids": ["system-a","system-b"],
  "item_ids": [0,1,2,...],
  "se": 0.02,
  "n_looks": 1}}

Rules:
  - "votes" is generators x items, integer labels, -1 for abstention.
  - at least three independent generators, or the audit is undecidable.
  - judge_ids and contestant_ids must be disjoint.
  - "se" is the standard error of the margin you report, if you can compute it.
  - deterministic given the seed; use random.Random(seed) or numpy Generator.
  - no network, no file writes outside the working directory.

Reply with the script only, no prose and no code fences."""


@dataclass
class ExperimentScript:
    """A runnable experiment, with the hash that will be pinned."""

    text: str
    path: Path | None = None
    source: str = "workspace"

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def materialise(self, directory: Path, name: str = "arx_experiment.py") -> Path:
        """Write the script to disk and return its path."""
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / name
        # Write bytes, not text: the pinned hash is over ``self.text`` encoded
        # as UTF-8, and a text-mode write is free to translate newlines on some
        # platforms, which would make the file hash differ from the pinned one
        # for reasons that have nothing to do with tampering.
        target.write_bytes(self.text.encode("utf-8"))
        self.path = target
        return target


class ExperimentProvider(Protocol):
    name: str

    def provide(self, hypothesis: Hypothesis) -> ExperimentScript | None: ...


class NullExperimentProvider:
    """Supplies nothing. The agent records that the hypothesis was untested."""

    name = "null"

    def provide(self, hypothesis: Hypothesis) -> ExperimentScript | None:
        return None


class WorkspaceExperimentProvider:
    """Reads a fixed script path from the workspace.

    The path is fixed on purpose. A provider that globbed for the newest
    ``.py`` file would silently pick up the user's editor scratch buffer.
    """

    name = "workspace"

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def provide(self, hypothesis: Hypothesis) -> ExperimentScript | None:
        if not self.path.exists():
            return None
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError:
            return None
        if not text.strip():
            return None
        return ExperimentScript(text=text, path=self.path, source=self.name)


class LLMExperimentProvider:
    """Asks a model to write the experiment.

    Returns ``None`` whenever the model is unavailable or its reply does not
    look like a script. It never falls back to a template — a template would
    produce a script that runs and reports something, and "something" is
    indistinguishable from a real result once it reaches the ledger.
    """

    name = "llm"

    def __init__(self, llm: ChatModel, *, workspace: str | Path | None = None):
        self.llm = llm
        self.workspace = Path(workspace) if workspace else None

    def provide(self, hypothesis: Hypothesis) -> ExperimentScript | None:
        if not self.llm.available:
            return None
        reply = self.llm.chat(
            EXPERIMENT_SYSTEM,
            EXPERIMENT_TEMPLATE.format(hypothesis=hypothesis.text),
        )
        text = _strip_fences(reply)
        if not _looks_like_script(text):
            return None
        return ExperimentScript(text=text, path=None, source=self.name)


def _strip_fences(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned.startswith("```"):
        return cleaned
    body = cleaned[3:]
    newline = body.find("\n")
    if newline != -1:
        body = body[newline + 1 :]
    if body.rstrip().endswith("```"):
        body = body.rstrip()[:-3]
    return body.strip()


def _looks_like_script(text: str) -> bool:
    if not text or len(text) < 40:
        return False
    has_code = any(
        marker in text
        for marker in ("import ", "def ", "print(", "=")
    )
    has_contract = '"votes"' in text or "'votes'" in text or "votes" in text
    return has_code and has_contract


def build_provider(kind: str, **kwargs) -> ExperimentProvider:
    if kind in ("null", "none"):
        return NullExperimentProvider()
    if kind in ("workspace", "file"):
        path = kwargs.get("path")
        if not path:
            raise ValueError("workspace provider requires a path")
        return WorkspaceExperimentProvider(path)
    if kind in ("llm", "model"):
        llm = kwargs.get("llm")
        if llm is None:
            raise ValueError("llm provider requires an llm")
        return LLMExperimentProvider(llm, workspace=kwargs.get("workspace"))
    raise ValueError(f"unknown experiment provider: {kind!r}")


__all__ = [
    "EXPERIMENT_SYSTEM",
    "EXPERIMENT_TEMPLATE",
    "ExperimentProvider",
    "ExperimentScript",
    "LLMExperimentProvider",
    "NullExperimentProvider",
    "WorkspaceExperimentProvider",
    "build_provider",
]
