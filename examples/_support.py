"""Shared offline fixtures for the examples.

Every example in this directory runs with no network, no API key and no model.
That is deliberate: an example that needs a credential is an example most
readers cannot run, and a demonstration whose numbers you cannot reproduce is
just a claim.

The two things the agent needs are a model and an experiment provider. Here
both are scripted, so the numbers in the ledger come from code you can read.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


# Make `python3 examples/XX.py` work without setting PYTHONPATH. The examples
# are the first thing a reader runs, so "forgot an env var" is a bad first
# experience to hand somebody. The bootstrap runs before the arx imports
# below, so it cannot be written as a plain statement without splitting the
# import block; wrapping it keeps the file lint-clean at E402.
def _bootstrap_src() -> None:
    src = Path(__file__).resolve().parent.parent / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_bootstrap_src()

from arx.config import AgentConfig  # noqa: E402
from arx.experiments.provider import ExperimentScript  # noqa: E402
from arx.hypothesis import Hypothesis  # noqa: E402
from arx.literature.search import Paper, SearchOutcome  # noqa: E402
from arx.llm import ScriptedLLM  # noqa: E402

# ---------------------------------------------------------------------------
# an experiment that really runs, and is really deterministic
# ---------------------------------------------------------------------------

#: Emits a two-system leaderboard graded by five generators. The seed is
#: derived from the hypothesis text, so different hypotheses get different
#: data while the same hypothesis always gets the same data.
_RUNNER = '''\
"""Synthetic leaderboard experiment (offline, deterministic).

Prints one JSON object on its last stdout line, which is the contract the
runner looks for. Details of the measurement:
  * GENERATORS independent gold producers label every item;
  * AGREEMENT is the per-item probability a generator matches the latent label;
  * two contestants answer with their own error rates.

Change AGREEMENT and the audit's verdict changes with it -- that is the whole
point of the noise floor.
"""
import json
import random

SEED = %(seed)d
N_ITEMS = %(n_items)d
GENERATORS = %(generators)d
AGREEMENT = %(agreement)r
CONTESTANT = %(contestant)r

rng = random.Random(SEED)

truth = [rng.randint(0, 1) for _ in range(N_ITEMS)]
votes = [
    [int(t) if rng.random() < AGREEMENT else 1 - int(t) for t in truth]
    for _ in range(GENERATORS)
]

# A contestant answers each item correctly with probability CONTESTANT.
strong = [int(t) if rng.random() < CONTESTANT["strong"] else 1 - int(t) for t in truth]
weak = [int(t) if rng.random() < CONTESTANT["weak"] else 1 - int(t) for t in truth]

# Progress output is allowed; the runner takes the LAST JSON object.
print("generated " + str(N_ITEMS) + " items with " + str(GENERATORS) + " generators")

print(json.dumps({
    "votes": votes,
    "scores": {
        "strong": sum(a == b for a, b in zip(strong, truth)) / N_ITEMS,
        "weak": sum(a == b for a, b in zip(weak, truth)) / N_ITEMS,
    },
    "judge_ids": ["g" + str(i) for i in range(GENERATORS)],
    "contestant_ids": ["strong", "weak"],
    "item_ids": list(range(N_ITEMS)),
    "se": 0.01,
%(k_max_field)s}))
'''

#: Deliberately declaring a look count that contradicts the pre-registration.
#: The script cannot know the plan, so the honest thing is to say nothing --
#: the ledger is the authority. A script that *does* assert a number is making
#: a claim, and a claim that disagrees with the ledger is refused.
K_MAX_FIELD = '    "k_max": %d,\n'



def leaderboard_script(
    hypothesis_text: str,
    *,
    n_items: int = 400,
    generators: int = 5,
    agreement: float = 0.93,
    strong: float = 0.96,
    weak: float = 0.60,
    declared_k_max: int | None = None,
) -> str:
    """Build a runnable experiment script, seeded by the hypothesis text.

    Uses SHA-256 rather than ``hash()`` so the seed is stable across
    processes; ``hash()`` is salted per run and would make the "same
    hypothesis, same data" property false in a way nobody would notice.

    ``declared_k_max`` adds a claim about how many looks were taken. The
    honest default is to omit it: the pre-registration in the ledger is the
    authority, and a script that contradicts it is refused.
    """
    digest = hashlib.sha256(hypothesis_text.encode("utf-8")).hexdigest()
    seed = int(digest[:8], 16)
    return _RUNNER % {
        "seed": seed,
        "n_items": n_items,
        "generators": generators,
        "agreement": agreement,
        "contestant": {"strong": strong, "weak": weak},
        "k_max_field": ("" if declared_k_max is None
                        else K_MAX_FIELD % int(declared_k_max)),
    }


class StaticProvider:
    """Supplies one fixed script for every hypothesis."""

    name = "static"

    def __init__(self, script: str):
        self.script = script
        self.asked: list[str] = []

    def provide(self, hypothesis: Hypothesis) -> ExperimentScript | None:
        self.asked.append(hypothesis.text)
        return ExperimentScript(text=self.script, path=None, source=self.name)


class FactoryProvider:
    """Builds a script per hypothesis from a callable.

    This is what makes ``leaderboard_script(hypothesis.text)`` useful: each
    hypothesis gets its own deterministic dataset rather than all three sharing
    one, which would quietly turn three tests into one test repeated.
    """

    name = "factory"

    def __init__(self, factory):
        self.factory = factory
        self.asked: list[str] = []

    def provide(self, hypothesis: Hypothesis) -> ExperimentScript | None:
        self.asked.append(hypothesis.text)
        return ExperimentScript(text=self.factory(hypothesis), path=None, source=self.name)


class ScriptProvider:
    """Supplies a script chosen per hypothesis, by substring match."""

    name = "scripted"

    def __init__(self, scripts: dict[str, str], default: str | None = None):
        self.scripts = dict(scripts)
        self.default = default

    def provide(self, hypothesis: Hypothesis) -> ExperimentScript | None:
        for needle, script in self.scripts.items():
            if needle.lower() in hypothesis.text.lower():
                return ExperimentScript(text=script, path=None, source=self.name)
        if self.default is None:
            return None
        return ExperimentScript(text=self.default, path=None, source=self.name)


# ---------------------------------------------------------------------------
# a scripted model
# ---------------------------------------------------------------------------

HYPOTHESES_REPLY = """\
{"hypotheses": [
  {"text": "Reranking by the judge's own confidence improves agreement with the majority gold",
   "rationale": "Confidence is a proxy for item difficulty, so weighting by it should reduce ties.",
   "confidence": 0.6, "expected_effect_se_ratio": 0.5},
  {"text": "A three-generator floor separates strong systems but not weak ones",
   "rationale": "With few generators the floor tracks one generator's errors rather than the consensus.",
   "confidence": 0.5},
  {"text": "Mean-probability aggregation changes the leader when generators disagree at the margin",
   "rationale": "The arithmetic mean of two confident opposite votes is decided by rounding.",
   "confidence": 0.4}
]}
"""


def make_llm() -> ScriptedLLM:
    """A model that answers the one prompt the agent actually sends."""
    return ScriptedLLM(
        replies={"hypothes": HYPOTHESES_REPLY, "hypothesis": HYPOTHESES_REPLY},
        default=HYPOTHESES_REPLY,
    )


# ---------------------------------------------------------------------------
# a searchable literature
# ---------------------------------------------------------------------------

PAPERS = [
    Paper(
        title="Reading a Leaderboard Before You Cite It: Noise Floors for LLM Judges",
        authors=["A. Author"],
        year=2026,
        abstract="Model-generated reference labels define a floor below which rank "
                 "differences are not resolvable.",
        url="https://example.invalid/noise-floor",
        source="example",
    ),
    Paper(
        title="Are Emergent Abilities of Large Language Models a Mirage?",
        authors=["R. Schaeffer"],
        year=2023,
        abstract="Reported sharp capability jumps depend on the metric chosen: "
                 "discrete exact-match versus continuous log-loss.",
        url="https://example.invalid/emergent",
        source="example",
    ),
    Paper(
        title="The False Discovery Rate: A Practical and Powerful Approach to Multiple Testing",
        authors=["Y. Benjamini", "Y. Hochberg"],
        year=1995,
        abstract="Controls the expected proportion of false discoveries among rejections.",
        url="https://example.invalid/bh",
        source="example",
    ),
]


def make_searcher(papers: list[Paper] | None = None):
    papers = list(papers if papers is not None else PAPERS)

    def search(query: str, max_results: int = 20) -> SearchOutcome:
        return SearchOutcome(papers=papers[:max_results], sources_ok=["example"])

    return search


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def offline_config(workspace: Path, **overrides) -> AgentConfig:
    """A config that touches nothing outside ``workspace``.

    ``semantic_duplicate_threshold=0`` matters: the default would open
    ``~/.arx/global_ledger.db``, and an example must not write to the reader's
    home directory.
    """
    settings = dict(
        workspace=str(workspace),
        semantic_duplicate_threshold=0.0,
        llm_base_url="",
        llm_model="",
        max_iterations=12,
        max_experiment_runs=4,
        max_literature_sweeps=1,
        max_llm_calls=50,
        max_wall_seconds=300.0,
        sandbox_timeout_s=60,
        verbose=False,
    )
    settings.update(overrides)
    return AgentConfig(**settings)


# ---------------------------------------------------------------------------
# presentation
# ---------------------------------------------------------------------------

def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def out_dir(name: str) -> Path:
    """A scratch directory beside the examples, cleaned on each run."""
    import shutil

    path = Path(__file__).resolve().parent / "_out" / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_src_on_path() -> None:
    """Let the examples run from the repository root as well."""
    root = Path(__file__).resolve().parent.parent / "src"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
