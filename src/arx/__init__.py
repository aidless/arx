"""arx — an autonomous research loop whose claims must clear a statistical gate.

The package is organised so that the part that matters most can be used
without the rest: :mod:`arx.audit` has no dependency on the agent, the
scheduler, the LLM layer, or the network. It imports numpy and nothing else
beyond the standard library, and it can be installed and cited on its own.

The layering, from the bottom up::

    arx.audit        noise-floor, power, sequential, pre-registration, Goodhart
                     — statistics only, no agent, no model, no network
    arx.experiments  ledger (Merkle + optional ed25519) and subprocess runner
    arx.scheduler    budget-aware action selection; deterministic, replayable
    arx.literature   search, Zotero library, citation verification
    arx.longterm     cross-run ledger with semantic recall of prior hypotheses
    arx.oracles      external usage accounting to check self-reported budgets
    arx.theory       theory -> hypothesis graph and falsification bookkeeping
    arx.agent        the loop that ties them together
    arx.gate         the refusal: a context manager that blocks bad claims

What this does *not* do is stated in the README and repeated in every report:
it addresses procedural self-deception, not epistemological self-deception.
"""

from __future__ import annotations

import importlib.util
import platform
import sys

__version__ = "0.3.0"

from .audit.verdict import AuditReport, UndecidableReason, Verdict  # noqa: E402
from .config import CONFIG, AgentConfig  # noqa: E402
from .gate import EvaluationGate, GateRejection  # noqa: E402


def _have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def probe_capabilities() -> dict:
    """Report what this installation can actually do, without guessing.

    Every value is measured, not assumed. Absent optional pieces are reported
    as absent rather than silently degrading, because a run that quietly used
    a weaker path produces a number that cannot be compared with a full one.
    """
    import numpy

    cfg = AgentConfig()
    caps = {
        "version": __version__,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": numpy.__version__,
        "scipy_present": _have("scipy"),
        "gymnasium_present": _have("gymnasium"),
        "stable_baselines3_present": _have("stable_baselines3"),
        "cryptography_present": _have("cryptography"),
        "sentence_transformers_present": _have("sentence_transformers"),
        "sqlite3_present": _have("sqlite3"),
        "embedder": cfg.embedder,
        "llm_model": cfg.llm_model,
        "llm_base_url": cfg.llm_base_url,
        "llm_configured": bool(cfg.llm_base_url and cfg.llm_model),
        "zotero_enabled": cfg.zotero_enabled,
        "semantic_recall": cfg.semantic_duplicate_threshold > 0,
        "notes": [],
    }
    if not caps["scipy_present"]:
        caps["notes"].append(
            "scipy absent: arx.audit does not need it (normal quantiles come "
            "from statistics.NormalDist), and no core path requires it."
        )
    if not caps["gymnasium_present"]:
        caps["notes"].append(
            "gymnasium absent: the RL demonstration is unavailable. "
            "Install 'arx-researcher[rl]' to enable it."
        )
    if not caps["sentence_transformers_present"]:
        caps["notes"].append(
            "sentence-transformers absent: cross-run recall falls back to the "
            "hashing embedder, which matches wording, not meaning. Treat its "
            "flags as recall aids, not as an effective-test-count estimate."
        )
    if caps["llm_configured"] and not cfg.llm_api_key:
        caps["notes"].append(
            "no API key set: the loop will run, but hypothesis generation "
            "returns nothing and every generated step is recorded as skipped."
        )
    return caps


__all__ = [
    "AgentConfig",
    "AuditReport",
    "CONFIG",
    "EvaluationGate",
    "GateRejection",
    "UndecidableReason",
    "Verdict",
    "__version__",
    "probe_capabilities",
]
