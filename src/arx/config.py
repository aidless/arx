"""Configuration: every tunable lives here, with its rationale.

A constant buried in a module is a constant nobody can audit. The values below
are the ones that decide when the loop stops, when a monitor fires, and how
wide a claim has to be before it counts. Each carries the reason it has the
value it has, and — where the value is a judgement call rather than a
derivation — that is said explicitly.

Environment overrides use the ``ARX_`` prefix so a container can be retargeted
without editing code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(f"ARX_{name}", default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(f"ARX_{name}")
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(f"ARX_{name}")
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(f"ARX_{name}")
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class AgentConfig:
    """Runtime configuration for one research run."""

    # -- LLM ---------------------------------------------------------------
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", ""))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", ""))
    llm_timeout_s: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT", 120.0))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.2))

    # -- workspace ---------------------------------------------------------
    workspace: Path = field(
        default_factory=lambda: Path(_env("WORKSPACE", "./arx_workspace"))
    )
    max_iterations: int = field(default_factory=lambda: _env_int("MAX_ITERATIONS", 40))

    # -- budget ------------------------------------------------------------
    max_llm_calls: int = field(default_factory=lambda: _env_int("MAX_LLM_CALLS", 200))
    max_wall_seconds: float = field(
        default_factory=lambda: _env_float("MAX_WALL_SECONDS", 1800.0)
    )
    max_experiment_runs: int = field(
        default_factory=lambda: _env_int("MAX_EXPERIMENT_RUNS", 12)
    )
    max_literature_sweeps: int = field(
        default_factory=lambda: _env_int("MAX_LITERATURE_SWEEPS", 3)
    )
    sandbox_timeout_s: float = field(
        default_factory=lambda: _env_float("SANDBOX_TIMEOUT", 600.0)
    )

    # -- the gate ----------------------------------------------------------
    #: Reported margins smaller than this are never called READABLE. 0.02 is a
    #: conventional reporting threshold in leaderboard papers (two accuracy
    #: points); it has no statistical content and exists so that the gate has a
    #: declared floor. The *effective* threshold is max(this, MDE).
    gate_delta_declared: float = field(default_factory=lambda: _env_float("GATE_DELTA", 0.02))
    #: Two-sided alpha for every interval the gate reports.
    gate_alpha: float = field(default_factory=lambda: _env_float("GATE_ALPHA", 0.05))
    #: Target power used when converting a standard error into an MDE.
    gate_power: float = field(default_factory=lambda: _env_float("GATE_POWER", 0.8))
    #: Strict mode rejects MARGINAL as well as UNREADABLE/UNDECIDABLE.
    gate_strict: bool = field(default_factory=lambda: _env_bool("GATE_STRICT", True))
    #: Master switch. With the gate off, arx is a study assistant, not an
    #: auditor — and the ledger says so, because the flag is recorded.
    gate_enabled: bool = field(default_factory=lambda: _env_bool("GATE_ENABLED", True))

    # -- stall detection ---------------------------------------------------
    #: Steps without progress (hypothesis set / accepted set unchanged) before
    #: the scheduler writes up and stops. 6 is a judgement call: high enough
    #: that a legitimate LITERATURE → HYPOTHESIZE → EXPERIMENT → GATE cycle is
    #: never mistaken for a stall, low enough that a spin ends in minutes.
    stall_limit: int = field(default_factory=lambda: _env_int("STALL_LIMIT", 6))

    # -- Goodhart monitor --------------------------------------------------
    #: Sliding window over (predicted IG, realised outcome) pairs.
    goodhart_window: int = field(default_factory=lambda: _env_int("GOODHART_WINDOW", 30))
    #: Spearman below this means the scheduler has drifted from reality. 0.3 is
    #: a judgement call, not a derivation; it is recorded in every status.
    goodhart_threshold: float = field(
        default_factory=lambda: _env_float("GOODHART_THRESHOLD", 0.3)
    )
    goodhart_min_samples: int = field(
        default_factory=lambda: _env_int("GOODHART_MIN_SAMPLES", 10)
    )

    # -- literature --------------------------------------------------------
    zotero_enabled: bool = field(default_factory=lambda: _env_bool("ZOTERO_ENABLED", False))
    zotero_api_url: str = field(
        default_factory=lambda: _env("ZOTERO_API_URL", "http://localhost:23119/api")
    )
    zotero_library_id: str = field(default_factory=lambda: _env("ZOTERO_LIBRARY_ID", ""))
    zotero_api_key: str = field(default_factory=lambda: _env("ZOTERO_API_KEY", ""))
    citation_check_enabled: bool = field(
        default_factory=lambda: _env_bool("CITATION_CHECK", True)
    )
    search_max_results: int = field(
        default_factory=lambda: _env_int("SEARCH_MAX_RESULTS", 20)
    )

    # -- long-term memory --------------------------------------------------
    global_ledger_path: Path = field(
        default_factory=lambda: Path(
            _env("GLOBAL_LEDGER", str(Path.home() / ".arx" / "global_ledger.db"))
        )
    )
    #: Cosine similarity above which two hypotheses count as the same line of
    #: enquiry. 0.80 is a judgement call, and the value that produced each
    #: decision is written into the ledger so it can be revised later.
    semantic_duplicate_threshold: float = field(
        default_factory=lambda: _env_float("DUPLICATE_THRESHOLD", 0.80)
    )
    embedder: str = field(default_factory=lambda: _env("EMBEDDER", "hash"))
    semantic_top_k: int = field(default_factory=lambda: _env_int("SEMANTIC_TOP_K", 10))

    # -- provenance --------------------------------------------------------
    #: Run signed ledgers? Requires the [provenance] extra.
    sign_ledger: bool = field(default_factory=lambda: _env_bool("SIGN_LEDGER", False))
    #: Optional external usage log written by the API gateway. When set, arx
    #: reconciles the self-reported call count against it.
    oracle_log_path: Path | None = field(
        default_factory=lambda: (
            Path(p) if (p := _env("ORACLE_LOG", "").strip()) else None
        )
    )

    # -- theory layer ------------------------------------------------------
    #: Optional theory root that generated hypotheses are attached to. Without
    #: one, every hypothesis is recorded as exploratory — which is the honest
    #: label for a test that no theory would notice the outcome of. The
    #: coverage ratio then reports the deficit instead of hiding it.
    theory_root: str = field(default_factory=lambda: _env("THEORY_ROOT", ""))
    #: The observation that would show the theory wrong. Required for the
    #: theory to be falsifiable; without it the theory is registered but cannot
    #: be refuted, and ``TheoryLayer`` reports it as such.
    theory_falsified_by: str = field(
        default_factory=lambda: _env("THEORY_FALSIFIED_BY", "")
    )

    # -- misc --------------------------------------------------------------
    seed: int = field(default_factory=lambda: _env_int("SEED", 0))
    verbose: bool = field(default_factory=lambda: _env_bool("VERBOSE", True))

    def as_dict(self) -> dict:
        out = {}
        for key, value in self.__dict__.items():
            out[key] = str(value) if isinstance(value, Path) else value
        return out

    def ledger_flags(self) -> dict:
        """The settings that change what a result *means*.

        Recorded on every run so that a reader can tell whether an experiment
        was gated, and with what threshold — a number without its gate setting
        is not interpretable.
        """
        return {
            "gate_enabled": self.gate_enabled,
            "gate_strict": self.gate_strict,
            "gate_delta_declared": self.gate_delta_declared,
            "gate_alpha": self.gate_alpha,
            "gate_power": self.gate_power,
            "stall_limit": self.stall_limit,
            "goodhart_window": self.goodhart_window,
            "goodhart_threshold": self.goodhart_threshold,
            "semantic_duplicate_threshold": self.semantic_duplicate_threshold,
            "seed": self.seed,
        }


CONFIG = AgentConfig()

__all__ = ["AgentConfig", "CONFIG"]
