"""LLM access, kept deliberately thin.

One interface, one protocol: an OpenAI-compatible ``/chat/completions``
endpoint. No vendor SDK is imported, so pointing arx at a local server, a
gateway, or a hosted API is a base-URL change rather than a code change.

Two behaviours matter more than the HTTP call:

* **Every call is accounted for at the moment it is made**, through a
  ``UsageSink``. arx cannot make a call that skips this, because there is no
  other way to reach the network from inside the package. What it *cannot*
  guarantee is that the count is honest — the number is self-reported. An
  independent count belongs at the gateway; ``arx.oracles.auditor`` reconciles
  the two and reports any gap rather than assuming agreement.

* **``NullLLM`` is a first-class backend.** Most of arx works with no model at
  all, and the CI must not depend on one. When no endpoint is configured the
  agent degrades to it and the ledger records that no model was consulted —
  which is different from pretending a model answered.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from .config import AgentConfig

UsageSink = Callable[[str, dict], None]


class ChatModel(Protocol):
    """The only thing the rest of arx assumes about a language model."""

    def chat(self, system: str, user: str, *, temperature: float | None = None) -> str: ...

    @property
    def available(self) -> bool: ...


@dataclass
class LLMStats:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    failures: int = 0
    by_model: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "failures": self.failures,
            "by_model": dict(self.by_model),
        }


class NullLLM:
    """A backend that answers nothing, and says so.

    Used when no endpoint is configured. Callers must treat an empty response
    as "no model consulted" — never as a model that had nothing to say.
    """

    name = "null"

    def __init__(self, reason: str = "no endpoint configured"):
        self.reason = reason
        self.stats = LLMStats()

    @property
    def available(self) -> bool:
        return False

    def chat(self, system: str, user: str, *, temperature: float | None = None) -> str:
        return ""


class OpenAICompatibleLLM:
    """Minimal client for an OpenAI-compatible chat endpoint (stdlib only)."""

    name = "openai-compatible"

    def __init__(
        self,
        cfg: AgentConfig,
        *,
        usage_sink: UsageSink | None = None,
        model: str | None = None,
    ):
        self.cfg = cfg
        self.model = model or cfg.llm_model
        self.base_url = cfg.llm_base_url.rstrip("/")
        self.api_key = cfg.llm_api_key
        self.usage_sink = usage_sink
        self.stats = LLMStats()

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.model)

    def chat(self, system: str, user: str, *, temperature: float | None = None) -> str:
        if not self.available:
            raise RuntimeError(
                "LLM endpoint not configured (set ARX_LLM_BASE_URL and ARX_LLM_MODEL)"
            )
        body = {
            "model": self.model,
            "temperature": self.cfg.llm_temperature if temperature is None else temperature,
            "messages": [
                *([{"role": "system", "content": system}] if system else []),
                {"role": "user", "content": user},
            ],
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
            method="POST",
        )

        # Account for the attempt before it happens: a call that times out still
        # consumed a slot at the gateway, and a ledger that only counts
        # successes will under-report exactly when it matters.
        self.stats.calls += 1
        self.stats.by_model[self.model] = self.stats.by_model.get(self.model, 0) + 1
        if self.usage_sink is not None:
            self.usage_sink("llm_call", {"model": self.model, "phase": "request"})

        try:
            with urllib.request.urlopen(req, timeout=self.cfg.llm_timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            self.stats.failures += 1
            raise

        usage = data.get("usage") or {}
        self.stats.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.stats.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        if self.usage_sink is not None:
            self.usage_sink("llm_tokens", {"model": self.model, **usage})

        choices = data.get("choices") or []
        if not choices:
            self.stats.failures += 1
            raise RuntimeError("LLM response contained no choices")
        return str(choices[0].get("message", {}).get("content", ""))


class ScriptedLLM:
    """A deterministic backend for tests and offline demos.

    Returns canned replies keyed by a substring of the prompt, falling back to
    ``default``. Every reply is recorded so a test can assert on what was asked.
    """

    name = "scripted"

    def __init__(self, replies: dict[str, str] | None = None, default: str = ""):
        self.replies = dict(replies or {})
        self.default = default
        self.prompts: list[tuple[str, str]] = []
        self.stats = LLMStats()

    @property
    def available(self) -> bool:
        return True

    def chat(self, system: str, user: str, *, temperature: float | None = None) -> str:
        self.prompts.append((system, user))
        self.stats.calls += 1
        haystack = f"{system}\n{user}".lower()
        for needle, reply in self.replies.items():
            if needle.lower() in haystack:
                return reply
        return self.default


def build_llm(
    cfg: AgentConfig,
    *,
    usage_sink: UsageSink | None = None,
    force_null: bool = False,
) -> ChatModel:
    """Pick a backend: real if configured, otherwise a null one."""
    if force_null:
        return NullLLM("forced offline")
    probe = OpenAICompatibleLLM(cfg, usage_sink=usage_sink)
    if probe.available:
        return probe
    return NullLLM("ARX_LLM_BASE_URL / ARX_LLM_MODEL not set")


def parse_json_object(text: str) -> dict | None:
    """Extract the first JSON object from a model reply.

    Models wrap JSON in prose and code fences. Rather than a prompt re-ask loop
    (which costs a call and can still fail), scan for the first balanced object.
    Returning ``None`` is a real outcome the caller must handle.
    """
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1] if "```" in cleaned[3:] else cleaned[3:]
        cleaned = cleaned.lstrip("json").strip() if cleaned.lower().startswith("json") else cleaned
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(cleaned):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(cleaned[start : i + 1])
                except json.JSONDecodeError:
                    start = -1
    return None


__all__ = [
    "ChatModel",
    "LLMStats",
    "NullLLM",
    "OpenAICompatibleLLM",
    "ScriptedLLM",
    "UsageSink",
    "build_llm",
    "parse_json_object",
]
