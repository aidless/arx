"""Independent usage accounting.

``Budget.charge`` is called by the system being audited. The LLM backend could
swap for one that forgets to call it, a retry path could bypass it, or the agent
could simply not call it. A budget the audited system reports about itself is
not a control — it is a promise.

The fix is not a better counter inside the same process. It is a *second*
counter kept somewhere the audited code cannot reach: at the API gateway, in a
proxy, in the provider's billing export. This module reconciles the two and
reports the gap.

What it can and cannot do:

* It can tell you the self-reported count and the external count disagree, by
  how much, and where the discrepancy first appears.
* It cannot make the external log trustworthy — if the agent can write to it,
  it is not external. The log path is taken as configuration precisely so the
  user can point it at something the agent has no business touching.

A non-zero gap is not proof of dishonesty. It is a number that requires an
explanation before the run's conclusions are used, which is the whole point.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OracleEntry:
    ts: float
    event: str
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"ts": self.ts, "event": self.event, "payload": self.payload}


def record_usage(path: str | Path, event: str, payload: dict | None = None) -> None:
    """Append one usage event to the external log.

    Intended to be called from the place that actually holds the credentials —
    a gateway, a proxy, or a thin ``UsageSink`` wired into the HTTP layer. If
    it is called from inside the agent loop, the log stops being external and
    the reconciliation below becomes theatre.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    entry = OracleEntry(ts=time.time(), event=event, payload=payload or {})
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry.to_dict(), sort_keys=True, default=str) + "\n")


def read_usage(path: str | Path) -> list[OracleEntry]:
    p = Path(path)
    if not p.exists():
        return []
    out: list[OracleEntry] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        out.append(
            OracleEntry(
                ts=float(obj.get("ts", 0.0)),
                event=str(obj.get("event", "")),
                payload=obj.get("payload") if isinstance(obj.get("payload"), dict) else {},
            )
        )
    return out


@dataclass
class Reconciliation:
    reported: dict
    external: dict
    gaps: dict
    tolerance: float
    ok: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "reported": dict(self.reported),
            "external": dict(self.external),
            "gaps": dict(self.gaps),
            "tolerance": self.tolerance,
            "ok": self.ok,
            "notes": list(self.notes),
        }

    def to_text(self) -> str:
        lines = ["usage reconciliation", f"  tolerance: {self.tolerance:g}"]
        for key in sorted(set(self.reported) | set(self.external)):
            rep = self.reported.get(key)
            ext = self.external.get(key)
            gap = self.gaps.get(key)
            mark = "" if gap is None or abs(gap) <= self.tolerance else "  <-- DISCREPANCY"
            lines.append(f"  {key}: reported={rep} external={ext} gap={gap}{mark}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        lines.append(f"  verdict: {'consistent' if self.ok else 'REQUIRES EXPLANATION'}")
        return "\n".join(lines)


def count_events(entries: list[OracleEntry]) -> dict:
    """Aggregate an external log into counters by event type and model.

    Token totals are summed, not counted: an ``llm_tokens`` event carries
    ``prompt_tokens``/``completion_tokens`` in its payload, and comparing those
    against the run's own tallies is the only way to catch a gateway that is
    billing for more work than the ledger admits to.
    """
    counts = Counter()
    models = Counter()
    tokens = Counter()
    for entry in entries:
        counts[entry.event] += 1
        model = entry.payload.get("model")
        if model:
            models[str(model)] += 1
        for key in ("prompt_tokens", "completion_tokens"):
            if key in entry.payload:
                try:
                    tokens[key] += float(entry.payload.get(key) or 0)
                except (TypeError, ValueError):
                    continue
    out = dict(counts)
    if models:
        out["by_model"] = dict(models)
    out.update({k: v for k, v in tokens.items()})
    return out


def reconcile(
    reported: dict,
    external_entries: list[OracleEntry],
    *,
    tolerance: float = 0.0,
    keys: tuple[str, ...] = ("llm_call",),
) -> Reconciliation:
    """Compare self-reported usage against the external log.

    ``tolerance`` is an absolute allowance on the count difference. It defaults
    to zero, because "the system over-reported by a little" is exactly the kind
    of fact that should be looked at rather than absorbed.
    """
    external = count_events(external_entries)
    gaps: dict[str, float] = {}
    notes: list[str] = []

    for key in keys:
        rep_raw = reported.get(key)
        ext_raw = external.get(key)
        if rep_raw is None and ext_raw is None:
            continue
        rep = float(rep_raw or 0)
        ext = float(ext_raw or 0)
        gaps[key] = rep - ext
        if rep == 0 and ext > 0:
            notes.append(
                f"{key}: external log shows {ext:g} call(s) but the run reported none"
            )
        elif ext == 0 and rep > 0:
            notes.append(
                f"{key}: run reported {rep:g} call(s) with no external record "
                "(no gateway log configured, or logging is not wired up)"
            )

    # Self-reported token counts, when the agent tracked them.
    for key in ("prompt_tokens", "completion_tokens"):
        if key in reported and key in external:
            gaps[key] = float(reported[key]) - float(external[key])

    ok = all(abs(gap) <= tolerance for gap in gaps.values()) if gaps else False
    if not gaps:
        notes.append("no comparable counters: nothing was logged on either side")
    return Reconciliation(
        reported=dict(reported),
        external=external,
        gaps={k: round(v, 4) for k, v in gaps.items()},
        tolerance=tolerance,
        ok=ok,
        notes=notes,
    )


class UsageSink:
    """A callable that writes arx usage events to an external log.

    Wiring this into :func:`arx.llm.build_llm` is what makes reconciliation
    possible. It does not by itself make the log independent — only the
    placement of the log file does.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def __call__(self, event: str, payload: dict | None = None) -> None:
        record_usage(self.path, event, payload or {})


__all__ = [
    "OracleEntry",
    "Reconciliation",
    "UsageSink",
    "count_events",
    "read_usage",
    "reconcile",
    "record_usage",
]
