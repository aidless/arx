"""Sandboxed experiment execution.

An experiment is a standalone script that prints one JSON object on its last
stdout line. arx runs it as a subprocess — never in-process — with a wall-clock
timeout, capturing stdout/stderr verbatim into the ledger.

What this module deliberately does **not** do: claim to be a security sandbox.
A subprocess with a timeout contains accidents (infinite loops, runaway memory
in the common case, crashes) and gives you a faithful record of what the script
printed. It does not contain hostile code. If you are about to execute
model-generated code, run it inside a container — see ``docs/architecture.md``.

The output contract is one line of JSON:

    {
      "votes":       [[0, 1, 0, ...], ...],   # generators x items, -1 = abstain
      "scores":      {"system-a": 0.91, ...}, # leaderboard
      "item_ids":    [0, 1, 2, ...],          # optional
      "judge_ids":   ["g1", "g2", "g3"],      # optional
      "contestant_ids": ["system-a", ...],    # optional
      "n_looks":     3,                       # optional, for the sequential gate
      "se":          0.021                    # optional, standard error
    }

Anything the contract does not cover is the experiment's business.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REQUIRED_KEYS = ("votes", "scores")


@dataclass
class RunResult:
    entrypoint: str
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float
    script_sha256: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def payload(self) -> dict | None:
        """The last parseable JSON object in stdout, or None.

        Scans backwards so that a script may print progress lines before its
        final result without breaking the contract.
        """
        for line in reversed(self.stdout.strip().splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
        return None

    def validate(self) -> tuple[bool, str]:
        obj = self.payload()
        if obj is None:
            return False, "no JSON object found on stdout"
        missing = [k for k in REQUIRED_KEYS if k not in obj]
        if missing:
            return False, f"output missing required key(s): {', '.join(missing)}"
        if not isinstance(obj["scores"], dict) or not obj["scores"]:
            return False, "'scores' must be a non-empty object"
        if not isinstance(obj["votes"], list) or not obj["votes"]:
            return False, "'votes' must be a non-empty list of rows"
        return True, "ok"

    def to_dict(self) -> dict:
        return {
            "entrypoint": self.entrypoint,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "duration_s": round(self.duration_s, 3),
            "script_sha256": self.script_sha256,
            "stdout_tail": self.stdout[-4000:],
            "stderr_tail": self.stderr[-2000:],
        }


def run_experiment(
    entrypoint: str | Path,
    *,
    timeout_s: float = 600.0,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    python: str | None = None,
) -> RunResult:
    """Execute ``python3 <entrypoint>`` and capture the result.

    Uses the interpreter running arx unless ``python`` is given, so the
    downstream experiment sees the same environment the gate does.
    """
    import hashlib
    import time

    ep = Path(entrypoint)
    if not ep.exists():
        raise FileNotFoundError(f"experiment entrypoint not found: {ep}")

    script_sha = hashlib.sha256(ep.read_bytes()).hexdigest()
    interp = python or sys.executable or "python3"
    workdir = str(cwd or ep.parent)

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            [interp, str(ep.resolve())],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
            check=False,
        )
        duration = time.monotonic() - t0
        return RunResult(
            entrypoint=str(ep),
            returncode=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            timed_out=False,
            duration_s=duration,
            script_sha256=script_sha,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - t0
        return RunResult(
            entrypoint=str(ep),
            returncode=-1,
            stdout=_as_text(exc.stdout),
            stderr=_as_text(exc.stderr),
            timed_out=True,
            duration_s=duration,
            script_sha256=script_sha,
        )


def _as_text(blob: object) -> str:
    if blob is None:
        return ""
    if isinstance(blob, bytes):
        return blob.decode("utf-8", errors="replace")
    return str(blob)


__all__ = ["REQUIRED_KEYS", "RunResult", "run_experiment"]
