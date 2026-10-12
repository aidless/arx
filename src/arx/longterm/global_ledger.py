"""A cross-run, append-only ledger of hypotheses and how they died.

The per-run ledger answers "what happened in this run". This one answers the
question that actually costs researchers time: *have I tried this before?*

Single-run protection has a blind spot that is easy to miss and expensive to
keep missing — the same line of enquiry returns under a new name. In one run it
is called ``reranker-v2``; three months later, in a different project, it is
``context-aware retrieval``. Each instance is treated as a fresh independent
test and spends a fresh alpha. The protection that exists per run never sees it,
because it is not per run.

So this ledger:

* lives outside any project (default ``~/.arx/global_ledger.db``),
* stores the hypothesis text **and its embedding**, so retrieval is by meaning
  rather than by string match,
* is append-only and hash-chained, so a prior negative result cannot be quietly
  edited away,
* records the embedder that produced each vector, so a later run knows whether
  the similarity it computes is comparable.

One honest limit, stated because it is the difference between a useful tool and
a misleading one: matching *semantically* is not the same as matching
*statistically*. An LLM can generate a thousand hypotheses that are worded
differently and are still one test. This ledger narrows the gap; it does not
close it, and nothing here should be reported as if it did.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .embedding import Embedder, build_embedder, cosine

SCHEMA = """
CREATE TABLE IF NOT EXISTS hypotheses (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL    NOT NULL,
    project       TEXT    NOT NULL DEFAULT '',
    run_id        TEXT    NOT NULL DEFAULT '',
    hypothesis    TEXT    NOT NULL,
    verdict       TEXT    NOT NULL DEFAULT '',
    reason        TEXT    NOT NULL DEFAULT '',
    embedder      TEXT    NOT NULL DEFAULT '',
    embedding     TEXT    NOT NULL DEFAULT '[]',
    meta          TEXT    NOT NULL DEFAULT '{}',
    prev_hash     TEXT    NOT NULL DEFAULT '',
    hash          TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_hyps_project ON hypotheses(project);
CREATE INDEX IF NOT EXISTS idx_hyps_verdict ON hypotheses(verdict);
"""


@dataclass
class PriorHypothesis:
    id: int
    ts: float
    project: str
    run_id: str
    hypothesis: str
    verdict: str
    reason: str
    similarity: float
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "ts": self.ts,
            "project": self.project,
            "run_id": self.run_id,
            "hypothesis": self.hypothesis,
            "verdict": self.verdict,
            "reason": self.reason,
            "similarity": round(self.similarity, 4),
            "meta": dict(self.meta),
        }


class GlobalLedger:
    """SQLite-backed, append-only, cross-project hypothesis memory."""

    def __init__(
        self,
        path: str | Path,
        *,
        embedder: Embedder | None = None,
    ):
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or build_embedder("hash")
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self._cache: list[tuple[int, list[float]]] = []
        self._load_cache()

    # -- cache ---------------------------------------------------------------
    def _load_cache(self) -> None:
        self._cache = []
        rows = self.conn.execute(
            "SELECT id, embedding FROM hypotheses WHERE embedder = ?",
            (self.embedder.name,),
        )
        for row in rows:
            try:
                vec = json.loads(row["embedding"])
            except json.JSONDecodeError:
                continue
            if isinstance(vec, list) and vec:
                self._cache.append((int(row["id"]), [float(x) for x in vec]))

    # -- hash chain ----------------------------------------------------------
    def _tail_hash(self) -> str:
        row = self.conn.execute(
            "SELECT hash FROM hypotheses ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row["hash"] if row else ""

    @staticmethod
    def _chain_hash(prev_hash: str, payload: dict) -> str:
        import hashlib

        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(f"{prev_hash}|{blob}".encode()).hexdigest()

    # -- writing -------------------------------------------------------------
    def record(
        self,
        hypothesis: str,
        *,
        verdict: str = "",
        reason: str = "",
        project: str = "",
        run_id: str = "",
        meta: dict | None = None,
    ) -> int:
        """Append one hypothesis and its outcome. Returns the row id."""
        text = (hypothesis or "").strip()
        if not text:
            raise ValueError("hypothesis text must be non-empty")
        vec = self.embedder.embed(text)
        ts = time.time()
        prev = self._tail_hash()
        payload = {
            "ts": round(ts, 6),
            "project": project,
            "run_id": run_id,
            "hypothesis": text,
            "verdict": verdict,
            "reason": reason,
            "meta": meta or {},
            "prev_hash": prev,
        }
        chain = self._chain_hash(prev, payload)
        cur = self.conn.execute(
            "INSERT INTO hypotheses (ts, project, run_id, hypothesis, verdict, reason,"
            " embedder, embedding, meta, prev_hash, hash)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ts, project, run_id, text, verdict, reason, self.embedder.name,
                json.dumps([round(v, 8) for v in vec]), json.dumps(meta or {}),
                prev, chain,
            ),
        )
        self.conn.commit()
        row_id = int(cur.lastrowid)
        self._cache.append((row_id, vec))
        return row_id

    # -- reading -------------------------------------------------------------
    def _row(self, row_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM hypotheses WHERE id = ?", (row_id,)
        ).fetchone()

    def _to_prior(self, row: sqlite3.Row, similarity: float) -> PriorHypothesis:
        try:
            meta = json.loads(row["meta"] or "{}")
        except json.JSONDecodeError:
            meta = {}
        return PriorHypothesis(
            id=int(row["id"]),
            ts=float(row["ts"]),
            project=str(row["project"]),
            run_id=str(row["run_id"]),
            hypothesis=str(row["hypothesis"]),
            verdict=str(row["verdict"]),
            reason=str(row["reason"]),
            similarity=float(similarity),
            meta=meta if isinstance(meta, dict) else {},
        )

    def all_rows(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM hypotheses ORDER BY id"))

    def similar(
        self,
        hypothesis: str,
        *,
        threshold: float = 0.80,
        top_k: int = 10,
        exclude_id: int | None = None,
        same_embedder_only: bool = True,
    ) -> list[PriorHypothesis]:
        """Prior hypotheses at or above ``threshold`` cosine similarity.

        Only rows embedded with the *same* embedder are compared by default:
        similarities across different embedding spaces are not comparable, and
        silently mixing them would produce confident nonsense.
        """
        query = self.embedder.embed(hypothesis)
        rows = {
            int(r["id"]): r
            for r in self.conn.execute("SELECT * FROM hypotheses")
        }
        scored: list[tuple[float, sqlite3.Row]] = []
        for row_id, vec in self._cache:
            if exclude_id is not None and row_id == exclude_id:
                continue
            row = rows.get(row_id)
            if row is None:
                continue
            if same_embedder_only and str(row["embedder"]) != self.embedder.name:
                continue
            sim = cosine(query, vec)
            if sim >= threshold:
                scored.append((sim, row))
        scored.sort(key=lambda t: (-t[0], int(t[1]["id"])))
        return [self._to_prior(row, sim) for sim, row in scored[:top_k]]

    def prior_negatives(
        self,
        hypothesis: str,
        *,
        threshold: float = 0.80,
        top_k: int = 5,
    ) -> list[PriorHypothesis]:
        """Prior occurrences of this idea that did *not* produce a finding."""
        negative = {"", "rejected", "undecidable", "unreadable", "marginal",
                    "no_data", "stalled"}
        return [
            p
            for p in self.similar(hypothesis, threshold=threshold, top_k=top_k * 3)
            if p.verdict.lower() in negative
        ][:top_k]

    def stats(self) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n, COUNT(DISTINCT project) AS projects FROM hypotheses"
        ).fetchone()
        by_verdict = {
            str(r["verdict"] or "(unset)"): int(r["n"])
            for r in self.conn.execute(
                "SELECT verdict, COUNT(*) AS n FROM hypotheses GROUP BY verdict"
            )
        }
        return {
            "n_hypotheses": int(row["n"]),
            "n_projects": int(row["projects"]),
            "by_verdict": by_verdict,
            "embedder": self.embedder.name,
            "path": str(self.path),
        }

    # -- integrity -----------------------------------------------------------
    def verify_chain(self) -> bool:
        prev = ""
        for row in self.all_rows():
            payload = {
                "ts": round(float(row["ts"]), 6),
                "project": str(row["project"]),
                "run_id": str(row["run_id"]),
                "hypothesis": str(row["hypothesis"]),
                "verdict": str(row["verdict"]),
                "reason": str(row["reason"]),
                "meta": json.loads(row["meta"] or "{}"),
                "prev_hash": prev,
            }
            if str(row["prev_hash"]) != prev:
                return False
            if self._chain_hash(prev, payload) != str(row["hash"]):
                return False
            prev = str(row["hash"])
        return True

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> GlobalLedger:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def summarise_prior(priors: Sequence[PriorHypothesis]) -> str:
    """A one-line warning suitable for putting in front of a proposer.

    The reason is included when there is one: ``rejected`` on its own tells the
    proposer to stop but not why, and 'why' is the part that prevents the same
    dead end being re-proposed under a new name.
    """
    if not priors:
        return ""
    worst = priors[0]
    detail = f"{worst.verdict or 'no verdict'}"
    if worst.reason:
        detail += f": {worst.reason}"
    scope = [f"similarity {worst.similarity:.2f}"]
    if worst.project:
        scope.append(f"project {worst.project}")
    if worst.run_id:
        scope.append(f"run {worst.run_id}")
    extra = f"; {len(priors) - 1} other prior occurrence(s)" if len(priors) > 1 else ""
    return f"closest prior work: {worst.hypothesis!r} ({detail}; {', '.join(scope)}){extra}"


__all__ = ["GlobalLedger", "PriorHypothesis", "SCHEMA", "summarise_prior"]
