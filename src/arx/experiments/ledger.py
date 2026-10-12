"""Append-only research ledger with a hash chain and Merkle inclusion proofs.

Why not just a log file: a log records what happened; a ledger lets a third
party *verify* that what happened is what is written down, without trusting the
person who wrote it.

Two guarantees, at different granularities:

* **Hash chain** (``verify_chain``). Each record commits to its own content and
  to the previous record's hash. Rewriting, deleting, or reordering any record
  breaks the chain. The record's own hash is inside the committed payload, so
  tampering with the *last* record is caught too — a chain that only commits to
  predecessors cannot see that.

* **Merkle tree** (``root``, ``inclusion_proof``). A published root plus an
  O(log n) proof lets someone verify a single claim ("record #7 says X") without
  reading the whole file.

Signing (``SignedLedger``) is optional and lives behind the ``[provenance]``
extra. The hash chain proves consistency; a signature proves *authorship*.
Conflating the two — shipping an unsigned chain and calling it attested — is the
mistake this split exists to prevent.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EMPTY_ROOT = hashlib.sha256(b"arx-ledger-empty").hexdigest()


# ---------------------------------------------------------------------------
# canonical encoding
# ---------------------------------------------------------------------------

def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _leaf_hash(record: dict) -> str:
    return _sha256(b"\x00" + _canonical(record))


def _node_hash(left: str, right: str) -> str:
    return _sha256(b"\x01" + bytes.fromhex(left) + bytes.fromhex(right))


def record_hash(seq: int, ts: float, kind: str, payload: dict, prev_hash: str) -> str:
    """The self-committed hash of one record."""
    return _sha256(
        _canonical(
            {
                "seq": int(seq),
                "ts": round(float(ts), 6),
                "kind": str(kind),
                "payload": payload,
                "prev_hash": str(prev_hash),
            }
        )
    )


# ---------------------------------------------------------------------------
# inclusion proofs
# ---------------------------------------------------------------------------

@dataclass
class InclusionProof:
    index: int
    leaf: str
    siblings: list[tuple[str, str]]  # (side in {"L","R"}, hash)
    root: str

    def verify(self, record: dict | LedgerRecord) -> bool:
        """Check a record against this proof.

        Accepts either a dict or a ``LedgerRecord``. Silently returning
        ``False`` for the wrong type is a trap: a caller who passes a
        ``LedgerRecord`` and gets ``False`` concludes "the proof is bad" when
        the truth is "I called it wrong", and that is the kind of false
        negative that makes people stop trusting a verification API.
        """
        if isinstance(record, LedgerRecord):
            record = record.to_dict()
        cur = _leaf_hash(record)
        if cur != self.leaf:
            return False
        for side, sib in self.siblings:
            cur = _node_hash(sib, cur) if side == "L" else _node_hash(cur, sib)
        return cur == self.root

    def to_text(self) -> str:
        return (
            f"InclusionProof(record #{self.index}, depth={len(self.siblings)}, "
            f"leaf={self.leaf[:16]}..., root={self.root[:16]}...)"
        )


# ---------------------------------------------------------------------------
# ledger
# ---------------------------------------------------------------------------

@dataclass
class LedgerRecord:
    seq: int
    ts: float
    kind: str
    payload: dict
    prev_hash: str
    hash: str
    signature: str | None = None

    def to_dict(self) -> dict:
        out = {
            "seq": self.seq,
            "ts": self.ts,
            "kind": self.kind,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
            "hash": self.hash,
        }
        if self.signature is not None:
            out["signature"] = self.signature
        return out


class Ledger:
    """A file-backed, append-only, hash-chained ledger."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # -- internal helpers ---------------------------------------------------
    def _raw_records(self) -> list[dict]:
        if not self.path.exists():
            return []
        out: list[dict] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # A truncated trailing line is a real failure, not something to
                # skip quietly: it means a writer died mid-append.
                out.append({"__corrupt__": line})
        return out

    def _tail(self) -> dict | None:
        recs = self._raw_records()
        return recs[-1] if recs else None

    # -- writing ------------------------------------------------------------
    def append(self, kind: str, payload: dict) -> LedgerRecord:
        tail = self._tail()
        prev_hash = tail.get("hash", "") if tail and "__corrupt__" not in tail else ""
        seq = (tail.get("seq", -1) + 1) if tail and "__corrupt__" not in tail else 0
        ts = time.time()
        h = record_hash(seq, ts, kind, payload, prev_hash)
        rec = LedgerRecord(seq=seq, ts=ts, kind=str(kind), payload=payload,
                           prev_hash=prev_hash, hash=h)
        rec.signature = self._sign_record(rec)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec.to_dict(), sort_keys=True, default=str) + "\n")
        return rec

    def _sign_record(self, rec: LedgerRecord) -> str | None:
        """Hook for subclasses. The base ledger is unsigned, and says so."""
        return None

    # -- reading ------------------------------------------------------------
    def iter_records(self, *, limit: int | None = None, offset: int = 0) -> Iterator[LedgerRecord]:
        """Stream records without loading the whole file into memory."""
        emitted = 0
        skipped = 0
        for raw in self._raw_records():
            if "__corrupt__" in raw:
                continue
            if skipped < offset:
                skipped += 1
                continue
            if limit is not None and emitted >= limit:
                break
            emitted += 1
            yield LedgerRecord(**raw)

    def records(self, *, limit: int | None = None, offset: int = 0) -> list[LedgerRecord]:
        return list(self.iter_records(limit=limit, offset=offset))

    def __len__(self) -> int:
        return sum(1 for _ in self.iter_records())

    def as_dicts(self, **kw) -> list[dict]:
        return [r.to_dict() for r in self.iter_records(**kw)]

    # -- integrity ----------------------------------------------------------
    def verify_chain(self) -> bool:
        """Recompute every hash and check linkage. False on any defect."""
        prev = ""
        for raw in self._raw_records():
            if "__corrupt__" in raw:
                return False
            try:
                rec = LedgerRecord(**raw)
            except TypeError:
                return False
            if rec.prev_hash != prev:
                return False
            expected = record_hash(rec.seq, rec.ts, rec.kind, rec.payload, rec.prev_hash)
            if expected != rec.hash:
                return False
            prev = rec.hash
        return True

    # -- merkle -------------------------------------------------------------
    @staticmethod
    def _levels(leaves: list[str]) -> list[list[tuple[str, int]]]:
        if not leaves:
            return [[(EMPTY_ROOT, 0)]]
        level = [(h, i) for i, h in enumerate(leaves)]
        levels = [level]
        while len(level) > 1:
            nxt: list[tuple[str, int]] = []
            for i in range(0, len(level), 2):
                if i + 1 < len(level):
                    nxt.append((_node_hash(level[i][0], level[i + 1][0]), i))
                else:
                    nxt.append(level[i])  # odd node promoted, not duplicated
            levels.append(nxt)
            level = nxt
        return levels

    def leaves(self) -> list[str]:
        return [_leaf_hash(r) for r in self.as_dicts()]

    def root(self) -> str:
        leaves = self.leaves()
        if not leaves:
            return EMPTY_ROOT
        return self._levels(leaves)[-1][0][0]

    def tail_hash(self) -> str:
        """Hash of the last record (cheap state pin), or the empty root."""
        recs = self.as_dicts()
        return recs[-1]["hash"] if recs else EMPTY_ROOT

    # -- external anchoring -------------------------------------------------
    def anchor_path(self) -> Path:
        return self.path.with_name(self.path.name + ".anchor.json")

    def anchor(self, *, note: str = "", path: str | Path | None = None) -> dict:
        """Publish this ledger's current state so *later* edits are detectable.

        The in-file chain cannot detect truncation of its own tail: delete the
        last record and the remaining records still form a perfectly valid
        chain. That is not a bug, it is what a hash chain without an outside
        reference can do — the same way a receipt is only evidence of a
        purchase if someone other than the shop holds it.

        So we write the root, tail hash and record count out, and
        :meth:`verify_anchor` compares. **Store the anchor somewhere the ledger
        writer does not control** (a paper appendix, a commit, a CI log, a
        co-author's inbox). An anchor sitting next to its own ledger is only
        evidence against accident, not against an interested party.
        """
        out_path = Path(path) if path is not None else self.anchor_path()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "root": self.root(),
            "tail_hash": self.tail_hash(),
            "n_records": len(self),
            "anchored_at": time.time(),
            "note": note,
        }
        out_path.write_text(json.dumps(payload, sort_keys=True) + "\n",
                            encoding="utf-8")
        return payload

    def verify_anchor(self, path: str | Path | None = None) -> dict:
        """Compare the current ledger against a previously published anchor."""
        anchor_path = Path(path) if path is not None else self.anchor_path()
        out: dict = {"anchor_path": str(anchor_path), "ok": False, "reasons": []}
        if not anchor_path.exists():
            out["reasons"].append("no anchor published; truncation is not detectable")
            return out
        try:
            anchored = json.loads(anchor_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            out["reasons"].append(f"anchor is not valid JSON: {exc}")
            return out

        current_root = self.root()
        current_tail = self.tail_hash()
        current_n = len(self)
        out["anchored"] = anchored
        out["current"] = {"root": current_root, "tail_hash": current_tail,
                          "n_records": current_n}

        if current_n < int(anchored.get("n_records", 0)):
            out["reasons"].append(
                f"records were removed: anchored {anchored.get('n_records')}, "
                f"now {current_n}"
            )
        if current_root != anchored.get("root"):
            out["reasons"].append("root differs from the anchored root")
        if current_tail != anchored.get("tail_hash"):
            out["reasons"].append("tail hash differs from the anchored tail")
        out["ok"] = not out["reasons"]
        return out

    def inclusion_proof(self, index: int) -> InclusionProof:
        leaves = self.leaves()
        if not leaves:
            raise ValueError("empty ledger has no inclusion proof")
        if not 0 <= index < len(leaves):
            raise IndexError(index)
        levels = self._levels(leaves)
        siblings: list[tuple[str, str]] = []
        pos = index
        for lvl in levels[:-1]:
            if pos % 2 == 0:
                if pos + 1 < len(lvl):
                    siblings.append(("R", lvl[pos + 1][0]))
            else:
                siblings.append(("L", lvl[pos - 1][0]))
            pos //= 2
        return InclusionProof(index=index, leaf=leaves[index], siblings=siblings,
                              root=self.root())

    def verify_inclusion(self, index: int, record: dict) -> bool:
        return self.inclusion_proof(index).verify(record)


class SignedLedger(Ledger):
    """Hash-chained ledger whose records are signed with an ed25519 key.

    Requires the ``[provenance]`` extra::

        pip install 'arx-researcher[provenance]'

    Pass ``sk_bytes`` (32 raw bytes) to keep a stable identity across runs;
    omit it and a fresh key is generated, which is fine for a single session
    but means the attestations cannot be verified by anyone else.
    """

    def __init__(self, path: str | Path, sk_bytes: bytes | None = None):
        super().__init__(path)
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "SignedLedger needs the provenance extra: "
                "pip install 'arx-researcher[provenance]'"
            ) from exc
        self._sk = (
            Ed25519PrivateKey.generate()
            if sk_bytes is None
            else Ed25519PrivateKey.from_private_bytes(sk_bytes)
        )

    def _sign_record(self, rec: LedgerRecord) -> str:
        msg = _canonical(
            {"seq": rec.seq, "ts": rec.ts, "kind": rec.kind,
             "payload": rec.payload, "prev_hash": rec.prev_hash, "hash": rec.hash}
        )
        return self._sk.sign(msg).hex()

    @property
    def public_key_hex(self) -> str:
        return self._sk.public_key().public_bytes_raw().hex()

    def verify_signatures(self) -> bool:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )

        pk = Ed25519PublicKey.from_public_bytes(
            self._sk.public_key().public_bytes_raw()
        )
        for rec in self.iter_records():
            if not rec.signature:
                return False
            msg = _canonical(
                {"seq": rec.seq, "ts": rec.ts, "kind": rec.kind,
                 "payload": rec.payload, "prev_hash": rec.prev_hash, "hash": rec.hash}
            )
            try:
                pk.verify(bytes.fromhex(rec.signature), msg)
            except Exception:
                return False
        return True


def verify_ledger_file(path: str | Path, *, index: int | None = None,
                       record: dict | None = None) -> dict:
    """Convenience for CLI use: chain validity plus an optional inclusion check."""
    led = Ledger(path)
    out = {
        "path": str(path),
        "n_records": len(led),
        "chain_valid": led.verify_chain(),
        "root": led.root(),
        "tail_hash": led.tail_hash(),
    }
    if index is not None and record is not None:
        out["inclusion_valid"] = led.verify_inclusion(index, record)
    return out


__all__ = [
    "EMPTY_ROOT",
    "InclusionProof",
    "Ledger",
    "LedgerRecord",
    "SignedLedger",
    "record_hash",
    "verify_ledger_file",
]
