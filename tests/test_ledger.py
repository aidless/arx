"""Tests for the Merkle ledger.

The properties that matter, in order:

1. Integrity — the whole record set has a root, and any edit, deletion,
   reorder, or truncation changes it.
2. Inclusion — a third party holding only the root can check a single
   record without reading the file.
3. Determinism — the same records give the same root, so two runs can be
   compared by root alone.
4. Domain separation — a leaf cannot be passed off as an internal node.
5. Signatures are about *ownership*, not consistency, and the two are not
   confused.

The v0.2 transcript recorded a real defect here: the chain covered only
*earlier* records, so tampering with the **last** record was undetectable.
`test_tampering_with_the_last_record_is_detected` is the regression witness.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from arx.experiments.ledger import (
    Ledger,
    SignedLedger,
    verify_ledger_file,
)

HAS_CRYPTO = importlib.util.find_spec("cryptography") is not None


def _ledger(tmp_path: Path, name: str = "ledger.jsonl") -> Ledger:
    return Ledger(tmp_path / name)


# ---------------------------------------------------------------------------
# basics
# ---------------------------------------------------------------------------

def test_a_fresh_ledger_is_empty_and_valid(tmp_path):
    led = _ledger(tmp_path)
    assert len(led) == 0
    assert led.records() == []
    assert led.verify_chain()
    assert led.root()  # the empty root is still a root


def test_append_returns_a_record_with_a_sequence_number(tmp_path):
    led = _ledger(tmp_path)
    first = led.append("hypothesis", {"text": "H1"})
    second = led.append("gate", {"verdict": "READABLE"})
    assert first.seq == 0 and second.seq == 1
    assert len(led) == 2
    assert led.records()[0].payload == {"text": "H1"}


def test_the_ledger_survives_a_reopen(tmp_path):
    path = tmp_path / "ledger.jsonl"
    Ledger(path).append("hypothesis", {"text": "H1"})
    reopened = Ledger(path)
    assert len(reopened) == 1
    assert reopened.records()[0].payload["text"] == "H1"
    assert reopened.verify_chain()


def test_records_support_limit_and_offset(tmp_path):
    led = _ledger(tmp_path)
    for i in range(5):
        led.append("step", {"i": i})
    assert [r.payload["i"] for r in led.records(limit=2)] == [0, 1]
    assert [r.payload["i"] for r in led.records(offset=3)] == [3, 4]
    assert [r.payload["i"] for r in led.records(limit=1, offset=2)] == [2]
    assert [r.payload["i"] for r in led.iter_records()] == [0, 1, 2, 3, 4]


def test_as_dicts_is_json_safe(tmp_path):
    led = _ledger(tmp_path)
    led.append("step", {"i": 0})
    json.dumps(led.as_dicts())


# ---------------------------------------------------------------------------
# integrity: tamper evidence
# ---------------------------------------------------------------------------

def test_tampering_with_the_last_record_is_detected(tmp_path):
    """The defect the transcript's v0.2 review named.

    A chain that only binds each record to its predecessor leaves the tail
    free: edit the last record and nothing downstream disagrees. Committing
    each record's own hash into the chain closes that.
    """
    path = tmp_path / "ledger.jsonl"
    led = Ledger(path)
    led.append("hypothesis", {"text": "original"})
    led.append("gate", {"verdict": "READABLE"})
    assert led.verify_chain()

    lines = path.read_text().splitlines()
    last = json.loads(lines[-1])
    last["payload"]["verdict"] = "UNDECIDABLE"
    lines[-1] = json.dumps(last, sort_keys=True)
    path.write_text("\n".join(lines) + "\n")

    assert not Ledger(path).verify_chain()


def test_tampering_with_an_early_record_is_detected(tmp_path):
    path = tmp_path / "ledger.jsonl"
    led = Ledger(path)
    led.append("hypothesis", {"text": "original"})
    led.append("gate", {"verdict": "READABLE"})

    lines = path.read_text().splitlines()
    first = json.loads(lines[0])
    first["payload"]["text"] = "TAMPERED"
    lines[0] = json.dumps(first, sort_keys=True)
    path.write_text("\n".join(lines) + "\n")

    assert not Ledger(path).verify_chain()


def test_deleting_the_last_record_is_not_detectable_from_the_file_alone(tmp_path):
    """An honest limitation, pinned so nobody 'fixes' it with a false claim.

    Remove the tail and the remaining records still form a valid chain: each
    one commits to its predecessor, and nothing commits to the future. No
    in-file scheme can close this, because whatever record declared 'there
    are N records' could itself be removed. The mitigation is an external
    anchor — see `test_a_published_anchor_makes_truncation_detectable`.
    """
    path = tmp_path / "ledger.jsonl"
    led = Ledger(path)
    for i in range(3):
        led.append("step", {"i": i})
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")
    # The chain is internally consistent; only the count changed.
    truncated = Ledger(path)
    assert truncated.verify_chain()
    assert len(truncated) == 2


def test_a_published_anchor_makes_truncation_detectable(tmp_path):
    """The mitigation for the limitation above."""
    path = tmp_path / "ledger.jsonl"
    external = tmp_path / "published_anchor.json"

    led = Ledger(path)
    for i in range(3):
        led.append("step", {"i": i})
    led.anchor(path=external, note="published with the paper")

    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")

    verdict = Ledger(path).verify_anchor(external)
    assert verdict["ok"] is False
    assert any("removed" in r for r in verdict["reasons"])


def test_a_published_anchor_accepts_an_untouched_ledger(tmp_path):
    external = tmp_path / "anchor.json"
    led = Ledger(tmp_path / "ledger.jsonl")
    led.append("step", {"i": 0})
    led.append("step", {"i": 1})
    led.anchor(path=external)
    verdict = Ledger(tmp_path / "ledger.jsonl").verify_anchor(external)
    assert verdict["ok"] is True


def test_growth_is_reported_until_the_anchor_is_refreshed(tmp_path):
    """Appending is the normal case, but it *does* move the root.

    Reporting it as a mismatch is the point: the anchor pins a specific
    state, and a reader who checks against an old pin should be told the
    ledger has moved rather than have it waved through. Re-anchoring is the
    deliberate act that supersedes the old pin.
    """
    external = tmp_path / "anchor.json"
    path = tmp_path / "ledger.jsonl"
    led = Ledger(path)
    led.append("step", {"i": 0})
    led.anchor(path=external)
    led.append("step", {"i": 1})
    assert Ledger(path).verify_anchor(external)["ok"] is False
    Ledger(path).anchor(path=external)
    assert Ledger(path).verify_anchor(external)["ok"] is True


def test_a_missing_anchor_is_reported_not_assumed_ok(tmp_path):
    verdict = Ledger(tmp_path / "ledger.jsonl").verify_anchor(tmp_path / "nope.json")
    assert verdict["ok"] is False
    assert any("no anchor" in r for r in verdict["reasons"])


def test_an_unreadable_anchor_is_reported(tmp_path):
    external = tmp_path / "anchor.json"
    external.write_text("{not json")
    verdict = Ledger(tmp_path / "ledger.jsonl").verify_anchor(external)
    assert verdict["ok"] is False
    assert any("not valid JSON" in r for r in verdict["reasons"])


def test_deleting_a_middle_record_is_detected(tmp_path):
    path = tmp_path / "ledger.jsonl"
    led = Ledger(path)
    for i in range(4):
        led.append("step", {"i": i})
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:2] + lines[3:]) + "\n")
    assert not Ledger(path).verify_chain()


def test_reordering_records_is_detected(tmp_path):
    path = tmp_path / "ledger.jsonl"
    led = Ledger(path)
    for i in range(3):
        led.append("step", {"i": i})
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[1], lines[0], lines[2]]) + "\n")
    assert not Ledger(path).verify_chain()


def test_a_truncated_final_line_is_detected(tmp_path):
    """A crash mid-write leaves half a line. That must not read as valid."""
    path = tmp_path / "ledger.jsonl"
    led = Ledger(path)
    led.append("step", {"i": 0})
    with path.open("a") as fh:
        fh.write('{"seq": 1, "kind": "step", "payl')
    assert not Ledger(path).verify_chain()


def test_appending_after_tampering_does_not_repair_the_chain(tmp_path):
    """A common failure mode: the writer happily appends to a broken file."""
    path = tmp_path / "ledger.jsonl"
    led = Ledger(path)
    led.append("step", {"i": 0})
    lines = path.read_text().splitlines()
    record = json.loads(lines[0])
    record["payload"]["i"] = 99
    path.write_text(json.dumps(record, sort_keys=True) + "\n")

    appended = Ledger(path)
    appended.append("step", {"i": 1})
    assert not appended.verify_chain()


# ---------------------------------------------------------------------------
# Merkle root and inclusion proofs
# ---------------------------------------------------------------------------

def test_same_records_give_the_same_root(tmp_path):
    """Determinism: the root is a pure function of the stored record bytes.

    Note that a root is *not* comparable across two independent runs — every
    record carries the wall-clock time at which it was written — so the
    comparison here is between two copies of the same ledger, which is what
    "did this file change?" actually requires.
    """
    path = tmp_path / "original.jsonl"
    led = Ledger(path)
    led.append("hypothesis", {"text": "H1"})
    led.append("gate", {"verdict": "READABLE"})
    copy_path = tmp_path / "copy.jsonl"
    copy_path.write_text(path.read_text())
    assert Ledger(copy_path).root() == led.root()
    assert Ledger(copy_path).verify_chain()


def test_the_root_changes_when_a_record_is_added(tmp_path):
    led = _ledger(tmp_path)
    before = led.root()
    led.append("step", {"i": 0})
    assert led.root() != before


def test_inclusion_proof_verifies_for_every_index(tmp_path):
    """Odd counts exercise the promote-unchanged branch."""
    led = _ledger(tmp_path)
    for i in range(7):
        led.append("step", {"i": i})
    records = led.records()
    for index in range(7):
        proof = led.inclusion_proof(index)
        assert proof.index == index
        assert proof.root == led.root()
        assert proof.verify(records[index].to_dict())
        # A caller who hands the proof a LedgerRecord instead of a dict gets
        # the same answer as one who calls it correctly. Returning False for
        # the wrong type reads as "the proof is bad" rather than "I called it
        # wrong", which is the false negative that makes people distrust a
        # verification API entirely.
        assert proof.verify(records[index])
        assert proof.verify(records[index].to_dict())
        assert led.verify_inclusion(index, records[index])
        assert led.verify_inclusion(index, records[index].to_dict())


def test_inclusion_proof_rejects_a_different_record(tmp_path):
    led = _ledger(tmp_path)
    led.append("step", {"i": 0})
    led.append("step", {"i": 1})
    proof = led.inclusion_proof(0)
    assert not proof.verify({"seq": 0, "kind": "step", "payload": {"i": 99},
                             "ts": 0.0, "prev_hash": "", "hash": ""})


def test_inclusion_proof_bounds_are_checked(tmp_path):
    led = _ledger(tmp_path)
    led.append("step", {"i": 0})
    with pytest.raises(IndexError):
        led.inclusion_proof(5)
    with pytest.raises(IndexError):
        led.inclusion_proof(-1)


def test_inclusion_proof_on_an_empty_ledger_raises(tmp_path):
    with pytest.raises(ValueError):
        _ledger(tmp_path).inclusion_proof(0)


def test_proof_text_is_human_readable(tmp_path):
    led = _ledger(tmp_path)
    led.append("step", {"i": 0})
    assert "InclusionProof" in led.inclusion_proof(0).to_text()


def test_leaf_and_node_hashes_are_domain_separated(tmp_path):
    """A crafted leaf must not be usable as an internal node.

    Without the prefix bytes a caller could submit a leaf whose content equals
    the concatenation an internal node hashes, and forge a proof.
    """
    from arx.experiments.ledger import _leaf_hash, _node_hash

    payload = {"seq": 0, "ts": 0.0, "kind": "step", "payload": {},
               "prev_hash": "", "hash": ""}
    leaf = _leaf_hash(payload)
    node = _node_hash(leaf, leaf)
    assert leaf != node
    assert len(leaf) == 64 and len(node) == 64


# ---------------------------------------------------------------------------
# tail hash and file-level helper
# ---------------------------------------------------------------------------

def test_tail_hash_tracks_the_last_record(tmp_path):
    led = _ledger(tmp_path)
    first_tail = led.tail_hash()
    led.append("step", {"i": 0})
    second_tail = led.tail_hash()
    assert first_tail != second_tail
    assert second_tail == led.records()[-1].hash


def test_verify_ledger_file_reports_a_verdict(tmp_path):
    path = tmp_path / "ledger.jsonl"
    led = Ledger(path)
    led.append("step", {"i": 0})
    result = verify_ledger_file(path)
    assert result["chain_valid"] is True
    assert result["n_records"] == 1
    assert result["root"] == led.root()


def test_verify_ledger_file_can_prove_one_record(tmp_path):
    path = tmp_path / "ledger.jsonl"
    led = Ledger(path)
    led.append("step", {"i": 0})
    led.append("step", {"i": 1})
    record = led.records()[1].to_dict()
    result = verify_ledger_file(path, index=1, record=record)
    assert result["inclusion_valid"] is True
    forged = dict(record, payload={"i": 99})
    assert verify_ledger_file(path, index=1, record=forged)["inclusion_valid"] is False


# ---------------------------------------------------------------------------
# SignedLedger — attestation is about ownership, not consistency
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_CRYPTO, reason="provenance extra not installed")
def test_signed_ledger_attestation_verifies(tmp_path):
    led = SignedLedger(tmp_path / "signed.jsonl")
    led.append("hypothesis", {"text": "H1"})
    assert led.verify_signatures()
    assert len(led.public_key_hex) == 64


@pytest.mark.skipif(not HAS_CRYPTO, reason="provenance extra not installed")
def test_signed_ledger_detects_an_unsigned_edit(tmp_path):
    path = tmp_path / "signed.jsonl"
    led = SignedLedger(path)
    led.append("hypothesis", {"text": "H1"})
    led.append("gate", {"verdict": "READABLE"})

    lines = path.read_text().splitlines()
    record = json.loads(lines[0])
    record["payload"]["text"] = "TAMPERED"
    lines[0] = json.dumps(record, sort_keys=True)
    path.write_text("\n".join(lines) + "\n")

    reopened = SignedLedger(path, sk_bytes=led._sk.private_bytes_raw())
    assert not reopened.verify_signatures()


@pytest.mark.skipif(not HAS_CRYPTO, reason="provenance extra not installed")
def test_signed_ledger_rejects_a_key_from_another_writer(tmp_path):
    """The signature is a claim about *who* wrote the record."""
    path = tmp_path / "signed.jsonl"
    SignedLedger(path).append("hypothesis", {"text": "H1"})
    other = SignedLedger(path)  # a fresh key
    assert not other.verify_signatures()


def test_unsigned_ledger_exposes_no_signature(tmp_path):
    """`_sign_record` returning None is how the base class says 'unsigned',
    and it must not be mistaken for a failed signature."""
    led = _ledger(tmp_path)
    record = led.append("step", {"i": 0})
    assert record.signature is None
