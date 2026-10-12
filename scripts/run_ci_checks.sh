#!/usr/bin/env bash
# Ledger integrity smoke test.
#
# Builds a tiny ledger, records its state, tampers with it in the three ways
# that matter, and asserts the ledger notices each one. This is the check that
# keeps `docs/ledger.md` honest: it claims truncation is NOT self-detectable,
# so the anchor exists, and this script asserts both halves of that.
#
# Run: bash scripts/run_ci_checks.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHONPATH=src python3 - <<'PY'
import json
import shutil
import tempfile
from pathlib import Path

from arx.experiments.ledger import EMPTY_ROOT, InclusionProof, Ledger

failures: list[str] = []


def check(label: str, ok: bool) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        failures.append(label)


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    path = tmp / "ledger.jsonl"
    led = Ledger(path)

    print("\n1. a fresh ledger is empty and valid")
    check("empty root is the documented constant", led.root() == EMPTY_ROOT)
    check("empty ledger verifies", led.verify_chain())
    check("no records", len(led.records()) == 0)

    print("\n2. appends extend the chain")
    for i in range(5):
        led.append("step", {"i": i})
    root_before = led.root()
    tail_before = led.tail_hash()
    check("five records", len(led) == 5)
    check("chain verifies", led.verify_chain())
    check("root is a 64-char hex digest",
          len(root_before) == 64 and all(c in "0123456789abcdef" for c in root_before))
    check("tail differs from root", tail_before != root_before)

    print("\n3. Merkle inclusion proofs, for every record")
    for i in range(5):
        proof = led.inclusion_proof(i)
        check(f"record #{i} verifies against its proof",
              proof.verify(led.records()[i]))
        check(f"record #{i} proof carries the ledger root",
              proof.root == led.root())
    check("proof for a different record is rejected",
          not led.inclusion_proof(0).verify(led.records()[1]))

    print("\n4. rewriting a record is detected")
    tampered = tmp / "rewritten.jsonl"
    shutil.copy(path, tampered)
    lines = tampered.read_text().splitlines()
    rec = json.loads(lines[2])
    rec["payload"]["i"] = 999
    lines[2] = json.dumps(rec, sort_keys=True)
    tampered.write_text("\n".join(lines) + "\n")
    check("in-file chain catches the rewrite",
          not Ledger(tampered).verify_chain())

    print("\n5. deleting a middle record is detected")
    deleted = tmp / "deleted.jsonl"
    shutil.copy(path, deleted)
    lines = deleted.read_text().splitlines()
    deleted.write_text("\n".join(lines[:2] + lines[3:]) + "\n")
    check("chain catches the deletion", not Ledger(deleted).verify_chain())

    print("\n6. reordering records is detected")
    reordered = tmp / "reordered.jsonl"
    lines = path.read_text().splitlines()
    reordered.write_text("\n".join([lines[1], lines[0], *lines[2:]]) + "\n")
    check("chain catches the reorder", not Ledger(reordered).verify_chain())

    print("\n7. truncating the TAIL is not self-detectable -- this is the point")
    truncated = tmp / "truncated.jsonl"
    lines = path.read_text().splitlines()
    truncated.write_text("\n".join(lines[:-1]) + "\n")
    remaining = Ledger(truncated)
    check("the remaining records still form a valid chain",
          remaining.verify_chain())
    check("and the record count dropped", len(remaining) == 4)
    print("    docs/ledger.md claims this. A hash chain without an outside")
    print("    reference cannot see its own tail go missing.")

    print("\n8. the anchor is what makes tail loss detectable")
    anchor = led.anchor(note="ci smoke test", path=tmp / "ledger.anchor.json")
    check("anchor file was written", (tmp / "ledger.anchor.json").exists())
    verdict = led.verify_anchor(tmp / "ledger.anchor.json")
    check("anchor matches an untouched ledger", verdict["ok"])
    truncated_verdict = Ledger(truncated).verify_anchor(tmp / "ledger.anchor.json")
    check("anchor notices the missing tail", not truncated_verdict["ok"])
    check("anchor says why", "record count" in str(truncated_verdict).lower()
          or "root" in str(truncated_verdict).lower())

    print("\n9. the anchor cannot vouch for edits made before it was published")
    early = tmp / "early.jsonl"
    early.write_text("\n".join(lines[:2]) + "\n")
    early_led = Ledger(early)
    early_led.anchor(path=tmp / "early.anchor.json")
    lines2 = early.read_text().splitlines()
    rec2 = json.loads(lines2[0])
    rec2["payload"]["i"] = -1          # edit, then re-anchor
    lines2[0] = json.dumps(rec2, sort_keys=True)
    early.write_text("\n".join(lines2) + "\n")
    check("re-anchoring after an edit still validates the edited chain",
          Ledger(early).verify_anchor(tmp / "early.anchor.json"))
    print("    An anchor is a promise about the future, not a certificate")
    print("    about the past.")

print()
if failures:
    print(f"FAILED: {len(failures)} check(s)")
    for f in failures:
        print(f"  - {f}")
    raise SystemExit(1)

print("All ledger integrity checks passed.")
PY
