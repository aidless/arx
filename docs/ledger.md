# The ledger

Every record in a run is append-only, self-hashing and hash-chained. This
document says what that buys, and — more usefully — what it does not.

## What it proves

**Consistency.** `verify_chain()` recomputes every hash and checks linkage.
Rewriting, deleting, or reordering any record breaks it, including the *last*
one: the record's own hash is inside the committed payload, so a chain that
only commits to predecessors cannot see a tampered tail. This package does
commit to it.

**Selective disclosure.** `root()` is a Merkle root over the leaves, and
`inclusion_proof(i).verify(record)` is an `O(log n)` proof that record `i` is
what the root says it is. A reader can check one claim without reading the
whole file — and, more to the point, without being given the whole file.

**Attestation** (optional, `[provenance]` extra). `SignedLedger.attest()` signs
the current root with an ed25519 key. The hash chain proves *consistency*; a
signature proves *authorship*. Shipping an unsigned chain and calling it
attested is the mistake the split exists to prevent.

## What it does not prove

**Tail truncation.** Delete the last record and the remaining records still
form a perfectly valid chain. There is no record after the deleted one to
complain, and nothing inside the file can say what the chain *should* end at.

This is not a bug and no amount of hashing fixes it. It is what a hash chain
without an outside reference can do — the same way a receipt is evidence of a
purchase only if someone other than the shop holds it.

`anchor()` exists for exactly this: write the root (and tail hash, and record
count) to a sidecar file that you keep out of reach of the thing being audited.
`verify_anchor()` then catches a tail that was trimmed after the fact. Note
the asymmetry: an anchor makes *later* edits detectable and says nothing about
edits *before* the anchor was published.

```
arx anchor      ledger.jsonl        # writes ledger.jsonl.anchor.json
arx verify-ledger ledger.jsonl      # checks the chain, the record count, the root
```

## The honest limit of the whole design

A ledger is evidence, not truth. It proves that a run happened the way its
records say, and that the records have not been edited since. It cannot prove
the records describe a real experiment — that is what the gate is for, and the
gate in turn cannot prove the *scores* it was handed. Each layer closes one
hole and opens a smaller one further out.

The useful question about any record is therefore not "is it tampered with" but
"who else holds a copy". For the reproducibility case, that is
[`examples/02_audit_a_leaderboard.py`](../examples/02_audit_a_leaderboard.py):
the claim a reader checks is one number from a synthetic board, and the
arithmetic that produced it is in the example source.

## Reading a ledger back

```python
from arx.experiments.ledger import Ledger

led = Ledger("workspace/ledger.jsonl")
led.verify_chain()                       # bool
led.root()                               # merkle root
for rec in led.iter_records():           # streaming, works on huge files
    print(rec.seq, rec.kind)

proof = led.inclusion_proof(7)
proof.verify(led.records()[7])           # O(log n); dict or LedgerRecord
```

## Tamper-evidence in the runner

Each `run` record carries `script_sha256` and `output_sha256` — the hash of the
file that executed and the hash of the bytes it printed. The first is measured
by the runner, not reported by the script, which is what makes it
pre-registration-grade: the pin is decided before the run and read from disk
afterwards, so a script swapping itself out between the two is a detectable
event rather than an undetected one.
