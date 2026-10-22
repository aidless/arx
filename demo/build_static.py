#!/usr/bin/env python3
"""Build the static arx demo payload from real audit calls."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "demo") not in sys.path:
    sys.path.insert(0, str(ROOT / "demo"))

from payload import build_payload, write_payload  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "demo" / "public" / "cases.json"))
    ap.add_argument("--generated-at", default=None)
    args = ap.parse_args()

    payload = build_payload(generated_at=args.generated_at)
    out = Path(args.out)
    write_payload(out, payload)
    summary = payload["summary"]
    verification = payload["verification"]
    print(f"wrote {out}")
    print(f"cases={summary['cases']} verdicts={summary['verdicts']}")
    print(f"verification={verification['status']}")
    if verification["status"] == "MISMATCH":
        print("mismatches=" + ", ".join(verification["mismatches"]))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
