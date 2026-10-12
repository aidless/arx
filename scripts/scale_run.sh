#!/usr/bin/env bash
# Scale run: repeat the full loop across topics and tally how the gate votes.
#
# The point is not to produce a benchmark nobody asked for. It is to answer one
# question with data: when the hypotheses and the experiment code are both
# produced by a local model, what fraction of the resulting claims does the gate
# refuse, and *why*? A gate that refuses everything is a gate that is not doing
# anything useful either.
#
# Requires: Ollama (or any OpenAI-compatible endpoint) at ARX_LLM_BASE_URL.
#
# Usage: bash scripts/scale_run.sh [runs]
set -uo pipefail

RUNS="${1:-5}"
MODEL="${ARX_LLM_MODEL:-llama3.2:3b}"
BASE="${ARX_LLM_BASE_URL:-http://localhost:11434/v1}"
OUT="${ARX_SCALE_OUT:-/tmp/arx_scale}"

mkdir -p "$OUT"

# The console script may not be on PATH (pip installs to ~/.local/bin on this
# machine). Resolve arx once, up front, rather than letting each run fail with
# a bare "command not found" that the tally below would misreport as a run that
# produced no ledger.
if command -v arx >/dev/null 2>&1; then
    ARX_BIN="arx"
elif [ -x "$HOME/.local/bin/arx" ]; then
    ARX_BIN="$HOME/.local/bin/arx"
elif python3 -c "import arx" 2>/dev/null; then
    ARX_BIN="python3 -m arx.cli"
else
    echo "arx not found. Run: python3 -m pip install -e ." >&2
    exit 1
fi
echo "using: $ARX_BIN"

TOPICS=(
  "reward hacking in reinforcement learning"
  "noise floors in llm-as-judge evaluation"
  "calibration of large language model confidence"
  "spurious correlations in emergent ability metrics"
  "reproducibility of reported benchmark margins"
)

printf '%s\n' "${TOPICS[@]}" > "$OUT/topics.txt"

for i in $(seq 1 "$RUNS"); do
  idx=$(( (i - 1) % ${#TOPICS[@]} ))
  topic="${TOPICS[$idx]}"
  ws="$OUT/run_$i"
  rm -rf "$ws"
  echo "=== run $i/$RUNS : $topic"
  ARX_LLM_BASE_URL="$BASE" ARX_LLM_MODEL="$MODEL" timeout 1800 $ARX_BIN run "$topic" \
      --provider llm \
      --workspace "$ws" \
      --out "$ws/report.md" \
      --theory-root "scale-run" \
      --falsified-by "a claim reaches the report without clearing the noise-floor check" \
      2>&1 | grep -E "papers reviewed|verified|unresolved|untested|steps|chain valid|merkle root|halt" | sed 's/^/    /'
done

echo
echo "=== tally ==="
PYTHONPATH=src python3 - "$OUT" "$RUNS" <<'PY'
import json
import sys
from pathlib import Path

from arx.experiments.ledger import Ledger
from arx.audit.verdict import Verdict

out = Path(sys.argv[1])
runs = int(sys.argv[2])

verdicts: dict[str, int] = {}
reasons: dict[str, int] = {}
hyp_tested = 0
papers_total = 0
exp_crashed = 0
runs_ok = 0
findings = 0

for i in range(1, runs + 1):
    led = out / f"run_{i}" / "ledger.jsonl"
    if not led.exists():
        print(f"  run {i}: NO LEDGER (run failed to start)")
        continue
    source = Ledger(led)
    if not source.verify_chain():
        print(f"  run {i}: CHAIN INVALID")
        continue
    runs_ok += 1
    for rec in source.iter_records():
        p = rec.payload
        if rec.kind == "literature":
            papers_total += p.get("n_papers", 0) or 0
        elif rec.kind == "experiment":
            if p.get("status") == "no_entrypoint":
                continue
            hyp_tested += 1
            if p.get("returncode") not in (0, None):
                exp_crashed += 1
        elif rec.kind == "gate":
            v = str(p.get("verdict", ""))
            verdicts[v] = verdicts.get(v, 0) + 1
            if v in ("READABLE", "STOP_EARLY"):
                findings += 1
            # Reasons live nested: payload['report']['reasons'] is a list, and
            # the flat payload['reason'] only carries the pre-registration
            # failures. Reading the flat key alone silently drops the reason for
            # every UNREADABLE verdict, which is precisely the one a reader
            # asks about.
            reasons_list = (p.get("report") or {}).get("reasons") or []
            if not reasons_list and p.get("reason"):
                reasons_list = [str(p["reason"])]
            for r in reasons_list:
                bucket = str(r).split(":")[0].split(".")[0][:48].strip()
                reasons[bucket] = reasons.get(bucket, 0) + 1

total = sum(verdicts.values())
print(f"  runs completed        : {runs_ok}/{runs}")
print(f"  literature retrieved  : {papers_total}")
print(f"  experiments executed  : {hyp_tested}")
print(f"  experiment crashes    : {exp_crashed}")
print(f"  gate decisions        : {total}")
for v, n in sorted(verdicts.items(), key=lambda kv: -kv[1]):
    print(f"    {v or '(empty)':16} {n:>4}   {n / total * 100:5.1f}%" if total else "")
print(f"  findings written      : {findings}")
print()
print("  refusal reasons:")
for r, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
    print(f"    {n:>4}  {r}")
PY
