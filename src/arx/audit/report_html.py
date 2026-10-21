"""Single-file HTML renderer for an :class:`AuditReport`.

The report is deliberately self-contained: no JavaScript, no external assets,
no network. Every dynamic string is escaped, so a leaderboard file cannot inject
markup into the page.
"""

from __future__ import annotations

import html
import json
from datetime import UTC, datetime

from .verdict import AuditReport

_MEANING = {
    "READABLE": (
        "The gap is larger than the measurement error. It does not mean the "
        "metric is the right one."
    ),
    "MARGINAL": (
        "The gap is positive but inside what this sample can resolve. Collect "
        "more items before claiming a difference."
    ),
    "UNREADABLE": (
        "The leader does not clear the noise floor. The ranking is not "
        "supported by this measurement."
    ),
    "UNDECIDABLE": (
        "The premises are violated; no floor is estimable. This is a finding, "
        "not a failure of the analysis."
    ),
    "STOP_EARLY": (
        "The sequential gate stopped early. See the sequential block for the "
        "boundary and the information used."
    ),
}

_COLOR = {
    "READABLE": "#0a7d32",
    "MARGINAL": "#9a6700",
    "UNREADABLE": "#b42318",
    "UNDECIDABLE": "#6b4fbb",
    "STOP_EARLY": "#0a7d32",
}


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _card(label: str, value: object, note: str = "") -> str:
    note_html = f'<div class="note">{_esc(note)}</div>' if note else ""
    return (
        '<div class="card">'
        f'<div class="label">{_esc(label)}</div>'
        f'<div class="value">{_esc(_fmt(value))}</div>'
        f"{note_html}</div>"
    )


def render_html(
    report: AuditReport,
    *,
    title: str = "arx benchmark audit",
    generated_at: str | None = None,
) -> str:
    d = report.to_dict()
    verdict = str(d["verdict"])
    color = _COLOR.get(verdict, "#333333")
    meaning = _MEANING.get(verdict, "No interpretation is registered for this verdict.")
    generated = generated_at or datetime.now(UTC).isoformat()

    reasons = d.get("reasons") or []
    reasons_html = "".join(f"<li>{_esc(r)}</li>" for r in reasons)
    if not reasons_html:
        reasons_html = "<li>No refusal reason was recorded.</li>"

    sequential = d.get("sequential")
    sequential_html = ""
    if sequential:
        rows = "".join(
            f"<tr><th>{_esc(k)}</th><td>{_esc(_fmt(v))}</td></tr>"
            for k, v in sequential.items()
        )
        sequential_html = (
            "<section><h2>Sequential gate</h2>"
            f'<table class="kv">{rows}</table></section>'
        )

    extra = d.get("extra") or {}
    extra_html = ""
    if extra:
        blob = json.dumps(extra, ensure_ascii=False, indent=2, default=str)
        extra_html = (
            "<details><summary>Extra diagnostics</summary>"
            f"<pre>{_esc(blob)}</pre></details>"
        )

    raw = json.dumps(d, ensure_ascii=False, indent=2, default=str)

    resolution_note = []
    if d.get("delta_declared") is not None:
        resolution_note.append(f"declared delta={_fmt(d['delta_declared'])}")
    if d.get("mde") is not None:
        resolution_note.append(f"MDE={_fmt(d['mde'])}")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)} — {_esc(verdict)}</title>
<style>
  :root {{
    color-scheme: light;
    --ink: #171717;
    --muted: #5f6368;
    --line: #d9dde3;
    --paper: #ffffff;
    --wash: #f6f7f9;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--wash);
    color: var(--ink);
    font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans CJK SC",
          "Noto Sans", Arial, sans-serif;
  }}
  main {{ max-width: 980px; margin: 0 auto; padding: 36px 22px 56px; }}
  .eyebrow {{ font-size: 12px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }}
  h1 {{ font-size: 30px; line-height: 1.15; margin: 8px 0 18px; }}
  h2 {{ font-size: 18px; margin: 30px 0 10px; }}
  .verdict {{
    display: inline-block; color: #fff; background: {color};
    padding: 7px 12px; border-radius: 999px; font-weight: 700;
    letter-spacing: .04em;
  }}
  .meaning {{ margin: 12px 0 0; max-width: 70ch; color: var(--muted); }}
  .grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px; margin: 26px 0 0;
  }}
  .card {{ background: var(--paper); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; }}
  .label {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }}
  .value {{ font-size: 22px; font-weight: 650; margin-top: 4px; font-variant-numeric: tabular-nums; }}
  .note {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
  section {{ background: var(--paper); border: 1px solid var(--line); border-radius: 12px; padding: 4px 18px 18px; margin-top: 16px; }}
  ul {{ margin: 8px 0 0; padding-left: 22px; }}
  li {{ margin: 6px 0; }}
  table.kv {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
  table.kv th, table.kv td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: left; }}
  table.kv th {{ width: 34%; color: var(--muted); font-weight: 600; }}
  details {{ margin-top: 16px; }}
  summary {{ cursor: pointer; color: var(--muted); }}
  pre {{ overflow: auto; background: var(--wash); border: 1px solid var(--line); border-radius: 10px; padding: 14px; font-size: 12px; }}
  .repro {{ background: var(--paper); border: 1px solid var(--line); border-radius: 12px; padding: 14px 18px; margin-top: 16px; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; }}
  footer {{ margin-top: 28px; color: var(--muted); font-size: 12px; }}
</style>
</head>
<body>
<main>
  <div class="eyebrow">arx / benchmark readability audit</div>
  <h1>{_esc(title)}</h1>
  <div class="verdict">{_esc(verdict)}</div>
  <p class="meaning">{_esc(meaning)}</p>

  <div class="grid">
    {_card("Leader", d.get("leader_system"), f"score={_fmt(d.get('leader_score'))}")}
    {_card("Noise floor", d.get("floor_estimate"), str(d.get("floor_method") or "unknown method"))}
    {_card("Margin over floor", d.get("margin"))}
    {_card("Resolution bound", d.get("threshold"), "; ".join(resolution_note))}
    {_card("Items", d.get("n_items"), f"floor on {_fmt(d.get('n_floor_items'))}")}
    {_card("Gold generators", d.get("n_generators"))}
    {_card("Contestants", d.get("n_contestants"))}
  </div>

  <section>
    <h2>Why this verdict</h2>
    <ul>{reasons_html}</ul>
  </section>

  {sequential_html}
  {extra_html}

  <div class="repro">
    <strong>Reproduce.</strong>
    <code>arx audit &lt;leaderboard.json&gt; --html &lt;report.html&gt;</code>
    <div class="note">The verdict is computed from the input JSON only. No network, no model.</div>
  </div>

  <details>
    <summary>Raw report</summary>
    <pre>{_esc(raw)}</pre>
  </details>

  <footer>
    Generated {_esc(generated)} by arx. A refusal is the tool working.
  </footer>
</main>
</body>
</html>
"""
