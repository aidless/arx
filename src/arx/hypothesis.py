"""Hypothesis generation, deduplication, and falsification hooks.

A hypothesis in arx is not just a sentence. It carries the two things the gate
needs in order to pre-register honestly:

* ``confidence`` — the proposer's prior, which sets how many looks are planned.
  Self-reported, so it is recorded rather than trusted.
* ``expected_effect_se_ratio`` — expected effect over expected standard error.
  This is what makes "how many looks do we need" answerable *before* the data
  exists. Proposing a hypothesis without it is allowed; the planner then falls
  back to the prior alone and says so in the ledger.

Deduplication runs on **normalised token sets**, not raw strings, so
"Reranker-v2 improves retrieval" and "retrieval improves with reranker v2" are
recognised as one line of enquiry. That is a weak notion of sameness — two
hypotheses can be lexically different and statistically the same — and the
limitation is stated here rather than papered over: see
``docs/architecture.md`` on why semantic dedup is an approximation to the
effective-test-count problem, not a solution to it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from .llm import ChatModel, parse_json_object

# Words that carry no discriminating power inside a hypothesis sentence.
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have if in into is it its of on or that
    the their there these this to using was we were will with without than then them
    our can may might should would could does do not no more most less least""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]+")


@dataclass
class Hypothesis:
    text: str
    rationale: str = ""
    confidence: float = 0.5
    expected_effect_se_ratio: float | None = None
    novelty: float | None = None
    duplicates: list[str] = field(default_factory=list)
    origin: str = "generated"

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ValueError("hypothesis text must be non-empty")
        self.text = self.text.strip()
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "expected_effect_se_ratio": self.expected_effect_se_ratio,
            "novelty": self.novelty,
            "duplicates": list(self.duplicates),
            "origin": self.origin,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Hypothesis:
        return cls(
            text=str(data["text"]),
            rationale=str(data.get("rationale", "")),
            confidence=float(data.get("confidence", 0.5)),
            expected_effect_se_ratio=data.get("expected_effect_se_ratio"),
            novelty=data.get("novelty"),
            duplicates=list(data.get("duplicates") or []),
            origin=str(data.get("origin", "generated")),
        )


# ---------------------------------------------------------------------------
# similarity
# ---------------------------------------------------------------------------

def token_set(text: str) -> set[str]:
    """Content tokens: lowercased, stopwords dropped, short tokens dropped."""
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2}


def jaccard(a: str, b: str) -> float:
    sa, sb = token_set(a), token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def containment(a: str, b: str) -> float:
    """Overlap coefficient: ``|A∩B| / min(|A|,|B|)``.

    Jaccard punishes length differences, which is wrong here — a long
    hypothesis and a short restatement of it are still the same idea. The
    overlap coefficient ignores the extra words on the longer side, so
    ``containment(long, short)`` stays near 1.0 exactly when Jaccard collapses.

    It is symmetric (``min`` is), unlike the asymmetric containment used in set
    theory; the name refers to "how much of the shorter statement is present in
    the longer one", which is the same number in either argument order.
    """
    sa, sb = token_set(a), token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

GENERATE_SYSTEM = (
    "You are a research methodologist. You propose hypotheses that could be "
    "falsified by a small, runnable experiment. You never claim a result. "
    "You reply with JSON only."
)

GENERATE_TEMPLATE = """Research topic:
{topic}

{context}

Propose {n} distinct, falsifiable hypotheses about the topic. For each one give:
  - text: one sentence, phrased so that an observation could contradict it
  - rationale: why it is plausible
  - confidence: your prior in [0,1] that it holds
  - expected_effect_se_ratio: expected effect size divided by expected standard
    error for a small experiment, or null if you cannot estimate it

Reply with JSON only, shaped exactly like:
{{"hypotheses": [{{"text": "...", "rationale": "...", "confidence": 0.5,
                   "expected_effect_se_ratio": null}}]}}"""


def _clamp01(value: object, default: float = 0.5) -> float:
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, v))


def parse_hypotheses(reply: str) -> list[Hypothesis]:
    """Parse a model reply into hypotheses. Returns [] rather than raising."""
    obj = parse_json_object(reply)
    if not obj:
        return []
    raw = obj.get("hypotheses")
    if not isinstance(raw, list):
        return []
    out: list[Hypothesis] = []
    for item in raw:
        if not isinstance(item, dict) or not str(item.get("text", "")).strip():
            continue
        ratio = item.get("expected_effect_se_ratio")
        try:
            ratio = float(ratio) if ratio is not None else None
        except (TypeError, ValueError):
            ratio = None
        if ratio is not None and ratio <= 0:
            ratio = None
        try:
            out.append(
                Hypothesis(
                    text=str(item["text"]),
                    rationale=str(item.get("rationale", "")),
                    confidence=_clamp01(item.get("confidence")),
                    expected_effect_se_ratio=ratio,
                    origin="llm",
                )
            )
        except ValueError:
            continue
    return out


def generate(
    llm: ChatModel,
    topic: str,
    *,
    n: int = 5,
    context: str = "",
) -> list[Hypothesis]:
    """Ask the model for hypotheses. Empty list if unavailable or unparseable."""
    if not llm.available:
        return []
    prompt = GENERATE_TEMPLATE.format(topic=topic, n=n, context=context)
    reply = llm.chat(GENERATE_SYSTEM, prompt)
    return parse_hypotheses(reply)


# ---------------------------------------------------------------------------
# deduplication
# ---------------------------------------------------------------------------

@dataclass
class DedupResult:
    kept: list[Hypothesis]
    dropped: list[Hypothesis]
    pairs: list[tuple[str, str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kept": [h.text for h in self.kept],
            "dropped": [h.text for h in self.dropped],
            "pairs": [
                {"kept": a, "dropped": b, "similarity": round(s, 4)}
                for a, b, s in self.pairs
            ],
        }


def deduplicate(
    hypotheses: Sequence[Hypothesis],
    *,
    threshold: float = 0.6,
    known: Iterable[str] = (),
) -> DedupResult:
    """Drop hypotheses that restate one already kept (or already known).

    ``known`` holds texts from previous runs. Passing them in is what turns
    single-run dedup into something that can catch the same line of enquiry
    coming back under a new name.
    """
    kept: list[Hypothesis] = []
    dropped: list[Hypothesis] = []
    pairs: list[tuple[str, str, float]] = []
    reference = list(known)

    for hyp in hypotheses:
        best_sim = 0.0
        best_match = ""
        for ref in reference:
            sim = max(jaccard(hyp.text, ref), containment(hyp.text, ref) * 0.9)
            if sim > best_sim:
                best_sim, best_match = sim, ref
        if best_sim >= threshold and best_match:
            hyp.duplicates = [best_match]
            dropped.append(hyp)
            pairs.append((best_match, hyp.text, best_sim))
        else:
            kept.append(hyp)
            reference.append(hyp.text)
    return DedupResult(kept=kept, dropped=dropped, pairs=pairs)


def from_texts(texts: Iterable[str]) -> list[Hypothesis]:
    """Wrap caller-supplied texts.

    ``origin="manual"`` rather than the dataclass default: these were not
    generated by a model, and a report that says "generated" about a
    hand-written hypothesis is describing something that did not happen.
    """
    return [Hypothesis(text=t, origin="manual") for t in texts if str(t).strip()]


__all__ = [
    "GENERATE_SYSTEM",
    "DedupResult",
    "Hypothesis",
    "containment",
    "deduplicate",
    "from_texts",
    "generate",
    "jaccard",
    "parse_hypotheses",
    "token_set",
]
