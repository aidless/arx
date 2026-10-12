"""Theory layer: what each test is supposed to constrain.

arx has a statistical layer. Left there, it can run a hundred hypotheses, gate
every one of them, control the family-wise error rate correctly — and produce a
statistically flawless data-dredging machine. Uniform p-values, no learned
theory. Formal compliance is not scientific compliance.

The distinction that matters is Popper's, and it is not the same as "every
hypothesis must be pre-registered". Pre-registration is a *procedure*; the
epistemic requirement is that **an observation must be able to constrain a
theory**. A pre-registered hypothesis that no theory would notice the fate of
can fail and change nothing.

So this module does three things, and none of them is a solution:

1. Every hypothesis may be attached to a **theory node** with an explicit
   observable: *if this theory is wrong, we will see X*. A hypothesis with a
   blank observable is recorded as such.
2. When every hypothesis derived from a theory has been rejected, the theory is
   marked **falsified** — the tree is falsifiable as a whole, not just leaf by
   leaf.
3. It **measures the deficit** rather than claiming to have fixed it:
   ``coverage()`` reports the fraction of tests that were theory-derived. A run
   whose coverage is 0.1 is a dredge with bookkeeping, and the number says so.

What this cannot do: tell you whether your theory is *good*. A bad theory with
a crisp observable satisfies every check here. That limit is documented, not
hidden — see ``docs/architecture.md``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

#: Falsification requires at least this many tested hypotheses. One rejected
#: hypothesis is a result, not a refuted theory; the number is a judgement call
#: and is recorded on every falsification report.
MIN_TESTS_TO_FALSIFY = 2

NEGATIVE_OUTCOMES = frozenset(
    {"rejected", "undecidable", "unreadable", "no_data", "stalled"}
)


@dataclass
class TheoryNode:
    """A theory and the hypotheses derived from it."""

    root: str
    statement: str = ""
    #: The observation that would show the theory is wrong. A theory without
    #: one is unfalsifiable by construction, and ``falsifiable`` says so.
    falsified_by: str = ""
    hypotheses: list[str] = field(default_factory=list)
    outcomes: dict[str, str] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    falsified: bool = False
    falsified_at: float | None = None
    #: Set when a theory that had been falsified later gains a supported
    #: hypothesis. The label is the *current* state of the evidence, so the
    #: theory stops being reported as falsified -- and this flag records that
    #: the reversal happened, because a theory that dies and then recovers is
    #: usually the interesting case.
    revived: bool = False

    # -- structure ----------------------------------------------------------
    def attach(self, hypothesis: str) -> None:
        text = (hypothesis or "").strip()
        if not text:
            raise ValueError("hypothesis text must be non-empty")
        if text not in self.hypotheses:
            self.hypotheses.append(text)

    @property
    def falsifiable(self) -> bool:
        return bool(self.falsified_by.strip())

    @property
    def n_tested(self) -> int:
        return len(self.outcomes)

    @property
    def n_rejected(self) -> int:
        return sum(1 for v in self.outcomes.values() if v in NEGATIVE_OUTCOMES)

    @property
    def n_supported(self) -> int:
        return sum(1 for v in self.outcomes.values() if v not in NEGATIVE_OUTCOMES)

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "statement": self.statement,
            "falsified_by": self.falsified_by,
            "falsifiable": self.falsifiable,
            "hypotheses": list(self.hypotheses),
            "outcomes": dict(self.outcomes),
            "n_tested": self.n_tested,
            "n_rejected": self.n_rejected,
            "n_supported": self.n_supported,
            "falsified": self.falsified,
            "falsified_at": self.falsified_at,
            "revived": self.revived,
        }


class TheoryLayer:
    """Registry of theories, their derived hypotheses, and their outcomes."""

    def __init__(self, *, min_tests_to_falsify: int = MIN_TESTS_TO_FALSIFY):
        if min_tests_to_falsify < 1:
            raise ValueError("min_tests_to_falsify must be >= 1")
        self.min_tests_to_falsify = int(min_tests_to_falsify)
        self.theories: dict[str, TheoryNode] = {}
        self._attachment: dict[str, str] = {}   # hypothesis -> theory root
        self.exploratory: list[str] = []

    # -- registration -------------------------------------------------------
    def register(
        self,
        root: str,
        *,
        statement: str = "",
        falsified_by: str = "",
    ) -> TheoryNode:
        key = (root or "").strip()
        if not key:
            raise ValueError("theory root must be non-empty")
        node = self.theories.get(key)
        if node is None:
            node = TheoryNode(root=key, statement=statement, falsified_by=falsified_by)
            self.theories[key] = node
        else:
            if statement:
                node.statement = statement
            if falsified_by:
                node.falsified_by = falsified_by
        return node

    def attach(self, hypothesis: str, theory_root: str) -> TheoryNode:
        node = self.theories.get((theory_root or "").strip())
        if node is None:
            raise KeyError(f"unknown theory: {theory_root!r}")
        text = (hypothesis or "").strip()
        node.attach(text)
        self._attachment[text] = node.root
        return node

    def mark_exploratory(self, hypothesis: str) -> None:
        """Record a hypothesis that belongs to no theory.

        Not an error. Exploratory work is legitimate, and pretending otherwise
        would just push it underground. But it is counted separately, because
        an exploratory finding is a hypothesis to be tested later, not a test
        of anything.
        """
        text = (hypothesis or "").strip()
        if text and text not in self.exploratory:
            self.exploratory.append(text)

    def theory_of(self, hypothesis: str) -> TheoryNode | None:
        root = self._attachment.get((hypothesis or "").strip())
        return self.theories.get(root) if root else None

    # -- outcomes -----------------------------------------------------------
    def record(
        self,
        hypothesis: str,
        verdict: str,
        *,
        reason: str = "",
    ) -> TheoryNode | None:
        """Record a test outcome. Returns the theory node, or None if
        the hypothesis was never attached (it becomes exploratory)."""
        text = (hypothesis or "").strip()
        node = self.theory_of(text)
        if node is None:
            self.mark_exploratory(text)
            return None
        node.outcomes[text] = str(verdict).lower()
        if reason:
            node.reasons[text] = reason
        self._refresh(node)
        return node

    def _refresh(self, node: TheoryNode) -> bool:
        """Recompute the falsified label from the evidence so far.

        Recomputing rather than latching matters: the criterion is "every test
        rejected", so one supporting result means the theory is *not* falsified
        and saying otherwise would leave a stale label on the record. The
        ``revived`` flag keeps the history visible instead of pretending the
        reversal never happened.
        """
        was = node.falsified
        now = self.evaluate(node)
        node.falsified = now
        if now and not was:
            node.falsified_at = time.time()
        elif was and not now:
            node.revived = True
            node.falsified_at = None
        elif not now:
            node.falsified_at = None
        return now

    def evaluate(self, node: TheoryNode) -> bool:
        """True when the theory has earned the label *falsified*.

        Requires: a stated falsifier, at least ``min_tests_to_falsify`` tested
        hypotheses, and every one of them rejected. Anything less is a
        suggestive result, not a refutation.
        """
        if not node.falsifiable:
            return False
        if node.n_tested < self.min_tests_to_falsify:
            return False
        return node.n_rejected == node.n_tested

    # -- reporting ----------------------------------------------------------
    def falsified_theories(self) -> list[TheoryNode]:
        return [t for t in self.theories.values() if t.falsified]

    def surviving_theories(self) -> list[TheoryNode]:
        return [t for t in self.theories.values() if not t.falsified]

    def unfalsifiable_theories(self) -> list[TheoryNode]:
        """Theories with no stated observable. These cannot be refuted here,
        which makes them the ones most likely to be believed for free."""
        return [t for t in self.theories.values() if not t.falsifiable]

    def coverage(self) -> dict:
        """How much of the work was theory-driven, and how much was dredging.

        This is the metric the module exists to expose. It is reported, never
        optimised against — a rising coverage number that came from attaching
        theories retroactively would be its own Goodhart failure.
        """
        attached = sum(len(t.hypotheses) for t in self.theories.values())
        tested_theory = sum(t.n_tested for t in self.theories.values())
        exploratory = len(self.exploratory)
        total_hypotheses = attached + exploratory
        total_tested = tested_theory + exploratory
        return {
            "n_theories": len(self.theories),
            "n_theories_falsifiable": sum(1 for t in self.theories.values() if t.falsifiable),
            "n_theories_falsified": len(self.falsified_theories()),
            "n_theories_unfalsifiable": len(self.unfalsifiable_theories()),
            "hypotheses_attached": attached,
            "hypotheses_exploratory": exploratory,
            "hypotheses_tested_theory_driven": tested_theory,
            "theory_coverage": (
                round(tested_theory / total_tested, 4) if total_tested else 0.0
            ),
            "hypothesis_coverage": (
                round(attached / total_hypotheses, 4) if total_hypotheses else 0.0
            ),
            "note": (
                "theory_coverage is the share of tested hypotheses that could "
                "constrain a theory. A high value with weak theories is still "
                "weak science; the number bounds one failure mode, not all."
            ),
        }

    def to_dict(self) -> dict:
        return {
            "theories": [t.to_dict() for t in self.theories.values()],
            "exploratory": list(self.exploratory),
            "coverage": self.coverage(),
        }

    # -- reconstruction -----------------------------------------------------
    @classmethod
    def from_ledger(cls, records, *, min_tests_to_falsify: int = MIN_TESTS_TO_FALSIFY):
        """Rebuild the layer from ``theory`` ledger records, in order."""
        layer = cls(min_tests_to_falsify=min_tests_to_falsify)
        for rec in records:
            kind = rec.get("kind") if isinstance(rec, dict) else getattr(rec, "kind", None)
            payload = rec.get("payload") if isinstance(rec, dict) else getattr(rec, "payload", None)
            if kind != "theory" or not isinstance(payload, dict):
                continue
            event = payload.get("event")
            if event == "register":
                layer.register(
                    str(payload.get("root", "")),
                    statement=str(payload.get("statement", "")),
                    falsified_by=str(payload.get("falsified_by", "")),
                )
            elif event == "attach":
                root = str(payload.get("root", ""))
                if root and root in layer.theories:
                    layer.attach(str(payload.get("hypothesis", "")), root)
            elif event == "outcome":
                layer.record(
                    str(payload.get("hypothesis", "")),
                    str(payload.get("verdict", "")),
                    reason=str(payload.get("reason", "")),
                )
            elif event == "exploratory":
                layer.mark_exploratory(str(payload.get("hypothesis", "")))
        return layer


__all__ = [
    "MIN_TESTS_TO_FALSIFY",
    "NEGATIVE_OUTCOMES",
    "TheoryLayer",
    "TheoryNode",
]
