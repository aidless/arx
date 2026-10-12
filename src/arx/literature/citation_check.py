"""Citation verification.

Two questions, asked in order, both of which must be answered before a
reference is allowed into a report:

1. **Existence** — does the paper resolve in an external index? A title that
   resolves nowhere is either a fabricated citation or a typo, and both are
   reasons to stop.
2. **Reading** — is it in the verified library (Zotero)? This is the step that
   catches a real paper cited for something it does not say.

When Zotero is unavailable the check reports ``skipped``, never ``passed``. An
unverifiable citation is not a verified one, and the report says which is which.

Every verdict carries the reason it was reached, because "this citation was
rejected" is not actionable while "this citation resolves to a different paper
with a similar title" is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .search import Paper, SearchOutcome, normalise_title, search_all, token_set
from .zotero import Library, ZoteroItem


def _significant_words(text: str) -> set[str]:
    """Words worth comparing two titles on.

    Reuses the search layer's tokenisation so that "significant" means the
    same thing on both sides of the comparison.
    """
    return token_set(text)


class CitationStatus(StrEnum):
    VERIFIED = "verified"
    UNRESOLVED = "unresolved"
    TITLE_MISMATCH = "title_mismatch"
    NOT_IN_LIBRARY = "not_in_library"
    LIBRARY_UNAVAILABLE = "library_unavailable"
    ERROR = "error"


@dataclass
class CitationVerdict:
    title: str
    status: CitationStatus
    reason: str
    resolved: Paper | None = None
    library_item: ZoteroItem | None = None
    library_checked: bool = False

    @property
    def ok(self) -> bool:
        """True only when the citation is verified *or* the library step was
        skipped because no library is configured. An unconfigured library is a
        declared absence of evidence, not a failure of the citation."""
        return self.status is CitationStatus.VERIFIED or (
            self.status is CitationStatus.LIBRARY_UNAVAILABLE
        )

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "status": self.status.value,
            "reason": self.reason,
            "resolved_title": self.resolved.title if self.resolved else None,
            "resolved_url": self.resolved.url if self.resolved else None,
            "library_key": self.library_item.key if self.library_item else None,
            "library_checked": self.library_checked,
        }


@dataclass
class CitationAudit:
    verdicts: list[CitationVerdict] = field(default_factory=list)
    sources_ok: list[str] = field(default_factory=list)
    sources_failed: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(v.ok for v in self.verdicts)

    @property
    def n_verified(self) -> int:
        return sum(1 for v in self.verdicts if v.status is CitationStatus.VERIFIED)

    @property
    def problems(self) -> list[CitationVerdict]:
        return [v for v in self.verdicts if not v.ok]

    def to_dict(self) -> dict:
        return {
            "n": len(self.verdicts),
            "n_verified": self.n_verified,
            "ok": self.ok,
            "sources_ok": list(self.sources_ok),
            "sources_failed": dict(self.sources_failed),
            "problems": [v.to_dict() for v in self.problems],
            "verdicts": [v.to_dict() for v in self.verdicts],
        }

    def to_text(self) -> str:
        lines = [f"citations: {len(self.verdicts)} checked, {self.n_verified} verified"]
        if self.sources_ok:
            lines.append(f"  sources ok: {', '.join(self.sources_ok)}")
        for name, err in self.sources_failed.items():
            lines.append(f"  source failed: {name} ({err})")
        for verdict in self.problems:
            lines.append(f"  [{verdict.status.value}] {verdict.title}: {verdict.reason}")
        return "\n".join(lines)


def verify_one(
    title: str,
    *,
    library: Library,
    search: SearchOutcome | None = None,
    timeout: float = 20.0,
) -> CitationVerdict:
    """Run both checks for a single citation."""
    target = normalise_title(title)
    if not target:
        return CitationVerdict(title, CitationStatus.ERROR, "empty citation title")

    if search is None:
        try:
            search = search_all(title, max_results=5, timeout=timeout)
        except Exception as exc:  # network layers are allowed to be hostile
            return CitationVerdict(
                title, CitationStatus.ERROR, f"search failed: {type(exc).__name__}: {exc}"
            )

    exact: Paper | None = None
    near: Paper | None = None
    near_why = ""
    target_words = _significant_words(title)
    for paper in search.papers:
        key = normalise_title(paper.title)
        if key == target:
            exact = paper
            break
        if near is not None:
            continue
        if key.startswith(target[:40]) or target.startswith(key[:40]):
            near, near_why = paper, "shared leading text"
            continue
        # A title that keeps most of the same significant words is more likely
        # a retitled or mistyped version of the same work than an unrelated
        # paper. This is a heuristic and the reason string says so -- reporting
        # it as a match would be the failure mode worth avoiding.
        words = _significant_words(paper.title)
        shared = target_words & words
        denom = min(len(target_words), len(words)) or 1
        if len(shared) >= 2 and len(shared) / denom >= 0.5:
            near = paper
            near_why = f"{len(shared)}/{denom} shared significant words"

    if exact is None and near is None:
        return CitationVerdict(
            title,
            CitationStatus.UNRESOLVED,
            "not found in any queried index"
            + (f" (failed: {', '.join(search.sources_failed)})" if search.sources_failed else ""),
        )
    if exact is None and near is not None:
        return CitationVerdict(
            title,
            CitationStatus.TITLE_MISMATCH,
            f"closest match has a different title: {near.title!r}"
            f" (heuristic: {near_why})",
            resolved=near,
        )

    resolved = exact or near
    if not library.available:
        return CitationVerdict(
            title,
            CitationStatus.LIBRARY_UNAVAILABLE,
            "resolves in the index; library check skipped (no library configured)",
            resolved=resolved,
            library_checked=False,
        )

    item = None
    if resolved.doi:
        item = _lookup_doi(library, resolved.doi)
    if item is None:
        item = _lookup_title(library, resolved.title)
    if item is None:
        return CitationVerdict(
            title,
            CitationStatus.NOT_IN_LIBRARY,
            "resolves in the index but is absent from the verified library",
            resolved=resolved,
            library_checked=True,
        )
    return CitationVerdict(
        title,
        CitationStatus.VERIFIED,
        "resolves in the index and is present in the verified library",
        resolved=resolved,
        library_item=item,
        library_checked=True,
    )


def _lookup_doi(library: Library, doi: str) -> ZoteroItem | None:
    finder = getattr(library, "find_by_doi", None)
    if callable(finder):
        try:
            return finder(doi)
        except Exception:
            return None
    return None


def _lookup_title(library: Library, title: str) -> ZoteroItem | None:
    finder = getattr(library, "find_by_title", None)
    if callable(finder):
        try:
            return finder(title)
        except Exception:
            return None
    target = normalise_title(title)
    for item in library.items():
        if normalise_title(item.title) == target:
            return item
    return None


def audit_references(
    titles: list[str],
    library: Library,
    *,
    timeout: float = 20.0,
) -> CitationAudit:
    """Check every title. One search per title; results are not cached across
    titles because two different titles can legitimately resolve to one paper."""
    audit = CitationAudit()
    seen_sources_ok: set[str] = set()
    for title in titles:
        try:
            outcome = search_all(title, max_results=5, timeout=timeout)
        except Exception as exc:
            audit.verdicts.append(
                CitationVerdict(
                    title, CitationStatus.ERROR,
                    f"search failed: {type(exc).__name__}: {exc}",
                )
            )
            continue
        seen_sources_ok.update(outcome.sources_ok)
        for name, err in outcome.sources_failed.items():
            audit.sources_failed.setdefault(name, err)
        audit.verdicts.append(verify_one(title, library=library, search=outcome,
                                         timeout=timeout))
    audit.sources_ok = sorted(seen_sources_ok)
    return audit


__all__ = [
    "CitationAudit",
    "CitationStatus",
    "CitationVerdict",
    "audit_references",
    "verify_one",
]
