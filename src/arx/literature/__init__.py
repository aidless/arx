"""Literature search, verified-library lookup, and citation checking."""

from .citation_check import (
    CitationAudit,
    CitationStatus,
    CitationVerdict,
    audit_references,
    verify_one,
)
from .search import (
    LocalCorpus,
    Paper,
    SearchOutcome,
    normalise_title,
    search_all,
    search_arxiv,
    search_semantic_scholar,
)
from .zotero import (
    Library,
    LocalLibrary,
    NullLibrary,
    ZoteroClient,
    ZoteroItem,
    build_library,
)

__all__ = [
    "CitationAudit",
    "CitationStatus",
    "CitationVerdict",
    "audit_references",
    "verify_one",
    "LocalCorpus",
    "Paper",
    "SearchOutcome",
    "normalise_title",
    "search_all",
    "search_arxiv",
    "search_semantic_scholar",
    "Library",
    "LocalLibrary",
    "NullLibrary",
    "ZoteroClient",
    "ZoteroItem",
    "build_library",
]
