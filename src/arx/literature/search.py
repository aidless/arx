"""Literature search over arXiv and Semantic Scholar.

Both sources are free and need no key. Both fail in the ordinary way — rate
limits, timeouts, maintenance windows — so every search returns whatever it
could get plus an explicit record of which sources failed. "Zero results from
one source" must never be reported as "no literature exists": that conflation
is how a novelty claim ends up resting on an outage.

No network access happens at import time, and nothing here requires an API key.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

ARXIV_API = "http://export.arxiv.org/api/query"
S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
_ATOM = {"a": "http://www.w3.org/2005/Atom"}


@dataclass
class Paper:
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    abstract: str = ""
    url: str = ""
    doi: str | None = None
    source: str = ""

    def key(self) -> str:
        return normalise_title(self.title)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "url": self.url,
            "doi": self.doi,
            "source": self.source,
        }


@dataclass
class SearchOutcome:
    papers: list[Paper]
    sources_ok: list[str] = field(default_factory=list)
    sources_failed: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_papers": len(self.papers),
            "sources_ok": list(self.sources_ok),
            "sources_failed": dict(self.sources_failed),
        }


_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalise_title(title: str) -> str:
    """Fold a title to a comparison key: lowercase alphanumerics only."""
    return _NON_ALNUM.sub("", (title or "").lower())


def _get(url: str, params: dict | None = None, timeout: float = 20.0) -> bytes:
    full = f"{url}?{urllib.parse.urlencode(params)}" if params else url
    req = urllib.request.Request(full, headers={"User-Agent": "arx-researcher/0.3"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def search_arxiv(query: str, max_results: int = 20, *, timeout: float = 20.0) -> list[Paper]:
    raw = _get(
        ARXIV_API,
        {
            "search_query": f"all:{query}",
            "max_results": int(max_results),
            "sortBy": "relevance",
        },
        timeout,
    )
    root = ET.fromstring(raw)  # noqa: S314 - XML from a fixed, trusted host
    out: list[Paper] = []
    for entry in root.findall("a:entry", _ATOM):
        title = (entry.findtext("a:title", default="", namespaces=_ATOM) or "").strip()
        if not title:
            continue
        published = entry.findtext("a:published", default="", namespaces=_ATOM) or ""
        year = int(published[:4]) if published[:4].isdigit() else None
        authors = [
            (a.findtext("a:name", default="", namespaces=_ATOM) or "").strip()
            for a in entry.findall("a:author", _ATOM)
        ]
        out.append(
            Paper(
                title=re.sub(r"\s+", " ", title),
                authors=[a for a in authors if a],
                year=year,
                abstract=re.sub(
                    r"\s+", " ",
                    (entry.findtext("a:summary", default="", namespaces=_ATOM) or "").strip(),
                ),
                url=(entry.findtext("a:id", default="", namespaces=_ATOM) or "").strip(),
                source="arxiv",
            )
        )
    return out


def search_semantic_scholar(
    query: str, limit: int = 20, *, timeout: float = 20.0
) -> list[Paper]:
    raw = _get(
        S2_API,
        {
            "query": query,
            "limit": int(limit),
            "fields": "title,authors,year,abstract,url,externalIds",
        },
        timeout,
    )
    data = json.loads(raw.decode("utf-8"))
    out: list[Paper] = []
    for item in data.get("data", []) or []:
        title = (item.get("title") or "").strip()
        if not title:
            continue
        ext = item.get("externalIds") or {}
        out.append(
            Paper(
                title=title,
                authors=[a.get("name", "") for a in (item.get("authors") or []) if a.get("name")],
                year=item.get("year"),
                abstract=item.get("abstract") or "",
                url=item.get("url") or "",
                doi=ext.get("DOI"),
                source="semantic_scholar",
            )
        )
    return out


def search_all(
    query: str,
    max_results: int = 20,
    *,
    timeout: float = 20.0,
    sources: tuple[str, ...] = ("arxiv", "semantic_scholar"),
) -> SearchOutcome:
    """Query every source, merge, deduplicate by normalised title.

    Failures are collected, not swallowed: the caller decides whether a partial
    result is good enough to act on.
    """
    outcome = SearchOutcome(papers=[])
    merged: dict[str, Paper] = {}

    for name in sources:
        try:
            if name == "arxiv":
                found = search_arxiv(query, max_results, timeout=timeout)
            elif name == "semantic_scholar":
                found = search_semantic_scholar(query, max_results, timeout=timeout)
            else:
                outcome.sources_failed[name] = "unknown source"
                continue
        except (urllib.error.URLError, TimeoutError, OSError, ET.ParseError,
                json.JSONDecodeError, ValueError) as exc:
            outcome.sources_failed[name] = f"{type(exc).__name__}: {exc}"
            continue
        outcome.sources_ok.append(name)
        for paper in found:
            merged.setdefault(paper.key(), paper)

    outcome.papers = list(merged.values())
    return outcome


class LocalCorpus:
    """An offline corpus, for tests and for runs that must not touch the network.

    Deterministic, dependency-free, and honest about being a stub: it searches
    titles and abstracts it was given, and nothing else.
    """

    def __init__(self, papers: list[Paper] | None = None):
        self.papers = list(papers or [])

    def add(self, *, title: str, authors: list[str] | None = None, year: int | None = None,
            abstract: str = "", url: str = "", doi: str | None = None) -> Paper:
        paper = Paper(title=title, authors=authors or [], year=year, abstract=abstract,
                      url=url, doi=doi, source="local")
        self.papers.append(paper)
        return paper

    def search(self, query: str, max_results: int = 20) -> list[Paper]:
        terms = token_set(query)
        scored: list[tuple[float, Paper]] = []
        for paper in self.papers:
            hay = token_set(f"{paper.title} {paper.abstract}")
            if not hay:
                continue
            overlap = len(terms & hay)
            if overlap:
                scored.append((overlap / max(len(terms), 1), paper))
        scored.sort(key=lambda t: (-t[0], t[1].title))
        return [p for _, p in scored[:max_results]]


def token_set(text: str) -> set[str]:
    return {t for t in _NON_ALNUM.split((text or "").lower()) if len(t) > 2}


__all__ = [
    "ARXIV_API",
    "S2_API",
    "LocalCorpus",
    "Paper",
    "SearchOutcome",
    "normalise_title",
    "search_all",
    "search_arxiv",
    "search_semantic_scholar",
    "token_set",
]
