"""Zotero integration.

Two access paths, one interface:

* **Local HTTP API** (Zotero 7, ``http://localhost:23119/api``) — no key, no
  network, works offline. This is the default.
* **Web API** (``https://api.zotero.org``) — needs a library id and key.

The role Zotero plays in arx is narrow and worth stating: it is a *verified
library*. A citation that resolves against arXiv proves the paper exists; a
citation present in the user's Zotero library proves the user has actually read
and kept it. The second check is the one that catches "I cited something I
skimmed and misremembered".

When Zotero is off, the checker says the Zotero step was skipped. It does not
silently pass, and it does not silently fail.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

from ..config import AgentConfig
from .search import normalise_title


@dataclass
class ZoteroItem:
    key: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    item_type: str = ""

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "doi": self.doi,
            "item_type": self.item_type,
        }


class Library(Protocol):
    """Anything that can answer "is this paper in my verified library?"."""

    @property
    def available(self) -> bool: ...

    def items(self, limit: int = 200) -> list[ZoteroItem]: ...


class NullLibrary:
    """No library configured. Lookups return nothing and say they are off."""

    name = "null"

    @property
    def available(self) -> bool:
        return False

    def items(self, limit: int = 200) -> list[ZoteroItem]:
        return []


class LocalLibrary:
    """An in-memory library. Used for tests and for air-gapped runs."""

    name = "local"

    def __init__(self, items: list[ZoteroItem] | None = None):
        self._items = list(items or [])

    @property
    def available(self) -> bool:
        return True

    def add(self, title: str, *, authors: list[str] | None = None, year: int | None = None,
            doi: str | None = None, key: str | None = None) -> ZoteroItem:
        item = ZoteroItem(
            key=key or f"local-{len(self._items) + 1}",
            title=title,
            authors=authors or [],
            year=year,
            doi=doi,
            item_type="journalArticle",
        )
        self._items.append(item)
        return item

    def items(self, limit: int = 200) -> list[ZoteroItem]:
        return self._items[:limit]


class ZoteroClient:
    """HTTP client for a Zotero library (local or web)."""

    name = "zotero"

    def __init__(self, cfg: AgentConfig, *, timeout: float = 15.0):
        self.cfg = cfg
        self.timeout = timeout
        self.base_url = cfg.zotero_api_url.rstrip("/")
        self._cache: list[ZoteroItem] | None = None
        self._error: str | None = None

    @property
    def available(self) -> bool:
        return bool(self.base_url)

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "Zotero-API-Version": "3"}
        if self.cfg.zotero_api_key:
            headers["Zotero-API-Key"] = self.cfg.zotero_api_key
        return headers

    def _prefix(self) -> str:
        if "/api" in self.base_url and self.cfg.zotero_library_id:
            return f"/users/{urllib.parse.quote(self.cfg.zotero_library_id)}"
        return ""

    def items(self, limit: int = 200) -> list[ZoteroItem]:
        if self._cache is not None:
            return self._cache[:limit]
        url = (
            f"{self.base_url}{self._prefix()}/items"
            f"?limit={int(limit)}&format=json&itemType=-attachment"
        )
        try:
            req = urllib.request.Request(url, headers=self._headers())
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError,
                json.JSONDecodeError, ValueError) as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            self._cache = []
            return []

        out: list[ZoteroItem] = []
        for entry in payload if isinstance(payload, list) else []:
            data = entry.get("data", {}) if isinstance(entry, dict) else {}
            if not data:
                continue
            date = str(data.get("date", ""))
            year = int(date[:4]) if date[:4].isdigit() else None
            creators = data.get("creators") or []
            out.append(
                ZoteroItem(
                    key=str(entry.get("key") or data.get("key") or ""),
                    title=str(data.get("title") or ""),
                    authors=[
                        str(c.get("lastName") or c.get("name") or "")
                        for c in creators
                        if isinstance(c, dict)
                    ],
                    year=year,
                    doi=data.get("DOI") or None,
                    item_type=str(data.get("itemType") or ""),
                )
            )
        self._cache = out
        return out[:limit]

    @property
    def last_error(self) -> str | None:
        return self._error

    def find_by_title(self, title: str) -> ZoteroItem | None:
        target = normalise_title(title)
        if not target:
            return None
        for item in self.items():
            if normalise_title(item.title) == target:
                return item
        # Fall back to a prefix match, which catches subtitle truncation.
        for item in self.items():
            key = normalise_title(item.title)
            if key and (key.startswith(target[:60]) or target.startswith(key[:60])):
                return item
        return None

    def find_by_doi(self, doi: str) -> ZoteroItem | None:
        target = (doi or "").strip().lower()
        if not target:
            return None
        for item in self.items():
            if (item.doi or "").strip().lower() == target:
                return item
        return None


def build_library(cfg: AgentConfig) -> Library:
    if cfg.zotero_enabled and cfg.zotero_api_url:
        return ZoteroClient(cfg)
    return NullLibrary()


__all__ = [
    "Library",
    "LocalLibrary",
    "NullLibrary",
    "ZoteroClient",
    "ZoteroItem",
    "build_library",
]
