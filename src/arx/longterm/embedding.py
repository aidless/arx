"""Deterministic, dependency-free text embeddings.

Why not a sentence-transformer: the point of storing hypotheses for years is
that a later run can re-derive the same similarity and get the same answer. A
model with weights that update, a tokeniser that changes, and a library that
must be installed is a reproducibility liability for a *ledger*.

So the default embedder is a hashed bag of tokens and character trigrams:

* deterministic — the same text always maps to the same vector, in any process,
  on any machine, because the hash is SHA-256 rather than Python's randomised
  ``hash()``;
* dependency-free — nothing to install, nothing to version;
* honest about its weakness — it matches wording, not meaning. Two hypotheses
  saying the same thing in unrelated vocabulary will look different.

That third point is the one that matters. This is a *recall* aid for the
"same line of enquiry, new name" failure, not a solution to the effective-
test-count problem. ``docs/architecture.md`` states where it stops being
useful. A ``[longterm]`` extra can swap in a transformer embedder when recall
matters more than bit-identical reproducibility; the choice is recorded per
row in the database.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Sequence
from typing import Protocol

DIM = 256
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-]*")


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, text: str) -> list[float]: ...

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]: ...


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def _features(text: str) -> list[str]:
    """Tokens plus character trigrams over the token stream."""
    tokens = tokenize(text)
    features = list(tokens)
    joined = " ".join(tokens)
    for n in (3, 4):
        for i in range(max(0, len(joined) - n + 1)):
            features.append(joined[i : i + n])
    return features


def _bucket(feature: str, dim: int) -> tuple[int, float]:
    """Stable bucket and sign for a feature. Signed hashing keeps collisions
    from systematically inflating similarity."""
    digest = hashlib.sha256(feature.encode("utf-8")).digest()
    idx = int.from_bytes(digest[:4], "big") % dim
    sign = 1.0 if digest[4] & 1 else -1.0
    return idx, sign


class HashingEmbedder:
    """Signed hashing over tokens and character n-grams, L2-normalised."""

    name = "hash"
    dim: int = DIM

    def __init__(self, dim: int = DIM):
        if dim < 16:
            raise ValueError("dim must be >= 16")
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for feature in _features(text):
            idx, sign = _bucket(feature, self.dim)
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class TransformerEmbedder:  # pragma: no cover - optional extra
    """sentence-transformers backend. Requires ``arx-researcher[longterm]``."""

    name = "transformer"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "TransformerEmbedder needs the longterm extra: "
                "pip install 'arx-researcher[longterm]'"
            ) from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> list[float]:
        return [float(x) for x in self._model.encode(text, normalize_embeddings=True)]

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        return [
            [float(x) for x in row]
            for row in self._model.encode(list(texts), normalize_embeddings=True)
        ]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity. Assumes L2-normalised inputs when it can skip the norm."""
    if len(a) != len(b):
        raise ValueError("vector length mismatch")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(dot / (na * nb))


def build_embedder(kind: str = "hash") -> Embedder:
    if kind in ("hash", "hashing", "default"):
        return HashingEmbedder()
    if kind in ("transformer", "sentence-transformers", "minilm"):
        return TransformerEmbedder()
    raise ValueError(f"unknown embedder kind: {kind!r}")


__all__ = [
    "DIM",
    "Embedder",
    "HashingEmbedder",
    "TransformerEmbedder",
    "build_embedder",
    "cosine",
    "tokenize",
]
