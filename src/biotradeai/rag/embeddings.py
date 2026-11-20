"""Switchable text embedders (same idea as the LLM spec strings).

    st:BAAI/bge-small-en-v1.5  local sentence-transformers (default; ~130 MB, one download)
    st:all-MiniLM-L6-v2        any sentence-transformers model id (faster, lower recall)
    ollama:nomic-embed-text    Ollama /api/embed
    hash                       deterministic offline hashing embedder (tests)

Resolution: explicit spec -> env ``BIOTRADEAI_EMBEDDER`` -> DEFAULT_EMBEDDER.
All vectors are L2-normalised so cosine similarity = dot product.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import os
import re
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import httpx

DEFAULT_EMBEDDER = "st:BAAI/bge-small-en-v1.5"


class EmbeddingError(RuntimeError):
    pass


@runtime_checkable
class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def embed_query(embedder: Embedder, text: str) -> list[float]:
    """Some models (BGE, E5, nomic) expect a different prefix for queries than passages."""
    fn = getattr(embedder, "embed_queries", None)
    vecs: list[list[float]] = fn([text]) if callable(fn) else embedder.embed([text])
    return vecs[0]


# Query instructions recommended by the model authors.
_QUERY_PREFIX = {
    "bge-": "Represent this sentence for searching relevant passages: ",
    "e5-": "query: ",
}


def _normalize(v: Sequence[float]) -> list[float]:
    n = math.sqrt(math.fsum(x * x for x in v))
    return [float(x) / n for x in v] if n > 0 else [0.0 for _ in v]


class HashingEmbedder:
    """Bag of words + bigrams hashed into ``dim`` buckets with signs. Fully offline."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hash-{dim}"

    def _vec(self, text: str) -> list[float]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        feats = words + [f"{a}_{b}" for a, b in itertools.pairwise(words)]
        v = [0.0] * self.dim
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return _normalize(v)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


class SentenceTransformerEmbedder:
    def __init__(self, model: str, *, local_files_only: bool = False) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise EmbeddingError("pip install -e .[rag] for sentence-transformers") from exc
        model_id = model if "/" in model else f"sentence-transformers/{model}"
        try:
            self._m: Any = SentenceTransformer(model_id, device="cpu",
                                               local_files_only=local_files_only)
        except Exception as exc:
            raise EmbeddingError(f"could not load embedding model {model_id}: {exc}") from exc
        short = model_id.split("/")[-1]
        self.name = f"st-{short}"
        self.query_prefix = next((p for k, p in _QUERY_PREFIX.items()
                                  if short.lower().startswith(k)), "")
        self._passage_prefix = "passage: " if short.lower().startswith("e5-") else ""

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._m.encode(list(texts), normalize_embeddings=True, batch_size=32,
                              show_progress_bar=False)
        return [[float(x) for x in v] for v in vecs]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode([self._passage_prefix + t for t in texts])

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode([self.query_prefix + t for t in texts])


class OllamaEmbedder:
    def __init__(self, model: str, host: str | None = None,
                 transport: httpx.BaseTransport | None = None) -> None:
        self.name = f"ollama-{model.replace(':', '-')}"
        self._model = model
        base = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self._http = httpx.Client(base_url=base, timeout=120, transport=transport)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            r = self._http.post("/api/embed", json={"model": self._model, "input": list(texts)})
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"ollama embed failed: {exc}") from exc
        return [_normalize(v) for v in r.json()["embeddings"]]


def make_embedder(spec: str | None = None, *, local_files_only: bool = False) -> Embedder:
    s = (spec or os.environ.get("BIOTRADEAI_EMBEDDER") or DEFAULT_EMBEDDER).strip()
    kind, _, model = s.partition(":")
    kind = kind.lower()
    if kind == "hash":
        return HashingEmbedder(int(model) if model else 512)
    if kind == "st" and model:
        return SentenceTransformerEmbedder(model, local_files_only=local_files_only)
    if kind == "ollama" and model:
        return OllamaEmbedder(model)
    raise EmbeddingError(f"bad embedder spec {s!r}; use st:<model>, ollama:<model> or hash")
