"""ChromaDB-backed vector index (local, persistent under ``data/chroma``).

Embeddings are computed by our ``Embedder`` and passed in explicitly, so Chroma never
downloads its own model. One collection per embedder (dimensions differ).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from biotradeai.paths import data_dir
from biotradeai.schemas import SourceTier


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    ticker: str
    entity_id: str
    tier: SourceTier
    published_at: datetime
    sha256: str  # of the chunk text
    doc_sha256: str
    url: str
    title: str
    doc_type: str
    claims: dict[str, str]

    def metadata(self) -> dict[str, str | int | float | bool]:
        return {
            "ticker": self.ticker,
            "entity_id": self.entity_id,
            "source_tier": int(self.tier),
            "published_at": self.published_at.isoformat(),
            "published_ts": int(self.published_at.timestamp()),
            "sha256": self.sha256,
            "doc_sha256": self.doc_sha256,
            "url": self.url,
            "title": self.title[:300],
            "doc_type": self.doc_type,
            "claims_json": json.dumps(self.claims, sort_keys=True),
            "has_claims": bool(self.claims),
        }

    @classmethod
    def from_record(cls, chunk_id: str, text: str, md: dict[str, Any]) -> Chunk:
        return cls(
            chunk_id=chunk_id, text=text, ticker=str(md["ticker"]),
            entity_id=str(md["entity_id"]), tier=SourceTier(int(md["source_tier"])),
            published_at=datetime.fromisoformat(str(md["published_at"])),
            sha256=str(md["sha256"]), doc_sha256=str(md["doc_sha256"]), url=str(md["url"]),
            title=str(md.get("title", "")), doc_type=str(md.get("doc_type", "")),
            claims=json.loads(str(md.get("claims_json") or "{}")),
        )


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    similarity: float


def where_filter(ticker: str, entity_id: str, as_of: datetime | None = None,
                 extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """The hard pre-filter: exact ticker AND exact entity_id (and point-in-time)."""
    clauses: list[dict[str, Any]] = [{"ticker": ticker}, {"entity_id": entity_id}]
    if as_of is not None:
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        clauses.append({"published_ts": {"$lte": int(as_of.timestamp())}})
    if extra:
        clauses.extend({k: v} for k, v in extra.items())
    return {"$and": clauses}


class VectorIndex:
    def __init__(self, embedder_name: str, *, path: Path | None = None,
                 client: Any | None = None) -> None:
        if client is None:
            import chromadb

            p = path or (data_dir() / "chroma")
            p.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(p))
        self._client = client
        slug = re.sub(r"[^a-zA-Z0-9_-]", "-", embedder_name)[:50].strip("-_")
        self.collection_name = f"biotradeai-{slug}"
        # Typed as Any: chromadb's stubs reject plain list[list[float]] embeddings.
        self._col: Any = client.get_or_create_collection(self.collection_name)

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length differ")
        self._col.upsert(ids=[c.chunk_id for c in chunks], embeddings=embeddings,
                         documents=[c.text for c in chunks],
                         metadatas=[c.metadata() for c in chunks])

    def query(self, embedding: list[float], where: dict[str, Any], k: int) -> list[Hit]:
        res: Any = self._col.query(query_embeddings=[embedding], n_results=k, where=where,
                              include=["documents", "metadatas", "distances"])
        hits: list[Hit] = []
        for cid, doc, md, dist in zip(res["ids"][0], res["documents"][0],
                                      res["metadatas"][0], res["distances"][0], strict=True):
            # Default space is squared L2; on unit vectors cos = 1 - d/2.
            hits.append(Hit(Chunk.from_record(cid, doc, dict(md)), 1.0 - float(dist) / 2.0))
        return hits

    def get(self, where: dict[str, Any]) -> list[Chunk]:
        res: Any = self._col.get(where=where, include=["documents", "metadatas"])
        return [Chunk.from_record(cid, doc, dict(md))
                for cid, doc, md in zip(res["ids"], res["documents"], res["metadatas"],
                                        strict=True)]

    def get_with_embeddings(self, where: dict[str, Any],
                            limit: int) -> list[tuple[Chunk, list[float]]]:
        res: Any = self._col.get(where=where, limit=limit,
                            include=["documents", "metadatas", "embeddings"])
        return [(Chunk.from_record(cid, doc, dict(md)), [float(x) for x in emb])
                for cid, doc, md, emb in zip(res["ids"], res["documents"], res["metadatas"],
                                             res["embeddings"], strict=True)]

    def count(self) -> int:
        n: int = self._col.count()
        return n


def utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
