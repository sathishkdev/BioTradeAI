"""Retrieval with a hard metadata pre-filter, tier resolution and a confidence score.

1. Resolve the drug name to its Entity_ID (ontology) and check it belongs to the ticker.
2. Search ONLY within ``ticker AND entity_id`` (and ``published_at <= as_of``): cosine
   similarity, fused with BM25 keyword ranking by Reciprocal Rank Fusion when
   ``rag.hybrid`` is on. Results are re-checked in code; any mismatch raises.
3. Claims (clinical hold, endpoint, PDUFA date, CRL) from ALL claim-bearing chunks of the
   entity are resolved:
     * within a tier, the most recent claim wins;
     * across tiers, the most authoritative tier's latest claim wins (T1 > T2 > T3);
     * EXCEPT a Tier-2 claim that is at least as recent as the Tier-1 winner and
       disagrees with it: authority alone cannot settle that -> unresolved.
4. confidence = authority x density
     authority = similarity-weighted mean of tier weights over relevant chunks
     density   = min(1, relevant chunks / min_supporting_chunks)
5. INCONCLUSIVE if no relevant chunk, confidence < min_confidence, or any unresolved
   Tier-1/Tier-2 conflict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from biotradeai.config import GuardrailsConfig
from biotradeai.rag.embeddings import Embedder, embed_query
from biotradeai.rag.index import Chunk, Hit, VectorIndex, where_filter
from biotradeai.rag.lexical import bm25_scores, rrf
from biotradeai.rag.ontology import Ontology, OntologyError
from biotradeai.schemas import EvidenceBundle, EvidenceStatus, SourceRecord, SourceTier


class SemanticBleedError(RuntimeError):
    """The index returned a chunk outside the requested ticker/entity. Never expected."""


@dataclass(frozen=True)
class ClaimEvidence:
    value: str
    tier: SourceTier
    published_at: datetime
    sha256: str
    url: str


@dataclass(frozen=True)
class ResolvedClaim:
    key: str
    value: str
    evidence: ClaimEvidence
    rule: str  # agreement | tier | recency
    dissent: tuple[ClaimEvidence, ...] = ()


@dataclass(frozen=True)
class Conflict:
    key: str
    winner: ClaimEvidence
    dissent: tuple[ClaimEvidence, ...]
    resolved: bool
    note: str


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    ticker: str
    entity_id: str
    hits: tuple[Hit, ...]
    relevant: tuple[Hit, ...]
    claims: dict[str, ResolvedClaim]
    conflicts: tuple[Conflict, ...]
    authority: float
    density: float
    confidence: float
    bundle: EvidenceBundle
    reasons: tuple[str, ...] = field(default=())

    @property
    def inconclusive(self) -> bool:
        return self.bundle.status is EvidenceStatus.INCONCLUSIVE


# --------------------------------------------------------------------------- claims


def resolve_claims(chunks: list[Chunk]) -> tuple[dict[str, ResolvedClaim], list[Conflict]]:
    by_key: dict[str, list[ClaimEvidence]] = {}
    seen: set[tuple[str, str]] = set()
    for c in chunks:
        if (c.sha256, c.entity_id) in seen:
            continue
        seen.add((c.sha256, c.entity_id))
        for k, v in c.claims.items():
            by_key.setdefault(k, []).append(
                ClaimEvidence(v, c.tier, c.published_at, c.sha256, c.url))

    resolved: dict[str, ResolvedClaim] = {}
    conflicts: list[Conflict] = []
    for key in sorted(by_key):
        entries = by_key[key]
        latest: dict[SourceTier, ClaimEvidence] = {}
        for e in sorted(entries, key=lambda x: (x.published_at, x.sha256)):
            latest[e.tier] = e  # ends as the most recent entry per tier
        winner = latest[min(latest)]
        dissent = tuple(e for e in entries if e.value != winner.value)
        if not dissent:
            resolved[key] = ResolvedClaim(key, winner.value, winner, "agreement")
            continue
        same_tier = any(e.tier == winner.tier for e in dissent)
        rule = "recency" if same_tier and len(latest) == 1 else "tier"
        t2 = latest.get(SourceTier.SEC)
        unresolved = (winner.tier is SourceTier.REGULATORY and t2 is not None
                      and t2.value != winner.value and t2.published_at >= winner.published_at)
        note = (f"Tier-2 ({t2.published_at.date()}) says {t2.value!r}, Tier-1 "
                f"({winner.published_at.date()}) says {winner.value!r}"
                if unresolved and t2 else
                f"{rule}: {winner.value!r} from tier {int(winner.tier)} "
                f"({winner.published_at.date()}) over {len(dissent)} dissenting claim(s)")
        conflicts.append(Conflict(key, winner, dissent, not unresolved, note))
        if not unresolved:
            resolved[key] = ResolvedClaim(key, winner.value, winner, rule, dissent)
    return resolved, conflicts


# --------------------------------------------------------------------------- retriever


def _source(c: Chunk) -> SourceRecord:
    return SourceRecord(tier=c.tier, published_at=c.published_at, sha256=c.sha256,
                        url=c.url, ticker=c.ticker, entity_id=c.entity_id)


class Retriever:
    def __init__(self, index: VectorIndex, embedder: Embedder, ontology: Ontology,
                 cfg: GuardrailsConfig) -> None:
        self._index = index
        self._embedder = embedder
        self._onto = ontology
        self._cfg = cfg

    def _entity(self, ticker: str, drug: str | None, entity_id: str | None) -> str:
        if entity_id is None:
            if drug is None:
                raise ValueError("pass drug or entity_id")
            ent = self._onto.require(drug)
            entity_id = ent.entity_id
        ent_obj = self._onto.entities.get(entity_id)
        if ent_obj is not None and ent_obj.ticker != ticker:
            raise OntologyError(f"{entity_id} belongs to {ent_obj.ticker}, not {ticker}")
        return entity_id

    def _rank(self, query: str, qvec: list[float], where: dict[str, Any]) -> list[Hit]:
        """Vector ranking (exact over the filtered pool, ANN if the pool is huge), optionally
        fused with BM25 via Reciprocal Rank Fusion. Similarity reported is always cosine."""
        rag = self._cfg.rag
        pool = self._index.get_with_embeddings(where, limit=rag.max_pool)
        if len(pool) >= rag.max_pool:  # too large for exact search: use the ANN index
            cand = self._index.query(qvec, where, rag.top_k * 4)
            chunks = {h.chunk.chunk_id: h.chunk for h in cand}
            sims = {h.chunk.chunk_id: h.similarity for h in cand}
        else:
            chunks = {c.chunk_id: c for c, _ in pool}
            sims = {c.chunk_id: math.fsum(a * b for a, b in zip(qvec, e, strict=True))
                    for c, e in pool}
        vec_rank = sorted(sims, key=lambda i: (-sims[i], i))
        order = vec_rank
        if rag.hybrid and chunks:
            ids = list(chunks)
            scores = bm25_scores(query, [chunks[i].text for i in ids])
            lex_rank = [i for s, i in sorted(zip(scores, ids, strict=True),
                                             key=lambda t: (-t[0], t[1])) if s > 0]
            order = rrf([vec_rank, lex_rank])
        return [Hit(chunks[i], sims[i]) for i in order[: rag.top_k]]

    def retrieve(self, query: str, *, ticker: str, drug: str | None = None,
                 entity_id: str | None = None, as_of: datetime | None = None) -> RetrievalResult:
        rag = self._cfg.rag
        ticker = ticker.upper()
        eid = self._entity(ticker, drug, entity_id)
        where = where_filter(ticker, eid, as_of)

        qvec = embed_query(self._embedder, query)
        hits = self._rank(query, qvec, where)
        claim_chunks = self._index.get(where_filter(ticker, eid, as_of, {"has_claims": True}))
        for c in [h.chunk for h in hits] + claim_chunks:
            if c.ticker != ticker or c.entity_id != eid or (
                    as_of is not None and c.published_at > as_of):
                raise SemanticBleedError(f"chunk {c.chunk_id} is {c.ticker}/{c.entity_id}")

        relevant = [h for h in hits if h.similarity >= rag.min_similarity]
        claims, conflicts = resolve_claims(claim_chunks)

        if relevant:
            wsum = sum(max(h.similarity, 1e-9) for h in relevant)
            authority = sum(rag.tier_weights[int(h.chunk.tier)] * max(h.similarity, 1e-9)
                            for h in relevant) / wsum
        else:
            authority = 0.0
        density = min(1.0, len(relevant) / rag.min_supporting_chunks)
        confidence = round(authority * density, 6)

        reasons: list[str] = []
        if not relevant:
            reasons.append("no relevant evidence above min_similarity")
        if confidence < self._cfg.min_confidence:
            reasons.append(f"confidence {confidence:.3f} < min {self._cfg.min_confidence}")
        reasons += [f"unresolved {c.key} conflict: {c.note}" for c in conflicts if not c.resolved]

        src_chunks: dict[str, Chunk] = {h.chunk.sha256: h.chunk for h in relevant}
        for rc in claims.values():
            for c in claim_chunks:
                if c.sha256 == rc.evidence.sha256:
                    src_chunks.setdefault(c.sha256, c)
        bundle = EvidenceBundle(
            ticker=ticker, entity_id=eid,
            sources=[_source(c) for c in src_chunks.values()],
            confidence=max(0.0, min(1.0, confidence)),
            status=EvidenceStatus.INCONCLUSIVE if reasons else EvidenceStatus.OK,
            inconclusive_reason="; ".join(reasons) or None,
        )
        return RetrievalResult(query, ticker, eid, tuple(hits), tuple(relevant), claims,
                               tuple(conflicts), authority, density, confidence, bundle,
                               tuple(reasons))
