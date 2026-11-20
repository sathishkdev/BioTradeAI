"""Turn source documents into tagged, embedded chunks.

Sources and their tiers:
  Tier 1  ClinicalTrials.gov records, FDA AdCom briefing documents (local PDF/HTML)
  Tier 2  SEC 10-K / 10-Q Item 1A "Risk Factors"
  Tier 3  PubMed abstracts, press releases

Entity tagging: if a document is about one known drug, pass ``entity_id``. Otherwise
each chunk is tagged with every entity of the document's ticker that it mentions (one
stored copy per entity), and chunks that mention none get the company-level entity.
"""

from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from biotradeai.config import RagConfig
from biotradeai.data.clinicaltrials import TrialRecord
from biotradeai.rag.chunking import chunk_text
from biotradeai.rag.claims import extract_claims
from biotradeai.rag.embeddings import Embedder
from biotradeai.rag.index import Chunk, VectorIndex
from biotradeai.rag.ontology import Ontology, company_entity
from biotradeai.schemas import SourceTier


@dataclass(frozen=True)
class Document:
    ticker: str
    tier: SourceTier
    published_at: datetime
    url: str
    title: str
    text: str
    doc_type: str
    entity_id: str | None = None
    meta: dict[str, str] = field(default_factory=dict)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- text utils

_TAG = re.compile(r"<[^>]+>")
_BLOCK = re.compile(r"</?(p|div|br|tr|li|h[1-6]|table|section)\b[^>]*>", re.I)


def html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", raw)
    raw = _BLOCK.sub("\n\n", raw)
    text = html.unescape(_TAG.sub(" ", raw)).replace("\xa0", " ")
    lines = [" ".join(ln.split()) for ln in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


_ITEM_1A = re.compile(r"item\s*1a\.?\s*[\-–—:]?\s*risk\s+factors", re.I)
_ITEM_END = re.compile(r"item\s*(1b\.?\s*[\-–—:]?\s*unresolved|1c\.?\s*[\-–—:]?\s*cyber"
                       r"|2\.?\s*[\-–—:]?\s*properties)", re.I)


def extract_item_1a(text: str) -> str | None:
    """Item 1A section of a 10-K/10-Q. Skips table-of-contents hits by taking the
    longest section between an Item 1A heading and the next item heading."""
    best: str | None = None
    for m in _ITEM_1A.finditer(text):
        end = _ITEM_END.search(text, m.end())
        section = text[m.end(): end.start() if end else len(text)].strip()
        if best is None or len(section) > len(best):
            best = section
    return best if best and len(best.split()) > 20 else None


def trial_results_lines(raw: dict[str, Any]) -> list[str]:
    """Primary endpoints and any POSTED results (outcome titles, time frames, p-values).

    Field names verified against a live v2 record (protocolSection.outcomesModule,
    resultsSection.outcomeMeasuresModule.outcomeMeasures[].{type,title,timeFrame,analyses}).
    """
    out: list[str] = []
    ps = raw.get("protocolSection") or {}
    for o in (ps.get("outcomesModule") or {}).get("primaryOutcomes") or []:
        if o.get("measure"):
            out.append(f"Primary endpoint: {o['measure']}"
                       + (f" (time frame: {o['timeFrame']})." if o.get("timeFrame") else "."))
    rs = raw.get("resultsSection") or {}
    for m in (rs.get("outcomeMeasuresModule") or {}).get("outcomeMeasures") or []:
        if str(m.get("type", "")).upper() != "PRIMARY":
            continue
        line = f"Posted primary outcome result: {m.get('title', '')}"
        if m.get("timeFrame"):
            line += f" (time frame: {m['timeFrame']})"
        stats = [f"p={a['pValue']}" + (f" ({a['statisticalMethod']})"
                                       if a.get("statisticalMethod") else "")
                 for a in m.get("analyses") or [] if a.get("pValue")]
        out.append(line + (f"; {', '.join(stats)}" if stats else "") + ".")
    if raw.get("hasResults") and not rs:
        out.append("Results are flagged as posted on ClinicalTrials.gov.")
    return out


def trial_to_document(rec: TrialRecord, ticker: str, entity_id: str) -> Document:
    lines = [
        f"ClinicalTrials.gov record {rec.nct_id}: {rec.title}.",
        f"Sponsor: {rec.sponsor}. Overall status: {rec.overall_status}. "
        f"Phases: {', '.join(rec.phases) or 'not specified'}.",
        f"Interventions: {', '.join(rec.interventions) or 'not listed'}. "
        f"Conditions: {', '.join(rec.conditions) or 'not listed'}.",
    ]
    if rec.primary_completion:
        lines.append(f"Primary completion date ({rec.primary_completion.kind.value.lower()}, "
                     f"sponsor-reported): {rec.primary_completion.raw}.")
    lines += trial_results_lines(rec.raw)
    published = datetime.combine(rec.last_update_posted, datetime.min.time(),
                                 rec.fetched_at.tzinfo) if rec.last_update_posted \
        else rec.fetched_at
    return Document(ticker=ticker, tier=SourceTier.REGULATORY, published_at=published,
                    url=f"https://clinicaltrials.gov/study/{rec.nct_id}", title=rec.title,
                    text="\n\n".join(lines), doc_type="ctgov", entity_id=entity_id)


def load_local_document(path: Path, *, ticker: str, tier: SourceTier, published_at: datetime,
                        url: str, doc_type: str, entity_id: str | None = None) -> Document:
    """FDA briefing docs, press releases, saved filings: .pdf, .html/.htm, .txt/.md."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        text = "\n\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    elif suffix in {".html", ".htm"}:
        text = html_to_text(path.read_text(encoding="utf-8", errors="replace"))
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    return Document(ticker=ticker, tier=tier, published_at=published_at, url=url,
                    title=path.stem, text=text, doc_type=doc_type, entity_id=entity_id)


# --------------------------------------------------------------------------- ingest


def build_chunks(doc: Document, ontology: Ontology, cfg: RagConfig) -> list[Chunk]:
    if doc.published_at.tzinfo is None:
        raise ValueError("Document.published_at must be timezone-aware")
    pieces = chunk_text(doc.text, chunk_tokens=cfg.chunk_tokens,
                        overlap_tokens=cfg.chunk_overlap_tokens)
    doc_sha = doc.sha256
    out: list[Chunk] = []
    for i, piece in enumerate(pieces):
        if doc.entity_id:
            entities = {doc.entity_id}
        else:
            entities = ontology.find_mentions(piece, doc.ticker) or {company_entity(doc.ticker)}
        claims = extract_claims(piece)
        for eid in sorted(entities):
            out.append(Chunk(
                chunk_id=_sha(f"{doc_sha}:{eid}:{i}")[:32], text=piece, ticker=doc.ticker,
                entity_id=eid, tier=doc.tier, published_at=doc.published_at,
                sha256=_sha(piece), doc_sha256=doc_sha, url=doc.url, title=doc.title,
                doc_type=doc.doc_type, claims=claims,
            ))
    return out


def ingest_documents(docs: list[Document], *, index: VectorIndex, embedder: Embedder,
                     ontology: Ontology, cfg: RagConfig) -> list[Chunk]:
    chunks = [c for d in docs for c in build_chunks(d, ontology, cfg)]
    # Embed each distinct text once even if it is stored for several entities.
    texts = sorted({c.text for c in chunks})
    vecs = dict(zip(texts, embedder.embed(texts), strict=True))
    index.upsert(chunks, [vecs[c.text] for c in chunks])
    return chunks
