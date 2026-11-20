"""PubMed abstracts via NCBI E-utilities (Tier 3 sources).

esearch:  /entrez/eutils/esearch.fcgi?db=pubmed&term=...&retmode=json
efetch:   /entrez/eutils/efetch.fcgi?db=pubmed&id=1,2&retmode=xml

NCBI asks for at most 3 requests/second without an API key (10 with one) and for
``tool`` and ``email`` parameters. We send ``tool=BioTradeAI``, plus ``email`` and
``api_key`` only if the user set NCBI_EMAIL / NCBI_API_KEY. Rate: data/netpolicy.py.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from datetime import UTC, datetime

import httpx

from biotradeai.data.http import HttpFetcher
from biotradeai.rag.ingest import Document
from biotradeai.schemas import SourceTier

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_MAX_REQUESTS_PER_SECOND = 3.0  # published ceiling without a key
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def _pub_date(article: ET.Element) -> datetime:
    for path in (".//ArticleDate", ".//JournalIssue/PubDate"):
        node = article.find(path)
        if node is None:
            continue
        year = node.findtext("Year")
        if not year or not year.isdigit():
            continue
        month_raw = (node.findtext("Month") or "1").strip()
        month = int(month_raw) if month_raw.isdigit() else _MONTHS.get(month_raw[:3].lower(), 1)
        day_raw = (node.findtext("Day") or "1").strip()
        day = int(day_raw) if day_raw.isdigit() else 1
        return datetime(int(year), month, day, tzinfo=UTC)
    return datetime(1970, 1, 1, tzinfo=UTC)


def parse_efetch(xml_text: str, *, ticker: str, entity_id: str | None) -> list[Document]:
    # Trusted host (NCBI) over HTTPS; Python's expat rejects entity-expansion bombs.
    root = ET.fromstring(xml_text)  # noqa: S314
    docs: list[Document] = []
    for art in root.iter("PubmedArticle"):
        pmid = art.findtext(".//MedlineCitation/PMID") or ""
        title_el = art.find(".//ArticleTitle")
        title = "".join(title_el.itertext()) if title_el is not None else ""
        parts = []
        for ab in art.iter("AbstractText"):
            label = ab.get("Label")
            body = "".join(ab.itertext()).strip()
            parts.append(f"{label}: {body}" if label else body)
        if not parts:
            continue
        docs.append(Document(
            ticker=ticker, tier=SourceTier.LITERATURE, published_at=_pub_date(art),
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", title=title,
            text=f"{title}\n\n" + "\n\n".join(parts), doc_type="pubmed",
            entity_id=entity_id, meta={"pmid": pmid},
        ))
    return docs


class PubMedClient:
    def __init__(self, *, transport: httpx.BaseTransport | None = None,
                 backoff_s: float = 1.0) -> None:
        self._f = HttpFetcher(BASE, source="ncbi", transport=transport, backoff_s=backoff_s)

    @staticmethod
    def _ident() -> dict[str, str]:
        p = {"tool": "BioTradeAI"}
        for env, key in (("NCBI_EMAIL", "email"), ("NCBI_API_KEY", "api_key")):
            if os.environ.get(env):
                p[key] = os.environ[env]
        return p

    def search(self, term: str, *, retmax: int = 20) -> list[str]:
        data = self._f.get_json("/esearch.fcgi", params={
            "db": "pubmed", "term": term, "retmode": "json", "retmax": retmax, "sort": "date",
            **self._ident()})
        return [str(x) for x in data.get("esearchresult", {}).get("idlist", [])]

    def fetch(self, pmids: list[str], *, ticker: str,
              entity_id: str | None = None) -> list[Document]:
        if not pmids:
            return []
        xml_text = self._f.get_text("/efetch.fcgi", params={
            "db": "pubmed", "id": ",".join(pmids), "retmode": "xml", **self._ident()})
        return parse_efetch(xml_text, ticker=ticker, entity_id=entity_id)
