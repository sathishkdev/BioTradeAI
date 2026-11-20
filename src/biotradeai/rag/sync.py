"""Automatic document collection for approved watchlist drugs (``rag sync``).

Sources, all through the guarded fetcher (allowlist, polite rates, budgets):
  Tier 1  ClinicalTrials.gov records for the drug's trials (incl. posted results)
  Tier 2  SEC: Item 1A risk factors of the latest 10-K / 10-Qs; recent 8-Ks with exhibits
  Tier 3  PubMed abstracts that name the drug

Every document keeps its REAL publication date (filing date, CT.gov last-update date,
PubMed date), never the download time, so point-in-time retrieval stays honest.
Each (document, drug) pair is ingested once (``ingested_docs`` fingerprint table).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from functools import partial
from typing import Any

from biotradeai.config import GuardrailsConfig
from biotradeai.data.clinicaltrials import ClinicalTrialsClient
from biotradeai.data.edgar import EdgarClient, Filing
from biotradeai.data.http import FetchError
from biotradeai.data.pacing import paced
from biotradeai.data.secindex import submission_text
from biotradeai.rag.embeddings import Embedder
from biotradeai.rag.index import VectorIndex
from biotradeai.rag.ingest import (
    Document,
    extract_item_1a,
    html_to_text,
    ingest_documents,
    trial_to_document,
)
from biotradeai.rag.ontology import load_ontology
from biotradeai.rag.pubmed import PubMedClient
from biotradeai.schemas import SourceTier
from biotradeai.store.sqlite_store import SQLiteStore
from biotradeai.watchlist.model import WatchEntry

EIGHT_K_LOOKBACK_DAYS = 180
PUBMED_MAX = 10


@dataclass
class SyncReport:
    entry: str
    added: dict[str, int] = field(default_factory=dict)
    skipped: int = 0
    errors: list[str] = field(default_factory=list)

    def bump(self, source: str) -> None:
        self.added[source] = self.added.get(source, 0) + 1


def _at(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


class DocumentSync:
    def __init__(self, *, store: SQLiteStore, cfg: GuardrailsConfig, index: VectorIndex,
                 embedder: Embedder, edgar: EdgarClient | None, ctgov: ClinicalTrialsClient,
                 pubmed: PubMedClient | None, sleep: Callable[[float], None] = time.sleep
                 ) -> None:
        self.store, self.cfg, self.index, self.embedder = store, cfg, index, embedder
        self.edgar, self.ctgov, self.pubmed = edgar, ctgov, pubmed
        self._sleep = sleep

    # ------------------------------------------------------------------ core

    def add(self, e: WatchEntry, doc: Document, source: str, rep: SyncReport) -> bool:
        """Ingest one document unless this (document, drug) pair was ingested before."""
        if not doc.text.strip():
            return False
        key = doc.entity_id or f"AUTO:{e.ticker}"
        if not self.store.mark_ingested(doc.sha256, key, ticker=e.ticker, source=source,
                                        url=doc.url, published_at=doc.published_at.isoformat()):
            rep.skipped += 1
            return False
        ingest_documents([doc], index=self.index, embedder=self.embedder,
                         ontology=load_ontology(), cfg=self.cfg.rag)
        rep.bump(source)
        return True

    def sec_8k(self, e: WatchEntry, *, path: str, filed: date, rep: SyncReport) -> str:
        """Fetch one 8-K full submission (with exhibits), ingest it, return its text."""
        if self.edgar is None:
            return ""
        text = submission_text(self.edgar.archive_text(path))
        url = f"https://www.sec.gov/Archives/{path.removeprefix('/Archives/')}"
        doc = Document(ticker=e.ticker, tier=SourceTier.SEC, published_at=_at(filed), url=url,
                       title=f"{e.company} 8-K {filed}", text=text, doc_type="8-K")
        self.add(e, doc, "sec-8k", rep)
        return text

    def sec_periodic(self, e: WatchEntry, f: Filing, rep: SyncReport) -> None:
        if self.edgar is None:
            return
        text = html_to_text(self.edgar.fetch_document(e.cik, f))
        risk = extract_item_1a(text)
        if risk:
            doc = Document(ticker=e.ticker, tier=SourceTier.SEC,
                           published_at=_at(f.filing_date), url=f.url(e.cik),
                           title=f"{e.company} {f.form} {f.filing_date} Item 1A",
                           text=risk, doc_type=f.form)
            self.add(e, doc, "sec-1a", rep)

    def trial(self, e: WatchEntry, nct_id: str, rep: SyncReport) -> dict[str, Any]:
        rec = self.ctgov.get_study(nct_id)
        self.add(e, trial_to_document(rec, e.ticker, e.entity_id), "ctgov", rep)
        return rec.raw

    def pubmed_refresh(self, e: WatchEntry, rep: SyncReport) -> None:
        if self.pubmed is None:
            return
        term = " OR ".join(f'"{n}"' for n in e.drug_names[:5])
        ids = self.pubmed.search(term, retmax=PUBMED_MAX)
        for doc in self.pubmed.fetch(ids, ticker=e.ticker, entity_id=e.entity_id):
            self.add(e, doc, "pubmed", rep)

    # ------------------------------------------------------------------ jobs

    def initial(self, e: WatchEntry, *, today: date) -> SyncReport:
        """First sync after approval: latest 10-K/10-Qs, 180 days of 8-Ks, trials, PubMed."""
        rep = SyncReport(e.id)
        steps: list[Callable[[], object]] = []
        if self.edgar is not None:
            subs = self.edgar.submissions(e.cik)
            periodic = subs.by_form("10-K")[:1] + subs.by_form("10-Q")[:2]
            steps += [partial(self.sec_periodic, e, f, rep) for f in periodic]
            since = today - timedelta(days=EIGHT_K_LOOKBACK_DAYS)
            steps += [partial(self.sec_8k, e, rep=rep, filed=f.filing_date,
                              path=f"edgar/data/{int(e.cik)}/{f.accession_number}.txt")
                      for f in subs.by_form("8-K") if f.filing_date >= since]
        steps += [partial(self.trial, e, nct, rep) for nct in e.trial_ids]
        steps.append(partial(self.pubmed_refresh, e, rep))
        self._run(steps, rep, window_s=0.0)
        self.store.append_audit("rag.sync", {"entry": e.id, "kind": "initial",
                                             "added": rep.added, "skipped": rep.skipped,
                                             "errors": rep.errors}, ticker=e.ticker)
        return rep

    def _run(self, steps: list[Callable[[], object]], rep: SyncReport, *,
             window_s: float) -> None:
        for step in paced(steps, window_s=window_s, sleep=self._sleep):
            try:
                step()
            except FetchError as exc:  # one bad document never stops the sync
                rep.errors.append(str(exc)[:200])
