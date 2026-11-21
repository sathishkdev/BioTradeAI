"""The research agent's tools, with real I/O (``agents/research_agent.Toolbox``).

Every tool is locked to one ticker and one drug entity, chosen by code:

* ``search``: the local Chroma index through ``rag.retrieve.Retriever`` with the same
  hard ``ticker AND entity_id`` filter (and point-in-time ``as_of``).
* ``list_trials`` / ``fetch_trial``: ClinicalTrials.gov. A trial is fetched only if its
  intervention names this drug; it is then ingested as a Tier-1 document.
* ``fetch_pubmed``: PubMed abstracts that name the drug, ingested as Tier 3.

Fetched documents go through the same pipeline as the follow job
(``rag.sync.DocumentSync.add``): clean, chunk, tag, extract claims, embed, upsert into
Chroma, record in ``ingested_docs``. The agent then reads them with ``search`` like any
other evidence, so every citation maps to a stored chunk with a hash.

Fetch tools need the drug's watchlist entry (company, CIK) and are switched off in
backtests (``live=False``), where only the point-in-time index search is allowed.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from biotradeai.agents.research_agent import (
    EvidenceUpdate,
    FoundChunk,
    ToolboxFactory,
    TrialBrief,
)
from biotradeai.config import GuardrailsConfig
from biotradeai.data.http import FetchError
from biotradeai.inputs import excerpts_for
from biotradeai.paper.gather import QUERY
from biotradeai.rag.index import Hit
from biotradeai.rag.ingest import trial_to_document
from biotradeai.rag.ontology import Ontology
from biotradeai.rag.retrieve import Retriever, score_evidence
from biotradeai.rag.sync import DocumentSync, SyncReport
from biotradeai.schemas import CatalystState
from biotradeai.watchlist.model import Status, WatchEntry, Watchlist, drug_key

MAX_TRIALS_LISTED = 10
PUBMED_PER_FETCH = 5


def _found(h: Hit) -> FoundChunk:
    c = h.chunk
    return FoundChunk(c.sha256, float(h.similarity), int(c.tier),
                      c.published_at.date().isoformat(), c.doc_type, c.title, c.text)


class EvidenceToolbox:
    def __init__(self, *, retriever: Retriever, cfg: GuardrailsConfig, ticker: str,
                 entity_id: str, drug_names: Sequence[str], as_of: datetime | None,
                 sync: DocumentSync | None = None, entry: WatchEntry | None = None,
                 live: bool = True, run_id: str | None = None) -> None:
        if not drug_names:
            raise ValueError("the toolbox needs at least one drug name")
        self._r, self._cfg = retriever, cfg
        self.ticker, self.entity_id = ticker.upper(), entity_id
        self.drug_names = tuple(drug_names)
        self._as_of, self._sync, self._entry, self._run_id = as_of, sync, entry, run_id
        self.live = bool(live and sync is not None and entry is not None)
        self._keys = {drug_key(n) for n in self.drug_names}

    # ------------------------------------------------------------------ local

    def baseline(self) -> list[FoundChunk]:
        return self.search(QUERY.format(drug=self.drug_names[0]))

    def search(self, query: str) -> list[FoundChunk]:
        r = self._r.retrieve(query, ticker=self.ticker, entity_id=self.entity_id,
                             as_of=self._as_of)
        return [_found(h) for h in r.hits]

    def known_trials(self) -> list[str]:
        return list(self._entry.trial_ids) if self._entry else []

    def score(self, shas: Sequence[str], baseline_n: int) -> EvidenceUpdate:
        """Confidence over everything found, computed here in code.

        Similarity is re-measured against the fixed decision question (the baseline
        query), so the agent cannot lift confidence by searching for loosely related
        but authoritative text. The baseline chunks keep the first places; the agent's
        finds fill the remaining ``agent.max_evidence`` slots by that similarity.
        """
        hits = self._r.hits_for(QUERY.format(drug=self.drug_names[0]), list(shas),
                                ticker=self.ticker, entity_id=self.entity_id,
                                as_of=self._as_of)
        base = [hits[s] for s in shas[:baseline_n] if s in hits]
        more = sorted((hits[s] for s in shas[baseline_n:] if s in hits),
                      key=lambda h: (-h.similarity, h.chunk.sha256))
        claim_chunks = self._r.claim_chunks(ticker=self.ticker, entity_id=self.entity_id,
                                            as_of=self._as_of)
        s = score_evidence(base + more, claim_chunks, ticker=self.ticker,
                           entity_id=self.entity_id, cfg=self._cfg,
                           limit=self._cfg.agent.max_evidence)
        wanted = {x.sha256 for x in s.bundle.sources}
        return EvidenceUpdate(s.bundle, excerpts_for((h.chunk for h in hits.values()), wanted),
                              {k: v.value for k, v in s.claims.items()},
                              s.authority, s.density, s.confidence)

    # ------------------------------------------------------------------ internet

    def _names_drug(self, names: Sequence[str]) -> bool:
        text = drug_key(" ".join(names))
        return any(k and k in text for k in self._keys)

    def list_trials(self) -> list[TrialBrief]:
        if not self.live or self._sync is None:
            raise PermissionError("fetch tools are off for this run")
        out: list[TrialBrief] = []
        seen: set[str] = set()
        for name in self.drug_names[:3]:
            for rec in self._sync.ctgov.search(intervention=name, page_size=20, max_pages=1):
                if rec.nct_id in seen or not self._names_drug(rec.interventions):
                    continue
                seen.add(rec.nct_id)
                out.append(TrialBrief(rec.nct_id, rec.title, rec.phases, rec.overall_status,
                                      rec.sponsor))
                if len(out) >= MAX_TRIALS_LISTED:
                    return out
        return out

    def fetch_trial(self, nct_id: str) -> str:
        if not self.live or self._sync is None or self._entry is None:
            raise PermissionError("fetch tools are off for this run")
        rec = self._sync.ctgov.get_study(nct_id)
        if not self._names_drug((*rec.interventions, rec.title)):
            return (f"refused: {nct_id} does not list {self.drug_names[0]} as an intervention"
                    " (not added)")
        rep = SyncReport(self._entry.id)
        doc = trial_to_document(rec, self.ticker, self.entity_id)
        added = self._sync.add(self._entry, doc, "ctgov-agent", rep)
        self._audit("agent.fetch", {"source": "ctgov", "nct_id": nct_id, "added": added,
                                    "url": doc.url, "doc_sha256": doc.sha256})
        return (f"added {nct_id} (tier 1, last update {doc.published_at.date()}); use "
                "search_evidence to read it" if added else
                f"{nct_id} is already in the evidence store; use search_evidence")

    def fetch_pubmed(self, query: str) -> str:
        if not self.live or self._sync is None or self._entry is None:
            raise PermissionError("fetch tools are off for this run")
        pm = self._sync.pubmed
        if pm is None:
            return "PubMed is not configured"
        term = "(" + " OR ".join(f'"{n}"' for n in self.drug_names[:5]) + ")"
        if query:
            term += f" AND ({query})"
        try:
            ids = pm.search(term, retmax=PUBMED_PER_FETCH)
            docs = pm.fetch(ids, ticker=self.ticker, entity_id=self.entity_id)
        except FetchError as exc:
            return f"error: PubMed request failed: {str(exc)[:160]}"
        rep = SyncReport(self._entry.id)
        added = sum(self._sync.add(self._entry, d, "pubmed-agent", rep) for d in docs)
        self._audit("agent.fetch", {"source": "pubmed", "term": term, "found": len(docs),
                                    "added": added})
        return (f"{len(docs)} abstract(s) found, {added} new added (tier 3); use "
                "search_evidence to read them" if docs else "no abstracts found")

    def _audit(self, event: str, payload: dict[str, object]) -> None:
        if self._sync is not None:
            self._sync.store.append_audit(event, dict(payload), ticker=self.ticker,
                                          run_id=self._run_id)


def toolbox_factory(*, retriever: Retriever, cfg: GuardrailsConfig, ontology: Ontology,
                    watchlist: Watchlist, sync: DocumentSync | None) -> ToolboxFactory:
    """Tools for one catalyst. Fetch tools only for drugs you approved on the watchlist."""

    def make(c: CatalystState) -> EvidenceToolbox | None:
        ent = ontology.entities.get(c.entity_id)
        if ent is None or ent.ticker != c.ticker:
            return None
        entry = next((e for e in watchlist.with_status(Status.APPROVED, Status.PAUSED)
                      if e.entity_id == c.entity_id and e.ticker == c.ticker), None)
        names = tuple(entry.drug_names) if entry and entry.drug_names else ent.all_names
        return EvidenceToolbox(retriever=retriever, cfg=cfg, ticker=c.ticker,
                               entity_id=c.entity_id, drug_names=names, as_of=c.as_of,
                               sync=sync, entry=entry, live=sync is not None,
                               run_id=c.run_id)
    return make
