"""Ingest + retrieve with the offline hashing embedder and an in-memory Chroma."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import chromadb
import pytest

from biotradeai.config import load_config, parse_config
from biotradeai.rag.embeddings import HashingEmbedder
from biotradeai.rag.evaluate import load_fixture, run_eval
from biotradeai.rag.index import VectorIndex, where_filter
from biotradeai.rag.ingest import Document, build_chunks, extract_item_1a, ingest_documents
from biotradeai.rag.ontology import Entity, Ontology, OntologyError, company_entity
from biotradeai.rag.retrieve import Retriever
from biotradeai.schemas import EvidenceStatus, SourceTier

CFG = load_config()
EMB = HashingEmbedder()
ONTO = Ontology({
    "ENT-ONCO-101": Entity("ENT-ONCO-101", "ONCO", ("ONC-101", "oncotinib"), ("XYZ-77",)),
    "ENT-ONCO-201": Entity("ENT-ONCO-201", "ONCO", ("ONC-201",)),
    "ENT-OTHR-1": Entity("ENT-OTHR-1", "OTHR", ("OT-1",)),
})


def _index() -> VectorIndex:
    return VectorIndex(f"t-{uuid.uuid4().hex[:12]}", client=chromadb.EphemeralClient())


def _d(text: str, *, ticker: str = "ONCO", tier: int = 2, when: str = "2025-01-01",
       entity: str | None = "ENT-ONCO-101", url: str = "https://e.org/x") -> Document:
    return Document(ticker=ticker, tier=SourceTier(tier),
                    published_at=datetime.fromisoformat(when).replace(tzinfo=UTC), url=url,
                    title="t", text=text, doc_type="test", entity_id=entity)


def _setup(docs: list[Document], cfg: Any = CFG) -> Retriever:
    idx = _index()
    ingest_documents(docs, index=idx, embedder=EMB, ontology=ONTO, cfg=cfg.rag)
    return Retriever(idx, EMB, ONTO, cfg)


HEPATIC = ("ONC-101 caused elevations of liver enzymes and drug-induced liver injury in "
           "several patients in the phase 2 trial.")


# ------------------------------------------------------------------ metadata & bleed


def test_chunk_metadata_tags() -> None:
    [c] = build_chunks(_d(HEPATIC), ONTO, CFG.rag)
    md = c.metadata()
    assert md["ticker"] == "ONCO" and md["entity_id"] == "ENT-ONCO-101"
    assert md["source_tier"] == 2 and len(str(md["sha256"])) == 64
    assert md["published_at"] == "2025-01-01T00:00:00+00:00"


def test_semantic_bleed_zero_chunks_from_other_ticker_or_entity() -> None:
    r = _setup([
        _d(HEPATIC),
        _d(HEPATIC.replace("ONC-101", "ONC-201"), entity="ENT-ONCO-201"),
        _d(HEPATIC.replace("ONC-101", "OT-1"), ticker="OTHR", entity="ENT-OTHR-1"),
        _d(HEPATIC, ticker="OTHR", entity="ENT-OTHR-1"),  # identical text, other ticker
    ])
    res = r.retrieve("drug-induced liver injury liver enzymes", ticker="ONCO", drug="ONC-101")
    assert res.hits
    assert all(h.chunk.ticker == "ONCO" and h.chunk.entity_id == "ENT-ONCO-101"
               for h in res.hits)
    assert all(s.ticker == "ONCO" and s.entity_id == "ENT-ONCO-101" for s in res.bundle.sources)


def test_wrong_ticker_for_drug_rejected() -> None:
    r = _setup([_d(HEPATIC)])
    with pytest.raises(OntologyError):
        r.retrieve("liver", ticker="OTHR", drug="ONC-101")


def test_renamed_drug_retrieves_old_code_risk_factor() -> None:
    old = _d("Risks related to XYZ-77. Two patients receiving XYZ-77 developed drug-induced "
             "liver injury meeting Hy's law.", entity=None, when="2022-03-01")
    r = _setup([old, _d("ONC-101 manufacturing scale-up is on track.")])
    res = r.retrieve("liver injury Hy's law", ticker="ONCO", drug="oncotinib")
    assert any("XYZ-77" in h.chunk.text for h in res.relevant)


def test_untagged_chunks_split_by_mention_or_company() -> None:
    doc = _d("ONC-101 and ONC-201 share a manufacturing site.\n\nWe may need to raise "
             "additional capital.", entity=None)
    chunks = build_chunks(doc, ONTO, CFG.rag)
    assert {c.entity_id for c in chunks} == {"ENT-ONCO-101", "ENT-ONCO-201"}
    lone = build_chunks(_d("We may need additional capital.", entity=None), ONTO, CFG.rag)
    assert {c.entity_id for c in lone} == {company_entity("ONCO")}


def test_point_in_time_filter_excludes_future_documents() -> None:
    r = _setup([_d(HEPATIC, when="2025-01-01"),
                _d(HEPATIC + " Updated analysis.", when="2026-01-01")])
    res = r.retrieve("liver injury", ticker="ONCO", drug="ONC-101",
                     as_of=datetime(2025, 6, 1, tzinfo=UTC))
    assert res.hits and all(h.chunk.published_at.year == 2025 for h in res.hits)
    with pytest.raises(ValueError):
        where_filter("ONCO", "ENT-ONCO-101", datetime(2025, 1, 1))


# ------------------------------------------------------------------ confidence


def test_confidence_two_tier2_chunks_ok() -> None:
    r = _setup([_d(HEPATIC), _d(HEPATIC + " Hepatic monitoring is required.")])
    res = r.retrieve("liver injury ONC-101", ticker="ONCO", drug="ONC-101")
    assert res.density == 1.0
    assert res.authority == pytest.approx(0.85)
    assert res.bundle.status is EvidenceStatus.OK


def test_single_chunk_is_inconclusive_by_density() -> None:
    res = _setup([_d(HEPATIC)]).retrieve("liver injury ONC-101", ticker="ONCO",
                                         drug="ONC-101")
    assert res.density == 0.5 and res.inconclusive


def test_press_only_evidence_is_inconclusive() -> None:
    res = _setup([_d(HEPATIC, tier=3), _d(HEPATIC + " More.", tier=3)]).retrieve(
        "liver injury ONC-101", ticker="ONCO", drug="ONC-101")
    assert res.authority == pytest.approx(0.6) and res.inconclusive


def test_no_relevant_evidence(raw_config: dict[str, Any]) -> None:
    raw_config["rag"]["min_similarity"] = 0.99
    res = _setup([_d(HEPATIC)], parse_config(raw_config)).retrieve(
        "completely unrelated words", ticker="ONCO", drug="ONC-101")
    assert res.inconclusive and "no relevant evidence" in res.reasons[0]


# ------------------------------------------------------------------ conflicts

HOLD = "In March the FDA placed a partial clinical hold on ONC-101 after a patient death."
LIFTED = "The FDA lifted the clinical hold on ONC-101 and dosing resumed."


def test_unresolved_tier1_vs_newer_tier2_is_inconclusive() -> None:
    res = _setup([
        _d(LIFTED, tier=1, when="2025-01-01", url="https://fda.example/1"),
        _d(HOLD, tier=2, when="2025-03-01"),
        _d(HEPATIC, tier=2), _d(HEPATIC + " x", tier=2),
    ]).retrieve("clinical hold status ONC-101", ticker="ONCO", drug="ONC-101")
    assert res.inconclusive
    assert any("unresolved clinical_hold" in r for r in res.reasons)
    assert "clinical_hold" not in res.claims


def test_newer_tier1_supersedes_older_tier2() -> None:
    res = _setup([
        _d(HOLD, tier=2, when="2025-01-01"),
        _d(LIFTED, tier=1, when="2025-03-01"),
    ]).retrieve("clinical hold ONC-101", ticker="ONCO", drug="ONC-101")
    assert res.claims["clinical_hold"].value == "lifted"
    assert res.claims["clinical_hold"].rule == "tier"
    assert all(c.resolved for c in res.conflicts)


def test_same_tier_conflict_most_recent_wins() -> None:
    res = _setup([
        _d(HOLD, tier=2, when="2025-01-01"),
        _d(LIFTED, tier=2, when="2025-05-01"),
    ]).retrieve("clinical hold ONC-101", ticker="ONCO", drug="ONC-101")
    rc = res.claims["clinical_hold"]
    assert rc.value == "lifted" and rc.rule == "recency" and rc.dissent


def test_tier2_beats_newer_tier3() -> None:
    res = _setup([
        _d(LIFTED, tier=2, when="2025-01-01"),
        _d(HOLD, tier=3, when="2025-06-01"),
    ]).retrieve("clinical hold ONC-101", ticker="ONCO", drug="ONC-101")
    assert res.claims["clinical_hold"].value == "lifted"


def test_claim_sources_are_in_bundle() -> None:
    res = _setup([_d(HOLD, tier=2), _d(HEPATIC), _d(HEPATIC + " y")]).retrieve(
        "liver injury", ticker="ONCO", drug="ONC-101")
    shas = {s.sha256 for s in res.bundle.sources}
    assert res.claims["clinical_hold"].evidence.sha256 in shas


# ------------------------------------------------------------------ item 1A


def test_extract_item_1a_skips_table_of_contents() -> None:
    body = " ".join(["risk"] * 40)
    text = (f"Table of contents Item 1A. Risk Factors 23 Item 1B. Unresolved Staff Comments 50 "
            f"... Item 1A. Risk Factors {body} Item 1B. Unresolved Staff Comments none "
            f"Item 2. Properties")
    out = extract_item_1a(text)
    assert out is not None and out.split() == ["risk"] * 40


# ------------------------------------------------------------------ eval harness


def test_eval_harness_runs_offline_with_zero_bleed() -> None:
    docs, keys, onto, queries = load_fixture()
    assert len(onto.entities) == 30 and len(keys) == len(docs)
    rep = run_eval(EMB, CFG, client=chromadb.EphemeralClient())
    assert rep.n_queries == len(queries) and rep.total_bleed == 0
    # The hashing embedder is only a smoke check; the real target is in tests/eval.
    assert rep.mean_recall > 0.5
