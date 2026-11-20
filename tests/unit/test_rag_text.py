"""Ontology, chunking and claim extraction (pure functions, no vector store)."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest
import yaml

from biotradeai.rag.chunking import chunk_text, estimate_tokens
from biotradeai.rag.claims import extract_claims
from biotradeai.rag.ontology import (
    Entity,
    Ontology,
    OntologyError,
    approve_proposal,
    list_proposals,
    load_ontology,
    normalize,
    propose,
)


def _onto() -> Ontology:
    return Ontology({
        "ENT-ONCO-101": Entity("ENT-ONCO-101", "ONCO", ("ONC-101", "oncotinib", "Oncora"),
                               ("XYZ-77",)),
        "ENT-ONCO-201": Entity("ENT-ONCO-201", "ONCO", ("ONC-201",)),
        "ENT-OTHR-1": Entity("ENT-OTHR-1", "OTHR", ("OT-1",)),
    })


# ------------------------------------------------------------------ ontology


@pytest.mark.parametrize("name", ["ONC-101", "onc 101", "ONC101", "Oncotinib", "ONCORA",
                                  "xyz_77", "XYZ-77"])
def test_all_names_resolve_to_one_entity(name: str) -> None:
    assert _onto().resolve(name) == "ENT-ONCO-101"


def test_unknown_name() -> None:
    assert _onto().resolve("ONC-102") is None
    with pytest.raises(OntologyError):
        _onto().require("ONC-102")


def test_duplicate_name_across_entities_rejected() -> None:
    with pytest.raises(OntologyError, match="maps to both"):
        Ontology({"ENT-A-1": Entity("ENT-A-1", "A", ("X-1",)),
                  "ENT-B-1": Entity("ENT-B-1", "B", ("x 1",))})


def test_find_mentions_word_boundaries_and_ticker_scope() -> None:
    o = _onto()
    text = "Data for ONC 101 and the older XYZ-77 program; ONC-1010 is unrelated; also OT-1."
    assert o.find_mentions(text) == {"ENT-ONCO-101", "ENT-OTHR-1"}
    assert o.find_mentions(text, "ONCO") == {"ENT-ONCO-101"}
    assert o.find_mentions("ONC-2010 only") == set()


def test_normalize() -> None:
    assert normalize(" Onc-101 ") == normalize("ONC101") == "onc101"


def test_proposal_requires_human_approval(tmp_path: Path) -> None:
    syn = tmp_path / "synonyms.yaml"
    syn.write_text("# header kept\nentities:\n  ENT-ONCO-101:\n    ticker: ONCO\n"
                   "    names: [ONC-101]\n", encoding="utf-8")
    props = tmp_path / "proposals.yaml"
    propose("oncotinib", "ENT-ONCO-101", ticker="ONCO", kind="name",
            rationale="INN assigned in 10-K", source="llm:test", path=props)
    # Proposal alone changes nothing.
    assert load_ontology(syn).resolve("oncotinib") is None
    assert list_proposals(props)[0]["status"] == "pending"
    approve_proposal(0, approver="human", proposals=props, synonyms=syn)
    assert load_ontology(syn).resolve("oncotinib") == "ENT-ONCO-101"
    assert syn.read_text(encoding="utf-8").startswith("# header kept")
    assert list_proposals(props)[0]["status"] == "approved"
    with pytest.raises(OntologyError):
        approve_proposal(0, approver="human", proposals=props, synonyms=syn)


def test_proposal_collision_rejected(tmp_path: Path) -> None:
    syn = tmp_path / "s.yaml"
    syn.write_text(yaml.safe_dump({"entities": {
        "ENT-A-1": {"ticker": "A", "names": ["A-1"]},
        "ENT-B-1": {"ticker": "B", "names": ["B-1"]}}}), encoding="utf-8")
    props = tmp_path / "p.yaml"
    propose("B-1", "ENT-A-1", ticker="A", kind="name", rationale="x", source="llm", path=props)
    with pytest.raises(OntologyError):
        approve_proposal(0, approver="h", proposals=props, synonyms=syn)


def test_repo_synonyms_file_loads() -> None:
    assert isinstance(load_ontology(), Ontology)


@pytest.mark.parametrize("body", [
    {"entities": {"bad-id": {"ticker": "A", "names": ["x"]}}},
    {"entities": {"ENT-A-1": {"ticker": "a", "names": ["x"]}}},
    {"entities": {"ENT-A-1": {"ticker": "A", "names": []}}},
])
def test_bad_synonym_files(tmp_path: Path, body: dict[str, object]) -> None:
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(body), encoding="utf-8")
    with pytest.raises(OntologyError):
        load_ontology(p)


# ------------------------------------------------------------------ chunking


def _para(n_words: int, tag: str) -> str:
    return " ".join(f"{tag}{i}" for i in range(n_words))


def test_chunks_respect_size_and_overlap() -> None:
    text = "\n\n".join(_para(120, f"p{k}w") for k in range(10))  # ~1560 tokens
    chunks = chunk_text(text, chunk_tokens=400, overlap_tokens=50)
    assert len(chunks) >= 4
    assert all(estimate_tokens(c) <= 400 for c in chunks)
    for a, b in itertools.pairwise(chunks):
        tail = a.split()[-38:]  # 50 tokens / 1.3 ~= 38 words
        assert b.split()[:38] == tail


def test_long_paragraph_split_on_sentences() -> None:
    para = " ".join(f"Sentence {i} has a few words in it." for i in range(150))
    chunks = chunk_text(para, chunk_tokens=300, overlap_tokens=50)
    assert len(chunks) > 1 and all(estimate_tokens(c) <= 300 for c in chunks)


def test_small_text_single_chunk() -> None:
    assert chunk_text("One short paragraph.", chunk_tokens=400, overlap_tokens=50) == [
        "One short paragraph."]
    with pytest.raises(ValueError):
        chunk_text("x", chunk_tokens=100, overlap_tokens=100)


def test_chunking_is_deterministic() -> None:
    text = "\n\n".join(_para(90, f"p{k}w") for k in range(8))
    assert chunk_text(text, chunk_tokens=400, overlap_tokens=50) == chunk_text(
        text, chunk_tokens=400, overlap_tokens=50)


# ------------------------------------------------------------------ claims


@pytest.mark.parametrize(("text", "expected"), [
    ("In March the FDA placed a partial clinical hold on ONC-101.", {"clinical_hold": "active"}),
    ("The FDA lifted the clinical hold on ONC-101 in June.", {"clinical_hold": "lifted"}),
    ("The FDA placed a clinical hold in March. The clinical hold was lifted in June.",
     {"clinical_hold": "lifted"}),
    ("The trial met its primary endpoint.", {"primary_endpoint": "met"}),
    ("The study did not meet its primary endpoint.", {"primary_endpoint": "not_met"}),
    ("The FDA assigned a PDUFA target action date of June 15, 2026.",
     {"pdufa_date": "2026-06-15"}),
    ("We received a complete response letter from the FDA.", {"crl": "received"}),
])
def test_claims(text: str, expected: dict[str, str]) -> None:
    assert extract_claims(text) == expected


@pytest.mark.parametrize("text", [
    "The FDA may place a clinical hold on our programs.",
    "If the trial does not meet its primary endpoint, our stock could decline.",
    "We could receive a complete response letter.",
    "There is no assurance the trial met its primary endpoint as defined.",
])
def test_hypothetical_risk_language_is_not_a_claim(text: str) -> None:
    assert extract_claims(text) == {}
