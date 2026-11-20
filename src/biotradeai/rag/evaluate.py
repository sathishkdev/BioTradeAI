"""Retrieval evaluation on the labeled fixture set (data/fixtures/rag)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from biotradeai.config import GuardrailsConfig
from biotradeai.paths import data_dir
from biotradeai.rag.embeddings import Embedder
from biotradeai.rag.index import VectorIndex
from biotradeai.rag.ingest import Document, ingest_documents
from biotradeai.rag.ontology import Ontology, load_ontology
from biotradeai.rag.retrieve import Retriever
from biotradeai.schemas import SourceTier

FIXTURE_DIR = data_dir() / "fixtures" / "rag"


@dataclass(frozen=True)
class QueryResult:
    query_id: str
    recall: float
    bleed: int
    missing: tuple[str, ...]


@dataclass(frozen=True)
class EvalReport:
    embedder: str
    top_k: int
    n_queries: int
    mean_recall: float
    total_bleed: int
    per_query: tuple[QueryResult, ...]

    def passed(self, target: float) -> bool:
        return self.mean_recall >= target and self.total_bleed == 0


def load_fixture(directory: Path = FIXTURE_DIR) -> tuple[list[Document], dict[str, str],
                                                        Ontology, list[dict[str, Any]]]:
    """Return (documents, doc_key -> doc sha256, ontology, queries)."""
    raw = json.loads((directory / "corpus.json").read_text(encoding="utf-8"))
    docs: list[Document] = []
    keys: dict[str, str] = {}
    for d in raw:
        doc = Document(ticker=d["ticker"], tier=SourceTier(d["tier"]),
                       published_at=datetime.fromisoformat(d["published_at"]), url=d["url"],
                       title=d["title"], text=d["text"], doc_type=d["doc_type"],
                       entity_id=d["entity_id"], meta={"key": d["key"]})
        docs.append(doc)
        keys[d["key"]] = doc.sha256
    queries = json.loads((directory / "queries.json").read_text(encoding="utf-8"))
    return docs, keys, load_ontology(directory / "synonyms.yaml"), queries


def run_eval(embedder: Embedder, cfg: GuardrailsConfig, *, client: Any | None = None,
             directory: Path = FIXTURE_DIR) -> EvalReport:
    import chromadb

    docs, keys, onto, queries = load_fixture(directory)
    index = VectorIndex(f"eval-{embedder.name}", client=client or chromadb.EphemeralClient())
    ingest_documents(docs, index=index, embedder=embedder, ontology=onto, cfg=cfg.rag)
    retriever = Retriever(index, embedder, onto, cfg)
    results: list[QueryResult] = []
    for q in queries:
        r = retriever.retrieve(q["query"], ticker=q["ticker"], drug=q["drug"])
        got = {h.chunk.doc_sha256 for h in r.hits}
        want = {keys[k]: k for k in q["relevant"]}
        hit = [k for sha, k in want.items() if sha in got]
        bleed = sum(1 for h in r.hits
                    if h.chunk.ticker != r.ticker or h.chunk.entity_id != r.entity_id)
        results.append(QueryResult(q["id"], len(hit) / len(want), bleed,
                                   tuple(k for k in want.values() if k not in hit)))
    mean = sum(x.recall for x in results) / len(results)
    return EvalReport(embedder.name, cfg.rag.top_k, len(results), mean,
                      sum(x.bleed for x in results), tuple(results))
