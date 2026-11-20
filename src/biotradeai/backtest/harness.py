"""Point-in-time backtest harness (Phase 7).

For each case it ingests ALL of the case's documents (including ones published after
the decision date) into a fresh in-memory vector index. It then runs the real
retriever with ``as_of = decision time`` and the real buy graph at simulated time.
Any retrieved chunk dated after ``as_of`` raises ``PointInTimeViolationError``. That is a
second guard on top of the retriever's own metadata filter.

The graph stops at the human-approval interrupt. Backtests never execute anything.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from langgraph.checkpoint.memory import InMemorySaver

from biotradeai.agents.base_rates import BaseRates
from biotradeai.agents.llm import FakeLLMClient, LLMClient
from biotradeai.config import GuardrailsConfig
from biotradeai.graph.buy_graph import BuyGraphDeps, build_buy_graph, start_run
from biotradeai.inputs import inputs_from_retrieval
from biotradeai.rag.embeddings import Embedder
from biotradeai.rag.index import VectorIndex
from biotradeai.rag.ingest import Document, ingest_documents
from biotradeai.rag.ontology import Entity, Ontology
from biotradeai.rag.retrieve import RetrievalResult, Retriever
from biotradeai.schemas import CatalystState, ResearchInputs, RunwaySnapshot, SourceTier
from biotradeai.store.sqlite_store import SQLiteStore

FIXTURE = Path(__file__).resolve().parents[3] / "data" / "fixtures" / "backtest" / "cases.yaml"
QUERY = "{drug} clinical trial results primary endpoint safety clinical hold"


class PointInTimeViolationError(RuntimeError):
    pass


@dataclass(frozen=True)
class CaseDoc:
    tier: int
    published_at: datetime
    url: str
    doc_type: str
    text: str
    leak: bool = False


@dataclass(frozen=True)
class BacktestCase:
    id: str
    ticker: str
    drug: str
    entity_id: str
    catalyst_event: str
    catalyst_date: date
    decision_at: datetime
    outcome: str  # "success" | "failure"
    expect_conflict: bool
    runway: dict[str, Any]
    market: dict[str, Any]
    portfolio: dict[str, Any]
    docs: tuple[CaseDoc, ...]


def load_cases(path: Path = FIXTURE) -> list[BacktestCase]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    out = []
    for c in raw["cases"]:
        docs = tuple(CaseDoc(int(d["tier"]), datetime.fromisoformat(str(d["published_at"])),
                             str(d["url"]), str(d["doc_type"]), str(d["text"]),
                             bool(d.get("leak", False))) for d in c["docs"])
        out.append(BacktestCase(
            id=str(c["id"]), ticker=str(c["ticker"]), drug=str(c["drug"]),
            entity_id=str(c["entity_id"]), catalyst_event=str(c["catalyst_event"]),
            catalyst_date=date.fromisoformat(str(c["catalyst_date"])),
            decision_at=datetime.fromisoformat(str(c["decision_at"])),
            outcome=str(c["outcome"]), expect_conflict=bool(c["expect_conflict"]),
            runway=dict(c["runway"]), market=dict(c["market"]),
            portfolio=dict(c["portfolio"]), docs=docs))
    return out


@dataclass
class CaseResult:
    case: BacktestCase
    state: CatalystState
    inputs: ResearchInputs
    retrieval: RetrievalResult
    risk: dict[str, Any] | None
    latency_s: float


def base_llm() -> FakeLLMClient:
    """'No adjustment' model: the tree stays exactly at base rates (the baseline)."""
    return FakeLLMClient(lambda msgs: '{"adjustments": [], "reviews": []}', model="base-rates")


def _ontology(cases: list[BacktestCase]) -> Ontology:
    return Ontology({c.entity_id: Entity(c.entity_id, c.ticker, (c.drug,)) for c in cases})


def build_index(cases: list[BacktestCase], *, embedder: Embedder, cfg: GuardrailsConfig,
                client: Any | None = None) -> tuple[VectorIndex, Ontology]:
    import chromadb

    onto = _ontology(cases)
    index = VectorIndex(f"backtest-{embedder.name}", client=client or chromadb.EphemeralClient())
    docs = [Document(ticker=c.ticker, tier=SourceTier(d.tier), published_at=d.published_at,
                     url=d.url, title=f"{c.id} {d.doc_type}", text=d.text, doc_type=d.doc_type,
                     entity_id=c.entity_id)
            for c in cases for d in c.docs]
    ingest_documents(docs, index=index, embedder=embedder, ontology=onto, cfg=cfg.rag)
    return index, onto


def leak_shas(case: BacktestCase) -> set[str]:
    """Document hashes the harness must never surface for this case."""
    return {Document(case.ticker, SourceTier(d.tier), d.published_at, d.url, "", d.text,
                     d.doc_type).sha256
            for d in case.docs if d.published_at > case.decision_at}


def run_case(case: BacktestCase, *, retriever: Retriever, cfg: GuardrailsConfig,
             rates: BaseRates, generator: LLMClient, critic: LLMClient, store: SQLiteStore,
             timer: Callable[[], float] = time.perf_counter) -> CaseResult:
    t0 = timer()
    r = retriever.retrieve(QUERY.format(drug=case.drug), ticker=case.ticker,
                           entity_id=case.entity_id, as_of=case.decision_at)
    late = [h.chunk.doc_sha256 for h in r.hits if h.chunk.published_at > case.decision_at]
    late += [s.sha256 for s in r.bundle.sources if s.published_at > case.decision_at]
    if late:
        raise PointInTimeViolationError(f"{case.id}: retrieved documents dated after "
                                   f"{case.decision_at}: {late[:3]}")
    rw = case.runway
    cash, burn = Decimal(str(rw["cash"])), Decimal(str(rw["quarterly_burn"]))
    months = float(cash / burn * 3) if burn > 0 else 999.0
    runway = RunwaySnapshot(months=months, cash=cash, quarterly_burn=burn,
                            observed_at=case.decision_at,
                            source_accns=[str(a) for a in rw.get("source_accns", [])])
    pf = case.portfolio
    inputs = inputs_from_retrieval(
        r, runway=runway, bid=Decimal(str(case.market["bid"])),
        ask=Decimal(str(case.market["ask"])), quote_at=case.decision_at,
        equity=Decimal(str(pf["equity"])), open_tickers=list(pf.get("open_tickers", [])),
        existing_value=Decimal(str(pf.get("existing_value", 0))), now=case.decision_at)
    catalyst = CatalystState(run_id=f"backtest-{case.id}", ticker=case.ticker,
                             entity_id=case.entity_id, catalyst_event=case.catalyst_event,
                             catalyst_date=case.catalyst_date, as_of=case.decision_at)
    deps = BuyGraphDeps(cfg, rates, generator, critic, store, executor=None,
                        clock=lambda: case.decision_at)
    out = start_run(build_buy_graph(deps, InMemorySaver()), catalyst, inputs)
    return CaseResult(case, out.state, inputs, r, out.risk, timer() - t0)
