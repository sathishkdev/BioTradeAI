"""Phase 7: point-in-time harness, metrics and report (offline: hashing embedder)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chromadb
import pytest

from biotradeai.agents.base_rates import BaseRates, load_base_rates
from biotradeai.agents.llm import FakeLLMClient, LLMMessage
from biotradeai.backtest import harness
from biotradeai.backtest.harness import (
    PointInTimeViolationError,
    base_llm,
    build_index,
    leak_shas,
    load_cases,
    run_case,
)
from biotradeai.backtest.metrics import breaches, brier, compute_metrics, grounding
from biotradeai.backtest.runner import run_backtest
from biotradeai.config import GuardrailsConfig, load_config
from biotradeai.rag.embeddings import HashingEmbedder
from biotradeai.rag.retrieve import Retriever
from biotradeai.schemas import Decision
from biotradeai.store.sqlite_store import SQLiteStore


@pytest.fixture(scope="module")
def cfg() -> GuardrailsConfig:
    # The toy hashing embedder scores low cosine similarities; relax only the similarity
    # floor so routing is exercised offline. Every other threshold is the real config.
    c = load_config()
    return c.model_copy(update={"rag": c.rag.model_copy(update={"min_similarity": -1.0})})


@pytest.fixture(scope="module")
def rates() -> BaseRates:
    return load_base_rates()


@pytest.fixture
def store(tmp_path: Path) -> SQLiteStore:
    return SQLiteStore(tmp_path / "a.db")


def bullish_llm() -> FakeLLMClient:
    """Scripted model that leans on evidence E1 at depth 1; the critic accepts."""
    def reply(msgs: Sequence[LLMMessage]) -> str:
        text = msgs[-1].content
        if "Proposed adjustments:" in text:
            return json.dumps({"reviews": [
                {"node": "success", "verdict": "accept", "reason": "E1"},
                {"node": "failure", "verdict": "accept", "reason": "E1"}]})
        if "Depth: 1" in text:
            return json.dumps({"adjustments": [
                {"node": "success", "adjustment": 0.15, "justification": "j", "evidence": ["E1"]},
                {"node": "failure", "adjustment": -0.15, "justification": "j",
                 "evidence": ["E1"]}]})
        return json.dumps({"adjustments": []})
    return FakeLLMClient(reply, model="bullish")


def _retriever(cfg: GuardrailsConfig) -> tuple[Retriever, list[harness.BacktestCase]]:
    cases = load_cases()
    emb = HashingEmbedder()
    index, onto = build_index(cases, embedder=emb, cfg=cfg, client=chromadb.EphemeralClient())
    return Retriever(index, emb, onto, cfg), cases


# ------------------------------------------------------------------ point in time


def test_every_case_has_a_future_leak_document() -> None:
    for c in load_cases():
        assert leak_shas(c), c.id
        assert all(d.published_at > c.decision_at for d in c.docs if d.leak)


def test_future_document_is_never_returned(cfg: GuardrailsConfig, rates: BaseRates,
                                           store: SQLiteStore) -> None:
    retriever, cases = _retriever(cfg)
    for c in cases:
        # Query with the leak's own wording: it would be the top hit without as_of.
        leak = next(d for d in c.docs if d.leak)
        r = retriever.retrieve(leak.text, ticker=c.ticker, entity_id=c.entity_id,
                               as_of=c.decision_at)
        assert all(h.chunk.published_at <= c.decision_at for h in r.hits)
        assert all(h.chunk.doc_sha256 not in leak_shas(c) for h in r.hits)
        unfiltered = retriever.retrieve(leak.text, ticker=c.ticker, entity_id=c.entity_id)
        assert any(h.chunk.doc_sha256 in leak_shas(c) for h in unfiltered.hits)


def test_harness_guard_catches_a_leaky_retriever(cfg: GuardrailsConfig, rates: BaseRates,
                                                 store: SQLiteStore,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    retriever, cases = _retriever(cfg)
    real = retriever.retrieve

    def leaky(query: str, **kw: Any) -> Any:
        kw["as_of"] = None  # simulate a broken point-in-time filter
        return real(leak_query, **kw)

    leak_query = next(d for d in cases[0].docs if d.leak).text
    monkeypatch.setattr(retriever, "retrieve", leaky)
    with pytest.raises(PointInTimeViolationError):
        run_case(cases[0], retriever=retriever, cfg=cfg, rates=rates, generator=base_llm(),
                 critic=base_llm(), store=store)


# ------------------------------------------------------------------ full run + metrics


def test_full_backtest_meets_structural_targets(cfg: GuardrailsConfig, rates: BaseRates,
                                                store: SQLiteStore, tmp_path: Path) -> None:
    llm = bullish_llm()
    run = run_backtest(load_cases(), cfg=cfg, rates=rates, embedder=HashingEmbedder(),
                       generator=llm, critic=llm, store=store, out_dir=tmp_path,
                       client=chromadb.EphemeralClient())
    m = run.metrics
    assert run.leaks == []
    assert m.breaches == [] and m.groundedness == 1.0
    assert m.escalation_accuracy == 1.0 and m.false_escalations == []
    assert m.latency_max_s <= 45
    by = {r.case.id: r for r in run.results}
    assert by["bt09"].state.decision is Decision.INCONCLUSIVE
    assert by["bt10"].state.decision is Decision.INCONCLUSIVE
    assert by["bt11"].state.decision is Decision.SKIP  # 1 month of cash
    assert m.decisions.get("BUY_CANDIDATE", 0) >= 1  # the buy path is exercised too
    assert m.brier is not None and m.brier_baseline is not None
    # report written as markdown + json
    assert run.report is not None and run.report.exists()
    js = json.loads(run.report.with_suffix(".json").read_text(encoding="utf-8"))
    assert js["checks"]["guardrail_compliance"] is True
    assert "Synthetic" in run.report.read_text(encoding="utf-8")
    assert any(e.event_type == "backtest.run" for e in store.read_audit())


def test_brier_math() -> None:
    assert brier([]) is None
    assert brier([(0.5, 1), (0.5, 0)]) == pytest.approx(0.25)
    assert brier([(1.0, 1), (0.0, 0)]) == 0.0
    assert brier([(0.8, 0)]) == pytest.approx(0.64)


def _buy_result(cfg: GuardrailsConfig, rates: BaseRates,
                store: SQLiteStore) -> harness.CaseResult:
    retriever, cases = _retriever(cfg)
    llm = bullish_llm()
    r = run_case(cases[0], retriever=retriever, cfg=cfg, rates=rates, generator=llm,
                 critic=llm, store=store)
    assert r.state.decision is Decision.BUY_CANDIDATE
    return r


def test_breach_detector_is_independent_of_risk_agent(cfg: GuardrailsConfig,
                                                      rates: BaseRates,
                                                      store: SQLiteStore) -> None:
    r = _buy_result(cfg, rates, store)
    assert breaches(r, cfg) == []
    # Pretend the Risk Agent let a low-EV trade through: the detector must flag it.
    assert r.state.scenarios is not None
    bad_sc = r.state.scenarios.model_copy(update={"scenarios": [
        s.model_copy(update={"expected_return": -0.5}) for s in r.state.scenarios.scenarios]})
    r.state = r.state.model_copy(update={"scenarios": bad_sc})
    found = breaches(r, cfg)
    assert any("EV" in b for b in found)


def test_grounding_flags_unevidenced_adjustment(cfg: GuardrailsConfig, rates: BaseRates,
                                                store: SQLiteStore) -> None:
    r = _buy_result(cfg, rates, store)
    assert grounding(r).grounded == grounding(r).total
    nodes = [n.model_copy(update={"evidence_ids": []}) if n.label == "success" else n
             for n in r.state.tot_nodes]
    r.state = r.state.model_copy(update={"tot_nodes": nodes})
    g = grounding(r)
    assert g.grounded == g.total - 1 and "success" in g.ungrounded[0]
    assert compute_metrics([r], cfg, rates).groundedness < 1.0
