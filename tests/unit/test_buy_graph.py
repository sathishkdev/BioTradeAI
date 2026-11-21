"""Phase 5 acceptance: the Supervisor graph end to end on the scenario fixtures.

* $ONCO (cash $40M, burn $10M/q) runs end to end with the failure branch in EV
* conflict fixture -> INCONCLUSIVE
* dilution trap (1 month of cash) -> SKIP
* EV mismatch between Research and Risk is logged and blocks the trade
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from biotradeai.agents import research
from biotradeai.agents.base_rates import BaseRates, load_base_rates
from biotradeai.brokers.approval import ApprovalAuthority
from biotradeai.brokers.execution import ExecutionService
from biotradeai.brokers.fake_broker import FakeBroker
from biotradeai.config import GuardrailsConfig, load_config
from biotradeai.graph.buy_graph import (
    BuyGraphDeps,
    RunOutcome,
    build_buy_graph,
    get_outcome,
    resume_run,
    sqlite_checkpointer,
    start_run,
)
from biotradeai.guardrails.rules import compute_ev
from biotradeai.paths import REPO_ROOT
from biotradeai.scenario_fixtures import ScenarioFixture, load_fixture
from biotradeai.schemas import ApprovalToken, Decision, StagedOrder
from biotradeai.store.sqlite_store import SQLiteStore

FIX = REPO_ROOT / "data" / "fixtures" / "scenarios"
SECRET = b"s" * 32


@pytest.fixture(scope="module")
def cfg() -> GuardrailsConfig:
    return load_config()


@pytest.fixture(scope="module")
def rates() -> BaseRates:
    return load_base_rates()


@pytest.fixture
def store(tmp_path: Path) -> SQLiteStore:
    return SQLiteStore(tmp_path / "audit.db")


class Harness:
    def __init__(self, name: str, cfg: GuardrailsConfig, rates: BaseRates, store: SQLiteStore,
                 saver: Any | None = None) -> None:
        self.fx: ScenarioFixture = load_fixture(FIX / f"{name}.yaml", cfg)
        self.llm = self.fx.fake_llm()
        self.store = store
        self.broker = FakeBroker(
            equity=self.fx.inputs.portfolio.equity, last_equity=self.fx.inputs.portfolio.equity,
            quotes={self.fx.catalyst.ticker: (self.fx.inputs.market.bid,
                                              self.fx.inputs.market.ask)})
        self.authority = ApprovalAuthority(SECRET, ttl=timedelta(minutes=15))
        self.executed: list[StagedOrder] = []
        deps = BuyGraphDeps(cfg, rates, self.llm, self.llm, store, executor=self._execute,
                            clock=lambda: self.fx.catalyst.as_of)
        self.app = build_buy_graph(deps, saver or InMemorySaver())
        self._cfg = cfg

    def _execute(self, order: StagedOrder, token: ApprovalToken, run_id: str) -> dict[str, Any]:
        self.executed.append(order)
        r = ExecutionService(self.broker, self.store, self._cfg, self.authority).execute(
            order, token, run_id=run_id)
        return {"status": r.status.value, "reasons": list(r.reasons), "order_id": r.order_id}

    def run(self) -> RunOutcome:
        return start_run(self.app, self.fx.catalyst, self.fx.inputs)

    def events(self) -> list[str]:
        return [e.event_type for e in self.store.read_audit(run_id=self.fx.catalyst.run_id,
                                                            limit=1000)]


Make = Callable[[str], Harness]


@pytest.fixture
def make(cfg: GuardrailsConfig, rates: BaseRates, store: SQLiteStore) -> Make:
    return lambda name: Harness(name, cfg, rates, store)


# ------------------------------------------------------------------ Phase 2


def test_phase2_readout_runs_end_to_end_and_base_rates_alone_skip(
        cfg: GuardrailsConfig, rates: BaseRates, store: SQLiteStore) -> None:
    from biotradeai.backtest.harness import base_llm

    fx = load_fixture(FIX / "onco.yaml", cfg)
    cat = fx.catalyst.model_copy(update={"catalyst_event": "phase2_readout"})
    llm = base_llm()
    deps = BuyGraphDeps(cfg, rates, llm, llm, store, executor=lambda *a: {},
                        clock=lambda: cat.as_of)
    out = start_run(build_buy_graph(deps, InMemorySaver()), cat, fx.inputs)
    s = out.state
    assert s.scenarios is not None and s.risk_ev is not None
    assert sum(x.probability for x in s.scenarios.scenarios) == pytest.approx(1.0)
    assert s.research_ev == pytest.approx(s.risk_ev, abs=1e-12)
    # Phase 2 odds are poor (BIO 2021: 28.9% reach Phase 3): unadjusted EV is negative.
    assert s.risk_ev < 0 and s.decision is Decision.SKIP and not out.awaiting_approval


# ------------------------------------------------------------------ $ONCO


def test_onco_end_to_end_reaches_human_approval(make: Make) -> None:
    h = make("onco")
    out = h.run()
    s = out.state
    assert s.runway_months == pytest.approx(12.0)
    assert s.decision is Decision.BUY_CANDIDATE and out.awaiting_approval
    assert s.scenarios is not None and s.research_ev is not None and s.risk_ev is not None
    # failure branch is part of EV, and so is the residual of the pruned 'partial' branch
    failure = [x for x in s.scenarios.scenarios if x.label.startswith("failure")]
    assert failure and sum(x.probability for x in failure) == pytest.approx(0.272)
    assert any(x.is_residual for x in s.scenarios.scenarios)
    assert sum(x.probability for x in s.scenarios.scenarios) == pytest.approx(1.0)
    assert s.risk_ev == pytest.approx(compute_ev(s.scenarios))
    assert s.research_ev == pytest.approx(s.risk_ev, abs=1e-12)
    assert s.risk_ev == pytest.approx(0.29755216)
    o = s.staged_order
    assert o is not None and o.order_type.value == "limit" and o.qty == 499
    assert str(o.limit_price) == "10.02"
    assert h.executed == []  # nothing happens before a human answers


def test_onco_approved_executes_once_through_execution_service(make: Make) -> None:
    h = make("onco")
    out = h.run()
    assert out.state.staged_order is not None
    token = h.authority.issue(out.state.staged_order)
    out = resume_run(h.app, h.fx.catalyst.run_id, approved=True, operator="tester", token=token)
    assert out.execution is not None and out.execution["status"] == "placed"
    assert len(h.executed) == 1 and h.broker.calls.count("place_order") == 1
    ev = h.events()
    for name in ("graph.ingest", "tot.generate", "tot.critique", "tot.prune", "tot.finalize",
                 "risk.verdict", "supervisor.decision", "hitl.request", "hitl.response",
                 "execution.result"):
        assert name in ev, name
    assert h.store.verify_audit_chain()


def test_onco_rejected_by_human_sends_nothing(make: Make) -> None:
    h = make("onco")
    h.run()
    out = resume_run(h.app, h.fx.catalyst.run_id, approved=False, operator="tester",
                     token=None)
    assert out.execution is None and h.executed == []
    assert "place_order" not in h.broker.calls


def test_waiting_run_survives_restart(cfg: GuardrailsConfig, rates: BaseRates,
                                      store: SQLiteStore, tmp_path: Path) -> None:
    db = tmp_path / "ckpt.db"
    with sqlite_checkpointer(db) as saver:
        h1 = Harness("onco", cfg, rates, store, saver)
        out = h1.run()
        assert out.awaiting_approval
        order = out.state.staged_order
    assert order is not None
    with sqlite_checkpointer(db) as saver:  # "process restart"
        h2 = Harness("onco", cfg, rates, store, saver)
        assert get_outcome(h2.app, h2.fx.catalyst.run_id).awaiting_approval
        token = h2.authority.issue(order)
        out = resume_run(h2.app, h2.fx.catalyst.run_id, approved=True, operator="t",
                         token=token)
    assert out.execution is not None and out.execution["status"] == "placed"


# ------------------------------------------------------------------ routing


def test_conflict_routes_to_inconclusive_without_tot(make: Make) -> None:
    h = make("conflict")
    out = h.run()
    s = out.state
    assert s.decision is Decision.INCONCLUSIVE and not out.awaiting_approval
    assert s.tot_nodes == [] and s.scenarios is None and s.staged_order is None
    assert h.llm.calls == []  # no LLM spend on unresolved evidence
    assert s.evidence is not None and "clinical_hold" in (s.evidence.inconclusive_reason or "")


def test_dilution_trap_routes_to_skip(make: Make, cfg: GuardrailsConfig) -> None:
    h = make("dilution_trap")
    out = h.run()
    s = out.state
    assert s.runway_months == pytest.approx(1.0)
    assert s.decision is Decision.SKIP and s.staged_order is None
    assert any(r.startswith("runway") for r in s.veto_reasons)
    assert s.scenarios is not None
    res = [x for x in s.scenarios.scenarios if x.is_residual]
    pruned = [n for n in s.tot_nodes if n.pruned and n.depth == 1]
    assert {n.label for n in pruned} == {"success", "partial"}
    assert res[0].probability == pytest.approx(sum(n.probability for n in pruned))
    assert res[0].expected_return == cfg.tot.residual_scenario_return


def test_ev_mismatch_is_logged_and_blocks_trade(make: Make,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    real = research.tree_ev
    monkeypatch.setattr(research, "tree_ev", lambda nodes, cfg: real(nodes, cfg) + 0.05)
    h = make("onco")
    out = h.run()
    assert out.state.decision is Decision.SKIP and out.state.staged_order is None
    assert any(r.startswith("ev_match") for r in out.state.veto_reasons)
    assert "risk.ev_mismatch" in h.events()


def test_stale_quote_vetoes(cfg: GuardrailsConfig, rates: BaseRates,
                            store: SQLiteStore) -> None:
    h = Harness("onco", cfg, rates, store)
    later = h.fx.catalyst.as_of + timedelta(hours=cfg.ttl_hours + 1)
    deps = BuyGraphDeps(cfg, rates, h.llm, h.llm, store, clock=lambda: later)
    app = build_buy_graph(deps, InMemorySaver())
    out = start_run(app, h.fx.catalyst, h.fx.inputs)
    assert out.state.decision is Decision.SKIP
    assert any(r.startswith("ttl") for r in out.state.veto_reasons)


def test_inputs_for_another_ticker_rejected(make: Make, cfg: GuardrailsConfig) -> None:
    h = make("onco")
    other = load_fixture(FIX / "dilution_trap.yaml", cfg)
    with pytest.raises(ValueError, match="semantic bleed"):
        start_run(h.app, h.fx.catalyst, other.inputs)
