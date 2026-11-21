"""Phase 8: paper-trading runner, live input gathering, daily review and gate.

No real keys and no network: FakeBroker, or the real AlpacaBroker against a mock
Alpaca server (dummy key strings), plus scripted LLM replies.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import chromadb
import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from biotradeai.agents.base_rates import BaseRates, load_base_rates
from biotradeai.brokers.alpaca_broker import AlpacaBroker
from biotradeai.brokers.approval import ApprovalAuthority
from biotradeai.brokers.base import Broker, BrokerOrder, Position
from biotradeai.brokers.fake_broker import FakeBroker
from biotradeai.config import GuardrailsConfig, Mode, load_config
from biotradeai.data.catalysts import Catalyst
from biotradeai.data.edgar import parse_company_facts
from biotradeai.data.outcomes import Outcome
from biotradeai.data.runway import compute_runway
from biotradeai.events import CompletionDateChange, FilingEvent, ReadoutResult
from biotradeai.graph.sell_graph import SellPosition, SellState
from biotradeai.paper.gather import GatherError, LiveGatherer
from biotradeai.paper.review import daily_review, paper_gate
from biotradeai.paper.runner import PaperRunner
from biotradeai.paths import REPO_ROOT
from biotradeai.rag.embeddings import HashingEmbedder
from biotradeai.rag.index import VectorIndex
from biotradeai.rag.ingest import Document, ingest_documents
from biotradeai.rag.ontology import Entity, Ontology
from biotradeai.rag.retrieve import Retriever
from biotradeai.scenario_fixtures import ScenarioFixture, load_fixture
from biotradeai.schemas import CatalystState, OrderSide, OrderType, ResearchInputs, SourceTier
from biotradeai.store.sqlite_store import SQLiteStore

FIX = REPO_ROOT / "data" / "fixtures" / "scenarios" / "onco.yaml"
SECRET = b"k" * 32
URL = "https://example.com/catalyst"
ONCO_CAT = Catalyst("ONCO", "ONC-101", "PHASE3_READOUT", date(2027, 1, 15), URL)


@pytest.fixture(scope="module")
def cfg() -> GuardrailsConfig:
    return load_config()


@pytest.fixture(scope="module")
def rates() -> BaseRates:
    return load_base_rates()


@pytest.fixture
def store(tmp_path: Path) -> SQLiteStore:
    return SQLiteStore(tmp_path / "a.db")


@pytest.fixture
def fx(cfg: GuardrailsConfig) -> ScenarioFixture:
    return load_fixture(FIX, cfg)


class StubGatherer:
    """Returns the ONCO fixture inputs (already validated contracts)."""

    def __init__(self, fx: ScenarioFixture) -> None:
        self.fx = fx
        self.calls = 0

    def gather(self, c: Catalyst, *, now: datetime,
               run_id: str) -> tuple[CatalystState, ResearchInputs]:
        self.calls += 1
        return self.fx.catalyst.model_copy(update={"run_id": run_id}), self.fx.inputs


class StubFilings:
    def __init__(self, events: list[FilingEvent] | None = None) -> None:
        self.events = events or []

    def poll(self, ticker: str, *, since: date) -> list[FilingEvent]:
        return [e for e in self.events if e.ticker == ticker]


class Approver:
    def __init__(self, answer: bool) -> None:
        self.answer = answer
        self.seen: list[dict[str, Any]] = []

    def __call__(self, order: Any, pack: dict[str, Any]) -> bool:
        self.seen.append(pack)
        return self.answer


def _runner(cfg: GuardrailsConfig, rates: BaseRates, store: SQLiteStore, fx: ScenarioFixture,
            broker: Broker, approver: Approver, *, now: datetime | None = None,
            filings: StubFilings | None = None) -> PaperRunner:
    llm = fx.fake_llm()
    clock_now = now or fx.catalyst.as_of
    return PaperRunner(cfg=cfg, store=store, broker=broker, gatherer=StubGatherer(fx),
                       rates=rates, generator=llm, critic=llm,
                       authority=ApprovalAuthority(SECRET, ttl=timedelta(minutes=15)),
                       approver=approver, checkpointer=InMemorySaver(), filings=filings,
                       clock=lambda: clock_now)


def _fake(**kw: Any) -> FakeBroker:
    return FakeBroker(quotes={"ONCO": (Decimal("9.98"), Decimal("10.02"))}, **kw)


# ------------------------------------------------------------------ buy cycle


def test_buy_cycle_places_approved_limit_order(cfg: GuardrailsConfig, rates: BaseRates,
                                               store: SQLiteStore, fx: ScenarioFixture) -> None:
    broker, ok = _fake(), Approver(True)
    rep = _runner(cfg, rates, store, fx, broker, ok).cycle([ONCO_CAT], [])
    assert rep.evaluated == [("ONCO", "BUY_CANDIDATE")]
    assert rep.orders[0]["status"] == "placed" and rep.breaches == [] and rep.halted is None
    placed = list(broker.orders.values())
    assert len(placed) == 1 and placed[0].order_type is OrderType.LIMIT
    assert placed[0].side is OrderSide.BUY and placed[0].qty == 499
    assert ok.seen[0]["kind"] == "entry" and ok.seen[0]["ev"] > 0.2
    saved = SellPosition.model_validate(store.load_sell_state("ONCO"))
    assert saved.state is SellState.HOLDING and saved.entry_price == Decimal("10.02")
    assert store.verify_audit_chain()


def test_open_buy_order_blocks_re_entry(cfg: GuardrailsConfig, rates: BaseRates,
                                        store: SQLiteStore, fx: ScenarioFixture) -> None:
    broker = _fake()
    r = _runner(cfg, rates, store, fx, broker, Approver(True))
    r.cycle([ONCO_CAT], [])
    r.cycle([ONCO_CAT], [])  # the unfilled buy counts as exposure
    assert broker.calls.count("place_order") == 1


def test_evaluated_once_per_day(cfg: GuardrailsConfig, rates: BaseRates, store: SQLiteStore,
                                fx: ScenarioFixture) -> None:
    r = _runner(cfg, rates, store, fx, _fake(), Approver(False))
    r.cycle([ONCO_CAT], [])
    rep = r.cycle([ONCO_CAT], [])
    assert ("ONCO", "already evaluated today") in rep.skipped


def test_human_rejection_places_nothing(cfg: GuardrailsConfig, rates: BaseRates,
                                        store: SQLiteStore, fx: ScenarioFixture) -> None:
    broker = _fake()
    rep = _runner(cfg, rates, store, fx, broker, Approver(False)).cycle([ONCO_CAT], [])
    assert rep.orders[0]["status"] == "human_rejected"
    assert "place_order" not in broker.calls


def test_tripped_killswitch_halts_cycle(cfg: GuardrailsConfig, rates: BaseRates,
                                       store: SQLiteStore, fx: ScenarioFixture) -> None:
    store.set_killswitch(True, "test", changed_by="test")
    g = StubGatherer(fx)
    r = _runner(cfg, rates, store, fx, _fake(), Approver(True))
    r._gatherer = g
    rep = r.cycle([ONCO_CAT], [])
    assert rep.halted and g.calls == 0


def test_drawdown_trips_killswitch(cfg: GuardrailsConfig, rates: BaseRates,
                                   store: SQLiteStore, fx: ScenarioFixture) -> None:
    broker = _fake(equity=Decimal("96000"), last_equity=Decimal("100000"))
    rep = _runner(cfg, rates, store, fx, broker, Approver(True)).cycle([ONCO_CAT], [])
    assert rep.halted and store.get_killswitch().tripped


def test_refuses_non_paper_mode(cfg: GuardrailsConfig, rates: BaseRates, store: SQLiteStore,
                                fx: ScenarioFixture) -> None:
    shadow = cfg.model_copy(update={"mode": Mode.SHADOW})
    with pytest.raises(RuntimeError, match="paper"):
        _runner(shadow, rates, store, fx, _fake(), Approver(True)).cycle([ONCO_CAT], [])


def test_breach_trips_killswitch(cfg: GuardrailsConfig, rates: BaseRates, store: SQLiteStore,
                                 fx: ScenarioFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    r = _runner(cfg, rates, store, fx, _fake(), Approver(True))
    monkeypatch.setattr(r, "_post_trade_check", lambda *a: ["simulated breach"])
    rep = r.cycle([ONCO_CAT], [])
    assert rep.breaches == ["simulated breach"] and store.get_killswitch().tripped


def test_post_trade_check_flags_oversized_order(cfg: GuardrailsConfig, rates: BaseRates,
                                                store: SQLiteStore,
                                                fx: ScenarioFixture) -> None:
    from biotradeai.schemas import StagedOrder
    r = _runner(cfg, rates, store, fx, _fake(), Approver(True))
    big = StagedOrder(ticker="ONCO", side=OrderSide.BUY, qty=10_000, order_type=OrderType.LIMIT,
                      limit_price=Decimal("10"))
    out = r._post_trade_check(big, fx.inputs, {"ONCO"})
    assert out and "cap" in out[0]
    assert any(e.event_type == "guardrail.breach" for e in store.read_audit())


# ------------------------------------------------------------------ sell cycle


def _holding(store: SQLiteStore, *, qty: int = 100, state: SellState = SellState.HOLDING,
             ) -> FakeBroker:
    broker = _fake()
    broker.positions["ONCO"] = Position("ONCO", qty, Decimal("10"), Decimal(10 * qty))
    store.save_sell_state("ONCO", SellPosition(
        ticker="ONCO", qty=qty, entry_price=Decimal("10"), catalyst_event="phase3_readout",
        catalyst_date=date(2027, 1, 15), state=state).model_dump(mode="json"))
    return broker


def _sells(broker: FakeBroker) -> list[BrokerOrder]:
    return [o for o in broker.orders.values() if o.side is OrderSide.SELL]


def test_profit_target_exit_is_approved_and_placed(cfg: GuardrailsConfig, rates: BaseRates,
                                                   store: SQLiteStore,
                                                   fx: ScenarioFixture) -> None:
    broker = _holding(store)
    broker.quotes["ONCO"] = (Decimal("13.00"), Decimal("13.02"))
    ok = Approver(True)
    now = datetime(2026, 12, 1, 15, tzinfo=UTC)
    rep = _runner(cfg, rates, store, fx, broker, ok, now=now).cycle([ONCO_CAT], [])
    assert rep.exits[0]["trigger"] == "profit_target" and rep.exits[0]["status"] == "placed"
    s = _sells(broker)
    assert len(s) == 1 and s[0].qty == 100 and s[0].limit_price == Decimal("13.00")
    assert ok.seen[0]["kind"] == "exit"


def test_crl_outcome_sells_everything(cfg: GuardrailsConfig, rates: BaseRates,
                                      store: SQLiteStore, fx: ScenarioFixture) -> None:
    broker = _holding(store)
    broker.quotes["ONCO"] = (Decimal("4.00"), Decimal("4.05"))
    now = datetime(2027, 1, 15, 21, tzinfo=UTC)
    crl = Outcome("ONCO", "ONC-101", date(2027, 1, 15), ReadoutResult.CRL, URL)
    rep = _runner(cfg, rates, store, fx, broker, Approver(True), now=now).cycle([ONCO_CAT],
                                                                                 [crl])
    assert [(e["trigger"], e["qty"]) for e in rep.exits] == [("negative_readout", 100)]
    assert _sells(broker)[0].limit_price == Decimal("4.00")


def test_dilutive_filing_sells(cfg: GuardrailsConfig, rates: BaseRates, store: SQLiteStore,
                               fx: ScenarioFixture) -> None:
    broker = _holding(store)
    now = datetime(2026, 12, 1, 15, tzinfo=UTC)
    f = FilingEvent(ticker="ONCO", ts=now - timedelta(days=1), form="S-3", accession="a-1",
                    dilutive=True, reason="shelf", url=URL)
    rep = _runner(cfg, rates, store, fx, broker, Approver(True), now=now,
                  filings=StubFilings([f])).cycle([ONCO_CAT], [])
    assert rep.exits[0]["trigger"] == "dilution" and rep.exits[0]["status"] == "placed"


def test_rejected_exit_is_flagged_next_cycle(cfg: GuardrailsConfig, rates: BaseRates,
                                             store: SQLiteStore, fx: ScenarioFixture) -> None:
    broker = _holding(store)
    broker.quotes["ONCO"] = (Decimal("13.00"), Decimal("13.02"))
    now = datetime(2026, 12, 1, 15, tzinfo=UTC)
    r = _runner(cfg, rates, store, fx, broker, Approver(False), now=now)
    first = r.cycle([ONCO_CAT], [])
    assert first.exits[0]["status"] == "human_rejected" and _sells(broker) == []
    rep = r.cycle([ONCO_CAT], [])
    assert any("review manually" in why for _, why in rep.skipped)
    assert any(e.event_type == "paper.exit_unfilled" for e in store.read_audit())


def test_open_sell_orders_reduce_sellable_qty(cfg: GuardrailsConfig, rates: BaseRates,
                                              store: SQLiteStore, fx: ScenarioFixture) -> None:
    broker = _holding(store)
    broker.orders["x"] = BrokerOrder("x", "ONCO", OrderSide.SELL, 60, OrderType.LIMIT,
                                     Decimal("11"), "new")
    broker.quotes["ONCO"] = (Decimal("13.00"), Decimal("13.02"))
    now = datetime(2026, 12, 1, 15, tzinfo=UTC)
    rep = _runner(cfg, rates, store, fx, broker, Approver(True), now=now).cycle([ONCO_CAT], [])
    assert rep.exits[0]["qty"] == 40


def test_position_without_catalyst_is_unmanaged(cfg: GuardrailsConfig, rates: BaseRates,
                                                store: SQLiteStore, fx: ScenarioFixture) -> None:
    broker = _fake()
    broker.positions["ZZZ"] = Position("ZZZ", 5, Decimal("3"), Decimal("15"))
    rep = _runner(cfg, rates, store, fx, broker, Approver(True)).cycle([], [])
    assert ("ZZZ", "position has no catalyst in catalysts.csv") in rep.skipped


# ------------------------------------------------------------------ review + gate


def test_daily_review_and_gate(cfg: GuardrailsConfig, rates: BaseRates, store: SQLiteStore,
                               fx: ScenarioFixture) -> None:
    _runner(cfg, rates, store, fx, _fake(), Approver(True)).cycle([ONCO_CAT], [])
    r = daily_review(store, datetime.now(UTC).date())
    assert r.chain_ok and r.decisions["BUY_CANDIDATE"] == 1 and len(r.placed) == 1
    assert r.events["paper.cycle_done"] == 1
    g = paper_gate(store, cfg)
    assert g.paper_days == 1 and not g.eligible  # needs 90 days
    one_day = cfg.model_copy(update={"live_gate": cfg.live_gate.model_copy(
        update={"min_paper_days": 1})})
    assert paper_gate(store, one_day).eligible
    store.append_audit("guardrail.breach", {"breach": "x"}, ticker="ONCO")
    assert not paper_gate(store, one_day).eligible


# ------------------------------------------------------------------ real AlpacaBroker, mock server


def _alpaca_server(posted: list[dict[str, Any]]) -> httpx.MockTransport:
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    def h(req: httpx.Request) -> httpx.Response:
        assert req.headers["APCA-API-KEY-ID"] == "dummy-key-id"  # dummy, not a real key
        path = req.url.path
        if path == "/v2/account":
            return httpx.Response(200, json={"account_number": "PA0000SECRET", "equity": "100000",
                                             "last_equity": "100000", "cash": "100000",
                                             "buying_power": "200000"})
        if path == "/v2/positions":
            return httpx.Response(200, json=[])
        if path == "/v2/orders" and req.method == "GET":
            return httpx.Response(200, json=[])
        if path == "/v2/orders" and req.method == "POST":
            body = json.loads(req.content)
            posted.append(body)
            return httpx.Response(200, json={"id": "ord-1", "symbol": body["symbol"],
                                             "side": body["side"], "qty": body["qty"],
                                             "type": body["type"],
                                             "limit_price": body["limit_price"],
                                             "status": "accepted",
                                             "client_order_id": body["client_order_id"]})
        if path.endswith("/quotes/latest"):
            return httpx.Response(200, json={"symbol": "ONCO", "quote": {
                "bp": 9.98, "ap": 10.02, "bs": 3, "as": 4, "t": now}})
        return httpx.Response(404, json={"message": f"unexpected {path}"})
    return httpx.MockTransport(h)


def test_full_cycle_against_mock_alpaca(cfg: GuardrailsConfig, rates: BaseRates,
                                        store: SQLiteStore, fx: ScenarioFixture) -> None:
    posted: list[dict[str, Any]] = []
    broker = AlpacaBroker("dummy-key-id", "dummy-secret", transport=_alpaca_server(posted))
    rep = _runner(cfg, rates, store, fx, broker, Approver(True)).cycle([ONCO_CAT], [])
    assert rep.orders[0]["status"] == "placed"
    assert posted == [{"symbol": "ONCO", "qty": "499", "side": "buy", "type": "limit",
                       "time_in_force": "day", "limit_price": "10.02", "extended_hours": False,
                       "client_order_id": posted[0]["client_order_id"]}]
    dump = json.dumps([e.payload for e in store.read_audit()])
    assert "PA0000SECRET" not in dump and "dummy-secret" not in dump


# ------------------------------------------------------------------ live gatherer


def _retriever(cfg: GuardrailsConfig) -> Retriever:
    c = cfg.model_copy(update={"rag": cfg.rag.model_copy(update={"min_similarity": -1.0})})
    emb = HashingEmbedder()
    onto = Ontology({"ENT-ONCO-ONC101": Entity("ENT-ONCO-ONC101", "ONCO", ("ONC-101",))})
    index = VectorIndex("t", client=chromadb.EphemeralClient())
    docs = [Document("ONCO", SourceTier.REGULATORY, datetime(2026, 9, 1, tzinfo=UTC), URL, "t",
                     "ONC-101 Phase 2 met its primary endpoint. No new safety signals.",
                     "ctgov", "ENT-ONCO-ONC101"),
            Document("ONCO", SourceTier.SEC, datetime(2026, 10, 1, tzinfo=UTC), URL, "t",
                     "ONC-101 Phase 3 topline data expected in January 2027.", "10-Q",
                     "ENT-ONCO-ONC101")]
    ingest_documents(docs, index=index, embedder=emb, ontology=onto, cfg=c.rag)
    return Retriever(index, emb, onto, c)


def test_live_gatherer_builds_inputs(cfg: GuardrailsConfig) -> None:
    facts = json.loads((REPO_ROOT / "data" / "fixtures" / "edgar" / "companyfacts" /
                        "ONCO.json").read_text(encoding="utf-8"))
    cf = parse_company_facts(facts, datetime.now(UTC))
    broker = _fake()
    broker.orders["o"] = BrokerOrder("o", "XYZ", OrderSide.BUY, 10, OrderType.LIMIT,
                                     Decimal("5"), "new")
    g = LiveGatherer(retriever=_retriever(cfg), cfg=cfg, broker=broker,
                     runway=lambda t, now: compute_runway(cf, lookback_quarters=4,
                                                          as_of=now.date()))
    now = datetime(2026, 11, 20, 15, tzinfo=UTC)
    state, inp = g.gather(ONCO_CAT, now=now, run_id="r1")
    assert state.entity_id == "ENT-ONCO-ONC101" and state.catalyst_event == "phase3_readout"
    assert len(inp.evidence.sources) == 2 and inp.claims.get("primary_endpoint") == "met"
    assert inp.runway.months > 0 and inp.runway.source_accns
    assert inp.portfolio.open_tickers == ["XYZ"]  # open buy orders count as exposure


def test_gatherer_rejects_unsupported_event_and_dead_quote(cfg: GuardrailsConfig) -> None:
    broker = _fake()
    g = LiveGatherer(retriever=_retriever(cfg), cfg=cfg, broker=broker,
                     runway=lambda t, now: (_ for _ in ()).throw(RuntimeError("no facts")))
    now = datetime(2026, 11, 20, 15, tzinfo=UTC)
    with pytest.raises(GatherError, match="no base rates"):
        g.gather(Catalyst("ONCO", "ONC-101", "ADCOM", date(2027, 1, 1), URL), now=now,
                 run_id="r")
    with pytest.raises(GatherError, match="runway"):
        g.gather(ONCO_CAT, now=now, run_id="r")
    with pytest.raises(GatherError, match="runway"):  # Phase 2 passes the event gate
        g.gather(Catalyst("ONCO", "ONC-101", "PHASE2_READOUT", date(2027, 1, 15), URL),
                 now=now, run_id="r")


class StubTrials:
    def __init__(self, events: list[CompletionDateChange]) -> None:
        self.events = events

    def poll(self, ticker: str) -> list[CompletionDateChange]:
        return [e for e in self.events if e.ticker == ticker]


def test_slipped_trial_triggers_time_stop_exit(cfg: GuardrailsConfig, rates: BaseRates,
                                               store: SQLiteStore, fx: ScenarioFixture) -> None:
    broker = _holding(store)
    now = datetime(2026, 12, 1, 15, tzinfo=UTC)
    slip = CompletionDateChange(ticker="ONCO", ts=now, nct_id="NCT09000001",
                                old_date=date(2026, 12, 1), new_date=date(2027, 7, 1))
    r = _runner(cfg, rates, store, fx, broker, Approver(True), now=now)
    r._trials = StubTrials([slip])
    rep = r.cycle([ONCO_CAT], [])
    assert [(e["trigger"], e["status"]) for e in rep.exits] == [("time_stop", "placed")]
    assert _sells(broker)[0].qty == 100


# ------------------------------------------------------------------ review regressions


def test_early_readout_in_window_still_sells(cfg: GuardrailsConfig, rates: BaseRates,
                                             store: SQLiteStore, fx: ScenarioFixture) -> None:
    """Window events use the window END as the date; a readout early in the window,
    after the position was opened, must still reach the sell machine."""
    broker = _fake()
    broker.positions["ONCO"] = Position("ONCO", 100, Decimal("10"), Decimal(1000))
    store.save_sell_state("ONCO", SellPosition(
        ticker="ONCO", qty=100, entry_price=Decimal("10"), catalyst_event="phase3_readout",
        catalyst_date=date(2027, 3, 31), opened_on=date(2026, 12, 1)).model_dump(mode="json"))
    now = datetime(2027, 1, 21, 15, tzinfo=UTC)
    early = Outcome("ONCO", "ONC-101", date(2027, 1, 20), ReadoutResult.NEGATIVE, URL)
    rep = _runner(cfg, rates, store, fx, broker, Approver(True), now=now).cycle(
        [Catalyst("ONCO", "ONC-101", "PHASE3_READOUT", date(2027, 3, 31), URL)], [early])
    assert [(e["trigger"], e["qty"]) for e in rep.exits] == [("negative_readout", 100)]


class FailingTrials:
    def poll(self, ticker: str) -> list[CompletionDateChange]:
        from biotradeai.data.http import FetchError
        raise FetchError("ctgov is switched off until tomorrow (UTC)")


def test_failed_poll_does_not_abort_cycle(cfg: GuardrailsConfig, rates: BaseRates,
                                          store: SQLiteStore, fx: ScenarioFixture) -> None:
    broker = _holding(store)
    broker.quotes["ONCO"] = (Decimal("13.00"), Decimal("13.02"))
    r = _runner(cfg, rates, store, fx, broker, Approver(True),
                now=datetime(2026, 12, 1, 15, tzinfo=UTC))
    r._trials = FailingTrials()
    rep = r.cycle([ONCO_CAT], [])
    assert rep.exits[0]["trigger"] == "profit_target"  # other events still processed
    assert any("switched off" in e for e in rep.errors)
