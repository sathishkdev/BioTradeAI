"""Phase 6 acceptance: replayed event streams fire exactly the expected exits, every sell
passes the order-type guardrail, and every exit is audit-logged."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from biotradeai.config import GuardrailsConfig, load_config
from biotradeai.events import (
    CatalystOutcome,
    CompletionDateChange,
    FilingEvent,
    PriceTick,
    ReadoutResult,
    classify_filing,
)
from biotradeai.graph.sell_graph import (
    SellEngine,
    SellPosition,
    SellState,
    Trigger,
    months_between,
    scale_out_qty,
    sell_limit,
    step,
)
from biotradeai.guardrails import rules
from biotradeai.paths import REPO_ROOT
from biotradeai.schemas import OrderSide, OrderType
from biotradeai.sell_replay import load_replay
from biotradeai.store.sqlite_store import SQLiteStore

FIX = REPO_ROOT / "data" / "fixtures" / "sell"
REPLAYS = sorted(FIX.glob("*.yaml"))
T0 = datetime(2026, 12, 1, 15, tzinfo=UTC)


@pytest.fixture(scope="module")
def cfg() -> GuardrailsConfig:
    return load_config()


@pytest.fixture
def store(tmp_path: Path) -> SQLiteStore:
    return SQLiteStore(tmp_path / "a.db")


def _pos(**kw: object) -> SellPosition:
    base: dict[str, object] = {"ticker": "ONCO", "qty": 100, "entry_price": Decimal("10"),
                               "catalyst_event": "phase3_readout",
                               "catalyst_date": date(2027, 1, 15)}
    return SellPosition.model_validate({**base, **kw})


def _tick(bid: str, ts: datetime = T0) -> PriceTick:
    b = Decimal(bid)
    return PriceTick(ticker="ONCO", ts=ts, bid=b, ask=b + Decimal("0.02"))


# ------------------------------------------------------------------ replay acceptance


def test_there_is_a_fixture_for_every_trigger() -> None:
    fired = {x.trigger for p in REPLAYS for x in load_replay(p).expected}
    assert fired == set(Trigger)


@pytest.mark.parametrize("path", REPLAYS, ids=lambda p: p.stem)
def test_replay_fires_exactly_the_expected_exits(path: Path, cfg: GuardrailsConfig,
                                                 store: SQLiteStore) -> None:
    r = load_replay(path)
    final, exits = SellEngine(cfg, store).process(r.position, r.events, run_id=r.name)
    got = [(e.trigger, e.qty, e.order.limit_price if e.order else None) for e in exits]
    want = [(x.trigger, x.qty, x.limit_price) for x in r.expected]
    assert got == want
    # every sell is a LIMIT order that passes the guardrails, and no shorting
    held = r.position.qty
    for e in exits:
        assert e.order is not None
        assert e.order.side is OrderSide.SELL and e.order.order_type is OrderType.LIMIT
        assert rules.check_order_type(e.order, cfg).ok
        assert rules.check_sell_quantity(e.order, held_qty=held).ok
        held -= e.qty
    assert held == final.qty >= 0
    # every exit audit-logged
    audited = [a for a in store.read_audit(run_id=r.name) if a.event_type == "sell.exit"]
    assert [a.payload["trigger"] for a in audited] == [t.value for t, _, _ in got]
    assert store.verify_audit_chain()
    if exits and final.qty == 0:
        assert final.state is SellState.CLOSED


def test_state_is_persisted(cfg: GuardrailsConfig, store: SQLiteStore) -> None:
    r = load_replay(FIX / "positive_trailing.yaml")
    events = r.events[:3]  # stop before the trailing stop fires
    final, _ = SellEngine(cfg, store).process(r.position, events)
    saved = SellPosition.model_validate(store.load_sell_state("ONCO"))
    assert saved == final and saved.state is SellState.TRAILING and saved.qty == 251


# ------------------------------------------------------------------ boundaries


@pytest.mark.parametrize(("bid", "fires"), [("12.99", False), ("13.00", True), ("13.01", True)])
def test_profit_target_boundary(cfg: GuardrailsConfig, bid: str, fires: bool) -> None:
    _, out = step(_pos(), _tick(bid), cfg)
    assert bool(out) is fires


@pytest.mark.parametrize(("bid", "fires"), [("8.99", True), ("9.00", True), ("9.01", False)])
def test_trailing_stop_boundary(cfg: GuardrailsConfig, bid: str, fires: bool) -> None:
    pos = _pos(state=SellState.TRAILING, high_water=Decimal("10"), last_bid=Decimal("10"))
    _, out = step(pos, _tick(bid, T0 + timedelta(days=60)), cfg)
    assert bool(out) is fires
    if fires:
        assert out[0].trigger is Trigger.TRAILING_STOP


@pytest.mark.parametrize(("days", "fires"), [(182, False), (183, True)])
def test_time_stop_fires_only_when_delay_exceeds(cfg: GuardrailsConfig, days: int,
                                                 fires: bool) -> None:
    base = date(2026, 12, 1)
    assert (months_between(base, base + timedelta(days=days)) > 6) is fires
    pos = _pos(baseline_completion=base, last_bid=Decimal("9"))
    ev = CompletionDateChange(ticker="ONCO", ts=T0, nct_id="NCT09990001", old_date=base,
                              new_date=base + timedelta(days=days))
    _, out = step(pos, ev, cfg)
    assert bool(out) is fires


def test_first_completion_date_sets_baseline(cfg: GuardrailsConfig) -> None:
    ev = CompletionDateChange(ticker="ONCO", ts=T0, nct_id="NCT09990001", old_date=None,
                              new_date=date(2027, 9, 1))
    pos, out = step(_pos(), ev, cfg)
    assert out == [] and pos.baseline_completion == date(2027, 9, 1)


@pytest.mark.parametrize("result", [ReadoutResult.NEGATIVE, ReadoutResult.CRL,
                                    ReadoutResult.MIXED])
def test_bad_readouts_sell_everything(cfg: GuardrailsConfig, result: ReadoutResult) -> None:
    ev = CatalystOutcome(ticker="ONCO", ts=T0, result=result, source_url="u")
    pos, out = step(_pos(last_bid=Decimal("3")), ev, cfg)
    assert out[0].qty == 100 and pos.qty == 0 and pos.state is SellState.CLOSED


def test_exit_without_a_quote_is_flagged_not_priced(cfg: GuardrailsConfig,
                                                    store: SQLiteStore) -> None:
    ev = CatalystOutcome(ticker="ONCO", ts=T0, result=ReadoutResult.CRL, source_url="u")
    _, out = SellEngine(cfg, store).process(_pos(), [ev])
    assert out[0].order is None
    assert store.read_audit()[-2].payload["needs_quote"] is True


def test_closed_position_ignores_everything(cfg: GuardrailsConfig) -> None:
    pos = _pos(qty=0, state=SellState.CLOSED)
    f = FilingEvent(ticker="ONCO", ts=T0, form="S-3", accession="a", dilutive=True,
                    reason="r", url="u")
    assert step(pos, f, cfg) == (pos, [])


# ------------------------------------------------------------------ helpers / properties


@given(st.integers(1, 10_000), st.floats(0.01, 1.0))
def test_scale_out_never_oversells(qty: int, frac: float) -> None:
    n = scale_out_qty(qty, frac)
    assert 1 <= n <= qty


@given(st.decimals(min_value=Decimal("0.01"), max_value=Decimal("9999"), places=4))
def test_sell_limit_is_marketable(bid: Decimal) -> None:
    lim = sell_limit(bid)
    assert lim <= bid and bid - lim < Decimal("0.01") and lim.as_tuple().exponent == -2


@pytest.mark.parametrize(("form", "text", "dilutive"), [
    ("S-3", None, True), ("S-3ASR", None, True), ("424B5", None, True), ("s-1", None, True),
    ("10-Q", None, False), ("8-K", "Appointment of a new director.", False),
    ("8-K", "Pricing of an underwritten public offering of common stock.", True),
    ("8-K", "The Company entered into an at-the-market sales agreement.", True),
    ("8-K", "Securities purchase agreement for a private placement.", True),
    ("8-K", "The company agreed to issue and sell 5,000,000 shares to investors.", True),
    ("8-K", None, False),
])
def test_classify_filing(form: str, text: str | None, dilutive: bool) -> None:
    assert classify_filing(form, text)[0] is dilutive
