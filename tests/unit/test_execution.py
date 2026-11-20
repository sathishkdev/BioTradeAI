from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from biotradeai.brokers.approval import ApprovalAuthority
from biotradeai.brokers.base import Position
from biotradeai.brokers.execution import ExecStatus, ExecutionService
from biotradeai.brokers.fake_broker import FakeBroker
from biotradeai.config import GuardrailsConfig, load_config, parse_config
from biotradeai.schemas import OrderSide, OrderType, StagedOrder
from biotradeai.store.sqlite_store import SQLiteStore
from tests.unit.conftest import make_order

CFG = load_config()


def _svc(broker: FakeBroker, store: SQLiteStore, authority: ApprovalAuthority,
         cfg: GuardrailsConfig = CFG) -> ExecutionService:
    return ExecutionService(broker, store, cfg, authority)


def _events(store: SQLiteStore) -> list[str]:
    return [e.event_type for e in store.read_audit()]


def test_happy_path_places_and_audits(broker: FakeBroker, store: SQLiteStore,
                                      authority: ApprovalAuthority) -> None:
    o = make_order()
    r = _svc(broker, store, authority).execute(o, authority.issue(o), run_id="r1")
    assert r.status is ExecStatus.PLACED and r.order_id
    assert broker.calls.count("place_order") == 1
    assert _events(store) == ["execution.request", "execution.placed"]
    assert store.verify_audit_chain()


def test_no_token_rejected(broker: FakeBroker, store: SQLiteStore,
                           authority: ApprovalAuthority) -> None:
    r = _svc(broker, store, authority).execute(make_order(), None)
    assert r.status is ExecStatus.REJECTED and "place_order" not in broker.calls


def test_reused_token_rejected(broker: FakeBroker, store: SQLiteStore,
                               authority: ApprovalAuthority) -> None:
    o = make_order()
    t = authority.issue(o)
    svc = _svc(broker, store, authority)
    assert svc.execute(o, t).status is ExecStatus.PLACED
    r = svc.execute(o, t)
    assert r.status is ExecStatus.REJECTED and "already used" in r.reasons[0]
    assert broker.calls.count("place_order") == 1


def test_forged_and_expired_tokens_rejected(broker: FakeBroker, store: SQLiteStore,
                                            authority: ApprovalAuthority, clock: Any) -> None:
    o = make_order()
    svc = _svc(broker, store, authority)
    forged = authority.issue(o).model_copy(update={"signature": "b" * 64})
    assert svc.execute(o, forged).status is ExecStatus.REJECTED
    t = authority.issue(o)
    clock.advance(minutes=16)
    r = svc.execute(o, t)
    assert r.status is ExecStatus.REJECTED and "expired" in r.reasons[0]
    assert "place_order" not in broker.calls


@pytest.mark.parametrize("otype", [OrderType.MARKET, OrderType.STOP, OrderType.STOP_LIMIT])
def test_non_limit_never_reaches_broker(otype: OrderType, broker: FakeBroker,
                                        store: SQLiteStore,
                                        authority: ApprovalAuthority) -> None:
    o = make_order(otype=otype)
    r = _svc(broker, store, authority).execute(o, None)
    assert r.status is ExecStatus.REJECTED
    assert broker.calls == []


def test_position_cap_enforced(broker: FakeBroker, store: SQLiteStore,
                               authority: ApprovalAuthority) -> None:
    o = make_order(qty=501)  # $5,010 > 5% of $100k
    r = _svc(broker, store, authority).execute(o, authority.issue(o))
    assert r.status is ExecStatus.REJECTED and any("position_size" in x for x in r.reasons)
    assert "place_order" not in broker.calls


def test_concurrency_enforced(broker: FakeBroker, store: SQLiteStore,
                              authority: ApprovalAuthority) -> None:
    for t in ("AAA", "BBB", "CCC"):
        broker.positions[t] = Position(t, 10, Decimal(1), Decimal(10))
    o = make_order()
    r = _svc(broker, store, authority).execute(o, authority.issue(o))
    assert r.status is ExecStatus.REJECTED and any("concurrency" in x for x in r.reasons)


def test_no_shorting(broker: FakeBroker, store: SQLiteStore,
                     authority: ApprovalAuthority) -> None:
    o = make_order(side=OrderSide.SELL, qty=10)
    r = _svc(broker, store, authority).execute(o, authority.issue(o))
    assert r.status is ExecStatus.REJECTED and any("sell_qty" in x for x in r.reasons)


def test_wide_spread_blocks_buys(broker: FakeBroker, store: SQLiteStore,
                                 authority: ApprovalAuthority) -> None:
    broker.quotes["ONCO"] = (Decimal("9.00"), Decimal("10.00"))
    o = make_order()
    r = _svc(broker, store, authority).execute(o, authority.issue(o))
    assert r.status is ExecStatus.REJECTED and any("spread" in x for x in r.reasons)


def test_killswitch_blocks_everything(broker: FakeBroker, store: SQLiteStore,
                                      authority: ApprovalAuthority) -> None:
    svc = _svc(broker, store, authority)
    svc.killswitch.trip("manual")
    o = make_order()
    r = svc.execute(o, authority.issue(o))
    assert r.status is ExecStatus.REJECTED and "tripped" in r.reasons[0]
    assert "place_order" not in broker.calls


def test_drawdown_during_execution_trips_and_cancels(broker: FakeBroker, store: SQLiteStore,
                                                     authority: ApprovalAuthority) -> None:
    resting = broker.place_order(make_order(ticker="OTHR"), client_order_id="x").order_id
    broker.calls.clear()
    broker.equity = Decimal("96000")  # 4% below last_equity 100k
    o = make_order()
    r = _svc(broker, store, authority).execute(o, authority.issue(o))
    assert r.status is ExecStatus.REJECTED
    assert broker.orders[resting].status == "canceled"
    assert "place_order" not in broker.calls
    assert store.get_killswitch().tripped


def test_consecutive_broker_errors_trip_killswitch(broker: FakeBroker, store: SQLiteStore,
                                                   authority: ApprovalAuthority) -> None:
    svc = _svc(broker, store, authority)
    for _ in range(CFG.max_api_errors):
        broker.fail_next = 1
        o = make_order()
        assert svc.execute(o, authority.issue(o)).status is ExecStatus.REJECTED
    assert store.get_killswitch().tripped


def test_shadow_mode_never_calls_place(broker: FakeBroker, store: SQLiteStore,
                                       authority: ApprovalAuthority,
                                       raw_config: dict[str, Any]) -> None:
    raw_config["mode"] = "shadow"
    o = make_order()
    r = _svc(broker, store, authority, parse_config(raw_config)).execute(o, authority.issue(o))
    assert r.status is ExecStatus.SHADOW_REVIEWED and r.review is not None
    assert "place_order" not in broker.calls and "review_order" in broker.calls
    assert "execution.would_have_placed" in _events(store)


def test_live_mode_blocked(broker: FakeBroker, store: SQLiteStore,
                           authority: ApprovalAuthority, raw_config: dict[str, Any]) -> None:
    raw_config["mode"] = "live"
    o = make_order(qty=10)
    r = _svc(broker, store, authority, parse_config(raw_config)).execute(o, authority.issue(o))
    assert r.status is ExecStatus.REJECTED and broker.calls == []


@settings(max_examples=60, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(qty=st.integers(1, 5000),
       px=st.decimals(min_value=Decimal("0.5"), max_value=Decimal("200"), places=2),
       otype=st.sampled_from(list(OrderType)),
       equity=st.decimals(min_value=Decimal("1000"), max_value=Decimal("1000000"), places=2))
def test_property_nothing_unsafe_is_ever_placed(tmp_path: Path, authority: ApprovalAuthority,
                                                qty: int, px: Decimal, otype: OrderType,
                                                equity: Decimal) -> None:
    broker = FakeBroker(equity=equity, last_equity=equity,
                        quotes={"ONCO": (px, px)})
    store = SQLiteStore(tmp_path / "prop.db")
    o = StagedOrder(ticker="ONCO", side=OrderSide.BUY, qty=qty, order_type=otype,
                    limit_price=px)
    token = authority.issue(o) if otype is OrderType.LIMIT else None
    r = _svc(broker, store, authority).execute(o, token)
    if r.status is ExecStatus.PLACED:
        assert otype is OrderType.LIMIT
        assert px * qty <= equity * Decimal("0.05")
    placed = [b for b in broker.orders.values()]
    assert all(b.order_type is OrderType.LIMIT for b in placed)
