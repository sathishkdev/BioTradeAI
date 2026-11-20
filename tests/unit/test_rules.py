"""Guardrail rules: at / just below / just above every threshold, plus property tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from biotradeai.config import GuardrailsConfig, load_config, parse_config
from biotradeai.guardrails import rules
from biotradeai.schemas import OrderSide, OrderType, Scenario, ScenarioSet, StagedOrder

CFG = load_config()
EPS = 1e-9


def _order(qty: int = 10, price: str = "10", side: OrderSide = OrderSide.BUY,
           otype: OrderType = OrderType.LIMIT, ext: bool = False) -> StagedOrder:
    return StagedOrder(ticker="ONCO", side=side, qty=qty, order_type=otype,
                       limit_price=Decimal(price), extended_hours=ext)


def _cfg(raw_config: dict[str, Any], **changes: Any) -> GuardrailsConfig:
    for k, v in changes.items():
        raw_config[k] = v
    return parse_config(raw_config)


# ------------------------------------------------------------------ thresholds


@pytest.mark.parametrize(("delta", "ok"), [(0, True), (-EPS, False), (EPS, True)])
def test_ev_boundary(delta: float, ok: bool) -> None:
    assert rules.check_ev(CFG.min_ev + delta, CFG).ok is ok


@pytest.mark.parametrize(("delta", "ok"), [(0, True), (-EPS, False), (EPS, True)])
def test_runway_veto_boundary(delta: float, ok: bool) -> None:
    assert rules.check_runway(CFG.min_runway_months + delta, CFG).ok is ok


@pytest.mark.parametrize(("delta", "fatal"), [(0, False), (-EPS, True), (EPS, False)])
def test_fatal_runway_boundary(delta: float, fatal: bool) -> None:
    assert rules.is_fatal_runway(CFG.fatal_runway_months + delta, CFG) is fatal


@pytest.mark.parametrize(("delta", "ok"), [(0, True), (-EPS, False), (EPS, True)])
def test_confidence_boundary(delta: float, ok: bool) -> None:
    assert rules.check_confidence(CFG.min_confidence + delta, CFG).ok is ok


@pytest.mark.parametrize(("ask", "ok"), [
    ("10.15", True),     # spread 1.489% < 1.5%
    ("10.1523", False),  # just above 1.5%
    ("10.00", True),     # zero spread
])
def test_spread_boundary(ask: str, ok: bool) -> None:
    assert rules.check_spread(bid=Decimal("10.00"), ask=Decimal(ask), cfg=CFG).ok is ok


def test_spread_exactly_at_limit(raw_config: dict[str, Any]) -> None:
    cfg = _cfg(raw_config, max_spread_pct=0.02)
    # bid 99, ask 101 -> spread 2 / mid 100 = exactly 2%
    assert rules.check_spread(bid=Decimal(99), ask=Decimal(101), cfg=cfg).ok


@pytest.mark.parametrize(("bid", "ask"), [("0", "1"), ("2", "1"), ("-1", "1")])
def test_invalid_quotes_fail(bid: str, ask: str) -> None:
    assert not rules.check_spread(bid=Decimal(bid), ask=Decimal(ask), cfg=CFG).ok


@pytest.mark.parametrize(("equity_after", "ok"), [
    ("97000", True),      # exactly 3.00% drawdown -> allowed
    ("97000.01", True),   # just below
    ("96999.99", False),  # just above -> trip
])
def test_drawdown_boundary(equity_after: str, ok: bool) -> None:
    r = rules.check_drawdown(start_equity=Decimal(100000), equity=Decimal(equity_after), cfg=CFG)
    assert r.ok is ok


def test_ttl_boundary() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    ttl = timedelta(hours=CFG.ttl_hours)
    assert rules.check_ttl(now - ttl, now=now, cfg=CFG).ok
    assert not rules.check_ttl(now - ttl - timedelta(seconds=1), now=now, cfg=CFG).ok


# ------------------------------------------------------------------ sizing


@pytest.mark.parametrize(("qty", "ok"), [(500, True), (499, True), (501, False)])
def test_position_cap_boundary(qty: int, ok: bool) -> None:
    # 5% of 100,000 = 5,000 -> 500 shares at $10
    r = rules.check_position_size(_order(qty=qty), equity=Decimal(100000),
                                  existing_value=Decimal(0), cfg=CFG)
    assert r.ok is ok


def test_position_cap_counts_existing_exposure() -> None:
    r = rules.check_position_size(_order(qty=101), equity=Decimal(100000),
                                  existing_value=Decimal(4000), cfg=CFG)
    assert not r.ok
    assert rules.max_position_qty(Decimal(100000), Decimal(10), Decimal(4000), CFG) == 100


def test_sells_not_capped_but_cannot_short() -> None:
    sell = _order(qty=10_000, side=OrderSide.SELL)
    assert rules.check_position_size(sell, equity=Decimal(1), existing_value=Decimal(0),
                                     cfg=CFG).ok
    assert rules.check_sell_quantity(sell, held_qty=10_000).ok
    assert not rules.check_sell_quantity(sell, held_qty=9_999).ok


@pytest.mark.parametrize(("open_n", "ok"), [(2, True), (3, False)])
def test_concurrency_boundary(open_n: int, ok: bool) -> None:
    held = {f"T{i}" for i in range(open_n)}
    assert rules.check_concurrency(open_tickers=held, ticker="NEW", side=OrderSide.BUY,
                                   cfg=CFG).ok is ok


def test_adding_to_existing_position_is_not_new() -> None:
    held = {"A", "B", "ONCO"}
    assert rules.check_concurrency(open_tickers=held, ticker="ONCO", side=OrderSide.BUY,
                                   cfg=CFG).ok


def test_live_mode_uses_pilot_caps(raw_config: dict[str, Any]) -> None:
    cfg = _cfg(raw_config, mode="live")
    lim = rules.effective_limits(cfg)
    assert lim.pilot and lim.position_pct == 0.01 and lim.max_positions == 1
    assert not rules.check_position_size(_order(qty=101), equity=Decimal(100000),
                                         existing_value=Decimal(0), cfg=cfg).ok
    assert not rules.check_concurrency(open_tickers={"A"}, ticker="B", side=OrderSide.BUY,
                                       cfg=cfg).ok


# ------------------------------------------------------------------ order type


@pytest.mark.parametrize("otype", [OrderType.MARKET, OrderType.STOP, OrderType.STOP_LIMIT])
def test_non_limit_rejected(otype: OrderType) -> None:
    assert not rules.check_order_type(_order(otype=otype), CFG).ok


def test_extended_hours_blocked_by_default() -> None:
    assert not rules.check_order_type(_order(ext=True), CFG).ok
    assert rules.check_order_type(_order(), CFG).ok


# ------------------------------------------------------------------ EV


def test_ev_over_all_outcomes() -> None:
    s = ScenarioSet(tolerance=0.001, scenarios=[
        Scenario(depth=3, label="approve", probability=0.6, expected_return=0.8),
        Scenario(depth=3, label="partial", probability=0.3, expected_return=0.1),
        Scenario(depth=3, label="crl", probability=0.1, expected_return=-0.7),
    ])
    # 0.48 + 0.03 - 0.07 = 0.44 (the PDF's 90%-only version would give 0.51)
    assert rules.compute_ev(s) == pytest.approx(0.44)


# ------------------------------------------------------------------ properties

money = st.decimals(min_value=Decimal("1000"), max_value=Decimal("10000000"), places=2)
price = st.decimals(min_value=Decimal("0.01"), max_value=Decimal("1000"), places=4)


@given(equity=money, px=price, qty=st.integers(1, 10_000_000),
       existing_frac=st.floats(0, 1))
def test_property_passing_order_never_exceeds_cap(equity: Decimal, px: Decimal, qty: int,
                                                  existing_frac: float) -> None:
    existing = (equity * Decimal(str(existing_frac)) * Decimal("0.06")).quantize(Decimal("0.01"))
    order = StagedOrder(ticker="ONCO", side=OrderSide.BUY, qty=qty, order_type=OrderType.LIMIT,
                        limit_price=px)
    r = rules.check_position_size(order, equity=equity, existing_value=existing, cfg=CFG)
    if r.ok:
        assert px * qty + existing <= equity * Decimal("0.05")


@given(equity=money, px=price, existing_frac=st.floats(0, 1))
def test_property_max_qty_is_tight(equity: Decimal, px: Decimal, existing_frac: float) -> None:
    existing = (equity * Decimal(str(existing_frac)) * Decimal("0.05")).quantize(Decimal("0.01"))
    n = rules.max_position_qty(equity, px, existing, CFG)
    cap = equity * Decimal("0.05")
    if existing >= cap:
        assert n == 0  # already at/over the cap: buy nothing
        return
    assert px * n + existing <= cap
    assert px * (n + 1) + existing > cap


@given(otype=st.sampled_from([OrderType.MARKET, OrderType.STOP, OrderType.STOP_LIMIT]),
       qty=st.integers(1, 1000), ext=st.booleans())
def test_property_non_limit_always_rejected(otype: OrderType, qty: int, ext: bool) -> None:
    order = StagedOrder(ticker="ONCO", side=OrderSide.BUY, qty=qty, order_type=otype,
                        limit_price=Decimal("5"), extended_hours=ext)
    assert not rules.check_order_type(order, CFG).ok
