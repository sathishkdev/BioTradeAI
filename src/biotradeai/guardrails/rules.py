"""Risk rules as pure functions.

Every function takes the validated ``GuardrailsConfig`` plus plain values, performs no
I/O, and returns a ``RuleResult``. The LLM never calls these and never sees their
thresholds as instructions; the Risk Agent, Supervisor and ``execution.py`` do.

Boundary convention (tested at, just below and just above every threshold):
* "minimum" rules pass AT the threshold  (EV >= min_ev, runway >= min_runway, ...)
* "maximum" rules pass AT the threshold  (spread <= max_spread, size <= cap, ...)
* the drawdown limit trips only when EXCEEDED (drawdown > limit)
* the fatal-runway prune fires when strictly BELOW fatal_runway_months
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_FLOOR, Decimal

from biotradeai.config import GuardrailsConfig, Mode
from biotradeai.data.freshness import is_fresh
from biotradeai.schemas import OrderSide, OrderType, ScenarioSet, StagedOrder


@dataclass(frozen=True)
class RuleResult:
    rule: str
    ok: bool
    reason: str

    def __bool__(self) -> bool:  # pragma: no cover - convenience only
        return self.ok


def _r(rule: str, ok: bool, reason: str) -> RuleResult:
    return RuleResult(rule, ok, reason)


# --------------------------------------------------------------------------- sizing limits


@dataclass(frozen=True)
class Limits:
    position_pct: float
    max_positions: int
    pilot: bool


def effective_limits(cfg: GuardrailsConfig) -> Limits:
    """Live mode always uses the (smaller) pilot caps."""
    if cfg.mode is Mode.LIVE:
        return Limits(cfg.live_gate.pilot_position_pct, cfg.live_gate.pilot_max_positions, True)
    return Limits(cfg.max_position_pct, cfg.max_concurrent_positions, False)


def max_position_qty(equity: Decimal, limit_price: Decimal, existing_value: Decimal,
                     cfg: GuardrailsConfig) -> int:
    """Largest whole-share quantity that keeps total ticker exposure within the cap."""
    if equity <= 0 or limit_price <= 0:
        return 0
    cap = equity * Decimal(str(effective_limits(cfg).position_pct))
    room = cap - max(existing_value, Decimal(0))
    if room <= 0:
        return 0
    return int((room / limit_price).to_integral_value(rounding=ROUND_FLOOR))


# --------------------------------------------------------------------------- order rules


def check_order_type(order: StagedOrder, cfg: GuardrailsConfig) -> RuleResult:
    if order.order_type.value not in cfg.orders.allowed_types:
        return _r("order_type", False, f"{order.order_type.value} orders are not allowed")
    if order.order_type is not OrderType.LIMIT or order.limit_price is None:
        return _r("order_type", False, "only limit orders with a limit price are allowed")
    if order.extended_hours and not cfg.orders.allow_extended_hours:
        return _r("order_type", False, "extended-hours orders are disabled")
    return _r("order_type", True, "limit order")


def check_position_size(order: StagedOrder, *, equity: Decimal, existing_value: Decimal,
                        cfg: GuardrailsConfig) -> RuleResult:
    if order.side is OrderSide.SELL:
        return _r("position_size", True, "sells reduce exposure")
    if order.limit_price is None:
        return _r("position_size", False, "no limit price")
    if equity <= 0:
        return _r("position_size", False, "account equity is not positive")
    pct = Decimal(str(effective_limits(cfg).position_pct))
    exposure = order.limit_price * order.qty + max(existing_value, Decimal(0))
    cap = equity * pct
    ok = exposure <= cap
    return _r("position_size", ok,
              f"exposure {exposure:.2f} vs cap {cap:.2f} ({pct:.2%} of equity {equity:.2f})")


def check_sell_quantity(order: StagedOrder, *, held_qty: int) -> RuleResult:
    if order.side is OrderSide.BUY:
        return _r("sell_qty", True, "not a sell")
    ok = order.qty <= held_qty
    return _r("sell_qty", ok, f"sell {order.qty} vs held {held_qty} (no shorting)")


def check_concurrency(*, open_tickers: set[str], ticker: str, side: OrderSide,
                      cfg: GuardrailsConfig) -> RuleResult:
    limit = effective_limits(cfg).max_positions
    if side is OrderSide.SELL or ticker in open_tickers:
        return _r("concurrency", True, "does not open a new position")
    ok = len(open_tickers) + 1 <= limit
    return _r("concurrency", ok, f"{len(open_tickers)} open + 1 new vs max {limit}")


def spread_pct(bid: Decimal, ask: Decimal) -> float | None:
    if bid <= 0 or ask <= 0 or ask < bid:
        return None
    mid = (bid + ask) / 2
    return float((ask - bid) / mid)


def check_spread(*, bid: Decimal, ask: Decimal, cfg: GuardrailsConfig) -> RuleResult:
    s = spread_pct(bid, ask)
    if s is None:
        return _r("spread", False, f"invalid quote bid={bid} ask={ask}")
    ok = s <= cfg.max_spread_pct
    return _r("spread", ok, f"spread {s:.4%} vs max {cfg.max_spread_pct:.4%}")


# --------------------------------------------------------------------------- thesis rules


def compute_ev(scenarios: ScenarioSet) -> float:
    """Expected return over ALL outcomes (the set is validated to sum to 1)."""
    return math.fsum(s.probability * s.expected_return for s in scenarios.scenarios)


def check_ev(ev: float, cfg: GuardrailsConfig) -> RuleResult:
    ok = ev >= cfg.min_ev
    return _r("ev", ok, f"EV {ev:+.4f} vs min {cfg.min_ev:+.4f}")


def is_fatal_runway(months: float, cfg: GuardrailsConfig) -> bool:
    return months < cfg.fatal_runway_months


def check_runway(months: float, cfg: GuardrailsConfig) -> RuleResult:
    ok = months >= cfg.min_runway_months
    return _r("runway", ok, f"runway {months:.2f} mo vs min {cfg.min_runway_months} mo")


def check_confidence(confidence: float, cfg: GuardrailsConfig) -> RuleResult:
    ok = confidence >= cfg.min_confidence
    return _r("confidence", ok, f"confidence {confidence:.3f} vs min {cfg.min_confidence:.3f}")


def check_ttl(observed_at: datetime, *, now: datetime, cfg: GuardrailsConfig,
              what: str = "data") -> RuleResult:
    ok = is_fresh(observed_at, now=now, ttl_hours=cfg.ttl_hours)
    return _r("ttl", ok, f"{what} observed {observed_at.isoformat()} (ttl {cfg.ttl_hours}h)")


def drawdown(start_equity: Decimal, equity: Decimal) -> float:
    if start_equity <= 0:
        return 0.0
    return float((start_equity - equity) / start_equity)


def check_drawdown(*, start_equity: Decimal, equity: Decimal,
                   cfg: GuardrailsConfig) -> RuleResult:
    d = drawdown(start_equity, equity)
    ok = d <= cfg.daily_drawdown_limit
    return _r("drawdown", ok, f"drawdown {d:.4%} vs limit {cfg.daily_drawdown_limit:.4%}")


def all_ok(results: list[RuleResult]) -> bool:
    return all(r.ok for r in results)


def failures(results: list[RuleResult]) -> list[RuleResult]:
    return [r for r in results if not r.ok]
