"""Deterministic, event-driven sell state machine (Phase 6). No LLM anywhere.

States per position:

    HOLDING ──price >= entry*(1+profit_target) before the catalyst──────▶ CLOSED (sell 100%)
    HOLDING ──readout positive──▶ TRAILING (sell scale_out_fraction, trail the rest)
    HOLDING ──readout negative / CRL / mixed──────────────────────────────▶ CLOSED (sell 100%)
    TRAILING ─bid <= high_water*(1-trailing_stop)─────────────────────────▶ CLOSED (sell rest)
    TRAILING ─next catalyst scheduled (never hold between readouts)──────▶ CLOSED (sell rest)
    any open ─dilutive filing (S-3 / 424B / offering 8-K)─────────────────▶ CLOSED (sell 100%)
    any open ─primary completion pushed > completion_delay_months─────────▶ CLOSED (sell 100%)

Every exit is a SELL LIMIT order at the last bid (marketable, never a market order) and
must still pass ``guardrails.rules.check_order_type``. ``step`` is a pure function;
``SellEngine`` adds audit logging and persistence. Orders are only *staged* here; placing
them is ``brokers/execution.py``'s job (with human approval), wired in Phase 8.

Boundaries (tested): profit target and trailing stop fire AT the threshold; the
time-stop fires only when the delay strictly EXCEEDS ``completion_delay_months``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_DOWN, Decimal
from enum import StrEnum

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from biotradeai.config import GuardrailsConfig
from biotradeai.events import (
    CatalystOutcome,
    CatalystScheduled,
    CompletionDateChange,
    FilingEvent,
    PriceTick,
    ReadoutResult,
    SellEvent,
)
from biotradeai.guardrails import rules
from biotradeai.schemas import OrderSide, OrderType, Price, StagedOrder, Ticker
from biotradeai.store.sqlite_store import SQLiteStore

DAYS_PER_MONTH = 365.25 / 12
CENT = Decimal("0.01")


class SellState(StrEnum):
    HOLDING = "holding"  # before the catalyst outcome
    TRAILING = "trailing"  # after a positive readout, remainder under a trailing stop
    CLOSED = "closed"


class Trigger(StrEnum):
    PROFIT_TARGET = "profit_target"
    POSITIVE_SCALE_OUT = "positive_scale_out"
    NEGATIVE_READOUT = "negative_readout"
    TRAILING_STOP = "trailing_stop"
    NEXT_CATALYST = "next_catalyst"
    DILUTION = "dilution"
    TIME_STOP = "time_stop"


class SellPosition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ticker: Ticker
    qty: int = Field(ge=0)
    entry_price: Price
    catalyst_event: str
    catalyst_date: date
    nct_id: str | None = None
    baseline_completion: date | None = None  # primary completion date at entry
    state: SellState = SellState.HOLDING
    high_water: Decimal | None = None  # highest bid since TRAILING began
    last_bid: Decimal | None = None
    last_quote_at: AwareDatetime | None = None


@dataclass(frozen=True)
class Exit:
    ticker: str
    trigger: Trigger
    qty: int
    reason: str
    ts: AwareDatetime
    order: StagedOrder | None  # None when no quote is known yet (needs a quote first)


def sell_limit(bid: Decimal) -> Decimal:
    """Marketable sell limit: the bid, rounded DOWN to the cent."""
    return max(CENT, bid.quantize(CENT, rounding=ROUND_DOWN))


def scale_out_qty(qty: int, fraction: float) -> int:
    """floor(qty * fraction), but at least one share and never more than held."""
    if qty <= 0:
        return 0
    return min(qty, max(1, math.floor(qty * fraction)))


def months_between(a: date, b: date) -> float:
    return (b - a).days / DAYS_PER_MONTH


def _exit(pos: SellPosition, trigger: Trigger, qty: int, reason: str,
          ts: AwareDatetime) -> Exit:
    order = None
    if pos.last_bid is not None and qty > 0:
        order = StagedOrder(ticker=pos.ticker, side=OrderSide.SELL, qty=qty,
                            order_type=OrderType.LIMIT, limit_price=sell_limit(pos.last_bid))
    return Exit(pos.ticker, trigger, qty, reason, ts, order)


def _close(pos: SellPosition, trigger: Trigger, reason: str,
           ts: AwareDatetime) -> tuple[SellPosition, list[Exit]]:
    ex = _exit(pos, trigger, pos.qty, reason, ts)
    return pos.model_copy(update={"qty": 0, "state": SellState.CLOSED}), [ex]


def step(pos: SellPosition, ev: SellEvent,
         cfg: GuardrailsConfig) -> tuple[SellPosition, list[Exit]]:
    """Apply one event. Pure: same inputs, same outputs."""
    if ev.ticker != pos.ticker or pos.state is SellState.CLOSED or pos.qty == 0:
        return pos, []
    sc = cfg.sell

    if isinstance(ev, PriceTick):
        pos = pos.model_copy(update={"last_bid": ev.bid, "last_quote_at": ev.ts})
        if pos.state is SellState.HOLDING:
            target = pos.entry_price * Decimal(str(1 + sc.profit_target_pct))
            if ev.ts.date() < pos.catalyst_date and ev.bid >= target:
                return _close(pos, Trigger.PROFIT_TARGET,
                              f"bid {ev.bid} >= target {target:.4f} "
                              f"(+{sc.profit_target_pct:.0%}) before the catalyst", ev.ts)
            return pos, []
        hw = max(pos.high_water or ev.bid, ev.bid)
        pos = pos.model_copy(update={"high_water": hw})
        stop = hw * Decimal(str(1 - sc.trailing_stop_pct))
        if ev.bid <= stop:
            return _close(pos, Trigger.TRAILING_STOP,
                          f"bid {ev.bid} <= stop {stop:.4f} ({sc.trailing_stop_pct:.0%} "
                          f"below high {hw})", ev.ts)
        return pos, []

    if isinstance(ev, CatalystOutcome):
        if pos.state is not SellState.HOLDING:
            return pos, []
        if ev.result is ReadoutResult.POSITIVE:
            n = scale_out_qty(pos.qty, sc.scale_out_fraction)
            ex = _exit(pos, Trigger.POSITIVE_SCALE_OUT, n,
                       f"positive readout: sell {n} of {pos.qty} "
                       f"({sc.scale_out_fraction:.0%}), trail the rest", ev.ts)
            left = pos.qty - n
            new = pos.model_copy(update={
                "qty": left, "state": SellState.TRAILING if left else SellState.CLOSED,
                "high_water": pos.last_bid})
            return new, [ex]
        return _close(pos, Trigger.NEGATIVE_READOUT,
                      f"readout {ev.result.value}: sell 100%", ev.ts)

    if isinstance(ev, FilingEvent):
        if ev.dilutive:
            return _close(pos, Trigger.DILUTION,
                          f"{ev.form} {ev.accession}: {ev.reason}", ev.ts)
        return pos, []

    if isinstance(ev, CompletionDateChange):
        if pos.nct_id is not None and ev.nct_id != pos.nct_id:
            return pos, []
        if pos.baseline_completion is None:
            return pos.model_copy(update={"baseline_completion": ev.new_date}), []
        delay = months_between(pos.baseline_completion, ev.new_date)
        if delay > sc.completion_delay_months:
            return _close(pos, Trigger.TIME_STOP,
                          f"primary completion moved {pos.baseline_completion} -> "
                          f"{ev.new_date} ({delay:.1f} mo > {sc.completion_delay_months} mo)",
                          ev.ts)
        return pos, []

    if isinstance(ev, CatalystScheduled):
        if pos.state is SellState.TRAILING:
            return _close(pos, Trigger.NEXT_CATALYST,
                          f"next catalyst {ev.event} on {ev.catalyst_date}: positions are "
                          f"never held between readouts", ev.ts)
        return pos, []

    return pos, []  # pragma: no cover - the union is exhaustive


# --------------------------------------------------------------------------- engine


class SellEngine:
    """Runs events through ``step`` and writes every exit (and state change) to the audit log."""

    def __init__(self, cfg: GuardrailsConfig, store: SQLiteStore) -> None:
        self._cfg = cfg
        self._store = store

    def process(self, pos: SellPosition, events: Iterable[SellEvent], *,
                run_id: str | None = None) -> tuple[SellPosition, list[Exit]]:
        exits: list[Exit] = []
        for ev in sorted(events, key=lambda e: e.ts):  # stable: equal ts keep input order
            before = pos.state
            pos, out = step(pos, ev, self._cfg)
            for ex in out:
                checks = []
                if ex.order is not None:
                    checks = [rules.check_order_type(ex.order, self._cfg)]
                    if not rules.all_ok(checks):  # pragma: no cover - step only builds limits
                        raise RuntimeError(f"sell order failed guardrail: {checks}")
                self._store.append_audit("sell.exit", {
                    "trigger": ex.trigger.value, "qty": ex.qty, "reason": ex.reason,
                    "event": ev.model_dump(mode="json"),
                    "order": ex.order.model_dump(mode="json") if ex.order else None,
                    "needs_quote": ex.order is None,
                    "checks": [{"rule": c.rule, "ok": c.ok, "reason": c.reason}
                               for c in checks],
                }, run_id=run_id, ticker=pos.ticker)
                exits.append(ex)
            if pos.state is not before:
                self._store.append_audit("sell.state", {
                    "from": before.value, "to": pos.state.value, "qty": pos.qty,
                    "event_kind": ev.kind}, run_id=run_id, ticker=pos.ticker)
        self._store.save_sell_state(pos.ticker, pos.model_dump(mode="json"))
        return pos, exits
