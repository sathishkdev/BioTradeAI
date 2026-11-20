"""Broker protocol and broker-neutral data types.

Agents never import this module. Only ``execution.py``, the kill-switch and read-only
data paths use a ``Broker``. Account identifiers are never part of these types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Protocol, runtime_checkable

from biotradeai.schemas import OrderSide, OrderType, StagedOrder


class BrokerError(RuntimeError):
    """Any failure talking to a broker (counts toward the kill-switch)."""


class OrderRejectedError(BrokerError):
    """The adapter itself refused the order (e.g. non-limit type)."""


class ShadowModeError(BrokerError):
    """place_order was called on a broker operating in shadow (read-only) mode."""


@dataclass(frozen=True)
class Account:
    equity: Decimal
    last_equity: Decimal  # equity at previous close = start-of-day baseline
    cash: Decimal
    buying_power: Decimal
    trading_blocked: bool = False


@dataclass(frozen=True)
class Quote:
    ticker: str
    bid: Decimal
    ask: Decimal
    bid_size: int
    ask_size: int
    timestamp: datetime


@dataclass(frozen=True)
class OrderBook:
    """Top of book (Alpaca's free data has no depth; Robinhood exposes a price book)."""

    ticker: str
    bids: tuple[tuple[Decimal, int], ...]
    asks: tuple[tuple[Decimal, int], ...]
    timestamp: datetime


@dataclass(frozen=True)
class OrderReview:
    ticker: str
    side: OrderSide
    qty: int
    limit_price: Decimal
    estimated_notional: Decimal
    warnings: tuple[str, ...] = ()
    simulated: bool = True
    raw: dict[str, object] = field(default_factory=dict, compare=False, repr=False)


@dataclass(frozen=True)
class BrokerOrder:
    order_id: str
    ticker: str
    side: OrderSide
    qty: int
    order_type: OrderType
    limit_price: Decimal | None
    status: str
    client_order_id: str | None = None


@dataclass(frozen=True)
class Position:
    ticker: str
    qty: int
    avg_entry_price: Decimal
    market_value: Decimal


@runtime_checkable
class Broker(Protocol):
    name: str

    def get_account(self) -> Account: ...
    def get_quote(self, ticker: str) -> Quote: ...
    def get_order_book(self, ticker: str) -> OrderBook: ...
    def review_order(self, order: StagedOrder) -> OrderReview: ...
    def place_order(self, order: StagedOrder, *, client_order_id: str) -> BrokerOrder: ...
    def cancel_order(self, order_id: str) -> None: ...
    def get_positions(self) -> list[Position]: ...
    def get_orders(self, *, open_only: bool = True) -> list[BrokerOrder]: ...


def ensure_limit_only(order: StagedOrder) -> None:
    """Adapter-level guard. Every broker calls this before anything leaves the process."""
    if order.order_type is not OrderType.LIMIT or order.limit_price is None:
        raise OrderRejectedError(
            f"adapter rejects {order.order_type.value} orders: limit orders only"
        )
