"""In-memory broker for tests and offline dry runs. Records every call."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

from biotradeai.brokers.base import (
    Account,
    BrokerError,
    BrokerOrder,
    OrderBook,
    OrderReview,
    Position,
    Quote,
    ensure_limit_only,
)
from biotradeai.schemas import StagedOrder


@dataclass
class FakeBroker:
    name: str = "fake"
    equity: Decimal = Decimal("100000")
    last_equity: Decimal = Decimal("100000")
    quotes: dict[str, tuple[Decimal, Decimal]] = field(default_factory=dict)
    positions: dict[str, Position] = field(default_factory=dict)
    orders: dict[str, BrokerOrder] = field(default_factory=dict)
    fail_next: int = 0  # raise BrokerError on the next N calls
    calls: list[str] = field(default_factory=list)
    _ids: itertools.count[int] = field(default_factory=lambda: itertools.count(1))

    def _call(self, name: str) -> None:
        self.calls.append(name)
        if self.fail_next > 0:
            self.fail_next -= 1
            raise BrokerError(f"simulated failure in {name}")

    def get_account(self) -> Account:
        self._call("get_account")
        return Account(self.equity, self.last_equity, self.equity, self.equity)

    def get_quote(self, ticker: str) -> Quote:
        self._call("get_quote")
        bid, ask = self.quotes.get(ticker, (Decimal("9.99"), Decimal("10.01")))
        return Quote(ticker, bid, ask, 100, 100, datetime.now(UTC))

    def get_order_book(self, ticker: str) -> OrderBook:
        q = self.get_quote(ticker)
        return OrderBook(ticker, ((q.bid, q.bid_size),), ((q.ask, q.ask_size),), q.timestamp)

    def review_order(self, order: StagedOrder) -> OrderReview:
        self._call("review_order")
        ensure_limit_only(order)
        assert order.limit_price is not None  # noqa: S101 - narrowed by ensure_limit_only
        return OrderReview(order.ticker, order.side, order.qty, order.limit_price,
                           order.limit_price * order.qty)

    def place_order(self, order: StagedOrder, *, client_order_id: str) -> BrokerOrder:
        self._call("place_order")
        ensure_limit_only(order)
        oid = f"fake-{next(self._ids)}"
        bo = BrokerOrder(oid, order.ticker, order.side, order.qty, order.order_type,
                         order.limit_price, "new", client_order_id)
        self.orders[oid] = bo
        return bo

    def cancel_order(self, order_id: str) -> None:
        self._call("cancel_order")
        bo = self.orders[order_id]
        self.orders[order_id] = BrokerOrder(bo.order_id, bo.ticker, bo.side, bo.qty,
                                            bo.order_type, bo.limit_price, "canceled",
                                            bo.client_order_id)

    def get_positions(self) -> list[Position]:
        self._call("get_positions")
        return list(self.positions.values())

    def get_orders(self, *, open_only: bool = True) -> list[BrokerOrder]:
        self._call("get_orders")
        return [o for o in self.orders.values()
                if not open_only or o.status in {"new", "accepted", "partially_filled"}]
