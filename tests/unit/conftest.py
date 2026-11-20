from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from biotradeai.brokers.approval import ApprovalAuthority
from biotradeai.brokers.fake_broker import FakeBroker
from biotradeai.schemas import OrderSide, OrderType, StagedOrder

SECRET = b"x" * 32


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 1, 5, 15, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kw: float) -> None:
        self.now += timedelta(**kw)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def authority(clock: Clock) -> ApprovalAuthority:
    return ApprovalAuthority(SECRET, ttl=timedelta(minutes=15), clock=clock)


@pytest.fixture
def broker() -> FakeBroker:
    return FakeBroker()


def make_order(qty: int = 100, price: str = "10.00", side: OrderSide = OrderSide.BUY,
               otype: OrderType = OrderType.LIMIT, ticker: str = "ONCO") -> StagedOrder:
    return StagedOrder(ticker=ticker, side=side, qty=qty, order_type=otype,
                       limit_price=Decimal(price))
