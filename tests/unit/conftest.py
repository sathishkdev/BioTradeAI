from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from biotradeai.brokers.approval import ApprovalAuthority
from biotradeai.brokers.fake_broker import FakeBroker
from biotradeai.data import netpolicy
from biotradeai.data.http_limiter import RateLimiter
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


@pytest.fixture(autouse=True)
def _offline_network_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit tests never touch the network: fresh in-memory ledger, no real waiting.

    The real per-source rates (e.g. 0.8 req/s for CT.gov) are covered by
    tests/unit/test_netpolicy.py; here they would only make the suite slow.
    """
    netpolicy.use_ledger(netpolicy.InMemoryLedger())
    netpolicy.reset_limiters()
    monkeypatch.setattr(netpolicy, "limiter_for", lambda source: RateLimiter(1e9))
