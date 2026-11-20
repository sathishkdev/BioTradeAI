from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

import httpx
import pytest

from biotradeai.brokers.alpaca_broker import AlpacaBroker
from biotradeai.brokers.base import (
    Broker,
    BrokerError,
    OrderRejectedError,
    ShadowModeError,
)
from biotradeai.brokers.fake_broker import FakeBroker
from biotradeai.brokers.robinhood_broker import REQUIRED_TOOLS, RobinhoodBroker
from biotradeai.config import Mode
from biotradeai.schemas import OrderSide, OrderType
from tests.unit.conftest import make_order

# ------------------------------------------------------------------ alpaca


def _alpaca(handler: Any) -> AlpacaBroker:
    return AlpacaBroker("PKTESTKEY", "SECRETTEST", transport=httpx.MockTransport(handler))


def test_protocol_conformance() -> None:
    assert isinstance(FakeBroker(), Broker)
    assert isinstance(_alpaca(lambda r: httpx.Response(200, json={})), Broker)


def test_alpaca_requires_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
    with pytest.raises(BrokerError):
        AlpacaBroker()
    with pytest.raises(BrokerError, match="live"):
        AlpacaBroker("k", "s", paper=False)


def test_alpaca_account_drops_identifiers() -> None:
    def h(req: httpx.Request) -> httpx.Response:
        assert req.url.host == "paper-api.alpaca.markets"
        assert req.headers["APCA-API-KEY-ID"] == "PKTESTKEY"
        return httpx.Response(200, json={
            "id": "uuid-123", "account_number": "PA1234567", "equity": "100000.50",
            "last_equity": "99000", "cash": "50000", "buying_power": "200000",
            "trading_blocked": False,
        })

    a = _alpaca(h).get_account()
    assert a.equity == Decimal("100000.50") and a.last_equity == Decimal(99000)
    assert "PA1234567" not in repr(a)


def test_alpaca_place_order_wire_format() -> None:
    seen: dict[str, Any] = {}

    def h(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        assert req.method == "POST" and req.url.path == "/v2/orders"
        return httpx.Response(200, json={
            "id": "ord-1", "symbol": "ONCO", "side": "buy", "qty": "100", "type": "limit",
            "limit_price": "10.25", "status": "accepted", "client_order_id": "btai-abc",
        })

    bo = _alpaca(h).place_order(make_order(price="10.25"), client_order_id="btai-abc")
    assert seen == {"symbol": "ONCO", "qty": "100", "side": "buy", "type": "limit",
                    "time_in_force": "day", "limit_price": "10.25", "extended_hours": False,
                    "client_order_id": "btai-abc"}
    assert bo.order_id == "ord-1" and bo.order_type is OrderType.LIMIT


@pytest.mark.parametrize("otype", [OrderType.MARKET, OrderType.STOP, OrderType.STOP_LIMIT])
def test_alpaca_rejects_non_limit_before_http(otype: OrderType) -> None:
    calls: list[httpx.Request] = []

    def h(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(200, json={})

    with pytest.raises(OrderRejectedError):
        _alpaca(h).place_order(make_order(otype=otype), client_order_id="x")
    assert calls == []


def test_alpaca_quote_and_nanosecond_timestamp() -> None:
    def h(req: httpx.Request) -> httpx.Response:
        assert req.url.host == "data.alpaca.markets"
        assert req.url.params["feed"] == "iex"
        return httpx.Response(200, json={"symbol": "ONCO", "quote": {
            "bp": 9.98, "bs": 3, "ap": 10.02, "as": 5, "t": "2026-01-05T15:00:01.123456789Z"}})

    q = _alpaca(h).get_quote("ONCO")
    assert q.bid == Decimal("9.98") and q.ask == Decimal("10.02")
    assert q.timestamp.tzinfo is not None


def test_alpaca_errors_are_redacted() -> None:
    def h(req: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"message": "forbidden", "account_number": "PA999"})

    with pytest.raises(BrokerError) as ei:
        _alpaca(h).get_account()
    assert "PA999" not in str(ei.value)


def test_alpaca_review_warns_on_unmarketable_limit() -> None:
    def h(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"quote": {"bp": 10, "bs": 1, "ap": 11, "as": 1,
                                                   "t": "2026-01-05T15:00:00Z"}})

    rv = _alpaca(h).review_order(make_order(price="10.50"))
    assert rv.simulated and rv.warnings


# ------------------------------------------------------------------ robinhood


class FakeMCP:
    def __init__(self, tools: set[str]) -> None:
        self.tools = tools
        self.calls: list[str] = []

    def list_tools(self) -> list[str]:
        return sorted(self.tools)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(name)
        return {"account_number": "5RH000", "ok": True}


def test_robinhood_shadow_never_calls_place() -> None:
    mcp = FakeMCP(set(REQUIRED_TOOLS))
    rh = RobinhoodBroker(mcp, mode=Mode.SHADOW)
    with pytest.raises(ShadowModeError):
        rh.place_order(make_order(), client_order_id="x")
    assert mcp.calls == []


@pytest.mark.parametrize("otype", [OrderType.MARKET, OrderType.STOP])
def test_robinhood_rejects_non_limit(otype: OrderType) -> None:
    rh = RobinhoodBroker(FakeMCP(set(REQUIRED_TOOLS)), mode=Mode.LIVE)
    with pytest.raises(OrderRejectedError):
        rh.place_order(make_order(otype=otype), client_order_id="x")


def test_robinhood_tool_verification() -> None:
    assert RobinhoodBroker(FakeMCP(set(REQUIRED_TOOLS)), mode=Mode.SHADOW).verify_tools() == set()
    missing = RobinhoodBroker(FakeMCP(set(REQUIRED_TOOLS)), mode=Mode.LIVE).verify_tools()
    assert missing == {"get_trade_approvals"}


def test_robinhood_not_for_paper_and_redacts() -> None:
    with pytest.raises(BrokerError):
        RobinhoodBroker(FakeMCP(set()), mode=Mode.PAPER)
    rh = RobinhoodBroker(FakeMCP(set()), mode=Mode.SHADOW)
    assert rh._call("get_accounts", {})["account_number"] == "[REDACTED]"


def test_fake_broker_sell_side() -> None:
    fb = FakeBroker()
    o = make_order(side=OrderSide.SELL)
    assert fb.place_order(o, client_order_id="c").side is OrderSide.SELL
