"""Alpaca paper-trading adapter over the REST API v2 (``httpx``, no SDK).

Paper only for now: live Alpaca is rejected at construction. Order POSTs are never
retried automatically (a retry could create a duplicate order); ``client_order_id``
makes a manual re-submit idempotent on Alpaca's side.

Endpoints used (Alpaca Trading API v2 / Market Data API v2):
  GET    /v2/account
  GET    /v2/positions
  GET    /v2/orders?status=open|all
  POST   /v2/orders
  DELETE /v2/orders/{id}
  GET    {data}/v2/stocks/{symbol}/quotes/latest?feed=iex
"""

from __future__ import annotations

import os
from datetime import datetime
from decimal import Decimal
from typing import Any

import httpx

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
from biotradeai.data import netpolicy
from biotradeai.redact import redact
from biotradeai.schemas import OrderSide, OrderType, StagedOrder

PAPER_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"


def _check_host(request: httpx.Request) -> None:
    netpolicy.check_url(str(request.url))


def _dec(v: Any) -> Decimal:
    return Decimal(str(v)) if v not in (None, "") else Decimal(0)


def _ts(v: str) -> datetime:
    # Alpaca timestamps are RFC 3339, sometimes with nanoseconds.
    v = v.replace("Z", "+00:00")
    if "." in v:
        head, rest = v.split(".", 1)
        frac, sep, tz = rest.partition("+") if "+" in rest else rest.partition("-")
        v = f"{head}.{frac[:6]}{sep}{tz}"
    return datetime.fromisoformat(v)


class AlpacaBroker:
    name = "alpaca-paper"

    def __init__(
        self,
        key_id: str | None = None,
        secret: str | None = None,
        *,
        paper: bool = True,
        feed: str = "iex",
        transport: httpx.BaseTransport | None = None,
        timeout_s: float = 20.0,
    ) -> None:
        if not paper:
            raise BrokerError("live Alpaca trading is not enabled in this build")
        key_id = key_id or os.environ.get("ALPACA_API_KEY", "")
        secret = secret or os.environ.get("ALPACA_API_SECRET", "")
        if not key_id or not secret:
            raise BrokerError("ALPACA_API_KEY / ALPACA_API_SECRET are not set")
        headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret}
        hooks = {"request": [_check_host]}
        self._trade = httpx.Client(base_url=PAPER_URL, headers=headers, timeout=timeout_s,
                                   transport=transport, event_hooks=hooks)
        self._data = httpx.Client(base_url=DATA_URL, headers=headers, timeout=timeout_s,
                                  transport=transport, event_hooks=hooks)
        self._feed = feed

    # ------------------------------------------------------------------ http

    def _req(self, client: httpx.Client, method: str, url: str, **kw: Any) -> Any:
        # Shared "alpaca" limiter; no daily budget so exits are never blocked by it.
        netpolicy.limiter_for("alpaca").acquire()
        try:
            resp = client.request(method, url, **kw)
        except netpolicy.BlockedHostError as exc:
            raise BrokerError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise BrokerError(f"alpaca {method} {url}: {exc}") from exc
        if resp.status_code >= 400:
            try:
                detail: Any = redact(resp.json())
            except ValueError:
                detail = resp.text[:200]
            raise BrokerError(f"alpaca {method} {url}: HTTP {resp.status_code} {detail}")
        return resp.json() if resp.content else None

    # ------------------------------------------------------------------ reads

    def get_account(self) -> Account:
        a = self._req(self._trade, "GET", "/v2/account")
        # account_number / id are deliberately dropped here.
        return Account(
            equity=_dec(a.get("equity")),
            last_equity=_dec(a.get("last_equity")),
            cash=_dec(a.get("cash")),
            buying_power=_dec(a.get("buying_power")),
            trading_blocked=bool(a.get("trading_blocked") or a.get("account_blocked")),
        )

    def get_quote(self, ticker: str) -> Quote:
        d = self._req(self._data, "GET", f"/v2/stocks/{ticker}/quotes/latest",
                      params={"feed": self._feed})
        q = d["quote"]
        return Quote(ticker, _dec(q.get("bp")), _dec(q.get("ap")), int(q.get("bs") or 0),
                     int(q.get("as") or 0), _ts(q["t"]))

    def get_order_book(self, ticker: str) -> OrderBook:
        q = self.get_quote(ticker)
        return OrderBook(ticker, ((q.bid, q.bid_size),), ((q.ask, q.ask_size),), q.timestamp)

    def review_order(self, order: StagedOrder) -> OrderReview:
        """Alpaca has no order preview; this is a local simulation with a fresh quote."""
        ensure_limit_only(order)
        price = order.limit_price or Decimal(0)
        q = self.get_quote(order.ticker)
        warnings = []
        if order.side is OrderSide.BUY and price < q.ask:
            warnings.append(f"limit {price} below ask {q.ask}: may not fill")
        if order.side is OrderSide.SELL and price > q.bid:
            warnings.append(f"limit {price} above bid {q.bid}: may not fill")
        return OrderReview(order.ticker, order.side, order.qty, price, price * order.qty,
                           tuple(warnings), simulated=True)

    def get_positions(self) -> list[Position]:
        rows = self._req(self._trade, "GET", "/v2/positions")
        return [Position(r["symbol"], int(Decimal(str(r["qty"]))), _dec(r.get("avg_entry_price")),
                         _dec(r.get("market_value"))) for r in rows]

    def get_orders(self, *, open_only: bool = True) -> list[BrokerOrder]:
        rows = self._req(self._trade, "GET", "/v2/orders",
                         params={"status": "open" if open_only else "all", "limit": 500})
        return [self._order(r) for r in rows]

    @staticmethod
    def _order(r: dict[str, Any]) -> BrokerOrder:
        otype = str(r.get("type") or r.get("order_type") or "limit")
        try:
            ot = OrderType(otype)
        except ValueError:
            ot = OrderType.MARKET  # unknown types are treated as non-limit
        return BrokerOrder(
            order_id=r["id"], ticker=r["symbol"], side=OrderSide(r["side"]),
            qty=int(Decimal(str(r.get("qty") or 0))), order_type=ot,
            limit_price=_dec(r["limit_price"]) if r.get("limit_price") else None,
            status=r["status"], client_order_id=r.get("client_order_id"),
        )

    # ------------------------------------------------------------------ writes

    def place_order(self, order: StagedOrder, *, client_order_id: str) -> BrokerOrder:
        ensure_limit_only(order)
        body = {
            "symbol": order.ticker,
            "qty": str(order.qty),
            "side": order.side.value,
            "type": "limit",
            "time_in_force": order.time_in_force.value,
            "limit_price": format(order.limit_price, "f"),
            "extended_hours": order.extended_hours,
            "client_order_id": client_order_id,
        }
        return self._order(self._req(self._trade, "POST", "/v2/orders", json=body))

    def cancel_order(self, order_id: str) -> None:
        self._req(self._trade, "DELETE", f"/v2/orders/{order_id}")
