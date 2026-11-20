"""Robinhood official Trading MCP adapter — mode gating only until the Phase 3 spike.

What is implemented now (safe regardless of Robinhood's exact tool schemas):
* tool discovery: ``verify_tools`` compares the server's ``list_tools`` with what we need
* adapter-level rejection of every non-limit order
* shadow mode: ``place_order`` raises ``ShadowModeError`` *before* touching the MCP session
* every tool result is redacted (account numbers etc.) before it is returned or logged

What is deliberately NOT implemented yet: the argument/result field mapping for each
tool. The plan says not to invent API fields; Phase 3 records the real schemas in
``docs/robinhood_spike.md`` and then fills in the ``_map_*`` functions.
"""

from __future__ import annotations

from typing import Any, Protocol

from biotradeai.brokers.base import (
    Account,
    BrokerError,
    BrokerOrder,
    OrderBook,
    OrderReview,
    Position,
    Quote,
    ShadowModeError,
    ensure_limit_only,
)
from biotradeai.config import Mode
from biotradeai.redact import redact
from biotradeai.schemas import StagedOrder

MCP_URL = "https://agent.robinhood.com/mcp/trading"

# From Robinhood's docs; verified against list_tools at runtime (see verify_tools).
REQUIRED_TOOLS = frozenset({
    "get_accounts", "get_portfolio", "get_equity_quotes", "get_equity_price_book",
    "review_equity_order", "place_equity_order", "cancel_equity_order",
    "get_equity_orders", "get_equity_positions",
})
LIVE_REQUIRED_TOOLS = frozenset({"get_trade_approvals"})


class MCPToolCaller(Protocol):
    """Minimal surface of an MCP client session (sync wrapper). Faked in tests."""

    def list_tools(self) -> list[str]: ...
    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


class SchemaNotVerifiedError(BrokerError):
    """Raised for calls whose Robinhood field mapping has not been verified (Phase 3)."""


class RobinhoodBroker:
    name = "robinhood-mcp"

    def __init__(self, session: MCPToolCaller, *, mode: Mode) -> None:
        if mode is Mode.PAPER:
            raise BrokerError("Robinhood adapter is for shadow/live modes; paper uses Alpaca")
        self._s = session
        self.mode = mode

    # ------------------------------------------------------------------ discovery

    def verify_tools(self) -> set[str]:
        """Return the missing tool names (empty set = OK)."""
        available = set(self._s.list_tools())
        needed = REQUIRED_TOOLS | (LIVE_REQUIRED_TOOLS if self.mode is Mode.LIVE else set())
        return set(needed - available)

    def _call(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            out = self._s.call_tool(tool, args)
        except Exception as exc:
            raise BrokerError(f"robinhood {tool} failed: {exc}") from exc
        cleaned: dict[str, Any] = redact(out)
        return cleaned

    def _unverified(self, what: str) -> SchemaNotVerifiedError:
        return SchemaNotVerifiedError(
            f"{what}: Robinhood MCP field mapping not verified yet (Phase 3 spike)"
        )

    # ------------------------------------------------------------------ protocol

    def get_account(self) -> Account:
        raise self._unverified("get_account")

    def get_quote(self, ticker: str) -> Quote:
        raise self._unverified("get_quote")

    def get_order_book(self, ticker: str) -> OrderBook:
        raise self._unverified("get_order_book")

    def review_order(self, order: StagedOrder) -> OrderReview:
        ensure_limit_only(order)
        raise self._unverified("review_order")

    def place_order(self, order: StagedOrder, *, client_order_id: str) -> BrokerOrder:
        ensure_limit_only(order)
        if self.mode is Mode.SHADOW:
            raise ShadowModeError("shadow mode: place_order is forbidden")
        raise self._unverified("place_order")

    def cancel_order(self, order_id: str) -> None:
        raise self._unverified("cancel_order")

    def get_positions(self) -> list[Position]:
        raise self._unverified("get_positions")

    def get_orders(self, *, open_only: bool = True) -> list[BrokerOrder]:
        raise self._unverified("get_orders")
