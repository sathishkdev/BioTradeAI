"""The ONLY module that may call ``Broker.place_order``.

``tests/unit/test_architecture.py`` fails the build if any other module under ``src/``
calls ``place_order``. Agents and the LangGraph supervisor never get a reference to a
broker; they hand a ``StagedOrder`` + human ``ApprovalToken`` to ``ExecutionService``.

Order of checks (any failure rejects, and every step is audit-logged):
 1. kill-switch not tripped
 2. mode allowed (live is blocked until the Phase 10 live gate exists)
 3. order type: limit only (config + adapter)
 4. approval token: authentic, unexpired, matches this exact order
 5. fresh broker state: account not blocked, drawdown within limit (may trip switch),
    position cap, concurrency cap, no shorting, spread (buys only)
 6. nonce consumed (single use) — then paper: place / shadow: review only
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import TypeVar

from biotradeai.brokers.approval import ApprovalAuthority, ApprovalError
from biotradeai.brokers.base import (
    Broker,
    BrokerError,
    OrderRejectedError,
    OrderReview,
    ShadowModeError,
)
from biotradeai.config import GuardrailsConfig, Mode
from biotradeai.guardrails import rules
from biotradeai.guardrails.killswitch import KillSwitch
from biotradeai.schemas import ApprovalToken, OrderSide, StagedOrder
from biotradeai.store.sqlite_store import SQLiteStore

T = TypeVar("T")


class ExecStatus(StrEnum):
    PLACED = "placed"
    SHADOW_REVIEWED = "shadow_reviewed"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ExecutionResult:
    status: ExecStatus
    reasons: tuple[str, ...] = ()
    order_id: str | None = None
    review: OrderReview | None = None
    checks: tuple[rules.RuleResult, ...] = field(default=(), repr=False)


class _RejectError(Exception):
    def __init__(self, *reasons: str, checks: list[rules.RuleResult] | None = None) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons
        self.checks = checks or []


class ExecutionService:
    def __init__(self, broker: Broker, store: SQLiteStore, cfg: GuardrailsConfig,
                 approvals: ApprovalAuthority) -> None:
        self._broker = broker
        self._store = store
        self._cfg = cfg
        self._approvals = approvals
        self.killswitch = KillSwitch(store, cfg, broker)

    # ------------------------------------------------------------------ helpers

    def _audit(self, event: str, order: StagedOrder, run_id: str | None, **extra: object) -> None:
        self._store.append_audit(event, {"order": order.model_dump(mode="json"), **extra},
                                 run_id=run_id, ticker=order.ticker)

    def _broker_call(self, what: str, fn: Callable[[], T]) -> T:
        try:
            out = fn()
        except (OrderRejectedError, ShadowModeError) as exc:
            raise _RejectError(str(exc)) from exc  # policy refusal, not an API fault
        except BrokerError as exc:
            self.killswitch.record_api_error(f"{what}: {exc}")
            raise _RejectError(f"broker error during {what}: {exc}") from exc
        self.killswitch.record_api_success()
        return out

    def _pre_trade_checks(self, order: StagedOrder) -> list[rules.RuleResult]:
        acct = self._broker_call("get_account", self._broker.get_account)
        if acct.trading_blocked:
            raise _RejectError("broker reports trading is blocked on this account")
        if self.killswitch.check_drawdown(start_equity=acct.last_equity, equity=acct.equity):
            raise _RejectError("kill-switch tripped on drawdown")
        positions = self._broker_call("get_positions", self._broker.get_positions)
        held = {p.ticker: p for p in positions if p.qty > 0}
        existing = held.get(order.ticker)
        results = [
            rules.check_position_size(
                order, equity=acct.equity,
                existing_value=existing.market_value if existing else Decimal(0), cfg=self._cfg),
            rules.check_concurrency(open_tickers=set(held), ticker=order.ticker,
                                    side=order.side, cfg=self._cfg),
            rules.check_sell_quantity(order, held_qty=existing.qty if existing else 0),
        ]
        if order.side is OrderSide.BUY:
            q = self._broker_call("get_quote", lambda: self._broker.get_quote(order.ticker))
            results.append(rules.check_spread(bid=q.bid, ask=q.ask, cfg=self._cfg))
            results.append(rules.check_ttl(q.timestamp, now=datetime.now(UTC), cfg=self._cfg,
                                           what="quote"))
        return results

    # ------------------------------------------------------------------ entry point

    def execute(self, order: StagedOrder, token: ApprovalToken | None, *,
                run_id: str | None = None) -> ExecutionResult:
        self._audit("execution.request", order, run_id, mode=self._cfg.mode.value,
                    broker=self._broker.name)
        checks: list[rules.RuleResult] = []
        try:
            self.killswitch.ensure_not_tripped()
            if self._cfg.mode is Mode.LIVE:
                raise _RejectError("live mode is blocked until the live gate (Phase 10) exists")

            t = rules.check_order_type(order, self._cfg)
            checks.append(t)
            if not t.ok:
                raise _RejectError(t.reason, checks=checks)

            if token is None:
                raise _RejectError("no human approval token", checks=checks)
            try:
                self._approvals.verify(token, order)
            except ApprovalError as exc:
                raise _RejectError(str(exc), checks=checks) from exc

            checks.extend(self._pre_trade_checks(order))
            bad = rules.failures(checks)
            if bad:
                raise _RejectError(*(f"{r.rule}: {r.reason}" for r in bad), checks=checks)

            if not self._store.consume_nonce(token.nonce):
                raise _RejectError("approval token already used", checks=checks)

            client_id = f"btai-{token.nonce}"
            if self._cfg.mode is Mode.SHADOW:
                review = self._broker_call("review_order",
                                           lambda: self._broker.review_order(order))
                self._audit("execution.would_have_placed", order, run_id,
                            client_order_id=client_id, warnings=list(review.warnings))
                return ExecutionResult(ExecStatus.SHADOW_REVIEWED, review=review,
                                       checks=tuple(checks))

            placed = self._broker_call(
                "place_order", lambda: self._broker.place_order(order, client_order_id=client_id))
            self._audit("execution.placed", order, run_id, broker_order_id=placed.order_id,
                        client_order_id=client_id, status=placed.status,
                        checks=[c.__dict__ for c in checks])
            return ExecutionResult(ExecStatus.PLACED, order_id=placed.order_id,
                                   checks=tuple(checks))
        except _RejectError as rej:
            all_checks = checks + [c for c in rej.checks if c not in checks]
            self._audit("execution.rejected", order, run_id, reasons=list(rej.reasons),
                        checks=[c.__dict__ for c in all_checks])
            return ExecutionResult(ExecStatus.REJECTED, reasons=tuple(rej.reasons),
                                   checks=tuple(all_checks))
        except Exception as exc:  # kill-switch tripped etc.
            self._audit("execution.rejected", order, run_id, reasons=[str(exc)])
            return ExecutionResult(ExecStatus.REJECTED, reasons=(str(exc),))
