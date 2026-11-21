"""Approve-later queue: orders the paper cycle staged and parked for a human decision.

Approving from the UI does exactly what typing ``approve`` in the terminal does, plus
two extra checks because time has passed since staging:

1. the order is still pending, not expired, and the mode has not changed;
2. you typed the confirmation (the ticker; for live orders, the ticker and LIVE);
3. the kill-switch is not tripped;
4. the price has not drifted more than ``orders.max_price_drift_pct`` since staging;
5. a single-use HMAC token is issued and the order goes to ``brokers/execution.py``,
   which re-runs every pre-trade rule on fresh broker data before placing it.

An entry resumes its checkpointed buy graph (so the audit trail is one run); an exit
goes straight to the execution service.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

from biotradeai.agents.base_rates import load_base_rates
from biotradeai.agents.llm import FakeLLMClient
from biotradeai.brokers.alpaca_broker import AlpacaBroker
from biotradeai.brokers.approval import ApprovalAuthority, load_secret
from biotradeai.brokers.base import Broker
from biotradeai.brokers.execution import ExecStatus, ExecutionService
from biotradeai.brokers.fake_broker import FakeBroker
from biotradeai.config import GuardrailsConfig, Mode, load_config
from biotradeai.graph.buy_graph import (
    BuyGraphDeps,
    build_buy_graph,
    get_outcome,
    resume_run,
    sqlite_checkpointer,
)
from biotradeai.guardrails.killswitch import KillSwitch
from biotradeai.operator.db import OperatorDB, Pending, utcnow
from biotradeai.paper.runner import DeferSink, record_entry
from biotradeai.schemas import ApprovalToken, StagedOrder

BrokerFactory = Callable[[str], Broker]


class TradeError(RuntimeError):
    pass


def default_broker(name: str) -> Broker:
    if name == "alpaca":
        return AlpacaBroker()
    if name == "fake":
        return FakeBroker()
    raise TradeError(f"unknown broker {name!r}")


@dataclass(frozen=True)
class Destination:
    label: str
    tone: str  # paper | dry | shadow | live
    note: str


def destination(broker: str, mode: str) -> Destination:
    if broker == "fake":
        return Destination("Fake broker · dry run", "dry", "Nothing leaves this computer.")
    if mode == Mode.SHADOW.value:
        return Destination("Robinhood · SHADOW", "shadow",
                           "Simulation only: the order is reviewed, never placed.")
    if mode == Mode.LIVE.value:
        return Destination("Robinhood · LIVE", "live",
                           "Real money. Robinhood will also ask you to approve in its app.")
    return Destination("Alpaca · PAPER", "paper", "Simulated money; fills like a real order.")


def required_confirmation(p: Pending) -> str:
    return f"{p.ticker} LIVE" if p.mode == Mode.LIVE.value else p.ticker


def make_defer_sink(db: OperatorDB, *, broker: str, cfg: GuardrailsConfig) -> DeferSink:
    def sink(kind: str, order: StagedOrder, pack: dict[str, Any], run_id: str) -> int:
        return db.add_pending(
            kind=kind, run_id=run_id, ticker=order.ticker,
            order=order.model_dump(mode="json"), pack=pack, broker=broker,
            mode=cfg.mode.value,
            expires_at=utcnow() + timedelta(hours=cfg.orders.pending_approval_hours))
    return sink


def expire_stale(db: OperatorDB) -> list[int]:
    out = []
    for p in db.pending():
        if p.expired and db.decide_pending(p.id, status="expired", by="system",
                                           note="nobody decided before it expired"):
            db.store.append_audit("approval.expired", {"pending_id": p.id, "kind": p.kind,
                                                       "order": p.order},
                                  ticker=p.ticker, run_id=p.run_id)
            out.append(p.id)
    return out


def _ref_price(bid: Decimal, ask: Decimal) -> Decimal:
    if bid > 0 and ask > 0:
        return (bid + ask) / 2
    return max(bid, ask)


class TradeApprovals:
    def __init__(self, db: OperatorDB, *, cfg: GuardrailsConfig | None = None,
                 broker_factory: BrokerFactory = default_broker,
                 authority: ApprovalAuthority | None = None,
                 checkpointer: Any | None = None) -> None:
        self.db = db
        self.cfg = cfg or load_config()
        self._brokers = broker_factory
        self._authority = authority
        self._saver = checkpointer  # tests pass one; production opens data/checkpoints.db

    def _auth(self) -> ApprovalAuthority:
        if self._authority is None:
            self._authority = ApprovalAuthority(
                load_secret(), ttl=timedelta(minutes=self.cfg.orders.approval_ttl_minutes))
        return self._authority

    def _load(self, pending_id: int) -> Pending:
        p = self.db.get_pending(pending_id)
        if p is None:
            raise TradeError("no such order")
        if p.status != "pending":
            raise TradeError(f"this order is already {p.status}")
        if p.expired:
            expire_stale(self.db)
            raise TradeError("this order expired; the next cycle re-evaluates the catalyst")
        return p

    # ------------------------------------------------------------------ approve

    def approve(self, pending_id: int, *, typed: str, operator: str) -> dict[str, Any]:
        p = self._load(pending_id)
        want = required_confirmation(p)
        if typed.strip().upper() != want.upper():
            raise TradeError(f"type {want!r} to confirm")
        if self.cfg.mode.value != p.mode:
            raise TradeError(f"the mode changed since staging ({p.mode} -> "
                             f"{self.cfg.mode.value}); reject it and let the next cycle "
                             "re-evaluate")
        if self.db.store.get_killswitch().tripped:
            raise TradeError("the kill-switch is tripped; nothing can be placed")
        order = StagedOrder.model_validate(p.order)
        broker = self._brokers(p.broker)
        q = broker.get_quote(order.ticker)
        ref, limit = _ref_price(q.bid, q.ask), order.limit_price or Decimal(0)
        if ref <= 0 or limit <= 0:
            raise TradeError("no usable quote right now; try again in a moment")
        drift = abs(ref - limit) / limit
        if drift > Decimal(str(self.cfg.orders.max_price_drift_pct)):
            raise TradeError(f"the price moved {drift:.1%} since staging (limit "
                             f"{limit}, now ~{ref:.2f}); reject it and re-run the cycle")
        token = self._auth().issue(order)
        self.db.store.append_audit("approval.granted", {
            "order": p.order, "expiry": token.expiry.isoformat(), "source": "ui",
            "operator": operator, "pending_id": p.id,
            "destination": destination(p.broker, p.mode).label},
            ticker=p.ticker, run_id=p.run_id)
        res = (self._resume_entry(p, broker, token, operator, approved=True)
               if p.kind == "entry" else self._execute(broker, order, token, p.run_id))
        ok = res["status"] in (ExecStatus.PLACED.value, ExecStatus.SHADOW_REVIEWED.value)
        if ok and p.kind == "entry" and res["status"] == ExecStatus.PLACED.value:
            record_entry(self.db.store, order, event=str(p.pack.get("event", "")),
                         catalyst_date=date.fromisoformat(str(p.pack["catalyst_date"])),
                         opened_on=datetime.now(UTC).date())
        self.db.decide_pending(p.id, status="approved" if ok else "failed",
                               by=f"ui:{operator}", note=None, result=res)
        return res

    def _execute(self, broker: Broker, order: StagedOrder, token: ApprovalToken,
                 run_id: str) -> dict[str, Any]:
        r = ExecutionService(broker, self.db.store, self.cfg, self._auth()).execute(
            order, token, run_id=run_id)
        return {"status": r.status.value, "reasons": list(r.reasons), "order_id": r.order_id}

    def _resume_entry(self, p: Pending, broker: Broker, token: ApprovalToken | None,
                      operator: str, *, approved: bool) -> dict[str, Any]:
        result: dict[str, Any] = {}

        def executor(order: StagedOrder, tok: ApprovalToken, run_id: str) -> dict[str, Any]:
            result.update(self._execute(broker, order, tok, run_id))
            return result

        # Only the human step and execution remain, so the LLM clients are never called.
        idle = FakeLLMClient(lambda _msgs: "{}", model="unused")
        deps = BuyGraphDeps(self.cfg, load_base_rates(), idle, idle, self.db.store,
                            executor=executor, clock=lambda: datetime.now(UTC))

        def go(saver: Any) -> None:
            app = build_buy_graph(deps, saver)
            if not get_outcome(app, p.run_id).awaiting_approval:
                raise TradeError("this run is no longer waiting for approval")
            resume_run(app, p.run_id, approved=approved, operator=f"ui:{operator}",
                       token=token)

        if self._saver is not None:
            go(self._saver)
        else:
            with sqlite_checkpointer() as saver:
                go(saver)
        return result or {"status": "human_rejected" if not approved else "rejected",
                          "reasons": [] if not approved else ["no execution result"]}

    # ------------------------------------------------------------------ reject

    def reject(self, pending_id: int, *, reason: str, operator: str) -> None:
        p = self._load(pending_id)
        if not reason.strip():
            raise TradeError("give a short reason (it is used in the daily review)")
        self.db.store.append_audit("approval.rejected", {
            "order": p.order, "reason": reason, "source": "ui", "operator": operator,
            "pending_id": p.id}, ticker=p.ticker, run_id=p.run_id)
        if p.kind == "entry":
            # If the run already ended, the rejection is still recorded below.
            with contextlib.suppress(TradeError):
                self._resume_entry(p, FakeBroker(), None, operator, approved=False)
        self.db.decide_pending(p.id, status="rejected", by=f"ui:{operator}", note=reason)

    # ------------------------------------------------------------------ helpers

    def killswitch(self) -> KillSwitch:
        return KillSwitch(self.db.store, self.cfg, None)
