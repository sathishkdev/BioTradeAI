"""Trading-related CLI commands: account, positions, order (with human approval), killswitch."""

from __future__ import annotations

import argparse
import getpass
from datetime import timedelta
from decimal import Decimal, InvalidOperation

from pydantic import ValidationError

from biotradeai.brokers.alpaca_broker import AlpacaBroker
from biotradeai.brokers.approval import ApprovalAuthority, load_secret
from biotradeai.brokers.base import Broker
from biotradeai.brokers.execution import ExecStatus, ExecutionService
from biotradeai.brokers.fake_broker import FakeBroker
from biotradeai.config import load_config
from biotradeai.guardrails import rules
from biotradeai.guardrails.killswitch import KillSwitch
from biotradeai.schemas import OrderSide, OrderType, StagedOrder
from biotradeai.store.sqlite_store import SQLiteStore


def _ask(prompt: str) -> str:
    # PowerShell can prefix piped stdin with a UTF-8 BOM; never let it change an answer.
    return input(prompt).lstrip("﻿").strip()


def _broker(name: str) -> Broker:
    if name == "fake":
        return FakeBroker()
    return AlpacaBroker()


def cmd_account(args: argparse.Namespace) -> int:
    b = _broker(args.broker)
    a = b.get_account()
    print(f"broker       : {b.name}")
    print(f"equity       : ${a.equity:,.2f}")
    print(f"last equity  : ${a.last_equity:,.2f}  (start-of-day baseline)")
    print(f"cash         : ${a.cash:,.2f}")
    print(f"buying power : ${a.buying_power:,.2f}")
    print(f"drawdown     : {rules.drawdown(a.last_equity, a.equity):.2%}")
    return 0


def cmd_positions(args: argparse.Namespace) -> int:
    b = _broker(args.broker)
    pos = b.get_positions()
    orders = b.get_orders(open_only=True)
    print(f"{len(pos)} position(s)")
    for p in pos:
        print(f"  {p.ticker:<6} qty {p.qty:>6}  avg ${p.avg_entry_price:,.2f}  "
              f"value ${p.market_value:,.2f}")
    print(f"{len(orders)} open order(s)")
    for o in orders:
        print(f"  {o.order_id}  {o.side.value} {o.qty} {o.ticker} {o.order_type.value} "
              f"@ {o.limit_price}  [{o.status}]")
    return 0


def cmd_order(args: argparse.Namespace) -> int:
    cfg = load_config()
    store = SQLiteStore()
    broker = _broker(args.broker)
    try:
        price = Decimal(args.limit) if args.limit is not None else None
        order = StagedOrder(ticker=args.ticker.upper(), side=OrderSide(args.side), qty=args.qty,
                            order_type=OrderType(args.type), limit_price=price)
    except (ValidationError, InvalidOperation, ValueError) as exc:
        print(f"invalid order: {exc}")
        return 2

    acct = broker.get_account()
    q = broker.get_quote(order.ticker)
    max_qty = rules.max_position_qty(acct.equity, price or Decimal(0), Decimal(0), cfg)
    spread = rules.spread_pct(q.bid, q.ask)
    print("=" * 60)
    print(f" ORDER FOR HUMAN APPROVAL   mode={cfg.mode.value}  broker={broker.name}")
    print("=" * 60)
    print(f" {order.side.value.upper()} {order.qty} {order.ticker} {order.order_type.value} "
          f"@ {order.limit_price}")
    if price is not None:
        print(f" notional        : ${price * order.qty:,.2f}")
    print(f" quote           : bid {q.bid} / ask {q.ask}  spread "
          f"{'n/a' if spread is None else f'{spread:.2%}'}")
    print(f" equity          : ${acct.equity:,.2f}   max qty at cap: {max_qty}")
    print(" EV / confidence / runway : n/a (manual order; agents arrive in Phase 5)")
    print("=" * 60)
    answer = _ask("Type 'approve' to approve, anything else rejects: ").lower()
    if answer != "approve":
        store.append_audit("approval.rejected", {"order": order.model_dump(mode="json")},
                           ticker=order.ticker)
        print("rejected by human. nothing sent.")
        return 1

    authority = ApprovalAuthority(load_secret(),
                                  ttl=timedelta(minutes=cfg.orders.approval_ttl_minutes))
    token = None
    if order.order_type is OrderType.LIMIT and order.limit_price is not None:
        token = authority.issue(order)
        store.append_audit("approval.granted", {"order": order.model_dump(mode="json"),
                                                "expiry": token.expiry.isoformat(),
                                                "operator": getpass.getuser()},
                           ticker=order.ticker)
    result = ExecutionService(broker, store, cfg, authority).execute(order, token)
    print(f"result: {result.status.value}")
    for c in result.checks:
        print(f"  [{'ok' if c.ok else 'FAIL'}] {c.rule:<14} {c.reason}")
    for r in result.reasons:
        print(f"  reason: {r}")
    if result.order_id:
        print(f"  broker order id: {result.order_id}")
    return 0 if result.status is not ExecStatus.REJECTED else 1


def cmd_killswitch(args: argparse.Namespace) -> int:
    cfg = load_config()
    store = SQLiteStore()
    if args.action == "status":
        ks = store.get_killswitch()
        print(f"tripped: {ks.tripped}  reason: {ks.reason}  changed: {ks.changed_at} "
              f"by {ks.changed_by}  consecutive api errors: {store.get_api_errors()}")
        return 0
    broker = _broker(args.broker)
    switch = KillSwitch(store, cfg, broker)
    if args.action == "trip":
        cancelled = switch.trip(f"manual trip by {getpass.getuser()}")
        print(f"tripped. cancelled {len(cancelled)} open order(s).")
        return 0
    confirm = _ask("Type RESET to clear the kill-switch: ")
    switch.reset_by_human(operator=getpass.getuser(), confirmation=confirm)
    print("kill-switch reset.")
    return 0


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    def broker_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument("--broker", choices=["alpaca", "fake"], default="alpaca",
                       help="alpaca = Alpaca paper (default); fake = in-memory dry run")

    a = sub.add_parser("account", help="show broker account summary")
    broker_arg(a)
    a.set_defaults(fn=cmd_account)
    p = sub.add_parser("positions", help="show positions and open orders")
    broker_arg(p)
    p.set_defaults(fn=cmd_positions)
    o = sub.add_parser("order", help="stage an order, approve it, execute via guardrails")
    o.add_argument("ticker")
    o.add_argument("--side", choices=["buy", "sell"], default="buy")
    o.add_argument("--qty", type=int, required=True)
    o.add_argument("--limit", help="limit price")
    o.add_argument("--type", default="limit", choices=[t.value for t in OrderType],
                   help="anything but limit will be rejected (try it)")
    broker_arg(o)
    o.set_defaults(fn=cmd_order)
    k = sub.add_parser("killswitch", help="status | trip | reset")
    k.add_argument("action", choices=["status", "trip", "reset"])
    broker_arg(k)
    k.set_defaults(fn=cmd_killswitch)
