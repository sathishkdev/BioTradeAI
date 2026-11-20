"""Agent CLI: run the buy graph (Research ToT -> Risk -> human approval) on a fixture.

    biotradeai evaluate --fixture data/fixtures/scenarios/onco.yaml
    biotradeai evaluate --fixture ... --llm ollama:llama3.2:3b   # real model answers

Fixture runs are a DRY RUN: execution goes to an in-memory FakeBroker seeded with the
fixture's quote and equity, never to Alpaca or Robinhood. Time is simulated at the
fixture's ``as_of`` so freshness checks behave as they did on that date.
"""

from __future__ import annotations

import argparse
import getpass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from biotradeai.agents.base_rates import load_base_rates
from biotradeai.agents.llm import LLMClient, make_llm_client
from biotradeai.brokers.approval import ApprovalAuthority, ApprovalError, load_secret
from biotradeai.brokers.execution import ExecutionService
from biotradeai.brokers.fake_broker import FakeBroker
from biotradeai.cli_trading import _ask
from biotradeai.config import load_config
from biotradeai.graph.buy_graph import (
    BuyGraphDeps,
    RunOutcome,
    build_buy_graph,
    resume_run,
    sqlite_checkpointer,
    start_run,
)
from biotradeai.scenario_fixtures import load_fixture
from biotradeai.schemas import ApprovalToken, Decision, StagedOrder
from biotradeai.store.sqlite_store import SQLiteStore


def _print_outcome(out: RunOutcome) -> None:
    s = out.state
    print("=" * 72)
    print(f" {s.ticker}  {s.entity_id}  {s.catalyst_event} on {s.catalyst_date}   "
          f"run {s.run_id}")
    print("=" * 72)
    if s.evidence:
        print(f" evidence: {s.evidence.status.value}  confidence {s.evidence.confidence:.3f}  "
              f"sources {len(s.evidence.sources)}")
        if s.evidence.inconclusive_reason:
            print(f"   reason: {s.evidence.inconclusive_reason}")
    if s.runway_months is not None:
        print(f" runway (code): {s.runway_months:.2f} months")
    if s.tot_nodes:
        print("\n Tree of Thoughts   (base -> final probability, conditional on parent)")
        for n in sorted(s.tot_nodes, key=lambda n: n.node_id):
            tag = f"  PRUNED: {n.prune_reason}" if n.pruned else ""
            print(f"   {'  ' * (n.depth - 1)}{n.label:<22} {n.base_rate:.3f} -> "
                  f"{n.probability:.3f}  step {n.step_return:+.2f}{tag}")
            if abs(n.adjustment) > 1e-9 or n.justification.startswith("critic"):
                print(f"   {'  ' * (n.depth - 1)}  why: {n.justification}")
    if s.scenarios:
        print("\n Scenarios (EV is over ALL of them, probabilities sum to 1)")
        for sc in s.scenarios.scenarios:
            print(f"   p={sc.probability:.4f}  return {sc.expected_return:+.3f}  {sc.label}")
    if s.research_ev is not None:
        print(f"\n EV  research {s.research_ev:+.4f}   risk (recomputed) "
              f"{(s.risk_ev if s.risk_ev is not None else float('nan')):+.4f}")
    if out.risk:
        print("\n Risk checks")
        for c in out.risk["checks"]:
            print(f"   [{'ok' if c['ok'] else 'FAIL'}] {c['rule']:<15} {c['reason']}")
    print(f"\n DECISION: {s.decision.value if s.decision else 'n/a'}")
    if s.decision is Decision.INCONCLUSIVE:
        print("   -> needs human review of the sources; nothing staged.")
    if s.staged_order:
        o = s.staged_order
        print(f"   staged: BUY {o.qty} {o.ticker} LIMIT @ {o.limit_price}  "
              f"(notional ${(o.limit_price or 0) * o.qty:,.2f})")


def _llm(spec: str | None, role: str, fixture_llm: LLMClient) -> LLMClient:
    return fixture_llm if spec in (None, "fixture") else make_llm_client(role, spec)


def cmd_evaluate(args: argparse.Namespace) -> int:
    cfg = load_config()
    rates = load_base_rates()
    fx = load_fixture(Path(args.fixture), cfg)
    gen = _llm(args.llm, "research", fx.fake_llm())
    crit = _llm(args.critic_llm or args.llm, "critic", fx.fake_llm())
    store = SQLiteStore()
    if not rates.reviewed:
        print("WARNING: data/base_rates.yaml is not marked reviewed by a human yet.")

    broker = FakeBroker(equity=fx.inputs.portfolio.equity,
                        last_equity=fx.inputs.portfolio.equity,
                        quotes={fx.catalyst.ticker: (fx.inputs.market.bid,
                                                     fx.inputs.market.ask)})
    authority: ApprovalAuthority | None = None

    def executor(order: StagedOrder, token: ApprovalToken, run_id: str) -> dict[str, Any]:
        if authority is None:  # pragma: no cover - set before any approval
            return {"status": "rejected", "reasons": ["no approval authority"]}
        r = ExecutionService(broker, store, cfg, authority).execute(order, token, run_id=run_id)
        return {"status": r.status.value, "reasons": list(r.reasons), "order_id": r.order_id,
                "broker": broker.name, "dry_run": True}

    as_of = fx.catalyst.as_of
    deps = BuyGraphDeps(cfg, rates, gen, crit, store, executor=executor, clock=lambda: as_of)
    run_id = f"{fx.catalyst.run_id}-{datetime.now(UTC):%Y%m%dT%H%M%S}"
    catalyst = fx.catalyst.model_copy(update={"run_id": run_id})
    print(f"generator {gen.provider}:{gen.model}  critic {crit.provider}:{crit.model}  "
          f"simulated time {as_of.isoformat()}  (DRY RUN, FakeBroker)")
    with sqlite_checkpointer() as saver_ctx:
        app = build_buy_graph(deps, saver_ctx)
        out = start_run(app, catalyst, fx.inputs)
        _print_outcome(out)
        if not out.awaiting_approval:
            return 0
        secret: bytes | None = None
        try:
            secret = load_secret()
        except ApprovalError as exc:
            print(f"\ncannot approve: {exc}\nRecording the order as rejected.")
        answer = "reject" if args.auto_reject or secret is None else _ask(
            "\nType 'approve' to approve this order, anything else rejects: ").lower()
        approved = answer == "approve"
        token = None
        if approved and secret is not None and out.state.staged_order is not None:
            authority = ApprovalAuthority(
                secret, ttl=timedelta(minutes=cfg.orders.approval_ttl_minutes))
            token = authority.issue(out.state.staged_order)
        out = resume_run(app, run_id, approved=approved, operator=getpass.getuser(),
                         token=token)
    if not approved:
        print("rejected by human. nothing sent.")
        return 1
    ex = out.execution or {}
    print(f"execution (dry run on {ex.get('broker', 'fake')}): {ex.get('status')}  "
          f"order id {ex.get('order_id')}")
    for r in ex.get("reasons", []):
        print(f"  reason: {r}")
    return 0 if ex.get("status") == "placed" else 1


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    ev = sub.add_parser("evaluate", help="run Research ToT -> Risk -> approval on a fixture")
    ev.add_argument("--fixture", required=True,
                    help=f"scenario YAML, e.g. {Path('data/fixtures/scenarios/onco.yaml')}")
    ev.add_argument("--llm", help="generator model spec, or 'fixture' (default: scripted)")
    ev.add_argument("--critic-llm", help="critic model spec (default: same as --llm)")
    ev.add_argument("--auto-reject", action="store_true", help="reject at the approval step")
    ev.set_defaults(fn=cmd_evaluate)

