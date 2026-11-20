"""Paper-trading CLI (Phase 8).

    biotradeai paper run              # one cycle: exits, then entries (asks you to approve)
    biotradeai paper run --broker fake --auto-reject   # dry run, never approves
    biotradeai paper review           # today's audit summary (or --date YYYY-MM-DD)
    biotradeai paper gate             # paper days / breaches toward the live gate

Needs: ALPACA_API_KEY/SECRET (paper), APPROVAL_HMAC_SECRET, SEC_USER_AGENT, rows in
data/catalysts.csv, drug names in data/synonyms.yaml, and documents ingested with
"biotradeai rag ingest". Schedule "paper run" with Task Scheduler / cron.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime, timedelta
from typing import Any

from biotradeai.agents.base_rates import load_base_rates
from biotradeai.agents.llm import make_llm_client
from biotradeai.brokers.alpaca_broker import AlpacaBroker
from biotradeai.brokers.approval import ApprovalAuthority, load_secret
from biotradeai.brokers.base import Broker
from biotradeai.brokers.fake_broker import FakeBroker
from biotradeai.cli_trading import _ask
from biotradeai.config import load_config
from biotradeai.data.catalysts import load_catalysts
from biotradeai.data.clinicaltrials import ClinicalTrialsClient
from biotradeai.data.edgar import EdgarClient
from biotradeai.data.outcomes import load_outcomes
from biotradeai.data.pollers import TrialDatePoller
from biotradeai.events import CompletionDateChange
from biotradeai.graph.buy_graph import sqlite_checkpointer
from biotradeai.paper.gather import EdgarFilings, LiveGatherer, edgar_runway
from biotradeai.paper.review import daily_review, paper_gate, render_review
from biotradeai.paper.runner import Approver, PaperRunner
from biotradeai.rag.embeddings import make_embedder
from biotradeai.rag.index import VectorIndex
from biotradeai.rag.ontology import load_ontology
from biotradeai.rag.retrieve import Retriever
from biotradeai.schemas import StagedOrder
from biotradeai.store.sqlite_store import SQLiteStore
from biotradeai.watchlist.catalysts import agent_catalysts, agent_outcomes
from biotradeai.watchlist.model import Status, Watchlist


def interactive_approver(auto_reject: bool) -> Approver:
    def approve(order: StagedOrder, pack: dict[str, Any]) -> bool:
        print("=" * 72)
        print(f" {str(pack['kind']).upper()} FOR HUMAN APPROVAL: {order.side.value.upper()} "
              f"{order.qty} {order.ticker} LIMIT @ {order.limit_price}")
        print("=" * 72)
        for k, v in pack.items():
            if k not in ("kind", "checks"):
                print(f"  {k:<14} {json.dumps(v, default=str)[:200]}")
        for c in pack.get("checks", []):
            print(f"  [{'ok' if c['ok'] else 'FAIL'}] {c['rule']:<15} {c['reason']}")
        if auto_reject:
            print("  --auto-reject: rejected")
            return False
        return _ask("Type 'approve' to approve, anything else rejects: ").lower() == "approve"
    return approve


class WatchlistTrials:
    """Time-stop source: the trials of watchlist drugs held as positions."""

    def __init__(self, wl: Watchlist, poller: TrialDatePoller) -> None:
        self._wl, self._poller = wl, poller

    def poll(self, ticker: str) -> list[CompletionDateChange]:
        out: list[CompletionDateChange] = []
        for e in self._wl.with_status(Status.APPROVED, Status.PAUSED):
            if e.ticker == ticker:
                for nct in e.trial_ids:
                    out += self._poller.poll(ticker, nct)
        return out


def cmd_paper_run(args: argparse.Namespace) -> int:
    cfg = load_config()
    store = SQLiteStore()
    broker: Broker = FakeBroker() if args.broker == "fake" else AlpacaBroker()
    edgar = EdgarClient()
    embedder = make_embedder(args.embedder)
    retriever = Retriever(VectorIndex(embedder.name), embedder, load_ontology(), cfg)
    gatherer = LiveGatherer(retriever=retriever, runway=edgar_runway(edgar, cfg),
                            broker=broker, cfg=cfg)
    authority = ApprovalAuthority(load_secret(),
                                  ttl=timedelta(minutes=cfg.orders.approval_ttl_minutes))
    with sqlite_checkpointer() as saver:
        runner = PaperRunner(
            cfg=cfg, store=store, broker=broker, gatherer=gatherer, rates=load_base_rates(),
            generator=make_llm_client("research", args.llm),
            critic=make_llm_client("critic", args.critic_llm or args.llm),
            authority=authority, approver=interactive_approver(args.auto_reject),
            checkpointer=saver, filings=EdgarFilings(edgar, store),
            trials=WatchlistTrials(Watchlist(), TrialDatePoller(ClinicalTrialsClient(), store)))
        # Human-curated rows + agent-managed events for APPROVED watchlist drugs.
        wl, today = Watchlist(), datetime.now(UTC).date()
        rep = runner.cycle(load_catalysts() + agent_catalysts(store, wl, today),
                           load_outcomes() + agent_outcomes(store, wl))
    print(f"\ncycle {rep.started_at:%Y-%m-%d %H:%M} UTC on {broker.name}"
          + (f"   HALTED: {rep.halted}" if rep.halted else ""))
    if not rep.evaluated and not rep.skipped and not rep.orders and not rep.exits:
        print("  nothing to do (no upcoming catalysts in data/catalysts.csv, no positions)")
    for t, d in rep.evaluated:
        print(f"  evaluated {t}: {d}")
    for t, why in rep.skipped:
        print(f"  skipped   {t}: {why}")
    for o in rep.orders + rep.exits:
        print(f"  order     {o}")
    for b in rep.breaches:
        print(f"  BREACH    {b}")
    return 1 if rep.halted else 0


def cmd_paper_review(args: argparse.Namespace) -> int:
    day = date.fromisoformat(args.date) if args.date else datetime.now(UTC).date()
    r = daily_review(SQLiteStore(), day)
    print(render_review(r))
    return 0 if r.chain_ok and not r.breaches else 1


def cmd_paper_gate(_: argparse.Namespace) -> int:
    g = paper_gate(SQLiteStore(), load_config())
    print(f"paper days {g.paper_days}/{g.required_days} (since {g.first_day})  "
          f"breaches {g.breaches}  ->  {'ELIGIBLE' if g.eligible else 'not eligible'} for "
          "the live gate's paper condition")
    return 0 if g.eligible else 1


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("paper", help="Alpaca paper trading: run | review | gate")
    ps = p.add_subparsers(dest="paper_cmd", required=True)
    run = ps.add_parser("run", help="one cycle: exits then entries, with approval prompts")
    run.add_argument("--broker", choices=["alpaca", "fake"], default="alpaca")
    run.add_argument("--llm", help="research generator spec (default: env / config)")
    run.add_argument("--critic-llm", help="critic spec (default: same as --llm)")
    run.add_argument("--embedder", help="embedder spec (default: env / bge-small)")
    run.add_argument("--auto-reject", action="store_true", help="never approve (dry run)")
    run.set_defaults(fn=cmd_paper_run)
    rv = ps.add_parser("review", help="summarise a day of the audit log")
    rv.add_argument("--date", help="YYYY-MM-DD (default: today, UTC)")
    rv.set_defaults(fn=cmd_paper_review)
    ps.add_parser("gate", help="paper days and breaches toward the live gate").set_defaults(
        fn=cmd_paper_gate)
