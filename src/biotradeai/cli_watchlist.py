"""Watchlist CLI: scout proposals, approval cards, and the daily follow loop.

    biotradeai watchlist scout                  # propose company+drug pairs (weekly)
    biotradeai watchlist list [--status S]      # what's proposed / approved / paused
    biotradeai watchlist show ID                # the full approval card
    biotradeai watchlist approve ID [--note T]  # you approve: names go into synonyms.yaml
    biotradeai watchlist reject ID --reason T
    biotradeai watchlist retire ID --reason T
    biotradeai watchlist follow                 # daily: filings, trials, events, life cycle
    biotradeai watchlist events [ID]            # agent-managed catalyst events

Network use follows data/netpolicy.py (allowlist, polite rates, budgets). Use
``--spread-minutes`` on scheduled runs to spread requests over a time window.
"""

from __future__ import annotations

import argparse
import getpass
import json
from datetime import UTC, datetime
from typing import Any

from biotradeai.brokers.alpaca_broker import AlpacaBroker
from biotradeai.brokers.base import Broker, BrokerError
from biotradeai.config import load_config
from biotradeai.data.clinicaltrials import ClinicalTrialsClient
from biotradeai.data.edgar import EdgarClient
from biotradeai.data.secindex import SecDirectory
from biotradeai.rag.embeddings import make_embedder
from biotradeai.rag.index import VectorIndex
from biotradeai.rag.ontology import Entity, upsert_entity
from biotradeai.rag.pubmed import PubMedClient
from biotradeai.rag.sync import DocumentSync
from biotradeai.store.sqlite_store import SQLiteStore
from biotradeai.watchlist.follow import Follower
from biotradeai.watchlist.model import Status, WatchEntry, Watchlist
from biotradeai.watchlist.scout import Scout


def _today() -> Any:
    return datetime.now(UTC).date()


def _line(label: str, value: Any) -> None:
    if value not in (None, "", [], {}):
        print(f"  {label:<16} {value}")


def print_card(e: WatchEntry) -> None:
    c = e.card
    print("=" * 76)
    print(f" {e.id}   [{e.status.value}]   stage: {e.stage.value}   score {e.score}")
    print("=" * 76)
    co = c.get("company") or {}
    _line("company", f"{co.get('name')}  ({co.get('ticker')}, "
                     f"{'/'.join(co.get('exchanges') or [])}, SIC {co.get('sic')}, "
                     f"CIK {co.get('cik')})")
    _line("found via", ", ".join(c.get("sources") or []))
    _line("match", c.get("match"))
    rw = c.get("runway") or {}
    if "cash" in rw:
        months = (f"{rw['months']} months" if rw.get("months") is not None
                  else "not burning cash")
        _line("cash runway", f"{months} (cash ${float(rw['cash']):,.0f}, burn "
                             f"${float(rw['quarterly_burn']):,.0f}/qtr, as of "
                             f"{rw['balance_sheet_date']})")
    else:
        _line("cash runway", rw.get("error"))
    mk = c.get("market") or {}
    if mk:
        cap = mk.get("market_cap")
        _line("market", f"bid {mk.get('bid')} / ask {mk.get('ask')}, spread "
                        f"{mk.get('spread_pct')}"
                        + (f", market cap ~${float(cap) / 1e6:,.0f}M" if cap else "")
                        if "error" not in mk else mk["error"])
    d = c.get("drug") or {}
    _line("drug names", ", ".join(d.get("names") or e.drug_names))
    for t in d.get("trials") or []:
        print(f"  trial            {t['nct']} [{t['status']}] {t['title'][:70]}")
        _line("  endpoint", t.get("endpoint"))
        _line("  given with", ", ".join(t.get("with") or []))
        _line("  enrollment", t.get("enrollment"))
        _line("  completion", f"{t.get('primary_completion')} (sponsor estimate)")
    for k in c.get("catalysts") or []:
        print(f"  catalyst         {k['event']}  {k['when']} ({k['precision']}), "
              f"8-K filed {k['filed']}")
        print(f"                   \"{k['sentence'][:220]}\"")
        print(f"                   {k['url']}")
    _line("note", c.get("ctgov_note"))
    _line("Phase 3s due", (c.get("concentration") or {}).get(
        "phase3_trials_completing_in_3_to_12_months"))
    for f in c.get("red_flags") or []:
        print(f"  RED FLAG         {f}")
    fit = c.get("fit") or {}
    if fit.get("has_tradable_date"):
        _line("tradable", "yes: supported event with a company-stated date")
    elif fit.get("supported_event"):
        _line("tradable", "supported event type; becomes tradable once a company filing "
                          "states a date (the follow loop watches for it)")
    else:
        _line("tradable", "no (tracked only until base rates exist)")
    if c.get("previously"):
        _line("previously", json.dumps(c["previously"]))
    _line("pause reason", e.pause_reason)
    _line("retire reason", e.retire_reason)
    _line("check-in due", e.checkin_due)
    _line("review after", e.review_after)


def _broker(enabled: bool) -> Broker | None:
    if not enabled:
        return None
    try:
        return AlpacaBroker()
    except BrokerError as exc:
        print(f"(no market data: {exc})")
        return None


def cmd_scout(args: argparse.Namespace) -> int:
    cfg = load_config()
    wl = Watchlist()
    scout = Scout(edgar=EdgarClient(), ctgov=ClinicalTrialsClient(), store=SQLiteStore(),
                  watchlist=wl, cfg=cfg, today=_today(), broker=_broker(args.with_quotes),
                  spread_minutes=args.spread_minutes, max_proposals=args.max)
    rep = scout.run()
    print(f"scanned SEC index days {rep.days_scanned or 'none new'}; read {rep.filings_read} "
          f"biotech 8-K(s); saw {rep.trials_seen} Phase 3 trial(s); {rep.candidates} "
          "candidate pair(s)")
    if rep.filtered:
        print(f"filtered out: {rep.filtered}")
    if rep.cap_reached:
        print("soft cap reached: approve, reject or retire entries before new proposals")
    for i in rep.proposed:
        print(f"  proposed {i}   ->  biotradeai watchlist show {i}")
    for err in rep.errors[:5]:
        print(f"  note: {err}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    wl = Watchlist()
    rows = wl.entries if not args.status else wl.with_status(Status(args.status))
    if not rows:
        print("watchlist is empty" if not wl.entries else f"no {args.status} entries")
    for e in sorted(rows, key=lambda x: (x.status.value, -x.score)):
        extra = e.pause_reason or e.retire_reason or ", ".join(e.card.get("sources") or [])
        print(f"  {e.status.value:<9} {e.stage.value:<11} {e.id:<24} {e.company[:28]:<28} "
              f"{extra[:60]}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    print_card(Watchlist().get(args.id))
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    wl = Watchlist()
    e = wl.get(args.id)
    upsert_entity(Entity(e.entity_id, e.ticker, tuple(e.drug_names)))
    wl.approve(e.id, by=getpass.getuser(), note=args.note)
    wl.save()
    SQLiteStore().append_audit("watchlist.approved", {"id": e.id, "names": e.drug_names,
                                                      "note": args.note}, ticker=e.ticker)
    print(f"approved {e.id}; names added to data/synonyms.yaml. Next "
          "`biotradeai watchlist follow` collects its documents and events.")
    return 0


def cmd_reject(args: argparse.Namespace) -> int:
    wl = Watchlist()
    e = wl.reject(args.id, by=getpass.getuser(), reason=args.reason)
    wl.save()
    SQLiteStore().append_audit("watchlist.rejected", {"id": e.id, "reason": args.reason},
                               ticker=e.ticker)
    print(f"rejected {e.id}; it won't be proposed again unless something material changes")
    return 0


def cmd_retire(args: argparse.Namespace) -> int:
    wl = Watchlist()
    e = wl.retire(args.id, by=getpass.getuser(), reason=args.reason)
    wl.save()
    SQLiteStore().append_audit("watchlist.retired", {"id": e.id, "reason": args.reason,
                                                     "by": "human"}, ticker=e.ticker)
    print(f"retired {e.id}")
    return 0


def make_sync(store: SQLiteStore, edgar: EdgarClient | None,
              ctgov: ClinicalTrialsClient, embedder_spec: str | None) -> DocumentSync:
    emb = make_embedder(embedder_spec)
    return DocumentSync(store=store, cfg=load_config(), index=VectorIndex(emb.name),
                        embedder=emb, edgar=edgar, ctgov=ctgov, pubmed=PubMedClient())


def cmd_follow(args: argparse.Namespace) -> int:
    store = SQLiteStore()
    edgar, ctgov = EdgarClient(), ClinicalTrialsClient()
    f = Follower(directory=SecDirectory(edgar, store), ctgov=ctgov, store=store,
                 watchlist=Watchlist(), sync=make_sync(store, edgar, ctgov, args.embedder),
                 today=_today(), spread_minutes=args.spread_minutes)
    rep = f.run()
    print(f"followed {rep.entries} approved drug(s); {rep.filings} new filing(s); "
          f"documents added {rep.docs or 'none'}")
    for title, items in (("events", rep.events), ("outcomes", rep.outcomes),
                         ("stage", rep.stage), ("PAUSED (needs you)", rep.paused),
                         ("retired", rep.retired)):
        for x in items:
            print(f"  {title:<18} {x}")
    for err in rep.errors[:5]:
        print(f"  note: {err}")
    return 0


def cmd_events(args: argparse.Namespace) -> int:
    store = SQLiteStore()
    wl = Watchlist()
    ents = [wl.get(args.id)] if args.id else wl.with_status(Status.APPROVED, Status.PAUSED)
    for e in ents:
        for ev in store.list_events(entity_id=e.entity_id):
            when = ev["date"] or f"{ev['window_start']}..{ev['window_end']}"
            prev = json.loads(ev["previous_dates"] or "[]")
            print(f"  {e.id:<22} {ev['event_type']:<15} {when:<23} {ev['status']:<11} "
                  f"{ev['outcome'] or ''}" + (f"  (moved {len(prev)}x)" if prev else ""))
            print(f"      \"{ev['source_sentence'][:150]}\"  {ev['source_url']}")
    return 0


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("watchlist", help="scout | list | show | approve | reject | retire | "
                                         "follow | events")
    ws = p.add_subparsers(dest="wl_cmd", required=True)
    s = ws.add_parser("scout", help="propose company+drug pairs with evidence cards")
    s.add_argument("--max", type=int, default=10, help="max proposals per run")
    s.add_argument("--spread-minutes", type=float, default=0.0,
                   help="spread requests over this many minutes (scheduled runs)")
    s.add_argument("--with-quotes", action="store_true",
                   help="add spread/market cap from Alpaca (needs keys)")
    s.set_defaults(fn=cmd_scout)
    ls = ws.add_parser("list", help="list entries")
    ls.add_argument("--status", choices=[x.value for x in Status])
    ls.set_defaults(fn=cmd_list)
    sh = ws.add_parser("show", help="show an approval card")
    sh.add_argument("id")
    sh.set_defaults(fn=cmd_show)
    ap = ws.add_parser("approve", help="approve (or re-approve) a pair")
    ap.add_argument("id")
    ap.add_argument("--note")
    ap.set_defaults(fn=cmd_approve)
    rj = ws.add_parser("reject", help="reject a proposal")
    rj.add_argument("id")
    rj.add_argument("--reason", required=True)
    rj.set_defaults(fn=cmd_reject)
    rt = ws.add_parser("retire", help="stop following a pair")
    rt.add_argument("id")
    rt.add_argument("--reason", required=True)
    rt.set_defaults(fn=cmd_retire)
    fo = ws.add_parser("follow", help="daily: filings, trials, events, life cycle")
    fo.add_argument("--spread-minutes", type=float, default=0.0)
    fo.add_argument("--embedder")
    fo.set_defaults(fn=cmd_follow)
    ev = ws.add_parser("events", help="agent-managed catalyst events")
    ev.add_argument("id", nargs="?")
    ev.set_defaults(fn=cmd_events)
