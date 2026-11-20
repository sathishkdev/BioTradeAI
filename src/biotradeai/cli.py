"""Command-line entry point. Grows each phase (evaluate, approve, positions, killswitch, replay).

Usage:
    python -m biotradeai config-check
    python -m biotradeai init-db
    python -m biotradeai audit-tail [-n 20]
    python -m biotradeai llm-ping [--llm ollama:llama3.2:3b] [--role research]
    python -m biotradeai runway TICKER [--facts-file F] [--as-of YYYY-MM-DD]
    python -m biotradeai trial NCT######## [--file F]
    python -m biotradeai catalysts
    python -m biotradeai account | positions [--broker alpaca|fake]
    python -m biotradeai order TICKER --qty N --limit P [--side buy|sell] [--broker ...]
    python -m biotradeai killswitch status|trip|reset
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

from biotradeai import cli_agents, cli_paper, cli_rag, cli_trading
from biotradeai.agents.base_rates import BaseRateError
from biotradeai.agents.llm import LLMError, LLMMessage, make_llm_client, resolve_spec
from biotradeai.brokers.approval import ApprovalError
from biotradeai.brokers.base import BrokerError
from biotradeai.config import ConfigError, load_config
from biotradeai.data.catalysts import CatalystFileError, load_catalysts
from biotradeai.data.clinicaltrials import ClinicalTrialsClient, parse_study
from biotradeai.data.edgar import EdgarClient, EdgarConfigError, parse_company_facts
from biotradeai.data.http import FetchError
from biotradeai.data.outcomes import OutcomeFileError
from biotradeai.data.runway import RunwayUnavailableError, compute_runway
from biotradeai.guardrails import rules
from biotradeai.paths import REPO_ROOT
from biotradeai.rag.embeddings import EmbeddingError
from biotradeai.rag.ontology import OntologyError
from biotradeai.store.sqlite_store import SQLiteStore


def _cmd_config_check(_: argparse.Namespace) -> int:
    cfg = load_config()
    print(json.dumps(cfg.model_dump(mode="json"), indent=2))
    print(f"\nOK: guardrails valid, mode={cfg.mode.value}")
    return 0


def _cmd_init_db(_: argparse.Namespace) -> int:
    store = SQLiteStore()
    store.append_audit("system.init_db", {"message": "database initialised"})
    print(f"OK: database ready at {store.path}")
    return 0


def _cmd_audit_tail(args: argparse.Namespace) -> int:
    store = SQLiteStore()
    entries = store.read_audit()[-args.n:]
    for e in entries:
        print(f"{e.id:>6} {e.ts} {e.event_type:<28} {e.ticker or '-':<6} {json.dumps(e.payload)}")
    print(f"chain valid: {store.verify_audit_chain()}")
    return 0


def _cmd_llm_ping(args: argparse.Namespace) -> int:
    spec = resolve_spec(args.role, args.llm)
    client = make_llm_client(args.role, args.llm)
    print(f"using {spec}")
    resp = client.complete(
        [LLMMessage("user", "Reply with exactly the word: pong")], max_tokens=10
    )
    print(f"{resp.provider}:{resp.model} -> {resp.text.strip()!r} usage={resp.usage}")
    return 0


def _cmd_runway(args: argparse.Namespace) -> int:
    cfg = load_config()
    as_of = date.fromisoformat(args.as_of) if args.as_of else None
    if args.facts_file:
        raw = json.loads(Path(args.facts_file).read_text(encoding="utf-8"))
        facts = parse_company_facts(raw, datetime.now(UTC))
    else:
        client = EdgarClient()
        facts = client.company_facts(client.ticker_to_cik(args.ticker))
    r = compute_runway(facts, lookback_quarters=cfg.runway.burn_lookback_quarters, as_of=as_of)
    m = Decimal(1_000_000)
    print(f"{args.ticker.upper()}  {facts.entity_name}  (CIK {r.cik})")
    print(f"  balance sheet date : {r.balance_sheet_date}")
    print(f"  cash               : ${r.cash / m:,.2f}M")
    print(f"  securities (cur)   : ${r.securities_current / m:,.2f}M")
    print(f"  securities (non)   : ${r.securities_noncurrent / m:,.2f}M")
    print(f"  liquidity          : ${r.liquidity / m:,.2f}M")
    print(f"  quarterly burn     : ${r.quarterly_burn / m:,.2f}M")
    print(f"  runway             : {r.runway_months:.2f} months")
    for q in r.quarters_used:
        print(f"    quarter {q.start}..{q.end}  OCF ${q.ocf / m:,.2f}M  [{q.method}]")
    for n in r.notes:
        print(f"  note: {n}")
    fatal = rules.is_fatal_runway(r.runway_months, cfg)
    veto = not rules.check_runway(r.runway_months, cfg).ok
    print(f"  vs guardrails      : fatal_prune={fatal}  risk_veto={veto}")
    return 0


def _cmd_trial(args: argparse.Namespace) -> int:
    if args.file:
        rec = parse_study(json.loads(Path(args.file).read_text(encoding="utf-8")),
                          datetime.now(UTC))
    else:
        rec = ClinicalTrialsClient().get_study(args.nct_id)
    print(f"{rec.nct_id}  {rec.title}")
    print(f"  sponsor  : {rec.sponsor}")
    print(f"  status   : {rec.overall_status}   phases: {', '.join(rec.phases) or '-'}")
    print(f"  drugs    : {', '.join(rec.interventions) or '-'}")
    for label, d in (("primary completion", rec.primary_completion),
                     ("completion", rec.completion)):
        if d:
            print(f"  {label:<19}: {d.raw} ({d.kind.value}, claimed by sponsor)")
    print(f"  last update posted : {rec.last_update_posted}")
    return 0


def _cmd_catalysts(_: argparse.Namespace) -> int:
    rows = load_catalysts()
    if not rows:
        print("data/catalysts.csv has no rows yet (header only).")
    for c in rows:
        print(f"{c.date}  {c.ticker:<6} {c.event:<15} {c.drug:<20} {c.source_url}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="biotradeai", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("config-check", help="validate config/guardrails.yaml").set_defaults(
        fn=_cmd_config_check
    )
    sub.add_parser("init-db", help="create the SQLite database").set_defaults(fn=_cmd_init_db)
    t = sub.add_parser("audit-tail", help="show the last audit entries")
    t.add_argument("-n", type=int, default=20)
    t.set_defaults(fn=_cmd_audit_tail)
    ping = sub.add_parser("llm-ping", help="check that the selected LLM answers")
    ping.add_argument("--llm", help="provider:model, e.g. ollama:llama3.2:3b")
    ping.add_argument("--role", help="agent role whose env override to use (research, risk)")
    ping.set_defaults(fn=_cmd_llm_ping)
    rw = sub.add_parser("runway", help="cash runway from SEC XBRL facts")
    rw.add_argument("ticker")
    rw.add_argument("--facts-file", help="offline companyfacts JSON instead of live SEC")
    rw.add_argument("--as-of", help="point-in-time date YYYY-MM-DD (facts filed on/before)")
    rw.set_defaults(fn=_cmd_runway)
    tr = sub.add_parser("trial", help="show a ClinicalTrials.gov study")
    tr.add_argument("nct_id")
    tr.add_argument("--file", help="offline study JSON instead of live CT.gov")
    tr.set_defaults(fn=_cmd_trial)
    sub.add_parser("catalysts", help="list data/catalysts.csv").set_defaults(fn=_cmd_catalysts)
    cli_trading.register(sub)
    cli_rag.register(sub)
    cli_agents.register(sub)
    cli_paper.register(sub)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(REPO_ROOT / ".env")
    args = build_parser().parse_args(argv)
    try:
        rc: int = args.fn(args)
    except (ConfigError, LLMError, EdgarConfigError, FetchError, RunwayUnavailableError,
            CatalystFileError, KeyError, BrokerError, ApprovalError, PermissionError,
            OntologyError, EmbeddingError, BaseRateError, OutcomeFileError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
