"""Command-line entry point. Grows each phase (evaluate, approve, positions, killswitch, replay).

Usage:
    python -m biotradeai config-check
    python -m biotradeai init-db
    python -m biotradeai audit-tail [-n 20]
    python -m biotradeai llm-ping [--llm ollama:llama3.2:3b] [--role research]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from dotenv import load_dotenv

from biotradeai.agents.llm import LLMError, LLMMessage, make_llm_client, resolve_spec
from biotradeai.config import ConfigError, load_config
from biotradeai.paths import REPO_ROOT
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
    return p


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(REPO_ROOT / ".env")
    args = build_parser().parse_args(argv)
    try:
        rc: int = args.fn(args)
    except (ConfigError, LLMError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
