"""Read models for the operator screens (no side effects)."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from biotradeai.config import ConfigError, GuardrailsConfig, load_config
from biotradeai.operator import jobs
from biotradeai.operator.db import OperatorDB, Pending, Run
from biotradeai.operator.host import HostStatus, host_status
from biotradeai.operator.settings import base_rates_summary, secrets_status
from biotradeai.operator.trades import Destination, destination, required_confirmation
from biotradeai.paper.review import paper_gate
from biotradeai.watchlist.catalysts import tradable_date
from biotradeai.watchlist.model import Status, WatchEntry, Watchlist


@dataclass
class JobView:
    id: str
    title: str
    summary: str
    enabled: bool
    when: str
    next_fire: datetime | None
    last: Run | None
    active: Run | None
    params: dict[str, Any]
    needs_human: str


@dataclass
class Attention:
    level: str  # warn | info | danger
    text: str
    href: str


@dataclass
class Overview:
    host: HostStatus
    mode: str
    killswitch: bool
    killswitch_reason: str | None
    jobs: list[JobView]
    attention: list[Attention]
    counts: dict[str, int]
    gate: Any
    decisions: list[dict[str, Any]]
    failures: list[Run] = field(default_factory=list)


def _cfg() -> GuardrailsConfig | None:
    try:
        return load_config()
    except ConfigError:
        return None


def current_destination(db: OperatorDB) -> Destination:
    """Where orders go right now: the paper cycle's broker setting + the config mode."""
    cfg = _cfg()
    mode = cfg.mode.value if cfg else "paper"
    broker = str(jobs.schedule_for(db, "paper").params.get("broker", "alpaca"))
    return destination(broker, mode)


def job_views(db: OperatorDB) -> list[JobView]:
    now = datetime.now().astimezone()
    out = []
    for jid, d in jobs.JOBS.items():
        s = jobs.schedule_for(db, jid)
        active = db.runs(job=jid, status=("queued", "running"), limit=1)
        out.append(JobView(jid, d.title, d.summary, s.enabled,
                           f"{jobs.describe_days(s.days)} at {s.at_time}",
                           jobs.next_fire(s, now), db.last_finished(jid),
                           active[0] if active else None, s.params, d.needs_human))
    return out


def recent_audit(db: OperatorDB, *, prefix: str = "", ticker: str = "", run_id: str = "",
                 limit: int = 100, before_id: int | None = None) -> list[dict[str, Any]]:
    sql = "SELECT id, ts, event_type, run_id, ticker, payload_json FROM audit_log"
    where: list[str] = []
    args: list[Any] = []
    if prefix:
        where.append("event_type LIKE ?")
        args.append(prefix.replace("%", "") + "%")
    if ticker:
        where.append("ticker = ?")
        args.append(ticker.upper())
    if run_id:
        where.append("run_id = ?")
        args.append(run_id)
    if before_id:
        where.append("id < ?")
        args.append(before_id)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    conn = sqlite3.connect(db.store.path)
    try:
        rows = conn.execute(sql, args).fetchall()
    finally:
        conn.close()
    return [{"id": r[0], "ts": r[1], "event": r[2], "run_id": r[3], "ticker": r[4],
             "payload": json.loads(r[5])} for r in rows]


def recent_decisions(db: OperatorDB, limit: int = 8) -> list[dict[str, Any]]:
    out = []
    for e in recent_audit(db, prefix="supervisor.decision", limit=60):
        if (e["run_id"] or "").startswith(("backtest-", "fixture-")):
            continue
        p = e["payload"]
        out.append({"ts": e["ts"], "ticker": e["ticker"], "run_id": e["run_id"],
                    "decision": p.get("decision"), "ev": p.get("risk_ev"),
                    "reasons": p.get("reasons") or []})
        if len(out) >= limit:
            break
    return out


def overview(db: OperatorDB, wl: Watchlist | None = None) -> Overview:
    wl = wl or Watchlist()
    cfg = _cfg()
    ks = db.store.get_killswitch()
    pend = [p for p in db.pending() if not p.expired]
    since = datetime.now(UTC) - timedelta(days=1)
    failures = [r for r in db.runs(status=("failed",), limit=20)
                if r.finished_at and datetime.fromisoformat(r.finished_at) >= since]
    counts = {
        "trades": len(pend),
        "proposed": len(wl.with_status(Status.PROPOSED)),
        "approved": len(wl.with_status(Status.APPROVED)),
        "paused": len(wl.with_status(Status.PAUSED)),
        "conflicts": sum(1 for ev in db.store.list_events() if ev["status"] == "conflicting"),
        "failures": len(failures),
    }
    host = host_status(db)
    att: list[Attention] = []
    if ks.tripped:
        att.append(Attention("danger", f"Kill-switch is tripped: {ks.reason}", "/safety"))
    if not host.running:
        att.append(Attention("warn", "Agent host is not running, so schedules won't fire.",
                             "/agents"))
    if counts["trades"]:
        att.append(Attention("warn", f"{counts['trades']} order(s) waiting for your approval",
                             "/trades"))
    if counts["failures"]:
        att.append(Attention("danger", f"{counts['failures']} run(s) failed in the last 24 h",
                             "/agents"))
    if counts["paused"]:
        att.append(Attention("warn", f"{counts['paused']} watchlist pair(s) paused, "
                                     "waiting for you", "/watchlist?tab=paused"))
    if counts["proposed"]:
        att.append(Attention("info", f"{counts['proposed']} watchlist proposal(s) to review",
                             "/watchlist"))
    if counts["conflicts"]:
        att.append(Attention("warn", f"{counts['conflicts']} catalyst date(s) conflicting",
                             "/catalysts"))
    missing = [s["label"] for s in secrets_status() if s["required"] and not s["set"]]
    if missing:
        att.append(Attention("danger", "Missing: " + ", ".join(missing), "/settings#secrets"))
    try:
        if not base_rates_summary()["reviewed"]:
            att.append(Attention("info", "Base rates not marked reviewed yet",
                                 "/settings#base-rates"))
    except Exception:
        att.append(Attention("danger", "Base rates file is invalid", "/settings#base-rates"))
    if cfg is None:
        att.append(Attention("danger", "config/guardrails.yaml is invalid", "/settings"))
    return Overview(host, cfg.mode.value if cfg else "?", ks.tripped, ks.reason,
                    job_views(db), att, counts, paper_gate(db.store, cfg) if cfg else None,
                    recent_decisions(db), failures)


# ---------------------------------------------------------------------- trades


@dataclass
class TradeCard:
    p: Pending
    dest: Destination
    confirm: str
    checks: list[dict[str, Any]]
    minutes_left: int


def trade_cards(db: OperatorDB) -> list[TradeCard]:
    out = []
    now = datetime.now(UTC)
    for p in db.pending():
        if p.expired:
            continue
        left = int((datetime.fromisoformat(p.expires_at) - now).total_seconds() // 60)
        out.append(TradeCard(p, destination(p.broker, p.mode), required_confirmation(p),
                             list(p.pack.get("checks") or []), max(left, 0)))
    return out


def trade_history(db: OperatorDB, limit: int = 30) -> list[tuple[Pending, Destination]]:
    rows = db.pending(status=("approved", "rejected", "expired", "failed"), limit=limit)
    return [(p, destination(p.broker, p.mode)) for p in rows]


# ---------------------------------------------------------------------- watchlist


def watchlist_tabs(wl: Watchlist) -> dict[str, list[WatchEntry]]:
    tabs: dict[str, list[WatchEntry]] = {s.value: [] for s in Status}
    for e in wl.entries:
        tabs[e.status.value].append(e)
    for v in tabs.values():
        v.sort(key=lambda e: (-e.score, e.id))
    return tabs


# ---------------------------------------------------------------------- catalysts


def catalyst_rows(db: OperatorDB, wl: Watchlist) -> list[dict[str, Any]]:
    by_entity = {e.entity_id: e for e in wl.entries}
    today = datetime.now(UTC).date()
    rows = []
    for ev in db.store.list_events():
        e = by_entity.get(ev["entity_id"])
        when = ev["date"] or f"{ev['window_start']} – {ev['window_end']}"
        rows.append({**ev, "when": when, "entry_id": e.id if e else ev["ticker"],
                     "tradable": tradable_date(ev, today) is not None,
                     "moved": len(json.loads(ev.get("previous_dates") or "[]"))})
    rows.sort(key=lambda r: (r["status"] != "conflicting", r["window_end"] or r["date"] or ""))
    return rows


def positions(db: OperatorDB) -> list[dict[str, Any]]:
    out = []
    for p in db.store.open_positions():
        st = db.store.load_sell_state(p["ticker"]) or {}
        out.append({**p, "sell_state": st.get("state"), "catalyst_event":
                    st.get("catalyst_event"), "high": st.get("high_water"),
                    "last_bid": st.get("last_bid")})
    return out


def ledger_today(db: OperatorDB) -> list[dict[str, Any]]:
    day = datetime.now(UTC).date().isoformat()
    conn = sqlite3.connect(db.store.path)
    try:
        rows = conn.execute("SELECT source, count, failures, tripped FROM request_ledger "
                            "WHERE day = ? ORDER BY source", (day,)).fetchall()
    finally:
        conn.close()
    return [{"source": r[0], "count": r[1], "failures": r[2], "tripped": bool(r[3])}
            for r in rows]
