"""Daily review of the audit log, and the paper-trading part of the live gate.

Live gate condition 2 (plan section 7): the audit log shows at least
``live_gate.min_paper_days`` days of paper operation with zero guardrail breaches.
A "paper day" is a UTC date with at least one completed ``paper.cycle_done``.
A breach is any ``guardrail.breach`` event, from the post-trade compliance check.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from biotradeai.config import GuardrailsConfig
from biotradeai.store.sqlite_store import AuditEntry, SQLiteStore


@dataclass
class DailyReview:
    day: date
    events: Counter[str] = field(default_factory=Counter)
    decisions: Counter[str] = field(default_factory=Counter)
    placed: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    human_rejected: list[str] = field(default_factory=list)
    exits: list[str] = field(default_factory=list)
    skips: list[str] = field(default_factory=list)
    breaches: list[str] = field(default_factory=list)
    killswitch: list[str] = field(default_factory=list)
    chain_ok: bool = True


def _order(e: AuditEntry) -> str:
    o: dict[str, Any] = e.payload.get("order") or {}
    return f"{o.get('side', '?')} {o.get('qty', '?')} {e.ticker} @ {o.get('limit_price', '?')}"


def daily_review(store: SQLiteStore, day: date) -> DailyReview:
    r = DailyReview(day)
    for e in store.read_audit():
        if e.ts[:10] != day.isoformat():
            continue
        r.events[e.event_type] += 1
        p = e.payload
        match e.event_type:
            case "supervisor.decision" if (e.run_id or "").startswith("paper-"):
                r.decisions[str(p.get("decision"))] += 1
            case "execution.placed":
                r.placed.append(_order(e))
            case "execution.rejected":
                r.rejected.append(f"{_order(e)}: {'; '.join(p.get('reasons') or [])}")
            case "approval.rejected":
                r.human_rejected.append(_order(e))
            case "sell.exit":
                r.exits.append(f"{e.ticker} {p.get('trigger')} {p.get('qty')}: {p.get('reason')}")
            case "paper.skip":
                r.skips.append(f"{e.ticker}: {p.get('reason')}")
            case "guardrail.breach":
                r.breaches.append(f"{e.ticker}: {p.get('breach')}")
            case "killswitch.trip" | "killswitch.reset":
                r.killswitch.append(f"{e.event_type}: {p.get('reason') or p.get('operator')}")
            case _:
                pass
    r.chain_ok = store.verify_audit_chain()
    return r


def render_review(r: DailyReview) -> str:
    lines = [f"Daily review {r.day}   audit chain {'OK' if r.chain_ok else 'BROKEN'}",
             f"  cycles: {r.events.get('paper.cycle_done', 0)}   decisions: "
             f"{dict(r.decisions) or 'none'}"]
    for title, items in (("orders placed", r.placed), ("orders rejected by guardrails",
                                                       r.rejected),
                         ("rejected by you", r.human_rejected), ("exits", r.exits),
                         ("skipped", r.skips), ("GUARDRAIL BREACHES", r.breaches),
                         ("kill-switch", r.killswitch)):
        lines.append(f"  {title}: {len(items)}")
        lines += [f"    - {x}" for x in items]
    return "\n".join(lines)


@dataclass(frozen=True)
class GateStatus:
    paper_days: int
    required_days: int
    breaches: int
    first_day: date | None
    eligible: bool


def paper_gate(store: SQLiteStore, cfg: GuardrailsConfig) -> GateStatus:
    days: set[date] = set()
    breaches = 0
    for e in store.read_audit():
        # Dry runs on the fake broker are not paper trading: they don't count.
        if e.event_type == "paper.cycle_done" and e.payload.get("broker") != "fake":
            days.add(date.fromisoformat(e.ts[:10]))
        elif e.event_type == "guardrail.breach":
            breaches += 1
    need = cfg.live_gate.min_paper_days
    return GateStatus(len(days), need, breaches, min(days) if days else None,
                      eligible=len(days) >= need and breaches == 0)
