"""Agent-managed catalyst events for approved watchlist drugs (SQLite ``catalyst_events``).

Rules (agreed with the user, replacing human-curated dates for approved pairs):
* Event identity = (drug entity, event type, trial id). The date is a field, so a moved
  date UPDATES the event (old dates kept in ``previous_dates``) instead of duplicating it.
* Dates come only from company filings/press releases with the exact sentence kept.
  ClinicalTrials.gov completion dates are never used as catalyst dates.
* The newest filing wins. Two filings on the same day that disagree mark the event
  ``conflicting``, and it is not traded until a newer source settles it.
* Tradable = upcoming, supported type (Phase 3 readout, PDUFA), and either an exact day
  or a company-stated window of at most one quarter (the window end is used as the date).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any

from biotradeai.data.catalysts import Catalyst
from biotradeai.data.outcomes import Outcome
from biotradeai.data.signals import CatalystMention, OutcomeKind, Precision
from biotradeai.events import ReadoutResult
from biotradeai.store.sqlite_store import SQLiteStore
from biotradeai.watchlist.model import Status, WatchEntry, Watchlist

SUPPORTED = frozenset({"PHASE3_READOUT", "PDUFA"})
TRADABLE_PRECISION = frozenset({Precision.DAY, Precision.MONTH, Precision.QUARTER})


class EventStatus:
    UPCOMING = "upcoming"
    CONFLICTING = "conflicting"
    OCCURRED = "occurred"


@dataclass(frozen=True)
class EventChange:
    kind: str  # new | changed | same | conflict | stale
    entity_id: str
    event_type: str
    detail: str


def record_catalyst(
    store: SQLiteStore,
    e: WatchEntry,
    m: CatalystMention,
    *,
    url: str,
    filed: date,
    today: date,
    trial_id: str = "",
) -> EventChange:
    if m.when.end < today:
        return EventChange("stale", e.entity_id, m.event_type, "date already passed")
    new = {
        "entity_id": e.entity_id,
        "event_type": m.event_type,
        "trial_id": trial_id,
        "cik": e.cik,
        "ticker": e.ticker,
        "drug": e.drug_names[0],
        "date": m.when.exact.isoformat() if m.when.exact else None,
        "window_start": m.when.start.isoformat(),
        "window_end": m.when.end.isoformat(),
        "precision": m.when.precision.value,
        "status": EventStatus.UPCOMING,
        "outcome": None,
        "source_url": url,
        "source_sentence": m.sentence[:600],
        "source_filed": filed.isoformat(),
        "previous_dates": "[]",
    }
    cur = store.get_event(e.entity_id, m.event_type, trial_id)
    if cur is None:
        store.put_event(new)
        return EventChange("new", e.entity_id, m.event_type, m.when.text)
    same_when = (cur["window_start"], cur["window_end"]) == (new["window_start"], new["window_end"])
    if cur["status"] == EventStatus.OCCURRED:
        # A new cycle of the same event type (e.g. a resubmission after a CRL).
        prev = [
            *json.loads(cur["previous_dates"] or "[]"),
            {"date": cur["date"] or cur["window_end"], "outcome": cur["outcome"]},
        ]
        store.put_event({**new, "previous_dates": json.dumps(prev)})
        return EventChange("new", e.entity_id, m.event_type, f"new cycle: {m.when.text}")
    if filed.isoformat() < str(cur["source_filed"]):
        return EventChange("same", e.entity_id, m.event_type, "older source ignored")
    if same_when:
        if cur["status"] == EventStatus.CONFLICTING:
            store.put_event({**new, "previous_dates": cur["previous_dates"]})
            return EventChange("changed", e.entity_id, m.event_type, "conflict resolved")
        return EventChange("same", e.entity_id, m.event_type, m.when.text)
    if filed.isoformat() == str(cur["source_filed"]) and cur["source_url"] != url:
        store.put_event(
            {
                **cur,
                "status": EventStatus.CONFLICTING,
                "source_sentence": f"{cur['source_sentence']} || {m.sentence}"[:1200],
            }
        )
        return EventChange(
            "conflict",
            e.entity_id,
            m.event_type,
            f"{cur['window_end']} vs {m.when.end} on the same day",
        )
    prev = [
        *json.loads(cur["previous_dates"] or "[]"),
        {
            "date": cur["date"] or cur["window_end"],
            "source": cur["source_url"],
            "filed": cur["source_filed"],
        },
    ]
    store.put_event({**new, "previous_dates": json.dumps(prev)})
    return EventChange(
        "changed", e.entity_id, m.event_type, f"{cur['date'] or cur['window_end']} -> {m.when.text}"
    )


_OUTCOME_EVENT = {
    OutcomeKind.POSITIVE: ("PHASE3_READOUT", "PHASE2_READOUT"),
    OutcomeKind.NEGATIVE: ("PHASE3_READOUT", "PHASE2_READOUT"),
    OutcomeKind.CRL: ("PDUFA",),
    OutcomeKind.APPROVED: ("PDUFA",),
}


def record_outcome(
    store: SQLiteStore, e: WatchEntry, kind: OutcomeKind, *, url: str, sentence: str, filed: date
) -> str | None:
    """Mark the matching upcoming event as occurred. Returns its event type, if any."""
    for etype in _OUTCOME_EVENT.get(kind, ()):
        for ev in store.list_events(entity_id=e.entity_id):
            if ev["event_type"] == etype and ev["status"] != EventStatus.OCCURRED:
                store.put_event(
                    {
                        **ev,
                        "status": EventStatus.OCCURRED,
                        "outcome": kind.value,
                        "source_url": url,
                        "source_sentence": sentence[:600],
                        "source_filed": filed.isoformat(),
                    }
                )
                return etype
    return None


def tradable_date(ev: dict[str, Any], today: date) -> date | None:
    if ev["status"] != EventStatus.UPCOMING or ev["event_type"] not in SUPPORTED:
        return None
    if Precision(ev["precision"]) not in TRADABLE_PRECISION:
        return None
    d = date.fromisoformat(ev["date"] or ev["window_end"])
    return d if d >= today else None


def agent_catalysts(store: SQLiteStore, wl: Watchlist, today: date) -> list[Catalyst]:
    """Catalysts the paper runner may evaluate: approved entries, tradable events only."""
    out: list[Catalyst] = []
    for e in wl.with_status(Status.APPROVED):
        for ev in store.list_events(entity_id=e.entity_id):
            d = tradable_date(ev, today)
            if d is not None:
                out.append(
                    Catalyst(e.ticker, e.drug_names[0], ev["event_type"], d, ev["source_url"])
                )
    return out


_RESULT = {
    "positive": ReadoutResult.POSITIVE,
    "negative": ReadoutResult.NEGATIVE,
    "crl": ReadoutResult.CRL,
}


def agent_outcomes(store: SQLiteStore, wl: Watchlist) -> list[Outcome]:
    """Readout results the sell machine should see (approved and paused entries)."""
    out: list[Outcome] = []
    for e in wl.with_status(Status.APPROVED, Status.PAUSED):
        for ev in store.list_events(entity_id=e.entity_id):
            res = _RESULT.get(ev["outcome"] or "")
            if ev["status"] == EventStatus.OCCURRED and res is not None:
                out.append(
                    Outcome(
                        e.ticker,
                        e.drug_names[0],
                        date.fromisoformat(ev["source_filed"]),
                        res,
                        ev["source_url"],
                    )
                )
    return out
