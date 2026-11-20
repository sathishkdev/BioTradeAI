"""Watchlist of company+drug pairs: proposals, human approval, and the drug's life cycle.

One entry per (SEC company ID, drug entity). An approval covers the drug through every
phase until it reaches the market plus one full quarter after launch; then it pauses for
re-approval. Material changes also pause it. Nothing here places orders.

Statuses
    proposed  -> approved | rejected                (human)
    approved  -> paused (material change, yearly check-in, post-launch quarter, ...)
    paused    -> approved (human re-approval) | retired
    approved  -> retired (program ended)
Drug stages
    development -> approved (FDA) -> launched

Stored as YAML (``data/watchlist.yaml``, git-ignored) so it is easy to read and diff.
Writes are atomic (temp file + replace).
"""

from __future__ import annotations

import os
import re
import tempfile
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from biotradeai.paths import data_dir

CHECKIN_DAYS = 365
LAUNCH_GRACE_DAYS = 183  # approved but not launched after ~6 months -> ask
DEFAULT_SOFT_CAP = 30


class Status(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    PAUSED = "paused"
    RETIRED = "retired"


class Stage(StrEnum):
    DEVELOPMENT = "development"
    FDA_APPROVED = "approved"
    LAUNCHED = "launched"


ACTIVE = frozenset({Status.PROPOSED, Status.APPROVED, Status.PAUSED})

_ALLOWED: dict[Status, frozenset[Status]] = {
    Status.PROPOSED: frozenset({Status.APPROVED, Status.REJECTED}),
    Status.APPROVED: frozenset({Status.PAUSED, Status.RETIRED}),
    Status.PAUSED: frozenset({Status.APPROVED, Status.RETIRED}),
    # back to proposed only via a material change; a human may approve after rejecting
    Status.REJECTED: frozenset({Status.PROPOSED, Status.APPROVED}),
    Status.RETIRED: frozenset({Status.PROPOSED}),  # only via a material change
}


class WatchlistError(RuntimeError):
    pass


def drug_key(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", name.upper())


def entity_id_for(ticker: str, drug: str) -> str:
    """ENT-<TICKER>-<DRUG>, letters/digits only, within the ontology's 41-char body."""
    t = re.sub(r"[^A-Z0-9]", "", ticker.upper())[:10] or "X"
    return f"ENT-{t}-{drug_key(drug)[: 40 - len(t)] or 'DRUG'}"


def quarter_end_after(d: date) -> date:
    """End of the first FULL calendar quarter after ``d`` (launch quarter excluded)."""
    q = (d.month - 1) // 3 + 1
    y, q = (d.year + 1, 1) if q == 4 else (d.year, q + 1)
    m = q * 3
    nxt = date(y + (m == 12), 1 if m == 12 else m + 1, 1)
    return nxt - timedelta(days=1)


class WatchEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str  # TICKER:DRUG, human-friendly
    cik: str
    ticker: str
    company: str
    entity_id: str
    drug_names: list[str]
    trial_ids: list[str] = Field(default_factory=list)
    status: Status = Status.PROPOSED
    stage: Stage = Stage.DEVELOPMENT
    score: float = 0.0
    proposed_at: datetime
    decided_at: datetime | None = None
    decided_by: str | None = None
    note: str | None = None
    approved_at: datetime | None = None
    checkin_due: date | None = None
    fda_approved_on: date | None = None
    launched_on: date | None = None
    review_after: date | None = None  # end of first full quarter after launch
    pause_reason: str | None = None
    retire_reason: str | None = None
    card: dict[str, Any] = Field(default_factory=dict)
    history: list[dict[str, Any]] = Field(default_factory=list)

    def matches_drug(self, names: list[str] | tuple[str, ...]) -> bool:
        mine = {drug_key(n) for n in self.drug_names}
        return any(drug_key(n) in mine for n in names)


def default_path() -> Path:
    return data_dir() / "watchlist.yaml"


class Watchlist:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_path()
        self.entries: list[WatchEntry] = []
        if self.path.exists():
            raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
            self.entries = [WatchEntry.model_validate(e) for e in raw.get("entries") or []]

    # ------------------------------------------------------------------ io

    def save(self) -> None:
        body = {"entries": [e.model_dump(mode="json") for e in self.entries]}
        text = ("# BioTradeAI watchlist. Managed by `biotradeai watchlist ...`; approve/reject\n"
                "# with the CLI so every decision is audit-logged.\n"
                + yaml.safe_dump(body, sort_keys=False, allow_unicode=True))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, self.path)

    # ------------------------------------------------------------------ lookups

    def get(self, entry_id: str) -> WatchEntry:
        for e in self.entries:
            if e.id.lower() == entry_id.lower():
                return e
        raise WatchlistError(f"no watchlist entry {entry_id!r}")

    def find(self, cik: str, names: list[str] | tuple[str, ...]) -> WatchEntry | None:
        for e in self.entries:
            if e.cik == cik and e.matches_drug(names):
                return e
        return None

    def with_status(self, *statuses: Status) -> list[WatchEntry]:
        return [e for e in self.entries if e.status in statuses]

    def active_count(self) -> int:
        return len(self.with_status(Status.APPROVED, Status.PAUSED))

    # ------------------------------------------------------------------ transitions

    def _move(self, e: WatchEntry, to: Status, *, by: str, reason: str,
              now: datetime | None = None) -> None:
        if to not in _ALLOWED[e.status]:
            raise WatchlistError(f"{e.id}: cannot go from {e.status.value} to {to.value}")
        now = now or datetime.now(UTC)
        e.history.append({"at": now.isoformat(), "from": e.status.value, "to": to.value,
                          "by": by, "reason": reason})
        e.status = to

    def add_proposal(self, e: WatchEntry, *, material_change: str | None = None) -> bool:
        """Add a proposal unless the pair is already known. A rejected/retired pair comes
        back only with ``material_change`` (the reason is shown on the card)."""
        cur = self.find(e.cik, e.drug_names)
        if cur is None:
            e.history.append({"at": e.proposed_at.isoformat(), "from": None,
                              "to": Status.PROPOSED.value, "by": "scout", "reason": "new"})
            self.entries.append(e)
            return True
        if cur.status in ACTIVE or material_change is None:
            return False
        e.card = {**e.card, "previously": {"status": cur.status.value,
                                           "reason": cur.note or cur.retire_reason,
                                           "material_change": material_change}}
        self.entries.remove(cur)
        e.history = [*cur.history, {"at": e.proposed_at.isoformat(), "from": cur.status.value,
                                    "to": Status.PROPOSED.value, "by": "scout",
                                    "reason": f"material change: {material_change}"}]
        self.entries.append(e)
        return True

    def approve(self, entry_id: str, *, by: str, note: str | None = None,
                now: datetime | None = None) -> WatchEntry:
        e = self.get(entry_id)
        now = now or datetime.now(UTC)
        self._move(e, Status.APPROVED, by=by, reason=note or "approved", now=now)
        e.decided_at, e.decided_by, e.note = now, by, note
        e.approved_at = now
        e.checkin_due = (now + timedelta(days=CHECKIN_DAYS)).date()
        e.pause_reason = None
        if e.stage is Stage.LAUNCHED:  # re-approved after the post-launch pause
            e.review_after = None
        return e

    def reject(self, entry_id: str, *, by: str, reason: str,
               now: datetime | None = None) -> WatchEntry:
        e = self.get(entry_id)
        now = now or datetime.now(UTC)
        self._move(e, Status.REJECTED, by=by, reason=reason, now=now)
        e.decided_at, e.decided_by, e.note = now, by, reason
        return e

    def pause(self, entry_id: str, *, reason: str, by: str = "agent",
              now: datetime | None = None) -> WatchEntry:
        e = self.get(entry_id)
        if e.status is Status.PAUSED:
            return e
        self._move(e, Status.PAUSED, by=by, reason=reason, now=now)
        e.pause_reason = reason
        return e

    def retire(self, entry_id: str, *, reason: str, by: str = "agent",
               now: datetime | None = None) -> WatchEntry:
        e = self.get(entry_id)
        self._move(e, Status.RETIRED, by=by, reason=reason, now=now)
        e.retire_reason = reason
        return e
