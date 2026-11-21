"""The agents the operator can schedule and run, and their editable settings.

Each job runs as ``python -m biotradeai <argv>`` in its own process. The argv is built
only from these definitions and validated values: nothing typed in the UI is passed to
a command line as free text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Any

from biotradeai.operator.db import OperatorDB, Run, Schedule

WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
MISSED_GRACE = timedelta(hours=2)  # a run missed while the host was off still fires within this
_TIME = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class JobError(ValueError):
    pass


@dataclass(frozen=True)
class Param:
    name: str
    label: str
    kind: str  # int | float | bool | choice
    default: Any
    help: str
    flag: str = ""  # CLI flag; for choice params the mapping below decides
    lo: float | None = None
    hi: float | None = None
    choices: tuple[tuple[str, str], ...] = ()  # (value, label)

    def coerce(self, raw: Any) -> Any:
        if self.kind == "bool":
            return raw in (True, "true", "on", "1", 1, "yes")
        if self.kind == "choice":
            v = str(raw)
            if v not in {c for c, _ in self.choices}:
                raise JobError(f"{self.label}: choose one of {[c for c, _ in self.choices]}")
            return v
        try:
            v_num: float = float(raw)
        except (TypeError, ValueError) as exc:
            raise JobError(f"{self.label}: a number is required") from exc
        if (self.lo is not None and v_num < self.lo) or (self.hi is not None and v_num > self.hi):
            raise JobError(f"{self.label}: must be between {self.lo:g} and {self.hi:g}")
        return int(v_num) if self.kind == "int" else v_num


@dataclass(frozen=True)
class JobDef:
    id: str
    title: str
    summary: str
    base: tuple[str, ...]
    params: tuple[Param, ...]
    default_days: tuple[int, ...]
    default_time: str
    needs_human: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def defaults(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}

    def validate(self, raw: dict[str, Any]) -> dict[str, Any]:
        out = self.defaults()
        for p in self.params:
            if p.kind == "bool":
                out[p.name] = p.coerce(raw.get(p.name, False))
            elif p.name in raw and raw[p.name] not in (None, ""):
                out[p.name] = p.coerce(raw[p.name])
        return out

    def argv(self, params: dict[str, Any]) -> list[str]:
        v = self.validate(params)
        args = list(self.base)
        for p in self.params:
            val = v[p.name]
            if p.kind == "bool":
                if val:
                    args.append(p.flag)
            elif p.kind == "choice":
                args += _CHOICE_ARGS[(self.id, p.name, val)]
            else:
                args += [p.flag, str(val)]
        return args


_CHOICE_ARGS: dict[tuple[str, str, str], list[str]] = {
    ("paper", "broker", "alpaca"): ["--broker", "alpaca"],
    ("paper", "broker", "fake"): ["--broker", "fake"],
    ("paper", "approvals", "queue"): ["--approve-later"],
    ("paper", "approvals", "reject"): ["--auto-reject"],
    ("paper", "research", "agent"): [],
    ("paper", "research", "single"): ["--no-agent"],
}

JOBS: dict[str, JobDef] = {j.id: j for j in (
    JobDef(
        id="scout", title="Watchlist scout",
        summary="Finds upcoming Phase 2/3 readouts and PDUFA dates in SEC 8-Ks and "
                "ClinicalTrials.gov, and proposes company + drug pairs for your approval.",
        base=("watchlist", "scout"),
        params=(
            Param("spread_minutes", "Spread requests over (minutes)", "int", 120,
                  "Keeps the load on SEC and ClinicalTrials.gov low.", "--spread-minutes",
                  0, 240),
            Param("max", "Max proposals per run", "int", 10, "", "--max", 1, 30),
            Param("max_market_cap_billions", "Market-cap ceiling ($B)", "float", 5.0,
                  "Needs market quotes.", "--max-market-cap-billions", 0.1, 50),
            Param("with_quotes", "Use market quotes (Alpaca)", "bool", True,
                  "Adds spread and market cap to each card.", "--with-quotes"),
        ),
        default_days=(6,), default_time="22:00"),
    JobDef(
        id="follow", title="Watchlist follow",
        summary="For approved drugs: reads new filings, trials and papers; updates catalyst "
                "dates and outcomes; pauses a pair on material changes.",
        base=("watchlist", "follow"),
        params=(
            Param("spread_minutes", "Spread requests over (minutes)", "int", 60, "",
                  "--spread-minutes", 0, 240),
        ),
        default_days=(0, 1, 2, 3, 4), default_time="06:00"),
    JobDef(
        id="paper", title="Paper trading cycle",
        summary="Exits first, then new entries: researches each catalyst, runs the risk "
                "checks and stages limit orders. Orders wait for your approval.",
        base=("paper", "run"),
        params=(
            Param("broker", "Broker", "choice", "alpaca", "Where approved orders go.",
                  choices=(("alpaca", "Alpaca paper"), ("fake", "Fake broker (dry run)"))),
            Param("approvals", "Order approvals", "choice", "queue",
                  "Approve later in this app, or reject everything (watch-only).",
                  choices=(("queue", "Wait for me in Trade approvals"),
                           ("reject", "Reject all (watch only)"))),
            Param("research", "Evidence gathering", "choice", "agent",
                  "The research agent searches and fetches evidence itself (needs a "
                  "tool-capable model), or one fixed search as before.",
                  choices=(("agent", "Research agent (tool loop)"),
                           ("single", "Single baseline search"))),
        ),
        default_days=(0, 1, 2, 3, 4), default_time="10:00",
        needs_human="Orders wait in Trade approvals."),
)}


def job(job_id: str) -> JobDef:
    try:
        return JOBS[job_id]
    except KeyError:
        raise JobError(f"unknown job {job_id!r}") from None


def parse_time(s: str) -> str:
    if not _TIME.match(s.strip()):
        raise JobError("time must be HH:MM (24-hour)")
    return s.strip()


def schedule_for(db: OperatorDB, job_id: str) -> Schedule:
    s = db.get_schedule(job_id)
    if s is None:
        d = job(job_id)
        db.put_schedule(job_id, enabled=False, days=d.default_days, at_time=d.default_time,
                        params=d.defaults())
        s = db.get_schedule(job_id)
        if s is None:  # pragma: no cover - just written
            raise JobError(f"could not create the schedule for {job_id}")
    return s


def _at(day: datetime, hhmm: str) -> datetime:
    h, m = (int(x) for x in hhmm.split(":"))
    return datetime.combine(day.date(), time(h, m), tzinfo=day.tzinfo)


def due_fire(s: Schedule, now_local: datetime) -> datetime | None:
    """The scheduled time to fire now, if any (today's slot, not yet fired, within grace)."""
    if not s.enabled or now_local.weekday() not in s.days:
        return None
    slot = _at(now_local, s.at_time)
    if not (slot <= now_local <= slot + MISSED_GRACE):
        return None
    if s.last_fired and datetime.fromisoformat(s.last_fired) >= slot:
        return None
    return slot


def next_fire(s: Schedule, now_local: datetime) -> datetime | None:
    if not s.enabled or not s.days:
        return None
    for i in range(8):
        day = now_local + timedelta(days=i)
        if day.weekday() in s.days:
            slot = _at(day, s.at_time)
            fired = s.last_fired and datetime.fromisoformat(s.last_fired) >= slot
            if slot > now_local or (i == 0 and due_fire(s, now_local) and not fired):
                return slot
    return None


def describe_days(days: tuple[int, ...]) -> str:
    ds = tuple(sorted(days))
    if ds == (0, 1, 2, 3, 4):
        return "Weekdays"
    if ds == tuple(range(7)):
        return "Every day"
    if not ds:
        return "No days"
    return ", ".join(WEEKDAYS[d] for d in ds)


def run_now(db: OperatorDB, job_id: str, *, by: str, params: dict[str, Any] | None = None
            ) -> int:
    d = job(job_id)
    if any(r.active for r in db.runs(job=job_id, status=("queued", "running"), limit=1)):
        raise JobError(f"{d.title} is already queued or running")
    p = d.validate(params if params is not None else schedule_for(db, job_id).params)
    rid = db.enqueue(job_id, trigger="manual", by=by, params=p, argv=d.argv(p))
    db.store.append_audit("operator.run_requested", {"job": job_id, "run": rid, "by": by,
                                                     "params": p})
    return rid


def cancel(db: OperatorDB, run_id: int, *, by: str) -> bool:
    ok = db.request_cancel(run_id)
    if ok:
        db.store.append_audit("operator.cancel_requested", {"run": run_id, "by": by})
    return ok


def save_schedule(db: OperatorDB, job_id: str, *, enabled: bool, days: list[int],
                  at_time: str, params: dict[str, Any], by: str) -> Schedule:
    d = job(job_id)
    if any(x not in range(7) for x in days):
        raise JobError("invalid weekday")
    if enabled and not days:
        raise JobError("pick at least one day, or turn the schedule off")
    old = schedule_for(db, job_id)
    new_params = d.validate(params)
    db.put_schedule(job_id, enabled=enabled, days=tuple(days), at_time=parse_time(at_time),
                    params=new_params)
    db.store.append_audit("operator.schedule_changed", {
        "job": job_id, "by": by,
        "before": {"enabled": old.enabled, "days": list(old.days), "time": old.at_time,
                   "params": old.params},
        "after": {"enabled": enabled, "days": sorted(days), "time": at_time,
                  "params": new_params}})
    return schedule_for(db, job_id)


def run_label(r: Run) -> str:
    return {"queued": "Queued", "running": "Running", "succeeded": "Succeeded",
            "failed": "Failed", "cancelled": "Cancelled"}.get(r.status, r.status)
