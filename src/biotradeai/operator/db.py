"""Operator tables in the main SQLite database (created on first use).

* ``op_schedules`` — one row per job: enabled, weekdays, local time, job settings.
* ``op_runs``      — every run: queued -> running -> succeeded | failed | cancelled.
* ``op_pending``   — orders parked for a human decision (approve-later queue).

The audit log stays the record of decisions; these tables are working state.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from biotradeai.store.sqlite_store import SQLiteStore

_DDL = """
CREATE TABLE IF NOT EXISTS op_schedules (
    job          TEXT PRIMARY KEY,
    enabled      INTEGER NOT NULL CHECK (enabled IN (0, 1)),
    days         TEXT NOT NULL,             -- JSON list of weekday numbers, Mon = 0
    at_time      TEXT NOT NULL,             -- HH:MM local time
    params_json  TEXT NOT NULL,
    last_fired   TEXT,                      -- local ISO datetime of the last scheduled fire
    updated_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS op_runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    job               TEXT NOT NULL,
    trigger           TEXT NOT NULL,        -- schedule | manual
    requested_by      TEXT NOT NULL,
    status            TEXT NOT NULL,        -- queued | running | succeeded | failed | cancelled
    params_json       TEXT NOT NULL,
    argv_json         TEXT NOT NULL,
    requested_at      TEXT NOT NULL,
    started_at        TEXT,
    finished_at       TEXT,
    pid               INTEGER,
    exit_code         INTEGER,
    log_path          TEXT,
    cancel_requested  INTEGER NOT NULL DEFAULT 0,
    error             TEXT
);
CREATE INDEX IF NOT EXISTS ix_op_runs_job ON op_runs(job, id);
CREATE INDEX IF NOT EXISTS ix_op_runs_status ON op_runs(status);
CREATE TABLE IF NOT EXISTS op_pending (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    kind          TEXT NOT NULL,            -- entry | exit
    run_id        TEXT NOT NULL,
    ticker        TEXT NOT NULL,
    order_json    TEXT NOT NULL,
    pack_json     TEXT NOT NULL,
    broker        TEXT NOT NULL,            -- alpaca | fake (adapter that will place it)
    mode          TEXT NOT NULL,            -- paper | shadow | live at staging time
    status        TEXT NOT NULL,            -- pending | approved | rejected | expired | failed
    created_at    TEXT NOT NULL,
    expires_at    TEXT NOT NULL,
    decided_at    TEXT,
    decided_by    TEXT,
    note          TEXT,
    result_json   TEXT
);
CREATE INDEX IF NOT EXISTS ix_op_pending_status ON op_pending(status);
"""

ACTIVE_RUN = ("queued", "running")


def utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class Schedule:
    job: str
    enabled: bool
    days: tuple[int, ...]
    at_time: str
    params: dict[str, Any]
    last_fired: str | None
    updated_at: str


@dataclass(frozen=True)
class Run:
    id: int
    job: str
    trigger: str
    requested_by: str
    status: str
    params: dict[str, Any]
    argv: list[str]
    requested_at: str
    started_at: str | None
    finished_at: str | None
    pid: int | None
    exit_code: int | None
    log_path: str | None
    cancel_requested: bool
    error: str | None

    @property
    def active(self) -> bool:
        return self.status in ACTIVE_RUN

    @property
    def duration_s(self) -> float | None:
        if not self.started_at:
            return None
        end = datetime.fromisoformat(self.finished_at) if self.finished_at else utcnow()
        return (end - datetime.fromisoformat(self.started_at)).total_seconds()


@dataclass(frozen=True)
class Pending:
    id: int
    kind: str
    run_id: str
    ticker: str
    order: dict[str, Any]
    pack: dict[str, Any]
    broker: str
    mode: str
    status: str
    created_at: str
    expires_at: str
    decided_at: str | None
    decided_by: str | None
    note: str | None
    result: dict[str, Any] | None

    @property
    def expired(self) -> bool:
        return self.status == "pending" and utcnow() >= datetime.fromisoformat(self.expires_at)


_RUN_COLS = ("id, job, trigger, requested_by, status, params_json, argv_json, requested_at, "
             "started_at, finished_at, pid, exit_code, log_path, cancel_requested, error")
_PEND_COLS = ("id, kind, run_id, ticker, order_json, pack_json, broker, mode, status, "
              "created_at, expires_at, decided_at, decided_by, note, result_json")


def _run(r: tuple[Any, ...]) -> Run:
    return Run(r[0], r[1], r[2], r[3], r[4], json.loads(r[5]), json.loads(r[6]), r[7], r[8],
               r[9], r[10], r[11], r[12], bool(r[13]), r[14])


def _pend(r: tuple[Any, ...]) -> Pending:
    return Pending(r[0], r[1], r[2], r[3], json.loads(r[4]), json.loads(r[5]), r[6], r[7],
                   r[8], r[9], r[10], r[11], r[12], r[13],
                   json.loads(r[14]) if r[14] else None)


class OperatorDB:
    def __init__(self, store: SQLiteStore | None = None) -> None:
        self.store = store or SQLiteStore()
        conn = sqlite3.connect(self.store.path)
        try:
            conn.executescript(_DDL)
        finally:
            conn.close()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.store.path, isolation_level=None, timeout=10)
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ schedules

    def get_schedule(self, job: str) -> Schedule | None:
        with self._conn() as c:
            r = c.execute("SELECT job, enabled, days, at_time, params_json, last_fired, "
                          "updated_at FROM op_schedules WHERE job = ?", (job,)).fetchone()
        if r is None:
            return None
        return Schedule(r[0], bool(r[1]), tuple(json.loads(r[2])), r[3], json.loads(r[4]),
                        r[5], r[6])

    def put_schedule(self, job: str, *, enabled: bool, days: tuple[int, ...], at_time: str,
                     params: dict[str, Any]) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO op_schedules(job, enabled, days, at_time, params_json, "
                "updated_at) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(job) DO UPDATE SET "
                "enabled = excluded.enabled, days = excluded.days, at_time = "
                "excluded.at_time, params_json = excluded.params_json, updated_at = "
                "excluded.updated_at",
                (job, int(enabled), json.dumps(sorted(days)), at_time, json.dumps(params),
                 utcnow().isoformat()))

    def mark_fired(self, job: str, when_local: datetime) -> None:
        with self._conn() as c:
            c.execute("UPDATE op_schedules SET last_fired = ? WHERE job = ?",
                      (when_local.isoformat(), job))

    # ------------------------------------------------------------------ runs

    def enqueue(self, job: str, *, trigger: str, by: str, params: dict[str, Any],
                argv: list[str]) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO op_runs(job, trigger, requested_by, status, params_json, "
                "argv_json, requested_at) VALUES (?, ?, ?, 'queued', ?, ?, ?)",
                (job, trigger, by, json.dumps(params), json.dumps(argv), utcnow().isoformat()))
            rid = cur.lastrowid
        if rid is None:  # pragma: no cover
            raise RuntimeError("run insert returned no id")
        return rid

    def get_run(self, run_id: int) -> Run | None:
        with self._conn() as c:
            r = c.execute(f"SELECT {_RUN_COLS} FROM op_runs WHERE id = ?",  # noqa: S608
                          (run_id,)).fetchone()
        return _run(r) if r else None

    def runs(self, *, job: str | None = None, status: tuple[str, ...] = (),
             limit: int = 50) -> list[Run]:
        sql, args = f"SELECT {_RUN_COLS} FROM op_runs", []  # noqa: S608
        where = []
        if job:
            where.append("job = ?")
            args.append(job)
        if status:
            where.append(f"status IN ({','.join('?' * len(status))})")
            args.extend(status)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(str(limit))
        with self._conn() as c:
            return [_run(r) for r in c.execute(sql, args).fetchall()]

    def last_run(self, job: str) -> Run | None:
        rs = self.runs(job=job, limit=1)
        return rs[0] if rs else None

    def last_finished(self, job: str) -> Run | None:
        rs = self.runs(job=job, status=("succeeded", "failed", "cancelled"), limit=1)
        return rs[0] if rs else None

    def mark_started(self, run_id: int, *, pid: int, log_path: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE op_runs SET status = 'running', started_at = ?, pid = ?, "
                      "log_path = ? WHERE id = ?",
                      (utcnow().isoformat(), pid, log_path, run_id))

    def mark_finished(self, run_id: int, *, status: str, exit_code: int | None,
                      error: str | None = None) -> None:
        with self._conn() as c:
            c.execute("UPDATE op_runs SET status = ?, finished_at = ?, exit_code = ?, "
                      "error = ? WHERE id = ?",
                      (status, utcnow().isoformat(), exit_code, error, run_id))

    def request_cancel(self, run_id: int) -> bool:
        with self._conn() as c:
            cur = c.execute("UPDATE op_runs SET cancel_requested = 1 WHERE id = ? AND status "
                            "IN ('queued', 'running')", (run_id,))
            return cur.rowcount > 0

    # ------------------------------------------------------------------ pending approvals

    def add_pending(self, *, kind: str, run_id: str, ticker: str, order: dict[str, Any],
                    pack: dict[str, Any], broker: str, mode: str, expires_at: datetime) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO op_pending(kind, run_id, ticker, order_json, pack_json, broker, "
                "mode, status, created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, "
                "'pending', ?, ?)",
                (kind, run_id, ticker, json.dumps(order, default=str),
                 json.dumps(pack, default=str), broker, mode, utcnow().isoformat(),
                 expires_at.isoformat()))
            pid = cur.lastrowid
        if pid is None:  # pragma: no cover
            raise RuntimeError("pending insert returned no id")
        return pid

    def get_pending(self, pending_id: int) -> Pending | None:
        with self._conn() as c:
            r = c.execute(f"SELECT {_PEND_COLS} FROM op_pending WHERE id = ?",  # noqa: S608
                          (pending_id,)).fetchone()
        return _pend(r) if r else None

    def pending(self, *, status: tuple[str, ...] = ("pending",), limit: int = 100
                ) -> list[Pending]:
        marks = ",".join("?" * len(status))
        with self._conn() as c:
            rows = c.execute(f"SELECT {_PEND_COLS} FROM op_pending WHERE status IN "  # noqa: S608
                             f"({marks}) ORDER BY id DESC LIMIT ?",
                             (*status, limit)).fetchall()
        return [_pend(r) for r in rows]

    def has_pending(self, ticker: str) -> bool:
        return any(p.ticker == ticker and not p.expired for p in self.pending())

    def decide_pending(self, pending_id: int, *, status: str, by: str, note: str | None,
                       result: dict[str, Any] | None = None) -> bool:
        """Move a pending row to a final status. False if it was already decided."""
        with self._conn() as c:
            cur = c.execute(
                "UPDATE op_pending SET status = ?, decided_at = ?, decided_by = ?, note = ?, "
                "result_json = ? WHERE id = ? AND status = 'pending'",
                (status, utcnow().isoformat(), by, note,
                 json.dumps(result, default=str) if result is not None else None, pending_id))
            return cur.rowcount > 0
