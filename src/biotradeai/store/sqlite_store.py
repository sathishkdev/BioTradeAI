"""SQLite persistence: append-only audit log, positions, evaluated NCTs, nonces, kill-switch.

The audit log is append-only twice over:
* SQL triggers ``RAISE(ABORT)`` on any UPDATE or DELETE of ``audit_log``.
* Each row stores ``prev_hash`` and ``entry_hash`` (SHA-256 chain), so tampering
  done outside SQLite (e.g. dropping the triggers) is detectable by ``verify_audit_chain``.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from biotradeai.paths import db_path
from biotradeai.redact import redact

GENESIS_HASH = "0" * 64
SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS audit_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT    NOT NULL,
    event_type    TEXT    NOT NULL,
    run_id        TEXT,
    ticker        TEXT,
    payload_json  TEXT    NOT NULL,
    prev_hash     TEXT    NOT NULL,
    entry_hash    TEXT    NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS ix_audit_run    ON audit_log(run_id);
CREATE INDEX IF NOT EXISTS ix_audit_ticker ON audit_log(ticker);

CREATE TRIGGER IF NOT EXISTS audit_log_no_update
BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only: UPDATE forbidden'); END;

CREATE TRIGGER IF NOT EXISTS audit_log_no_delete
BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only: DELETE forbidden'); END;

CREATE TABLE IF NOT EXISTS positions (
    ticker         TEXT PRIMARY KEY,
    entity_id      TEXT,
    qty            INTEGER NOT NULL CHECK (qty >= 0),
    avg_price      TEXT    NOT NULL,
    catalyst_date  TEXT,
    status         TEXT    NOT NULL CHECK (status IN ('open', 'closed')),
    opened_at      TEXT    NOT NULL,
    updated_at     TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluated_ncts (
    nct_id                 TEXT PRIMARY KEY,
    ticker                 TEXT NOT NULL,
    last_completion_date   TEXT,
    decision               TEXT,
    evaluated_at           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS approval_nonces (
    nonce    TEXT PRIMARY KEY,
    used_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS killswitch (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    tripped     INTEGER NOT NULL CHECK (tripped IN (0, 1)),
    reason      TEXT,
    changed_at  TEXT NOT NULL,
    changed_by  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS killswitch_counters (
    id                       INTEGER PRIMARY KEY CHECK (id = 1),
    consecutive_api_errors   INTEGER NOT NULL CHECK (consecutive_api_errors >= 0)
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _entry_hash(prev_hash: str, ts: str, event_type: str, run_id: str | None,
                ticker: str | None, payload_json: str) -> str:
    material = "\x1f".join([prev_hash, ts, event_type, run_id or "", ticker or "", payload_json])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AuditEntry:
    id: int
    ts: str
    event_type: str
    run_id: str | None
    ticker: str | None
    payload: dict[str, Any]
    prev_hash: str
    entry_hash: str


@dataclass(frozen=True)
class KillSwitchState:
    tripped: bool
    reason: str | None
    changed_at: str
    changed_by: str


class SQLiteStore:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # executescript() manages its own transaction, so run DDL on a plain connection.
        ddl = sqlite3.connect(self.path)
        try:
            ddl.executescript(_SCHEMA)
        finally:
            ddl.close()
        with self._connect() as conn:
            if conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 0:
                conn.execute("INSERT INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,))
            conn.execute(
                "INSERT OR IGNORE INTO killswitch(id, tripped, reason, changed_at, changed_by) "
                "VALUES (1, 0, NULL, ?, 'init')",
                (_now(),),
            )
            conn.execute(
                "INSERT OR IGNORE INTO killswitch_counters(id, consecutive_api_errors) "
                "VALUES (1, 0)"
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, isolation_level=None, timeout=10)
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ audit

    def append_audit(self, event_type: str, payload: dict[str, Any], *,
                     run_id: str | None = None, ticker: str | None = None) -> AuditEntry:
        """Append one event. Payload is redacted before it is written."""
        payload_json = _canonical(redact(payload))
        ts = _now()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT entry_hash FROM audit_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
            prev = row[0] if row else GENESIS_HASH
            h = _entry_hash(prev, ts, event_type, run_id, ticker, payload_json)
            cur = conn.execute(
                "INSERT INTO audit_log(ts, event_type, run_id, ticker, payload_json, "
                "prev_hash, entry_hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ts, event_type, run_id, ticker, payload_json, prev, h),
            )
            new_id = cur.lastrowid
        if new_id is None:  # pragma: no cover - sqlite always sets it after INSERT
            raise RuntimeError("audit insert returned no row id")
        return AuditEntry(new_id, ts, event_type, run_id, ticker, json.loads(payload_json), prev, h)

    def read_audit(self, *, run_id: str | None = None, ticker: str | None = None,
                   limit: int | None = None) -> list[AuditEntry]:
        sql = ("SELECT id, ts, event_type, run_id, ticker, payload_json, prev_hash, entry_hash "
               "FROM audit_log")
        clauses, args = [], []
        if run_id is not None:
            clauses.append("run_id = ?")
            args.append(run_id)
        if ticker is not None:
            clauses.append("ticker = ?")
            args.append(ticker)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id"
        if limit is not None:
            sql += " LIMIT ?"
            args.append(str(limit))
        with self._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [AuditEntry(r[0], r[1], r[2], r[3], r[4], json.loads(r[5]), r[6], r[7])
                for r in rows]

    def verify_audit_chain(self) -> bool:
        """Recompute the hash chain. False means rows were altered, removed or reordered."""
        prev = GENESIS_HASH
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ts, event_type, run_id, ticker, payload_json, prev_hash, entry_hash "
                "FROM audit_log ORDER BY id"
            ).fetchall()
        for ts, et, rid, tk, pj, ph, eh in rows:
            if ph != prev or _entry_hash(ph, ts, et, rid, tk, pj) != eh:
                return False
            prev = eh
        return True

    # ------------------------------------------------------------------ nonces

    def consume_nonce(self, nonce: str) -> bool:
        """Mark an approval nonce used. Returns False if it was already used."""
        try:
            with self._connect() as conn:
                conn.execute("INSERT INTO approval_nonces(nonce, used_at) VALUES (?, ?)",
                             (nonce, _now()))
        except sqlite3.IntegrityError:
            return False
        return True

    # ------------------------------------------------------------------ kill-switch

    def get_killswitch(self) -> KillSwitchState:
        with self._connect() as conn:
            r = conn.execute(
                "SELECT tripped, reason, changed_at, changed_by FROM killswitch WHERE id = 1"
            ).fetchone()
        return KillSwitchState(bool(r[0]), r[1], r[2], r[3])

    def set_killswitch(self, tripped: bool, reason: str | None, changed_by: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE killswitch SET tripped = ?, reason = ?, changed_at = ?, changed_by = ? "
                "WHERE id = 1",
                (int(tripped), reason, _now(), changed_by),
            )

    def increment_api_errors(self) -> int:
        with self._connect() as conn:
            conn.execute("UPDATE killswitch_counters SET consecutive_api_errors = "
                         "consecutive_api_errors + 1 WHERE id = 1")
            n: int = conn.execute("SELECT consecutive_api_errors FROM killswitch_counters "
                                  "WHERE id = 1").fetchone()[0]
        return n

    def reset_api_errors(self) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE killswitch_counters SET consecutive_api_errors = 0 WHERE id = 1")

    def get_api_errors(self) -> int:
        with self._connect() as conn:
            n: int = conn.execute("SELECT consecutive_api_errors FROM killswitch_counters "
                                  "WHERE id = 1").fetchone()[0]
        return n

    # ------------------------------------------------------------------ positions

    def upsert_position(self, ticker: str, qty: int, avg_price: str, *, status: str = "open",
                        entity_id: str | None = None, catalyst_date: str | None = None) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO positions(ticker, entity_id, qty, avg_price, catalyst_date, status, "
                "opened_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(ticker) DO UPDATE SET qty = excluded.qty, "
                "avg_price = excluded.avg_price, status = excluded.status, "
                "entity_id = COALESCE(excluded.entity_id, positions.entity_id), "
                "catalyst_date = COALESCE(excluded.catalyst_date, positions.catalyst_date), "
                "updated_at = excluded.updated_at",
                (ticker, entity_id, qty, avg_price, catalyst_date, status, now, now),
            )

    def open_positions(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM positions WHERE status = 'open' ORDER BY ticker"
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ evaluated NCTs

    def record_nct(self, nct_id: str, ticker: str, *, last_completion_date: str | None,
                   decision: str | None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO evaluated_ncts(nct_id, ticker, last_completion_date, decision, "
                "evaluated_at) VALUES (?, ?, ?, ?, ?) ON CONFLICT(nct_id) DO UPDATE SET "
                "last_completion_date = excluded.last_completion_date, "
                "decision = excluded.decision, evaluated_at = excluded.evaluated_at",
                (nct_id, ticker, last_completion_date, decision, _now()),
            )

    def get_nct(self, nct_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            r = conn.execute("SELECT * FROM evaluated_ncts WHERE nct_id = ?", (nct_id,)).fetchone()
        return dict(r) if r else None
