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
from datetime import UTC, date, datetime
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

CREATE TABLE IF NOT EXISTS sell_state (
    ticker      TEXT PRIMARY KEY,
    state_json  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS seen_filings (
    accession   TEXT PRIMARY KEY,
    ticker      TEXT NOT NULL,
    form        TEXT NOT NULL,
    filed_on    TEXT NOT NULL,
    seen_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS request_ledger (
    source    TEXT NOT NULL,
    day       TEXT NOT NULL,
    count     INTEGER NOT NULL DEFAULT 0,
    failures  INTEGER NOT NULL DEFAULT 0,
    tripped   INTEGER NOT NULL DEFAULT 0 CHECK (tripped IN (0, 1)),
    PRIMARY KEY (source, day)
);

CREATE TABLE IF NOT EXISTS sec_companies (
    cik         TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    sic         TEXT,
    tickers     TEXT NOT NULL,
    exchanges   TEXT NOT NULL,
    fetched_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS catalyst_events (
    entity_id       TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    trial_id        TEXT NOT NULL DEFAULT '',
    cik             TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    drug            TEXT NOT NULL,
    date            TEXT,
    window_start    TEXT,
    window_end      TEXT,
    precision       TEXT NOT NULL,
    status          TEXT NOT NULL,
    outcome         TEXT,
    source_url      TEXT NOT NULL,
    source_sentence TEXT NOT NULL,
    source_filed    TEXT NOT NULL,
    previous_dates  TEXT NOT NULL DEFAULT '[]',
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (entity_id, event_type, trial_id)
);

CREATE TABLE IF NOT EXISTS ingested_docs (
    doc_sha      TEXT NOT NULL,
    entity_id    TEXT NOT NULL,
    ticker       TEXT NOT NULL,
    source       TEXT NOT NULL,
    url          TEXT NOT NULL,
    published_at TEXT NOT NULL,
    ingested_at  TEXT NOT NULL,
    PRIMARY KEY (doc_sha, entity_id)
);

CREATE TABLE IF NOT EXISTS kv (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
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

    # ------------------------------------------------------------------ sell machine

    def save_sell_state(self, ticker: str, state: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sell_state(ticker, state_json, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(ticker) DO UPDATE SET state_json = excluded.state_json, "
                "updated_at = excluded.updated_at",
                (ticker, _canonical(state), _now()),
            )

    def load_sell_state(self, ticker: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            r = conn.execute("SELECT state_json FROM sell_state WHERE ticker = ?",
                             (ticker,)).fetchone()
        if r is None:
            return None
        out: dict[str, Any] = json.loads(r[0])
        return out

    def mark_filing_seen(self, accession: str, ticker: str, form: str, filed_on: str) -> bool:
        """True the first time an accession number is seen, False afterwards."""
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO seen_filings(accession, ticker, form, filed_on, seen_at) "
                "VALUES (?, ?, ?, ?, ?)", (accession, ticker, form, filed_on, _now()))
            return cur.rowcount == 1

    # ------------------------------------------------------------------ request ledger

    def _ledger_bump(self, source: str, day: str, col: str) -> int:
        if col not in ("count", "failures"):  # pragma: no cover - internal guard
            raise ValueError(col)
        with self._connect() as conn:
            conn.execute("INSERT OR IGNORE INTO request_ledger(source, day) VALUES (?, ?)",
                         (source, day))
            conn.execute(f"UPDATE request_ledger SET {col} = {col} + 1 "  # noqa: S608
                         "WHERE source = ? AND day = ?", (source, day))
            row = conn.execute(f"SELECT {col} FROM request_ledger "  # noqa: S608
                               "WHERE source = ? AND day = ?", (source, day)).fetchone()
        return int(row[0])

    def ledger_row(self, source: str, day: str) -> tuple[int, int, bool]:
        with self._connect() as conn:
            row = conn.execute("SELECT count, failures, tripped FROM request_ledger "
                               "WHERE source = ? AND day = ?", (source, day)).fetchone()
        return (0, 0, False) if row is None else (int(row[0]), int(row[1]), bool(row[2]))

    def ledger_consume(self, source: str, day: str) -> int:
        return self._ledger_bump(source, day, "count")

    def ledger_fail(self, source: str, day: str) -> int:
        return self._ledger_bump(source, day, "failures")

    def ledger_set(self, source: str, day: str, *, failures: int | None = None,
                   tripped: bool | None = None) -> None:
        with self._connect() as conn:
            conn.execute("INSERT OR IGNORE INTO request_ledger(source, day) VALUES (?, ?)",
                         (source, day))
            if failures is not None:
                conn.execute("UPDATE request_ledger SET failures = ? WHERE source = ? "
                             "AND day = ?", (failures, source, day))
            if tripped is not None:
                conn.execute("UPDATE request_ledger SET tripped = ? WHERE source = ? "
                             "AND day = ?", (int(tripped), source, day))

    # ------------------------------------------------------------------ key/value

    def kv_get(self, key: str) -> str | None:
        with self._connect() as conn:
            r = conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return None if r is None else str(r[0])

    def kv_set(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute("INSERT INTO kv(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE "
                         "SET value = excluded.value", (key, value))

    # ------------------------------------------------------------------ SEC company cache

    def upsert_company(self, cik: str, name: str, sic: str | None, tickers: list[str],
                       exchanges: list[str]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sec_companies(cik, name, sic, tickers, exchanges, fetched_at) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(cik) DO UPDATE SET name = excluded.name, "
                "sic = excluded.sic, tickers = excluded.tickers, "
                "exchanges = excluded.exchanges, fetched_at = excluded.fetched_at",
                (cik, name, sic, json.dumps(tickers), json.dumps(exchanges), _now()))

    def get_company(self, cik: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            r = conn.execute("SELECT * FROM sec_companies WHERE cik = ?", (cik,)).fetchone()
        if r is None:
            return None
        d = dict(r)
        d["tickers"] = json.loads(d["tickers"])
        d["exchanges"] = json.loads(d["exchanges"])
        return d

    # ------------------------------------------------------------------ ingested documents

    def mark_ingested(self, doc_sha: str, entity_id: str, *, ticker: str, source: str,
                      url: str, published_at: str) -> bool:
        """True the first time (doc, entity) is recorded; False if already ingested."""
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO ingested_docs(doc_sha, entity_id, ticker, source, url, "
                "published_at, ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (doc_sha, entity_id, ticker, source, url, published_at, _now()))
            return cur.rowcount == 1

    def ingested_count(self, entity_id: str) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute("SELECT source, COUNT(*) FROM ingested_docs WHERE entity_id = ? "
                                "GROUP BY source", (entity_id,)).fetchall()
        return {str(s): int(n) for s, n in rows}

    # ------------------------------------------------------------------ catalyst events

    def get_event(self, entity_id: str, event_type: str,
                  trial_id: str = "") -> dict[str, Any] | None:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            r = conn.execute("SELECT * FROM catalyst_events WHERE entity_id = ? AND "
                             "event_type = ? AND trial_id = ?",
                             (entity_id, event_type, trial_id)).fetchone()
        return None if r is None else dict(r)

    def put_event(self, row: dict[str, Any]) -> None:
        cols = ("entity_id", "event_type", "trial_id", "cik", "ticker", "drug", "date",
                "window_start", "window_end", "precision", "status", "outcome", "source_url",
                "source_sentence", "source_filed", "previous_dates")
        vals = [row.get(c) for c in cols]
        vals[2] = vals[2] or ""
        vals[-1] = vals[-1] or "[]"
        with self._connect() as conn:
            conn.execute(
                f"INSERT INTO catalyst_events({', '.join(cols)}, updated_at) "  # noqa: S608
                f"VALUES ({', '.join('?' for _ in cols)}, ?) ON CONFLICT(entity_id, "
                "event_type, trial_id) DO UPDATE SET "
                + ", ".join(f"{c} = excluded.{c}" for c in cols[3:])
                + ", updated_at = excluded.updated_at", (*vals, _now()))

    def list_events(self, *, entity_id: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM catalyst_events"
        args: tuple[str, ...] = ()
        if entity_id is not None:
            sql += " WHERE entity_id = ?"
            args = (entity_id,)
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql + " ORDER BY COALESCE(date, window_end), entity_id",
                                args).fetchall()
        return [dict(r) for r in rows]

    def get_nct(self, nct_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            r = conn.execute("SELECT * FROM evaluated_ncts WHERE nct_id = ?", (nct_id,)).fetchone()
        return dict(r) if r else None


class SQLiteLedger:
    """Request budgets and circuit-breaker state that survive between runs.

    Opens the store lazily, so commands that never touch the network never create
    the database just to satisfy the ledger.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = path
        self._store: SQLiteStore | None = None
        self._tripped_logged: set[tuple[str, str]] = set()

    @property
    def store(self) -> SQLiteStore:
        if self._store is None:
            self._store = SQLiteStore(self._path)
        return self._store

    def consume(self, source: str, day: date) -> int:
        return self.store.ledger_consume(source, day.isoformat())

    def failures(self, source: str, day: date) -> int:
        return self.store.ledger_row(source, day.isoformat())[1]

    def record_failure(self, source: str, day: date) -> int:
        return self.store.ledger_fail(source, day.isoformat())

    def record_success(self, source: str, day: date) -> None:
        if self.failures(source, day):
            self.store.ledger_set(source, day.isoformat(), failures=0)

    def is_tripped(self, source: str, day: date) -> bool:
        return self.store.ledger_row(source, day.isoformat())[2]

    def trip(self, source: str, day: date) -> None:
        self.store.ledger_set(source, day.isoformat(), tripped=True)
        key = (source, day.isoformat())
        if key not in self._tripped_logged:
            self._tripped_logged.add(key)
            self.store.append_audit("network.source_off", {
                "source": source, "day": day.isoformat(),
                "reason": "repeated failed requests (circuit breaker)"})
