from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from biotradeai.redact import REDACTED
from biotradeai.store.sqlite_store import SQLiteStore


def test_append_and_read(store: SQLiteStore) -> None:
    e1 = store.append_audit("research.start", {"x": 1}, run_id="r1", ticker="ONCO")
    store.append_audit("risk.veto", {"reason": "runway"}, run_id="r1", ticker="ONCO")
    store.append_audit("other", {}, run_id="r2", ticker="ABCD")
    assert e1.id == 1
    assert [e.event_type for e in store.read_audit(run_id="r1")] == ["research.start", "risk.veto"]
    assert len(store.read_audit(ticker="ABCD")) == 1
    assert store.verify_audit_chain()


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE audit_log SET event_type = 'tampered' WHERE id = 1",
        "DELETE FROM audit_log WHERE id = 1",
        "DELETE FROM audit_log",
    ],
)
def test_audit_log_is_append_only(store: SQLiteStore, sql: str) -> None:
    store.append_audit("evt", {"a": 1})
    conn = sqlite3.connect(store.path)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute(sql)
    finally:
        conn.close()
    assert len(store.read_audit()) == 1


def test_hash_chain_detects_tampering_if_triggers_dropped(store: SQLiteStore) -> None:
    for i in range(3):
        store.append_audit("evt", {"i": i})
    conn = sqlite3.connect(store.path)
    conn.execute("DROP TRIGGER audit_log_no_update")
    conn.execute("UPDATE audit_log SET payload_json = '{\"i\":99}' WHERE id = 2")
    conn.commit()
    conn.close()
    assert not store.verify_audit_chain()


def test_payload_is_redacted(store: SQLiteStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_API_SECRET", "supersecretvalue123")
    e = store.append_audit("broker.call", {
        "api_key": "AKXXXX",
        "account_number": "5RH12345678",
        "nested": {"Authorization": "Bearer abc"},
        "note": "used key supersecretvalue123 today",
        "ticker": "ONCO",
    })
    assert e.payload["api_key"] == REDACTED
    assert e.payload["account_number"] == REDACTED
    assert e.payload["nested"]["Authorization"] == REDACTED
    assert "supersecretvalue123" not in e.payload["note"]
    assert e.payload["ticker"] == "ONCO"


def test_nonce_single_use(store: SQLiteStore) -> None:
    assert store.consume_nonce("abc" * 6)
    assert not store.consume_nonce("abc" * 6)


def test_killswitch_persists_across_restart(tmp_path: Path) -> None:
    db = tmp_path / "ks.db"
    s1 = SQLiteStore(db)
    assert not s1.get_killswitch().tripped
    s1.set_killswitch(True, "drawdown 3.4%", changed_by="killswitch")
    s2 = SQLiteStore(db)  # simulated restart
    ks = s2.get_killswitch()
    assert ks.tripped and ks.reason == "drawdown 3.4%"


def test_positions_and_ncts(store: SQLiteStore) -> None:
    store.upsert_position("ONCO", 100, "4.25", entity_id="ENT-001", catalyst_date="2026-03-01")
    store.upsert_position("ONCO", 50, "4.25")
    [p] = store.open_positions()
    assert p["qty"] == 50 and p["entity_id"] == "ENT-001"
    store.upsert_position("ONCO", 0, "4.25", status="closed")
    assert store.open_positions() == []

    store.record_nct("NCT01234567", "ONCO", last_completion_date="2026-06-01", decision="SKIP")
    nct = store.get_nct("NCT01234567")
    assert nct is not None and nct["decision"] == "SKIP"
    assert store.get_nct("NCT00000000") is None
