from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from biotradeai.brokers.fake_broker import FakeBroker
from biotradeai.config import load_config
from biotradeai.guardrails.killswitch import KillSwitch, KillSwitchTrippedError
from biotradeai.store.sqlite_store import SQLiteStore
from tests.unit.conftest import make_order

CFG = load_config()


def _with_open_orders(broker: FakeBroker, n: int) -> list[str]:
    return [broker.place_order(make_order(), client_order_id=f"c{i}").order_id
            for i in range(n)]


def test_forced_drawdown_trips_and_cancels_orders(store: SQLiteStore,
                                                  broker: FakeBroker) -> None:
    ids = _with_open_orders(broker, 2)
    ks = KillSwitch(store, CFG, broker)
    assert not ks.check_drawdown(start_equity=Decimal(100000), equity=Decimal(97000))
    assert ks.check_drawdown(start_equity=Decimal(100000), equity=Decimal("96999.99"))
    assert all(broker.orders[i].status == "canceled" for i in ids)
    with pytest.raises(KillSwitchTrippedError):
        ks.ensure_not_tripped()
    [trip] = [e for e in store.read_audit() if e.event_type == "killswitch.trip"]
    assert sorted(trip.payload["cancelled_orders"]) == sorted(ids)


def test_consecutive_api_errors_trip(store: SQLiteStore, broker: FakeBroker) -> None:
    ks = KillSwitch(store, CFG, broker)
    for _ in range(CFG.max_api_errors - 1):
        assert not ks.record_api_error("timeout")
    assert ks.record_api_error("timeout")


def test_success_resets_error_streak(store: SQLiteStore) -> None:
    ks = KillSwitch(store, CFG)
    for _ in range(CFG.max_api_errors - 1):
        ks.record_api_error("x")
    ks.record_api_success()
    for _ in range(CFG.max_api_errors - 1):
        assert not ks.record_api_error("x")


def test_state_survives_restart(tmp_path: Path) -> None:
    db = tmp_path / "ks.db"
    KillSwitch(SQLiteStore(db), CFG).trip("test")
    restarted = KillSwitch(SQLiteStore(db), CFG)
    assert restarted.tripped
    with pytest.raises(KillSwitchTrippedError, match="test"):
        restarted.ensure_not_tripped()


def test_error_counter_survives_restart(tmp_path: Path) -> None:
    db = tmp_path / "ks.db"
    for _ in range(CFG.max_api_errors - 1):
        KillSwitch(SQLiteStore(db), CFG).record_api_error("x")
    assert KillSwitch(SQLiteStore(db), CFG).record_api_error("x")


def test_reset_requires_typed_confirmation(store: SQLiteStore) -> None:
    ks = KillSwitch(store, CFG)
    ks.trip("t")
    with pytest.raises(PermissionError):
        ks.reset_by_human(operator="me", confirmation="reset")
    with pytest.raises(PermissionError):
        ks.reset_by_human(operator=" ", confirmation="RESET")
    assert ks.tripped
    ks.reset_by_human(operator="me", confirmation="RESET")
    assert not ks.tripped
    assert store.get_killswitch().changed_by == "human:me"


def test_trip_survives_cancel_failures(store: SQLiteStore, broker: FakeBroker) -> None:
    _with_open_orders(broker, 1)
    broker.fail_next = 5
    ks = KillSwitch(store, CFG, broker)
    ks.trip("api down")
    assert ks.tripped
    [trip] = [e for e in store.read_audit() if e.event_type == "killswitch.trip"]
    assert trip.payload["cancel_failures"]
