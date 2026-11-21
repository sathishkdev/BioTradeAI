"""Kill-switch and live-gate status for the operator UI."""

from __future__ import annotations

from dataclasses import dataclass

from biotradeai.brokers.alpaca_broker import AlpacaBroker
from biotradeai.brokers.base import Broker
from biotradeai.config import GuardrailsConfig
from biotradeai.guardrails.killswitch import KillSwitch
from biotradeai.paper.review import GateStatus, paper_gate
from biotradeai.store.sqlite_store import SQLiteStore


@dataclass(frozen=True)
class TripResult:
    cancelled: int
    broker_reached: bool


def _broker() -> Broker | None:
    try:
        return AlpacaBroker()
    except Exception:  # no keys or offline: still trip locally
        return None


def trip(store: SQLiteStore, cfg: GuardrailsConfig, *, by: str) -> TripResult:
    broker = _broker()
    cancelled = KillSwitch(store, cfg, broker).trip(f"manual trip by {by} (operator UI)")
    return TripResult(len(cancelled), broker is not None)


def reset(store: SQLiteStore, cfg: GuardrailsConfig, *, by: str, typed: str) -> None:
    KillSwitch(store, cfg, None).reset_by_human(operator=f"ui:{by}", confirmation=typed)


def gate(store: SQLiteStore, cfg: GuardrailsConfig) -> GateStatus:
    return paper_gate(store, cfg)
