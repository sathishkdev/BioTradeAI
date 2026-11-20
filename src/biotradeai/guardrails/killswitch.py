"""Circuit breaker.

Trips on (a) daily drawdown beyond ``daily_drawdown_limit`` or (b) ``max_api_errors``
consecutive broker failures. When tripped it cancels all open orders (best effort),
blocks every new order, and persists in SQLite so a restart does not clear it.
Only ``reset_by_human`` (wired to the CLI, requires typing RESET) clears it.
"""

from __future__ import annotations

from decimal import Decimal

from biotradeai.brokers.base import Broker
from biotradeai.config import GuardrailsConfig
from biotradeai.guardrails.rules import check_drawdown
from biotradeai.store.sqlite_store import SQLiteStore

RESET_CONFIRMATION = "RESET"


class KillSwitchTrippedError(RuntimeError):
    pass


class KillSwitch:
    def __init__(self, store: SQLiteStore, cfg: GuardrailsConfig,
                 broker: Broker | None = None) -> None:
        self._store = store
        self._cfg = cfg
        self._broker = broker

    # ------------------------------------------------------------------ state

    @property
    def tripped(self) -> bool:
        return self._store.get_killswitch().tripped

    def ensure_not_tripped(self) -> None:
        ks = self._store.get_killswitch()
        if ks.tripped:
            raise KillSwitchTrippedError(f"kill-switch is tripped: {ks.reason}")

    # ------------------------------------------------------------------ triggers

    def trip(self, reason: str) -> list[str]:
        """Trip (idempotent), cancel open orders, return the cancelled order ids."""
        already = self.tripped
        if not already:
            self._store.set_killswitch(True, reason, changed_by="killswitch")
        cancelled: list[str] = []
        failed: list[str] = []
        if self._broker is not None:
            try:
                open_orders = self._broker.get_orders(open_only=True)
            except Exception as exc:
                open_orders = []
                failed.append(f"get_orders: {exc}")
            for o in open_orders:
                try:
                    self._broker.cancel_order(o.order_id)
                    cancelled.append(o.order_id)
                except Exception as exc:
                    failed.append(f"{o.order_id}: {exc}")
        self._store.append_audit("killswitch.trip", {
            "reason": reason, "already_tripped": already,
            "cancelled_orders": cancelled, "cancel_failures": failed,
        })
        return cancelled

    def check_drawdown(self, *, start_equity: Decimal, equity: Decimal) -> bool:
        """Trip if drawdown exceeds the limit. Returns True if (now) tripped."""
        res = check_drawdown(start_equity=start_equity, equity=equity, cfg=self._cfg)
        if not res.ok:
            self.trip(f"daily drawdown breach: {res.reason}")
        return self.tripped

    def record_api_error(self, error: str) -> bool:
        n = self._store.increment_api_errors()
        self._store.append_audit("killswitch.api_error", {"count": n, "error": error[:500]})
        if n >= self._cfg.max_api_errors:
            self.trip(f"{n} consecutive API errors (last: {error[:200]})")
        return self.tripped

    def record_api_success(self) -> None:
        if self._store.get_api_errors():
            self._store.reset_api_errors()

    # ------------------------------------------------------------------ human reset

    def reset_by_human(self, *, operator: str, confirmation: str) -> None:
        if confirmation != RESET_CONFIRMATION:
            raise PermissionError(f"type {RESET_CONFIRMATION!r} to reset the kill-switch")
        if not operator.strip():
            raise PermissionError("operator name is required")
        prev = self._store.get_killswitch()
        self._store.set_killswitch(False, None, changed_by=f"human:{operator}")
        self._store.reset_api_errors()
        self._store.append_audit("killswitch.reset", {"operator": operator,
                                                      "previous_reason": prev.reason})
