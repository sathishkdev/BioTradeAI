"""Network guardrails: who we may call, how fast, and how much per day.

Every outbound data request goes through ``data/http.py``, which consults this module:

* **Host allowlist**: only official APIs / bulk files are reachable. Anything else is
  refused before a socket opens, including redirects to other hosts. ``*.test`` and
  ``localhost`` are allowed for tests and local services (never routable on the internet).
* **Per-source politeness**: one shared, jittered rate limiter per source, set at
  roughly half of each provider's published limit.
* **Daily budgets**: a hard cap of requests per source per UTC day. When it's spent, the
  source is off until tomorrow, so even a bug can't turn into a flood.
* **Circuit breaker**: after ``BREAKER_THRESHOLD`` consecutive failed requests (after
  retries) a source is switched off for the rest of the UTC day.

Budget and breaker state lives in a ``Ledger``: in memory by default, or in SQLite
(``store.sqlite_store.SQLiteLedger``) so it survives between scheduled runs.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Protocol
from urllib.parse import urlsplit

from biotradeai.data.http_limiter import RateLimiter

BREAKER_THRESHOLD = 3


@dataclass(frozen=True)
class SourcePolicy:
    name: str
    hosts: tuple[str, ...]
    max_per_second: float
    daily_budget: int
    published_rule: str  # what the provider says, shown in docs and errors


def _ncbi_rate() -> float:
    return 5.0 if os.environ.get("NCBI_API_KEY") else 2.0


def _openfda_budget() -> int:
    return 50_000 if os.environ.get("OPENFDA_API_KEY") else 500


def policies() -> dict[str, SourcePolicy]:
    """Current policies (a function so optional API keys in the env are honoured)."""
    return {p.name: p for p in (
        SourcePolicy("sec", ("data.sec.gov", "www.sec.gov"), 5.0, 20_000,
                     "SEC fair access: <= 10 requests/s per user, declared User-Agent"),
        SourcePolicy("ctgov", ("clinicaltrials.gov",), 0.8, 5_000,
                     "no official number found; ~50/min is the most conservative cited"),
        SourcePolicy("ncbi", ("eutils.ncbi.nlm.nih.gov",), _ncbi_rate(), 5_000,
                     "NCBI E-utilities: <= 3 requests/s without a key, <= 10 with a key"),
        SourcePolicy("openfda", ("api.fda.gov",), 1.0, _openfda_budget(),
                     "openFDA: 240/min and 1,000/day per IP without a key; 120,000/day with"),
        SourcePolicy("alpaca", ("paper-api.alpaca.markets", "data.alpaca.markets"), 1.0, 5_000,
                     "Alpaca: per-account API limits; we only read quotes/account"),
        SourcePolicy("robinhood", ("agent.robinhood.com",), 0.5, 1_000,
                     "Robinhood Trading MCP (final phases only)"),
    )}


# Hosts that are never on the public internet: test doubles and local services (Ollama).
_LOCAL_SUFFIXES = (".test", ".localhost")
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class BlockedHostError(RuntimeError):
    pass


def source_for_host(host: str) -> str | None:
    h = host.lower().rstrip(".")
    for p in policies().values():
        if h in p.hosts:
            return p.name
    return None


def is_allowed_host(host: str) -> bool:
    h = host.lower().rstrip(".")
    return (h in _LOCAL_HOSTS or h.endswith(_LOCAL_SUFFIXES)
            or source_for_host(h) is not None)


def check_url(url: str) -> None:
    host = urlsplit(url).hostname or ""
    if not is_allowed_host(host):
        raise BlockedHostError(
            f"{host or url!r} is not an allowed data source; see data/netpolicy.py")


# --------------------------------------------------------------------------- limiters


_limiters: dict[str, RateLimiter] = {}
_lock = threading.Lock()


def limiter_for(source: str) -> RateLimiter:
    """One process-wide limiter per source, shared by every client of that source."""
    with _lock:
        if source not in _limiters:
            p = policies().get(source)
            rate = p.max_per_second if p else 1.0
            _limiters[source] = RateLimiter(rate, jitter=0.1)
        return _limiters[source]


def reset_limiters() -> None:  # tests only
    with _lock:
        _limiters.clear()


# --------------------------------------------------------------------------- ledger


class BudgetExceededError(RuntimeError):
    pass


class SourceUnavailableError(RuntimeError):
    pass


class Ledger(Protocol):
    def consume(self, source: str, day: date) -> int: ...
    def failures(self, source: str, day: date) -> int: ...
    def record_failure(self, source: str, day: date) -> int: ...
    def record_success(self, source: str, day: date) -> None: ...
    def is_tripped(self, source: str, day: date) -> bool: ...
    def trip(self, source: str, day: date) -> None: ...


class InMemoryLedger:
    def __init__(self) -> None:
        self._counts: dict[tuple[str, date], int] = {}
        self._fails: dict[tuple[str, date], int] = {}
        self._tripped: set[tuple[str, date]] = set()
        self._lock = threading.Lock()

    def consume(self, source: str, day: date) -> int:
        with self._lock:
            k = (source, day)
            self._counts[k] = self._counts.get(k, 0) + 1
            return self._counts[k]

    def failures(self, source: str, day: date) -> int:
        return self._fails.get((source, day), 0)

    def record_failure(self, source: str, day: date) -> int:
        with self._lock:
            k = (source, day)
            self._fails[k] = self._fails.get(k, 0) + 1
            return self._fails[k]

    def record_success(self, source: str, day: date) -> None:
        with self._lock:
            self._fails.pop((source, day), None)

    def is_tripped(self, source: str, day: date) -> bool:
        return (source, day) in self._tripped

    def trip(self, source: str, day: date) -> None:
        with self._lock:
            self._tripped.add((source, day))


_ledger: Ledger = InMemoryLedger()
def _utc_today() -> date:
    return datetime.now(UTC).date()


_today: Callable[[], date] = _utc_today


def use_ledger(ledger: Ledger) -> None:
    global _ledger
    _ledger = ledger


def get_ledger() -> Ledger:
    return _ledger


def set_clock(fn: Callable[[], date]) -> None:  # tests only
    global _today
    _today = fn


def before_request(source: str) -> None:
    """Called just before each outbound request. Raises if the source is off today."""
    day = _today()
    p = policies().get(source)
    if _ledger.is_tripped(source, day):
        raise SourceUnavailableError(
            f"{source} is switched off until tomorrow (UTC) after repeated failures")
    n = _ledger.consume(source, day)
    if p is not None and n > p.daily_budget:
        raise BudgetExceededError(
            f"{source} daily request budget of {p.daily_budget} is spent; resumes tomorrow (UTC)")


def after_failure(source: str) -> bool:
    """A request failed for good (after retries). Returns True if the breaker tripped."""
    day = _today()
    if _ledger.record_failure(source, day) >= BREAKER_THRESHOLD:
        _ledger.trip(source, day)
        return True
    return False


def after_success(source: str) -> None:
    _ledger.record_success(source, _today())
