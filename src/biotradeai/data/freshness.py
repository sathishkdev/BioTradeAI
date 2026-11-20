"""Time-to-live checks.

TTL here means *observation* freshness: how long ago we fetched the record from its
source. It does not judge how old the underlying fact is (a 10-Q balance sheet is
always weeks old); that is the runway calculation's ``as_of`` date.
"""

from __future__ import annotations

from datetime import datetime, timedelta

# Tolerated clock skew for timestamps slightly in the future (seconds).
MAX_FUTURE_SKEW = timedelta(minutes=5)


class StaleDataError(RuntimeError):
    pass


def _require_aware(name: str, ts: datetime) -> None:
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def age(observed_at: datetime, now: datetime) -> timedelta:
    _require_aware("observed_at", observed_at)
    _require_aware("now", now)
    return now - observed_at


def is_fresh(observed_at: datetime, *, now: datetime, ttl_hours: float) -> bool:
    """Fresh if fetched no more than ``ttl_hours`` ago (inclusive) and not from the future."""
    a = age(observed_at, now)
    if a < -MAX_FUTURE_SKEW:
        return False
    return a <= timedelta(hours=ttl_hours)


def require_fresh(observed_at: datetime, *, now: datetime, ttl_hours: float, what: str) -> None:
    if not is_fresh(observed_at, now=now, ttl_hours=ttl_hours):
        raise StaleDataError(
            f"{what} observed at {observed_at.isoformat()} is older than {ttl_hours}h "
            f"TTL (or in the future) as of {now.isoformat()}"
        )
