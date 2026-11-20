from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from biotradeai.data.freshness import StaleDataError, is_fresh, require_fresh
from biotradeai.data.http import FetchError, HttpFetcher, RateLimiter

# ------------------------------------------------------------------ rate limiter


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


def test_rate_limiter_spaces_calls() -> None:
    c = FakeClock()
    rl = RateLimiter(10, clock=c.now, sleep=c.sleep)
    for _ in range(11):
        rl.acquire()
    # 11 calls at 10/s must take at least 1.0s of simulated time.
    assert c.t == pytest.approx(1.0)
    assert all(s == pytest.approx(0.1) for s in c.sleeps)


def test_rate_limiter_no_sleep_when_idle() -> None:
    c = FakeClock()
    rl = RateLimiter(10, clock=c.now, sleep=c.sleep)
    rl.acquire()
    c.t += 5
    rl.acquire()
    assert c.sleeps == []


# ------------------------------------------------------------------ fetcher


def _fetcher(handler: httpx.MockTransport, attempts: int = 3) -> HttpFetcher:
    c = FakeClock()
    return HttpFetcher("https://x.test", transport=handler, max_attempts=attempts,
                       backoff_s=0, limiter=RateLimiter(1000, clock=c.now, sleep=c.sleep))


def test_retries_then_succeeds() -> None:
    calls = {"n": 0}

    def h(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] < 3 else httpx.Response(200, json={"ok": 1})

    assert _fetcher(httpx.MockTransport(h)).get_json("/a") == {"ok": 1}
    assert calls["n"] == 3


def test_gives_up_after_max_attempts() -> None:
    f = _fetcher(httpx.MockTransport(lambda r: httpx.Response(429)), attempts=2)
    with pytest.raises(FetchError) as ei:
        f.get_json("/a")
    assert ei.value.status_code == 429
    assert f.request_count == 2


def test_404_not_retried() -> None:
    f = _fetcher(httpx.MockTransport(lambda r: httpx.Response(404)))
    with pytest.raises(FetchError, match="not found"):
        f.get_json("/a")
    assert f.request_count == 1


def test_transport_error_retried_and_wrapped() -> None:
    def h(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    f = _fetcher(httpx.MockTransport(h), attempts=2)
    with pytest.raises(FetchError):
        f.get_json("/a")
    assert f.request_count == 2


# ------------------------------------------------------------------ freshness

NOW = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)
TTL = 24.0


@pytest.mark.parametrize(
    ("age", "fresh"),
    [
        (timedelta(hours=23, minutes=59), True),   # just below TTL
        (timedelta(hours=24), True),               # at TTL (inclusive)
        (timedelta(hours=24, seconds=1), False),   # just above TTL
        (timedelta(0), True),
        (-timedelta(minutes=4), True),             # small clock skew tolerated
        (-timedelta(hours=1), False),              # from the future -> invalid
    ],
)
def test_ttl_boundaries(age: timedelta, fresh: bool) -> None:
    assert is_fresh(NOW - age, now=NOW, ttl_hours=TTL) is fresh


def test_require_fresh_raises() -> None:
    with pytest.raises(StaleDataError):
        require_fresh(NOW - timedelta(days=2), now=NOW, ttl_hours=TTL, what="CT.gov study")
    require_fresh(NOW, now=NOW, ttl_hours=TTL, what="x")


def test_naive_timestamps_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        is_fresh(datetime(2026, 1, 1), now=NOW, ttl_hours=TTL)
