"""Shared HTTP plumbing for public data sources, with the network guardrails built in.

Every request made through ``HttpFetcher``:
  1. is checked against the host allowlist (also on redirects)      -> netpolicy
  2. passes the source's daily budget and circuit breaker           -> netpolicy
  3. waits on the source's shared, jittered rate limiter
  4. on 429/5xx/transport errors is retried with exponential backoff, honouring the
     server's ``Retry-After`` header (capped), never retrying other 4xx
  5. reports success/failure so repeated failures switch the source off for the day

Clients take an optional ``httpx.BaseTransport`` so tests can run fully offline.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from biotradeai.data import netpolicy
from biotradeai.data.http_limiter import RateLimiter

__all__ = ["FetchError", "HttpFetcher", "RateLimiter", "parse_retry_after"]

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_RETRY_AFTER_S = 300.0


class FetchError(RuntimeError):
    """A request failed permanently (after retries, a non-retryable status, or a
    guardrail refusal: blocked host, budget spent, source switched off)."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class _RetryableStatusError(Exception):
    def __init__(self, response: httpx.Response) -> None:
        super().__init__(f"HTTP {response.status_code}")
        self.response = response
        self.retry_after = parse_retry_after(response.headers.get("Retry-After"))


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    """Seconds to wait from a ``Retry-After`` header (delta-seconds or HTTP-date)."""
    if not value:
        return None
    v = value.strip()
    if v.isdigit():
        return min(float(v), MAX_RETRY_AFTER_S)
    try:
        when = parsedate_to_datetime(v)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    secs = (when - (now or datetime.now(UTC))).total_seconds()
    return min(max(secs, 0.0), MAX_RETRY_AFTER_S)


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, _RetryableStatusError | httpx.TransportError)


def _check_request(request: httpx.Request) -> None:
    netpolicy.check_url(str(request.url))


class HttpFetcher:
    """GET JSON/text from one allowed source, politely."""

    def __init__(
        self,
        base_url: str,
        *,
        source: str | None = None,
        headers: dict[str, str] | None = None,
        max_per_second: float | None = None,
        max_attempts: int = 4,
        backoff_s: float = 1.0,
        timeout_s: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        limiter: RateLimiter | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        netpolicy.check_url(base_url)
        host = httpx.URL(base_url).host
        self.source = source or netpolicy.source_for_host(host) or host
        self._http = httpx.Client(
            base_url=base_url, headers=headers or {}, timeout=timeout_s, transport=transport,
            follow_redirects=True, event_hooks={"request": [_check_request]},
        )
        if limiter is not None:
            self._limiter = limiter
        elif max_per_second is not None:
            self._limiter = RateLimiter(max_per_second, jitter=0.1)
        else:
            self._limiter = netpolicy.limiter_for(self.source)
        self._max_attempts = max_attempts
        self._backoff_s = backoff_s
        self._sleep = sleep
        self.request_count = 0

    def _once(self, url: str, params: dict[str, Any] | None) -> httpx.Response:
        try:
            netpolicy.before_request(self.source)
        except (netpolicy.BudgetExceededError, netpolicy.SourceUnavailableError) as exc:
            raise FetchError(str(exc)) from exc
        self._limiter.acquire()
        self.request_count += 1
        resp = self._http.get(url, params=params)
        if resp.status_code in RETRYABLE_STATUS:
            raise _RetryableStatusError(resp)
        return resp

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        return self._get(url, params).json()

    def get_text(self, url: str, params: dict[str, Any] | None = None) -> str:
        return self._get(url, params).text

    def _wait(self, state: RetryCallState) -> float:
        backoff = wait_exponential(multiplier=self._backoff_s, max=30)(state)
        exc = state.outcome.exception() if state.outcome else None
        hinted = exc.retry_after if isinstance(exc, _RetryableStatusError) else None
        return max(backoff, hinted or 0.0)

    def _fail(self, message: str, status: int | None) -> FetchError:
        if netpolicy.after_failure(self.source):
            message += f" ({self.source} switched off for the rest of the UTC day)"
        return FetchError(message, status)

    def _get(self, url: str, params: dict[str, Any] | None) -> httpx.Response:
        retrying = Retrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=self._wait,
            retry=retry_if_exception(_is_retryable),
            reraise=True,
            sleep=self._sleep,
        )
        try:
            resp = retrying(self._once, url, params)
        except _RetryableStatusError as exc:
            raise self._fail(f"GET {url} failed after {self._max_attempts} attempts: {exc}",
                             exc.response.status_code) from exc
        except httpx.TransportError as exc:
            raise self._fail(f"GET {url} failed: {exc}", None) from exc
        except netpolicy.BlockedHostError as exc:
            raise FetchError(str(exc)) from exc
        if resp.status_code == 404:
            netpolicy.after_success(self.source)  # the source answered; not a fault
            raise FetchError(f"GET {url}: not found", 404)
        if resp.status_code >= 400:
            raise FetchError(f"GET {url}: HTTP {resp.status_code}", resp.status_code)
        netpolicy.after_success(self.source)
        return resp

    def close(self) -> None:
        self._http.close()
