"""Shared HTTP plumbing: polite rate limiting and retries for public data sources.

Clients take an optional ``httpx.BaseTransport`` so tests can swap the network for
``httpx.MockTransport`` and run fully offline.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import httpx
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class FetchError(RuntimeError):
    """A request failed permanently (after retries, or with a non-retryable status)."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class _RetryableStatusError(Exception):
    def __init__(self, response: httpx.Response) -> None:
        super().__init__(f"HTTP {response.status_code}")
        self.response = response


class RateLimiter:
    """Allows at most ``max_per_second`` calls by spacing them evenly."""

    def __init__(
        self,
        max_per_second: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_per_second <= 0:
            raise ValueError("max_per_second must be > 0")
        self.interval = 1.0 / max_per_second
        self._clock = clock
        self._sleep = sleep
        self._next = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = self._clock()
            wait = self._next - now
            if wait > 0:
                self._sleep(wait)
                now += wait
            self._next = now + self.interval


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, _RetryableStatusError | httpx.TransportError)


class HttpFetcher:
    """GET JSON with rate limiting and exponential-backoff retries."""

    def __init__(
        self,
        base_url: str,
        *,
        headers: dict[str, str] | None = None,
        max_per_second: float = 5.0,
        max_attempts: int = 4,
        backoff_s: float = 1.0,
        timeout_s: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        limiter: RateLimiter | None = None,
    ) -> None:
        self._http = httpx.Client(
            base_url=base_url, headers=headers or {}, timeout=timeout_s, transport=transport,
            follow_redirects=True,
        )
        self._limiter = limiter or RateLimiter(max_per_second)
        self._max_attempts = max_attempts
        self._backoff_s = backoff_s
        self.request_count = 0

    def _once(self, url: str, params: dict[str, Any] | None) -> httpx.Response:
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

    def _get(self, url: str, params: dict[str, Any] | None) -> httpx.Response:
        retrying = Retrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(multiplier=self._backoff_s, max=30),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        )
        try:
            resp = retrying(self._once, url, params)
        except _RetryableStatusError as exc:
            raise FetchError(
                f"GET {url} failed after {self._max_attempts} attempts: {exc}",
                exc.response.status_code,
            ) from exc
        except httpx.TransportError as exc:
            raise FetchError(f"GET {url} failed: {exc}") from exc
        if resp.status_code == 404:
            raise FetchError(f"GET {url}: not found", 404)
        if resp.status_code >= 400:
            raise FetchError(f"GET {url}: HTTP {resp.status_code}", resp.status_code)
        return resp

    def close(self) -> None:
        self._http.close()
