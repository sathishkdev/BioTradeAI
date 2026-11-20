"""Thread-safe request spacing with optional jitter (no bursts, no lockstep)."""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable


class RateLimiter:
    """At most ``max_per_second`` calls, spaced evenly.

    ``jitter`` (0..1) randomly *stretches* each gap by up to that fraction, so clients
    never fall into a fixed rhythm. It never makes a gap shorter than the limit allows.
    """

    def __init__(
        self,
        max_per_second: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        *,
        jitter: float = 0.0,
        rng: random.Random | None = None,
    ) -> None:
        if max_per_second <= 0:
            raise ValueError("max_per_second must be > 0")
        if not 0.0 <= jitter <= 1.0:
            raise ValueError("jitter must be in [0, 1]")
        self.interval = 1.0 / max_per_second
        self.jitter = jitter
        self._rng = rng or random.Random()  # noqa: S311 - spacing, not security
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
            gap = self.interval * (1.0 + self.jitter * self._rng.random())
            self._next = now + gap
