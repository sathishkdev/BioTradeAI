"""Balanced scheduling for bulk jobs (scout, universe refresh, document sync).

Even off-peak, a job should not fire all its requests at the source's maximum rate.
Two tools keep traffic flat:

* ``paced(items, window_s=...)``: spreads a job's requests evenly across a time window,
  e.g. 800 requests over 2 hours is one every ~9 s, with random jitter so the gaps are
  irregular. It is never faster than the source's own rate limiter, which still applies.
* ``rolling_batch(keys, day=..., parts=N)``: a stable 1-in-N slice of a large list, so
  a refresh of 800 companies with ``parts=7`` touches ~115 per night and every company
  once a week. The slice is a deterministic hash, so it's the same across runs.
"""

from __future__ import annotations

import hashlib
import random
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from datetime import date
from typing import TypeVar

T = TypeVar("T")

DEFAULT_JITTER = 0.25  # each gap stretched by 0-25%


def pace_interval(n_items: int, window_s: float, *, min_interval_s: float = 0.0) -> float:
    """Seconds between requests to spread ``n_items`` across ``window_s``."""
    if n_items <= 1:
        return min_interval_s
    return max(window_s / n_items, min_interval_s)


def paced(items: Sequence[T], *, window_s: float, min_interval_s: float = 0.0,
          jitter: float = DEFAULT_JITTER, sleep: Callable[[float], None] = time.sleep,
          rng: random.Random | None = None) -> Iterator[T]:
    """Yield items one by one, sleeping between them so the job spans ~``window_s``.

    The gap is only ever *stretched* by jitter, so the total may run a little past the
    window; it never bursts. No sleep before the first item.
    """
    if window_s < 0 or not 0.0 <= jitter <= 1.0:
        raise ValueError("window_s must be >= 0 and jitter in [0, 1]")
    r = rng or random.Random()  # noqa: S311 - spacing, not security
    gap = pace_interval(len(items), window_s, min_interval_s=min_interval_s)
    for i, item in enumerate(items):
        if i and gap > 0:
            sleep(gap * (1.0 + jitter * r.random()))
        yield item


def _bucket(key: str, parts: int) -> int:
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % parts


def rolling_batch(keys: Iterable[str], *, day: date, parts: int) -> list[str]:
    """Keys whose stable hash bucket matches today's slot (day ordinal mod ``parts``)."""
    if parts < 1:
        raise ValueError("parts must be >= 1")
    slot = day.toordinal() % parts
    return [k for k in keys if _bucket(k, parts) == slot]
