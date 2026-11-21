"""Network guardrails: allowed hosts, polite rates, daily budgets, circuit breaker,
Retry-After, jittered spacing and balanced pacing. All offline (MockTransport)."""

from __future__ import annotations

import random
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from biotradeai.data import netpolicy
from biotradeai.data.http import FetchError, HttpFetcher, parse_retry_after
from biotradeai.data.http_limiter import RateLimiter
from biotradeai.data.pacing import pace_interval, paced, rolling_batch
from biotradeai.paths import REPO_ROOT
from biotradeai.store.sqlite_store import SQLiteLedger

SRC = REPO_ROOT / "src" / "biotradeai"
# Hosts that appear in source only as citation links stored with documents; the code
# never sends requests to them.
# Shown to the human as links (opened in their browser), never fetched by the agents.
LINK_ONLY_HOSTS = {"pubmed.ncbi.nlm.nih.gov", "app.alpaca.markets", "robinhood.com"}


class Server:
    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        return self.responses.pop(0) if self.responses else httpx.Response(200, json={})


def _fetcher(server: Server, *, base: str = "https://clinicaltrials.gov/api/v2",
             attempts: int = 1, sleeps: list[float] | None = None) -> HttpFetcher:
    return HttpFetcher(base, transport=httpx.MockTransport(server), max_attempts=attempts,
                       backoff_s=0, limiter=RateLimiter(1e9),
                       sleep=(sleeps.append if sleeps is not None else lambda s: None))


# ------------------------------------------------------------------ allowlist


def test_every_url_in_source_is_an_allowed_host() -> None:
    hosts = set()
    for p in SRC.rglob("*.py"):
        hosts |= set(re.findall(r"https://([A-Za-z0-9.-]+)", p.read_text(encoding="utf-8")))
    bad = sorted(h for h in hosts - LINK_ONLY_HOSTS if not netpolicy.is_allowed_host(h))
    assert bad == [], f"add these to data/netpolicy.py or remove them: {bad}"


@pytest.mark.parametrize("host", ["example.com", "investors.somebiotech.com",
                                  "www.google.com", "biopharmcatalyst.com"])
def test_unlisted_hosts_are_refused(host: str) -> None:
    assert not netpolicy.is_allowed_host(host)
    with pytest.raises(netpolicy.BlockedHostError):
        HttpFetcher(f"https://{host}")


def test_redirect_to_another_host_is_blocked() -> None:
    server = Server(httpx.Response(302, headers={"Location": "https://evil.example/x"}))
    with pytest.raises(FetchError, match="not an allowed data source"):
        _fetcher(server).get_json("/studies")
    assert server.calls == 1  # the redirect target was never contacted


def test_sources_map_to_hosts() -> None:
    assert netpolicy.source_for_host("data.sec.gov") == "sec"
    assert netpolicy.source_for_host("WWW.SEC.GOV") == "sec"
    assert netpolicy.source_for_host("eutils.ncbi.nlm.nih.gov") == "ncbi"
    assert netpolicy.source_for_host("api.fda.gov") == "openfda"


# ------------------------------------------------------------------ polite rates


def test_rates_stay_well_under_published_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    monkeypatch.delenv("OPENFDA_API_KEY", raising=False)
    p = netpolicy.policies()
    assert p["sec"].max_per_second <= 10 / 2
    assert p["ctgov"].max_per_second * 60 < 50  # most conservative figure found
    assert p["ncbi"].max_per_second < 3
    assert p["openfda"].max_per_second * 60 < 240 and p["openfda"].daily_budget < 1000
    monkeypatch.setenv("NCBI_API_KEY", "k")
    monkeypatch.setenv("OPENFDA_API_KEY", "k")
    p = netpolicy.policies()
    assert p["ncbi"].max_per_second <= 10 / 2
    assert p["openfda"].daily_budget < 120_000


def test_jitter_only_ever_lengthens_gaps() -> None:
    t = [0.0]
    sleeps: list[float] = []

    def sleep(s: float) -> None:
        sleeps.append(s)
        t[0] += s

    rl = RateLimiter(2.0, clock=lambda: t[0], sleep=sleep, jitter=0.5,
                     rng=random.Random(7))
    for _ in range(50):
        rl.acquire()
    assert all(0.5 - 1e-9 <= s <= 0.75 + 1e-9 for s in sleeps)
    assert len(set(round(s, 6) for s in sleeps)) > 10  # irregular, no lockstep


# ------------------------------------------------------------------ budgets + breaker


def _small_budget(monkeypatch: pytest.MonkeyPatch, n: int) -> None:
    real = netpolicy.policies

    def patched() -> dict[str, netpolicy.SourcePolicy]:
        out = real()
        c = out["ctgov"]
        out["ctgov"] = netpolicy.SourcePolicy(c.name, c.hosts, c.max_per_second, n,
                                              c.published_rule)
        return out
    monkeypatch.setattr(netpolicy, "policies", patched)


def test_daily_budget_stops_requests_before_they_are_sent(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _small_budget(monkeypatch, 2)
    server = Server()
    f = _fetcher(server)
    f.get_json("/a")
    f.get_json("/b")
    with pytest.raises(FetchError, match="budget"):
        f.get_json("/c")
    assert server.calls == 2


def test_budget_resets_next_day(monkeypatch: pytest.MonkeyPatch) -> None:
    _small_budget(monkeypatch, 1)
    day = [date(2026, 11, 20)]
    monkeypatch.setattr(netpolicy, "_today", lambda: day[0])
    f = _fetcher(Server())
    f.get_json("/a")
    with pytest.raises(FetchError):
        f.get_json("/b")
    day[0] += timedelta(days=1)
    f.get_json("/c")


def test_breaker_switches_source_off_after_repeated_failures() -> None:
    server = Server(*[httpx.Response(503) for _ in range(3)])
    f = _fetcher(server)
    for _ in range(netpolicy.BREAKER_THRESHOLD - 1):
        with pytest.raises(FetchError):
            f.get_json("/x")
    with pytest.raises(FetchError, match="switched off"):
        f.get_json("/x")
    calls = server.calls
    with pytest.raises(FetchError, match="switched off until tomorrow"):
        f.get_json("/x")
    assert server.calls == calls  # nothing sent while the breaker is open


def test_success_resets_failure_count() -> None:
    server = Server(httpx.Response(503), httpx.Response(503), httpx.Response(200, json={}),
                    httpx.Response(503), httpx.Response(503), httpx.Response(200, json={}))
    f = _fetcher(server)
    for _ in range(2):
        for _ in range(2):
            with pytest.raises(FetchError):
                f.get_json("/x")
        f.get_json("/ok")  # never three in a row, so the breaker stays closed


def test_not_found_is_not_a_failure() -> None:
    server = Server(*[httpx.Response(404) for _ in range(5)])
    f = _fetcher(server)
    for _ in range(5):
        with pytest.raises(FetchError, match="not found"):
            f.get_json("/missing")


def test_sqlite_ledger_persists_between_runs(tmp_path: Path) -> None:
    db = tmp_path / "l.db"
    day = date(2026, 11, 20)
    a = SQLiteLedger(db)
    assert a.consume("sec", day) == 1 and a.consume("sec", day) == 2
    for _ in range(3):
        a.record_failure("sec", day)
    a.trip("sec", day)
    b = SQLiteLedger(db)  # a later scheduled run
    assert b.consume("sec", day) == 3
    assert b.is_tripped("sec", day) and not b.is_tripped("sec", day + timedelta(days=1))
    b.record_success("sec", day)
    assert b.failures("sec", day) == 0
    assert any(e.event_type == "network.source_off" for e in b.store.read_audit())


# ------------------------------------------------------------------ Retry-After


def test_parse_retry_after() -> None:
    now = datetime(2026, 11, 20, 12, 0, tzinfo=UTC)
    assert parse_retry_after("7") == 7.0
    assert parse_retry_after("99999") == 300.0  # capped
    assert parse_retry_after("Fri, 20 Nov 2026 12:00:30 GMT", now=now) == 30.0
    assert parse_retry_after("Fri, 20 Nov 2026 11:00:00 GMT", now=now) == 0.0
    assert parse_retry_after(None) is None and parse_retry_after("soon") is None


def test_retry_after_is_honoured() -> None:
    sleeps: list[float] = []
    server = Server(httpx.Response(429, headers={"Retry-After": "12"}),
                    httpx.Response(200, json={"ok": 1}))
    assert _fetcher(server, attempts=3, sleeps=sleeps).get_json("/x") == {"ok": 1}
    assert sleeps == [12.0]


def test_other_4xx_are_never_retried() -> None:
    server = Server(httpx.Response(403), httpx.Response(200, json={}))
    with pytest.raises(FetchError, match="403"):
        _fetcher(server, attempts=4).get_json("/x")
    assert server.calls == 1


# ------------------------------------------------------------------ pacing


def test_paced_spreads_a_job_across_the_window() -> None:
    sleeps: list[float] = []
    items = list(range(800))
    out = list(paced(items, window_s=7200, sleep=sleeps.append, rng=random.Random(1)))
    assert out == items and len(sleeps) == 799
    base = 7200 / 800  # 9 s
    assert all(base - 1e-9 <= s <= base * 1.25 + 1e-9 for s in sleeps)
    assert 7200 * 0.95 <= sum(sleeps) <= 7200 * 1.25


def test_paced_never_beats_the_minimum_interval() -> None:
    assert pace_interval(10_000, 60, min_interval_s=0.2) == 0.2
    sleeps: list[float] = []
    list(paced(list(range(5)), window_s=0, min_interval_s=1.0, jitter=0,
               sleep=sleeps.append))
    assert sleeps == [1.0] * 4


def test_rolling_batch_covers_everyone_once_per_cycle() -> None:
    keys = [f"CIK{i:07d}" for i in range(800)]
    start = date(2026, 11, 16)
    batches = [rolling_batch(keys, day=start + timedelta(days=d), parts=7) for d in range(7)]
    seen = [k for b in batches for k in b]
    assert sorted(seen) == sorted(keys)  # every key exactly once a week
    assert max(len(b) for b in batches) < 160  # ~115 a night, never the whole list
    assert rolling_batch(keys, day=start, parts=7) == batches[0]  # stable across runs
