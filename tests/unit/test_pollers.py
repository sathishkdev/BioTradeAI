"""EDGAR filing poller and CT.gov completion-date poller (offline, MockTransport)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from biotradeai.data.clinicaltrials import ClinicalTrialsClient
from biotradeai.data.edgar import EdgarClient
from biotradeai.data.pollers import FilingPoller, TrialDatePoller
from biotradeai.store.sqlite_store import SQLiteStore

UA = "BioTradeAI tests ops@example.com"
CIK = "0009990003"


@pytest.fixture
def store(tmp_path: Path) -> SQLiteStore:
    return SQLiteStore(tmp_path / "a.db")


def _subs() -> dict[str, Any]:
    rows = [  # accession, form, filed, doc
        ("0009990003-26-000001", "10-Q", "2026-11-10", "q.htm"),
        ("0009990003-26-000002", "8-K", "2026-11-12", "board.htm"),
        ("0009990003-26-000003", "8-K", "2026-11-14", "offer.htm"),
        ("0009990003-26-000004", "S-3", "2026-11-15", "s3.htm"),
        ("0009990003-25-000099", "S-3", "2025-06-01", "old.htm"),
    ]
    return {"cik": CIK, "name": "Dilution Co", "tickers": ["DLTN"], "filings": {"recent": {
        "accessionNumber": [r[0] for r in rows], "form": [r[1] for r in rows],
        "filingDate": [r[2] for r in rows], "reportDate": ["" for _ in rows],
        "primaryDocument": [r[3] for r in rows]}}}


DOCS = {
    "board.htm": "<html><body><p>The Board appointed a new director.</p></body></html>",
    "offer.htm": "<html><body><p>Pricing of $40 million underwritten public offering of "
                 "common stock.</p></body></html>",
}


def _edgar(seen: list[str]) -> EdgarClient:
    def h(req: httpx.Request) -> httpx.Response:
        seen.append(req.url.path)
        if req.url.path.startswith("/submissions/"):
            return httpx.Response(200, json=_subs())
        doc = req.url.path.rsplit("/", 1)[-1]
        return httpx.Response(200, text=DOCS.get(doc, "<p>nothing</p>"))
    return EdgarClient(UA, transport=httpx.MockTransport(h), backoff_s=0)


def test_filing_poller_classifies_and_reports_each_filing_once(store: SQLiteStore) -> None:
    seen: list[str] = []
    poller = FilingPoller(_edgar(seen), store)
    evs = poller.poll("DLTN", CIK, since=date(2026, 11, 1))
    assert [(e.form, e.dilutive) for e in evs] == [("8-K", False), ("8-K", True), ("S-3", True)]
    assert "underwritten public offering" in evs[1].reason
    # only 8-K documents are downloaded
    assert sum(p.endswith((".htm",)) for p in seen) == 2
    assert poller.poll("DLTN", CIK, since=date(2026, 11, 1)) == []  # already seen
    assert sum(1 for a in store.read_audit() if a.event_type == "poll.filing") == 3


def _study(primary: str) -> dict[str, Any]:
    p = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "ctgov" / "NCT09000001.json"
    data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    data["protocolSection"]["statusModule"]["primaryCompletionDateStruct"] = {
        "date": primary, "type": "ESTIMATED"}
    return data


def test_trial_poller_reports_changes_after_baseline(store: SQLiteStore) -> None:
    dates = iter(["2027-01", "2027-01", "2027-08-15"])

    def h(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_study(next(dates)))

    poller = TrialDatePoller(ClinicalTrialsClient(transport=httpx.MockTransport(h),
                                                  backoff_s=0), store)
    assert poller.poll("ONCO", "NCT09000001") == []  # first sighting = baseline
    assert poller.poll("ONCO", "NCT09000001") == []  # unchanged
    evs = poller.poll("ONCO", "NCT09000001")
    assert len(evs) == 1
    assert evs[0].old_date is not None and evs[0].new_date == date(2027, 8, 15)
    assert store.get_nct("NCT09000001")["last_completion_date"] == "2027-08-15"
