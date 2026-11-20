from __future__ import annotations

import json
from datetime import UTC, date, datetime
from typing import Any

import httpx
import pytest

from biotradeai.data.edgar import (
    EdgarClient,
    EdgarConfigError,
    cik10,
    parse_company_facts,
    parse_submissions,
    require_user_agent,
)
from biotradeai.paths import data_dir

FIX = data_dir() / "fixtures" / "edgar"
UA = "BioTradeAI-tests tests@example.org"


def _j(name: str) -> Any:
    return json.loads((FIX / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("bad", ["", "BioTradeAI", "BioTradeAI research you@example.com"])
def test_user_agent_required(bad: str) -> None:
    with pytest.raises(EdgarConfigError):
        require_user_agent(bad)
    assert require_user_agent(UA) == UA


def test_client_refuses_without_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(EdgarConfigError):
        EdgarClient()


def test_client_refuses_above_sec_rate_limit() -> None:
    with pytest.raises(ValueError):
        EdgarClient(UA, max_per_second=11)


def test_cik_padding() -> None:
    assert cik10(320193) == "0000320193"
    assert cik10("0000320193") == "0000320193"


def test_parse_submissions_and_filter() -> None:
    s = parse_submissions(_j("submissions_ONCO.json"), datetime.now(UTC))
    assert s.cik == "0009000001" and s.tickers == ("ONCO",)
    [s3] = s.by_form("S-3")
    assert s3.filing_date == date(2026, 4, 14) and s3.report_date is None
    assert s3.url(s.cik).endswith("/9000001/000900000126000006/s3.htm")
    assert [f.form for f in s.by_form("10-K", "10-q")] == ["10-K", "10-Q"]


def test_parse_submissions_rejects_ragged_columns() -> None:
    raw = _j("submissions_ONCO.json")
    raw["filings"]["recent"]["form"].pop()
    with pytest.raises(ValueError, match="mismatch"):
        parse_submissions(raw, datetime.now(UTC))


def test_parse_company_facts_usd_only() -> None:
    raw = _j("companyfacts/ONCO.json")
    raw["facts"]["us-gaap"]["SharesOutstanding"] = {"units": {"shares": [
        {"end": "2025-12-31", "val": 1, "accn": "a", "form": "10-K", "filed": "2026-03-02"}]}}
    cf = parse_company_facts(raw, datetime.now(UTC))
    assert "SharesOutstanding" not in cf.facts
    ocf = cf.get("NetCashProvidedByUsedInOperatingActivities")
    assert len(ocf) == 4 and all(f.start is not None for f in ocf)


def test_client_wire_calls_and_headers() -> None:
    seen: list[httpx.Request] = []

    def h(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        path = req.url.path
        if path == "/files/company_tickers.json":
            return httpx.Response(200, json=_j("company_tickers.json"))
        if path == "/submissions/CIK0009000001.json":
            return httpx.Response(200, json=_j("submissions_ONCO.json"))
        if path == "/api/xbrl/companyfacts/CIK0009000001.json":
            return httpx.Response(200, json=_j("companyfacts/ONCO.json"))
        return httpx.Response(404)

    c = EdgarClient(UA, transport=httpx.MockTransport(h), backoff_s=0)
    cik = c.ticker_to_cik("onco")
    assert cik == "0009000001"
    c.ticker_to_cik("DILU")  # cached map: no second download
    assert c.submissions(cik).name == "Onco Example Therapeutics"
    assert c.company_facts(cik).entity_name == "Onco Example Therapeutics"
    assert c.request_count == 3
    assert all(r.headers["User-Agent"] == UA for r in seen)
    assert seen[0].url.host == "www.sec.gov" and seen[1].url.host == "data.sec.gov"
    with pytest.raises(KeyError):
        c.ticker_to_cik("ZZZZ")
