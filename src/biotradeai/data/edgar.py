"""SEC EDGAR client: ticker->CIK map, submissions (filing index), XBRL companyfacts.

SEC fair-access rules: a descriptive ``User-Agent`` with contact email is required, and
clients must stay at or below 10 requests/second.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import httpx

from biotradeai.data import netpolicy
from biotradeai.data.http import FetchError, HttpFetcher, RateLimiter

DATA_URL = "https://data.sec.gov"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_MAX_REQUESTS_PER_SECOND = 10.0
# Half the published ceiling (see data/netpolicy.py, source "sec").
DEFAULT_REQUESTS_PER_SECOND = 5.0

_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


class EdgarConfigError(RuntimeError):
    pass


def require_user_agent(value: str | None = None) -> str:
    ua = (value if value is not None else os.environ.get("SEC_USER_AGENT", "")).strip()
    if not ua or not _EMAIL.search(ua) or "you@example.com" in ua:
        raise EdgarConfigError(
            "SEC_USER_AGENT must be set to '<app or name> <contact email>' "
            "(SEC requires it). Put it in .env."
        )
    return ua


def cik10(cik: int | str) -> str:
    return str(int(cik)).zfill(10)


# --------------------------------------------------------------------------- models


@dataclass(frozen=True)
class Filing:
    accession_number: str
    form: str
    filing_date: date
    report_date: date | None
    primary_document: str

    def url(self, cik: int | str) -> str:
        acc = self.accession_number.replace("-", "")
        return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{self.primary_document}"


@dataclass(frozen=True)
class Submissions:
    cik: str
    name: str
    tickers: tuple[str, ...]
    filings: tuple[Filing, ...]
    fetched_at: datetime
    sic: str | None = None
    sic_description: str = ""
    exchanges: tuple[str, ...] = ()

    def by_form(self, *forms: str) -> list[Filing]:
        wanted = {f.upper() for f in forms}
        return [f for f in self.filings if f.form.upper() in wanted]


@dataclass(frozen=True)
class Fact:
    """One XBRL data point. Instants (balance sheet) have ``start=None``."""

    tag: str
    unit: str
    start: date | None
    end: date
    val: Decimal
    accn: str
    fy: int | None
    fp: str | None
    form: str
    filed: date
    frame: str | None = None


@dataclass(frozen=True)
class CompanyFacts:
    cik: str
    entity_name: str
    facts: dict[str, list[Fact]]  # us-gaap tag -> facts (USD only)
    fetched_at: datetime
    raw: dict[str, Any] = field(repr=False, compare=False)

    def get(self, tag: str) -> list[Fact]:
        return self.facts.get(tag, [])


def _d(s: str | None) -> date | None:
    return date.fromisoformat(s) if s else None


def parse_submissions(data: dict[str, Any], fetched_at: datetime) -> Submissions:
    recent = (data.get("filings") or {}).get("recent") or {}
    cols = ("accessionNumber", "form", "filingDate", "reportDate", "primaryDocument")
    n = len(recent.get("accessionNumber", []))
    for c in cols:
        if len(recent.get(c, [])) != n:
            raise ValueError(f"submissions column {c!r} length mismatch")
    filings = tuple(
        Filing(
            accession_number=recent["accessionNumber"][i],
            form=recent["form"][i],
            filing_date=date.fromisoformat(recent["filingDate"][i]),
            report_date=_d(recent["reportDate"][i] or None),
            primary_document=recent["primaryDocument"][i],
        )
        for i in range(n)
    )
    return Submissions(
        cik=cik10(data["cik"]),
        name=data.get("name", ""),
        tickers=tuple(data.get("tickers") or ()),
        filings=filings,
        fetched_at=fetched_at,
        sic=str(data["sic"]) if data.get("sic") else None,
        sic_description=str(data.get("sicDescription") or ""),
        exchanges=tuple(str(x) for x in (data.get("exchanges") or ()) if x),
    )


def parse_company_facts(data: dict[str, Any], fetched_at: datetime) -> CompanyFacts:
    gaap = (data.get("facts") or {}).get("us-gaap") or {}
    out: dict[str, list[Fact]] = {}
    for tag, body in gaap.items():
        usd = (body.get("units") or {}).get("USD")
        if not usd:
            continue
        out[tag] = [
            Fact(
                tag=tag,
                unit="USD",
                start=_d(f.get("start")),
                end=date.fromisoformat(f["end"]),
                val=Decimal(str(f["val"])),
                accn=f["accn"],
                fy=f.get("fy"),
                fp=f.get("fp"),
                form=f["form"],
                filed=date.fromisoformat(f["filed"]),
                frame=f.get("frame"),
            )
            for f in usd
        ]
    return CompanyFacts(
        cik=cik10(data["cik"]),
        entity_name=data.get("entityName", ""),
        facts=out,
        fetched_at=fetched_at,
        raw=data,
    )


# --------------------------------------------------------------------------- client


class EdgarClient:
    def __init__(
        self,
        user_agent: str | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        backoff_s: float = 1.0,
        max_per_second: float | None = None,
    ) -> None:
        if max_per_second is not None and max_per_second > SEC_MAX_REQUESTS_PER_SECOND:
            raise ValueError("SEC allows at most 10 requests/second")
        ua = require_user_agent(user_agent)
        headers = {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}
        # data.sec.gov and www.sec.gov share one fair-access budget, so one limiter:
        # by default the process-wide "sec" limiter shared with every other SEC client.
        limiter = (RateLimiter(max_per_second, jitter=0.1) if max_per_second is not None
                   else netpolicy.limiter_for("sec"))
        self._data = HttpFetcher(DATA_URL, source="sec", headers=headers,
                                 transport=transport, backoff_s=backoff_s, limiter=limiter)
        self._www = HttpFetcher("https://www.sec.gov", source="sec", headers=headers,
                                transport=transport, backoff_s=backoff_s, limiter=limiter)
        self._ticker_map: dict[str, str] | None = None

    def ticker_to_cik(self, ticker: str) -> str:
        if self._ticker_map is None:
            raw = self._www.get_json(TICKERS_URL.removeprefix("https://www.sec.gov"))
            self._ticker_map = {
                str(row["ticker"]).upper(): cik10(row["cik_str"]) for row in raw.values()
            }
        try:
            return self._ticker_map[ticker.upper()]
        except KeyError:
            raise KeyError(f"ticker {ticker!r} not found in SEC ticker map") from None

    def submissions(self, cik: int | str) -> Submissions:
        data = self._data.get_json(f"/submissions/CIK{cik10(cik)}.json")
        return parse_submissions(data, datetime.now(UTC))

    def company_facts(self, cik: int | str) -> CompanyFacts:
        data = self._data.get_json(f"/api/xbrl/companyfacts/CIK{cik10(cik)}.json")
        return parse_company_facts(data, datetime.now(UTC))

    def fetch_document(self, cik: int | str, filing: Filing) -> str:
        """Raw primary document (usually HTML) of a filing from www.sec.gov/Archives."""
        return self._www.get_text(filing.url(cik).removeprefix("https://www.sec.gov"))

    def daily_form_index(self, day: date) -> str | None:
        """SEC daily form index (one file lists every filing that day). None on days
        without an index (weekends, holidays) - SEC answers 404 / 403 for those."""
        q = (day.month - 1) // 3 + 1
        path = f"/Archives/edgar/daily-index/{day.year}/QTR{q}/form.{day:%Y%m%d}.idx"
        try:
            return self._www.get_text(path)
        except FetchError as exc:
            if exc.status_code in (403, 404):
                return None
            raise

    def archive_text(self, path: str) -> str:
        """A file under www.sec.gov/Archives, e.g. a full submission .txt with exhibits."""
        rel = path if path.startswith("/") else f"/Archives/{path}"
        return self._www.get_text(rel)

    def ticker_exchange_rows(self) -> list[tuple[str, str, str, str]]:
        """(cik10, company name, ticker, exchange) for every listed company (1 request)."""
        raw = self._www.get_json("/files/company_tickers_exchange.json")
        fields = list(raw["fields"])
        i = {f: fields.index(f) for f in ("cik", "name", "ticker", "exchange")}
        return [(cik10(r[i["cik"]]), str(r[i["name"]]), str(r[i["ticker"]]),
                 str(r[i["exchange"]] or "")) for r in raw["data"]]

    @property
    def request_count(self) -> int:
        return self._data.request_count + self._www.request_count
