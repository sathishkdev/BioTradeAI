"""SEC daily filing index + a cached company directory.

Instead of polling every company, one request per business day returns the list of
every filing (form, company, CIK, date, path). Company facts needed for filtering
(industry code, tickers, exchanges) are cached in SQLite and refreshed only when older
than ``COMPANY_TTL_DAYS``, so each company costs one request per month at most.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from biotradeai.data.edgar import EdgarClient, cik10
from biotradeai.rag.ingest import html_to_text
from biotradeai.store.sqlite_store import SQLiteStore

# Pharmaceutical preparations, biological products, commercial biological research.
BIOTECH_SIC = frozenset({"2834", "2836", "8731"})
LISTED_EXCHANGES = frozenset({"Nasdaq", "NYSE", "NYSE American"})
COMPANY_TTL_DAYS = 30

_LINE = re.compile(r"^(?P<form>\S(?:.*?\S)?)\s{2,}(?P<company>\S.*?)\s{2,}(?P<cik>\d+)\s+"
                   r"(?P<filed>\d{8})\s+(?P<path>edgar/\S+)\s*$")
_UUENCODED = re.compile(r"(?ms)^begin \d{3} .*?^end$")


@dataclass(frozen=True)
class IndexEntry:
    form: str
    company: str
    cik: str  # 10-digit
    filed: date
    path: str  # edgar/data/<cik>/<accession>.txt  (full submission incl. exhibits)

    @property
    def url(self) -> str:
        return f"https://www.sec.gov/Archives/{self.path}"

    @property
    def accession(self) -> str:
        return self.path.rsplit("/", 1)[-1].removesuffix(".txt")


def parse_form_index(text: str) -> list[IndexEntry]:
    out: list[IndexEntry] = []
    body = text.split("\n---", 1)[-1] if "\n---" in text else text
    for line in body.splitlines():
        m = _LINE.match(line.rstrip())
        if not m:
            continue
        try:
            filed = datetime.strptime(m["filed"], "%Y%m%d").date()
        except ValueError:
            continue
        out.append(IndexEntry(m["form"].strip(), m["company"].strip(), cik10(m["cik"]),
                              filed, m["path"]))
    return out


def business_days(start: date, end: date) -> list[date]:
    days, d = [], start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def submission_text(raw: str) -> str:
    """Readable text of a full submission .txt (drops uuencoded images, HTML tags)."""
    return html_to_text(_UUENCODED.sub(" ", raw))


@dataclass(frozen=True)
class Company:
    cik: str
    name: str
    sic: str | None
    tickers: tuple[str, ...]
    exchanges: tuple[str, ...]

    @property
    def is_biotech(self) -> bool:
        return self.sic in BIOTECH_SIC

    @property
    def is_listed(self) -> bool:
        return bool(set(self.exchanges) & LISTED_EXCHANGES) and bool(self.tickers)

    @property
    def ticker(self) -> str | None:
        return self.tickers[0] if self.tickers else None


class SecDirectory:
    def __init__(self, edgar: EdgarClient, store: SQLiteStore, *,
                 now: datetime | None = None) -> None:
        self.edgar = edgar
        self.store = store
        self._now = now or datetime.now(UTC)

    def company(self, cik: str) -> Company:
        cik = cik10(cik)
        row = self.store.get_company(cik)
        if row is not None:
            age = self._now - datetime.fromisoformat(str(row["fetched_at"]))
            if age <= timedelta(days=COMPANY_TTL_DAYS):
                return self._from_row(row)
        s = self.edgar.submissions(cik)
        self.store.upsert_company(cik, s.name, s.sic, list(s.tickers), list(s.exchanges))
        return Company(cik, s.name, s.sic, tuple(s.tickers), tuple(s.exchanges))

    @staticmethod
    def _from_row(row: dict[str, Any]) -> Company:
        return Company(str(row["cik"]), str(row["name"]), row.get("sic"),
                       tuple(row["tickers"]), tuple(row["exchanges"]))

    def filings(self, day: date) -> list[IndexEntry]:
        text = self.edgar.daily_form_index(day)
        return parse_form_index(text) if text else []
