"""openFDA drug approvals (``/drug/drugsfda.json``), to spot drugs already on the market.

Verified live (2026-11-20): resmetirom -> application NDA217785, sponsor MADRIGAL, brand
REZDIFFRA, ``submissions`` with ``submission_type`` ORIG + ``submission_status`` AP on
20240314. A name with no record returns HTTP 404.

Coverage limit: Drugs@FDA covers CDER drugs and CDER-regulated biologics. Vaccines, cell
and gene therapies (CBER) are not in it, so "no record" never proves "not approved".

Polite use: source "openfda" in data/netpolicy.py (1 req/s, 500/day without a key), and
answers are cached in SQLite for ``CACHE_DAYS`` so each name costs ~1 request a month.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta

import httpx

from biotradeai.data.http import FetchError, HttpFetcher
from biotradeai.store.sqlite_store import SQLiteStore

BASE_URL = "https://api.fda.gov"
CACHE_DAYS = 30


@dataclass(frozen=True)
class Approval:
    name: str  # the name that was searched
    brand: str
    generic: str
    application: str  # e.g. NDA217785 / BLA761234
    sponsor: str
    approved_on: date


def _searchable(name: str) -> bool:
    """Generic or brand names only: codes like ABC-123 are never in Drugs@FDA."""
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z \-]{3,}", name.strip())) and \
        not re.search(r"\d", name)


class OpenFDAClient:
    def __init__(self, store: SQLiteStore | None = None, *,
                 transport: httpx.BaseTransport | None = None, backoff_s: float = 1.0,
                 now: datetime | None = None) -> None:
        self._f = HttpFetcher(BASE_URL, source="openfda", transport=transport,
                              backoff_s=backoff_s)
        self._store = store
        self._now = now or datetime.now(UTC)

    def _cached(self, key: str) -> tuple[bool, Approval | None]:
        if self._store is None:
            return False, None
        raw = self._store.kv_get(key)
        if raw is None:
            return False, None
        d = json.loads(raw)
        if self._now - datetime.fromisoformat(d["at"]) > timedelta(days=CACHE_DAYS):
            return False, None
        a = d.get("approval")
        if a is None:
            return True, None
        return True, Approval(**{**a, "approved_on": date.fromisoformat(a["approved_on"])})

    def _remember(self, key: str, a: Approval | None) -> None:
        if self._store is not None:
            body = {"at": self._now.isoformat(),
                    "approval": None if a is None else {**asdict(a),
                                                        "approved_on": a.approved_on.isoformat()}}
            self._store.kv_set(key, json.dumps(body))

    def approval(self, name: str) -> Approval | None:
        """Earliest original approval for a generic or brand name, or None."""
        if not _searchable(name):
            return None
        n = " ".join(name.lower().split())
        key = f"openfda.approval.{n}"
        hit, cached = self._cached(key)
        if hit:
            return cached
        params = {"search": f'openfda.generic_name:"{n}" openfda.brand_name:"{n}"',
                  "limit": 5}
        if os.environ.get("OPENFDA_API_KEY"):
            params["api_key"] = os.environ["OPENFDA_API_KEY"]
        try:
            data = self._f.get_json("/drug/drugsfda.json", params=params)
        except FetchError as exc:
            if exc.status_code == 404:
                self._remember(key, None)
                return None
            raise
        best: Approval | None = None
        for r in data.get("results") or []:
            ofda = r.get("openfda") or {}
            for s in r.get("submissions") or []:
                if s.get("submission_type") != "ORIG" or s.get("submission_status") != "AP":
                    continue
                try:
                    d = datetime.strptime(str(s.get("submission_status_date")), "%Y%m%d").date()
                except ValueError:
                    continue
                if best is None or d < best.approved_on:
                    best = Approval(name=name, brand=(ofda.get("brand_name") or [""])[0],
                                    generic=(ofda.get("generic_name") or [""])[0],
                                    application=str(r.get("application_number", "")),
                                    sponsor=str(r.get("sponsor_name", "")), approved_on=d)
        self._remember(key, best)
        return best
