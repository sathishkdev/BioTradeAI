"""ClinicalTrials.gov API v2 client (``https://clinicaltrials.gov/api/v2``).

Trial dates are *sponsor-reported claims* (plan correction #3): they are often stale
and are never treated as a PDUFA/catalyst date. Catalyst dates come from
``data/catalysts.csv`` and filings.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any

import httpx

from biotradeai.data.http import FetchError, HttpFetcher

BASE_URL = "https://clinicaltrials.gov/api/v2"
NCT_PATTERN = re.compile(r"^NCT\d{8}$")
# No published hard limit; stay polite.
CTGOV_MAX_REQUESTS_PER_SECOND = 0.8  # see data/netpolicy.py


class DatePrecision(StrEnum):
    DAY = "day"
    MONTH = "month"


class DateKind(StrEnum):
    ACTUAL = "ACTUAL"
    ANTICIPATED = "ANTICIPATED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ClaimedDate:
    """A sponsor-reported date. Month-precision dates are stored as the 1st of the month."""

    value: date
    precision: DatePrecision
    kind: DateKind
    raw: str

    @property
    def claimed(self) -> bool:
        return True


@dataclass(frozen=True)
class TrialRecord:
    nct_id: str
    title: str
    sponsor: str
    overall_status: str
    phases: tuple[str, ...]
    conditions: tuple[str, ...]
    interventions: tuple[str, ...]
    primary_completion: ClaimedDate | None
    completion: ClaimedDate | None
    last_update_posted: date | None
    fetched_at: datetime
    raw: dict[str, Any] = field(repr=False, compare=False)


def parse_ctgov_date(raw: str | None) -> tuple[date, DatePrecision] | None:
    if not raw:
        return None
    parts = raw.split("-")
    try:
        if len(parts) == 3:
            return date(int(parts[0]), int(parts[1]), int(parts[2])), DatePrecision.DAY
        if len(parts) == 2:
            return date(int(parts[0]), int(parts[1]), 1), DatePrecision.MONTH
    except ValueError:
        return None
    return None


def _claimed(struct: dict[str, Any] | None) -> ClaimedDate | None:
    if not struct:
        return None
    raw = struct.get("date")
    parsed = parse_ctgov_date(raw)
    if parsed is None or raw is None:
        return None
    kind_raw = str(struct.get("type", "")).upper()
    kind = DateKind(kind_raw) if kind_raw in DateKind.__members__ else DateKind.UNKNOWN
    return ClaimedDate(value=parsed[0], precision=parsed[1], kind=kind, raw=raw)


def parse_study(study: dict[str, Any], fetched_at: datetime) -> TrialRecord:
    ps = study.get("protocolSection") or {}
    ident = ps.get("identificationModule") or {}
    status = ps.get("statusModule") or {}
    sponsor = (ps.get("sponsorCollaboratorsModule") or {}).get("leadSponsor") or {}
    design = ps.get("designModule") or {}
    conds = ps.get("conditionsModule") or {}
    arms = ps.get("armsInterventionsModule") or {}
    nct_id = ident.get("nctId")
    if not nct_id or not NCT_PATTERN.match(nct_id):
        raise ValueError(f"study has missing/invalid nctId: {nct_id!r}")
    last_update = parse_ctgov_date((status.get("lastUpdatePostDateStruct") or {}).get("date"))
    return TrialRecord(
        nct_id=nct_id,
        title=ident.get("briefTitle") or ident.get("officialTitle") or "",
        sponsor=sponsor.get("name", ""),
        overall_status=status.get("overallStatus", "UNKNOWN"),
        phases=tuple(design.get("phases") or ()),
        conditions=tuple(conds.get("conditions") or ()),
        interventions=tuple(i.get("name", "") for i in arms.get("interventions") or ()),
        primary_completion=_claimed(status.get("primaryCompletionDateStruct")),
        completion=_claimed(status.get("completionDateStruct")),
        last_update_posted=last_update[0] if last_update else None,
        fetched_at=fetched_at,
        raw=study,
    )


class ClinicalTrialsClient:
    def __init__(
        self,
        *,
        transport: httpx.BaseTransport | None = None,
        fetcher: HttpFetcher | None = None,
        backoff_s: float = 1.0,
    ) -> None:
        self._f = fetcher or HttpFetcher(
            BASE_URL,
            source="ctgov",
            headers={"Accept": "application/json"},
            transport=transport,
            backoff_s=backoff_s,
        )

    def get_study(self, nct_id: str) -> TrialRecord:
        if not NCT_PATTERN.match(nct_id):
            raise ValueError(f"invalid NCT id: {nct_id!r}")
        data = self._f.get_json(f"/studies/{nct_id}", params={"format": "json"})
        return parse_study(data, datetime.now(UTC))

    def search(
        self,
        *,
        sponsor: str | None = None,
        intervention: str | None = None,
        term: str | None = None,
        advanced: str | None = None,
        statuses: tuple[str, ...] = (),
        page_size: int = 50,
        max_pages: int = 10,
    ) -> Iterator[TrialRecord]:
        """``advanced`` is the API's ``filter.advanced`` Essie expression, e.g.
        ``AREA[Phase]PHASE3 AND AREA[LeadSponsorClass]INDUSTRY`` (verified live)."""
        params: dict[str, Any] = {"format": "json", "pageSize": page_size}
        if advanced:
            params["filter.advanced"] = advanced
        if statuses:
            params["filter.overallStatus"] = "|".join(statuses)
        if sponsor:
            params["query.spons"] = sponsor
        if intervention:
            params["query.intr"] = intervention
        if term:
            params["query.term"] = term
        if not ({"query.spons", "query.intr", "query.term", "filter.advanced"} & set(params)):
            raise ValueError("search needs sponsor, intervention, term or advanced")
        for _ in range(max_pages):
            data = self._f.get_json("/studies", params=params)
            now = datetime.now(UTC)
            for s in data.get("studies", []):
                yield parse_study(s, now)
            token = data.get("nextPageToken")
            if not token:
                return
            params["pageToken"] = token


__all__ = [
    "ClaimedDate",
    "ClinicalTrialsClient",
    "DateKind",
    "DatePrecision",
    "FetchError",
    "TrialRecord",
    "parse_ctgov_date",
    "parse_study",
]
