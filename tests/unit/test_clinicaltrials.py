from __future__ import annotations

import json
from datetime import UTC, date, datetime
from typing import Any

import httpx
import pytest

from biotradeai.data.clinicaltrials import (
    ClinicalTrialsClient,
    DateKind,
    DatePrecision,
    parse_ctgov_date,
    parse_study,
)
from biotradeai.paths import data_dir

FIX = data_dir() / "fixtures" / "ctgov"


def _load(nct: str) -> dict[str, Any]:
    return json.loads((FIX / f"{nct}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-06", (date(2026, 6, 1), DatePrecision.MONTH)),
        ("2025-11-14", (date(2025, 11, 14), DatePrecision.DAY)),
        ("2026", None),
        ("2026-13", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_dates(raw: str | None, expected: object) -> None:
    assert parse_ctgov_date(raw) == expected


def test_parse_study_anticipated_month_precision() -> None:
    rec = parse_study(_load("NCT09000001"), datetime.now(UTC))
    assert rec.nct_id == "NCT09000001"
    assert rec.sponsor == "Onco Example Therapeutics"
    assert rec.phases == ("PHASE3",)
    assert rec.interventions == ("ONC-101", "Placebo")
    pc = rec.primary_completion
    assert pc is not None
    assert pc.kind is DateKind.ANTICIPATED and pc.precision is DatePrecision.MONTH
    assert pc.value == date(2026, 6, 1) and pc.claimed


def test_parse_study_actual_and_official_title_fallback() -> None:
    rec = parse_study(_load("NCT09000002"), datetime.now(UTC))
    assert rec.title.startswith("A Phase 2/3 Trial")
    assert rec.primary_completion is not None
    assert rec.primary_completion.kind is DateKind.ACTUAL
    assert rec.last_update_posted == date(2026, 2, 11)


def test_parse_study_rejects_bad_nct() -> None:
    with pytest.raises(ValueError):
        parse_study({"protocolSection": {"identificationModule": {"nctId": "X1"}}},
                    datetime.now(UTC))


def test_get_study_offline() -> None:
    def h(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/v2/studies/NCT09000001"
        return httpx.Response(200, json=_load("NCT09000001"))

    rec = ClinicalTrialsClient(transport=httpx.MockTransport(h), backoff_s=0).get_study(
        "NCT09000001")
    assert rec.fetched_at.tzinfo is not None


def test_get_study_validates_id_before_network() -> None:
    c = ClinicalTrialsClient(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    with pytest.raises(ValueError):
        c.get_study("NCT123")


def test_search_paginates() -> None:
    pages = [
        {"studies": [_load("NCT09000001")], "nextPageToken": "tok2"},
        {"studies": [_load("NCT09000002")]},
    ]
    seen: list[dict[str, str]] = []

    def h(req: httpx.Request) -> httpx.Response:
        seen.append(dict(req.url.params))
        return httpx.Response(200, json=pages[len(seen) - 1])

    c = ClinicalTrialsClient(transport=httpx.MockTransport(h), backoff_s=0)
    ids = [r.nct_id for r in c.search(sponsor="Example")]
    assert ids == ["NCT09000001", "NCT09000002"]
    assert seen[0]["query.spons"] == "Example" and "pageToken" not in seen[0]
    assert seen[1]["pageToken"] == "tok2"


def test_search_requires_a_query() -> None:
    with pytest.raises(ValueError):
        list(ClinicalTrialsClient(transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={}))).search())
