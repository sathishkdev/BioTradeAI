"""Runway math vs hand-verified values in data/fixtures/edgar/expected_runway.yaml."""

from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import pytest
import yaml

from biotradeai.config import load_config
from biotradeai.data.edgar import CompanyFacts, parse_company_facts
from biotradeai.data.runway import RunwayUnavailableError, compute_runway, discrete_quarters
from biotradeai.paths import data_dir

FIX = data_dir() / "fixtures" / "edgar"
EXPECTED: dict[str, dict[str, Any]] = yaml.safe_load(
    (FIX / "expected_runway.yaml").read_text(encoding="utf-8"))
LOOKBACK = load_config().runway.burn_lookback_quarters
M = Decimal(1_000_000)


def _facts(ticker: str) -> CompanyFacts:
    raw = json.loads((FIX / "companyfacts" / f"{ticker}.json").read_text(encoding="utf-8"))
    return parse_company_facts(raw, datetime.now(UTC))


def test_ten_tickers_covered() -> None:
    tickers = {k.split("@")[0] for k in EXPECTED}
    assert len(tickers) == 10


@pytest.mark.parametrize("case", sorted(EXPECTED))
def test_runway_matches_hand_verified(case: str) -> None:
    exp = EXPECTED[case]
    ticker, _, as_of_s = case.partition("@")
    as_of = date.fromisoformat(as_of_s) if as_of_s else None
    cf = _facts(ticker)
    if "error" in exp:
        with pytest.raises(RunwayUnavailableError):
            compute_runway(cf, lookback_quarters=LOOKBACK, as_of=as_of)
        return
    r = compute_runway(cf, lookback_quarters=LOOKBACK, as_of=as_of)
    assert r.balance_sheet_date == exp["balance_sheet_date"]
    assert r.liquidity == Decimal(str(exp["liquidity"])) * M
    assert r.quarterly_burn == Decimal(str(exp["quarterly_burn"])) * M
    if math.isinf(exp["runway_months"]):
        assert math.isinf(r.runway_months) and not r.burning
    else:
        assert r.runway_months == pytest.approx(exp["runway_months"], abs=0.005)


def test_restatement_is_noted_and_wins() -> None:
    r = compute_runway(_facts("RSTA"), lookback_quarters=LOOKBACK)
    assert r.cash == 45 * M
    assert any("superseded" in n for n in r.notes)


def test_identical_duplicates_are_silent() -> None:
    r = compute_runway(_facts("DUPE"), lookback_quarters=LOOKBACK)
    assert not any("superseded" in n for n in r.notes)


def test_derivation_methods_recorded() -> None:
    qs = discrete_quarters(_facts("ONCO"))
    assert [q.method for q in qs] == ["reported", "derived", "derived", "derived"]
    assert all(q.ocf == -10 * M for q in qs)


def test_fallback_and_gap_notes() -> None:
    falb = compute_runway(_facts("FALB"), lookback_quarters=LOOKBACK)
    assert any("fallback tag" in n for n in falb.notes)
    gapq = compute_runway(_facts("GAPQ"), lookback_quarters=LOOKBACK)
    assert any("gap" in n for n in gapq.notes)
    annl = compute_runway(_facts("ANNL"), lookback_quarters=LOOKBACK)
    assert annl.quarters_used[0].method == "annual/4"


def test_point_in_time_excludes_later_filings() -> None:
    # Before the 10-Q/A (2025-12-15), RSTA's original Q3 numbers are used.
    r = compute_runway(_facts("RSTA"), lookback_quarters=LOOKBACK, as_of=date(2025, 12, 1))
    assert r.cash == 50 * M and r.quarterly_burn == 15 * M
    assert r.runway_months == pytest.approx(10.0)
    with pytest.raises(RunwayUnavailableError):
        compute_runway(_facts("ONCO"), lookback_quarters=LOOKBACK, as_of=date(2025, 1, 1))


def test_lookback_one_uses_latest_only() -> None:
    r = compute_runway(_facts("DUPE"), lookback_quarters=1)
    assert r.quarterly_burn == 6 * M and len(r.quarters_used) == 1


def test_sources_are_traceable() -> None:
    r = compute_runway(_facts("ONCO"), lookback_quarters=LOOKBACK)
    assert r.source_accns and all(a.startswith("0009000001-") for a in r.source_accns)


def test_bad_lookback() -> None:
    with pytest.raises(ValueError):
        compute_runway(_facts("ONCO"), lookback_quarters=0)
