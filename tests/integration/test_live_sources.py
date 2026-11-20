"""Live probes against the real public APIs. Run with:  pytest -m integration

These check that our parsers still match the real response shapes. They need network;
the EDGAR test also needs SEC_USER_AGENT in .env.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from biotradeai.data.clinicaltrials import ClinicalTrialsClient
from biotradeai.data.edgar import EdgarClient
from biotradeai.data.runway import compute_runway
from biotradeai.paths import REPO_ROOT

pytestmark = pytest.mark.integration
load_dotenv(REPO_ROOT / ".env")


def test_ctgov_search_and_get() -> None:
    c = ClinicalTrialsClient()
    first = next(iter(c.search(intervention="pembrolizumab", page_size=5, max_pages=1)))
    again = c.get_study(first.nct_id)
    assert again.nct_id == first.nct_id and again.sponsor


@pytest.mark.skipif(not os.environ.get("SEC_USER_AGENT"), reason="SEC_USER_AGENT not set")
def test_edgar_runway_for_a_real_biotech() -> None:
    # Vertex: large, long-lived XBRL history -> good shape check (cash-flow positive).
    c = EdgarClient()
    cik = c.ticker_to_cik("VRTX")
    subs = c.submissions(cik)
    assert subs.by_form("10-K")
    r = compute_runway(c.company_facts(cik), lookback_quarters=4)
    assert r.liquidity > 0 and r.source_accns
