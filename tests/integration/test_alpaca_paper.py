"""Read-only probe of the Alpaca PAPER account. Run with:  pytest -m integration

Needs ALPACA_API_KEY / ALPACA_API_SECRET (paper keys) in .env. Places no orders.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from biotradeai.brokers.alpaca_broker import AlpacaBroker
from biotradeai.paths import REPO_ROOT

pytestmark = pytest.mark.integration
load_dotenv(REPO_ROOT / ".env")


@pytest.mark.skipif(not os.environ.get("ALPACA_API_KEY"), reason="Alpaca paper keys not set")
def test_paper_account_reads() -> None:
    b = AlpacaBroker()
    a = b.get_account()
    assert a.equity > 0
    assert isinstance(b.get_positions(), list)
    assert isinstance(b.get_orders(open_only=True), list)
    q = b.get_quote("AAPL")
    assert q.ask >= q.bid >= 0
