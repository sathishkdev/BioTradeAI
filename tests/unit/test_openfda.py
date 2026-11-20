"""openFDA approval lookups (offline) and the scout's approved-drug / market-cap filters."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from biotradeai.data.openfda import OpenFDAClient
from biotradeai.store.sqlite_store import SQLiteStore

NOW = datetime(2026, 11, 20, 15, tzinfo=UTC)
REZ = {"application_number": "NDA217785", "sponsor_name": "MADRIGAL",
       "openfda": {"brand_name": ["REZDIFFRA"], "generic_name": ["RESMETIROM"]},
       "submissions": [{"submission_type": "SUPPL", "submission_status": "AP",
                        "submission_status_date": "20251219"},
                       {"submission_type": "ORIG", "submission_status": "AP",
                        "submission_status_date": "20240314"}]}


class Fda:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.calls += 1
        q = req.url.params.get("search", "")
        if "resmetirom" in q or "rezdiffra" in q:
            return httpx.Response(200, json={"meta": {"results": {"total": 1}},
                                             "results": [REZ]})
        return httpx.Response(404, json={"error": {"code": "NOT_FOUND"}})


@pytest.fixture
def store(tmp_path: Path) -> SQLiteStore:
    return SQLiteStore(tmp_path / "a.db")


def _client(fda: Fda, store: SQLiteStore | None, now: datetime = NOW) -> OpenFDAClient:
    return OpenFDAClient(store, transport=httpx.MockTransport(fda), backoff_s=0, now=now)


def test_approved_drug_found_by_generic_or_brand(store: SQLiteStore) -> None:
    fda = Fda()
    a = _client(fda, store).approval("Resmetirom")
    assert a is not None and a.application == "NDA217785"
    assert a.approved_on == date(2024, 3, 14)  # the ORIG approval, not the supplement
    assert _client(fda, store).approval("Rezdiffra") is not None


def test_unknown_drug_is_none_and_codes_are_not_searched(store: SQLiteStore) -> None:
    fda = Fda()
    c = _client(fda, store)
    assert c.approval("azetukalner") is None
    assert c.approval("XEN1101") is None and c.approval("ABC-123") is None
    assert fda.calls == 1  # codes never hit the API


def test_answers_are_cached_for_30_days(store: SQLiteStore) -> None:
    fda = Fda()
    _client(fda, store).approval("Resmetirom")
    _client(fda, store).approval("azetukalner")
    _client(fda, store, NOW + timedelta(days=10)).approval("resmetirom")
    _client(fda, store, NOW + timedelta(days=10)).approval("Azetukalner")
    assert fda.calls == 2
    _client(fda, store, NOW + timedelta(days=31)).approval("resmetirom")
    assert fda.calls == 3


def test_scout_filters_approved_drugs(tmp_path: Path, store: SQLiteStore) -> None:
    from biotradeai.config import load_config
    from biotradeai.watchlist.model import Watchlist
    from tests.unit.test_watchlist import Mock, _scout
    mock = Mock()
    mock.study["protocolSection"]["armsInterventionsModule"]["interventions"] = [
        {"type": "DRUG", "name": "ONC-101 (resmetirom)"}]
    rep = _scout(mock, store, Watchlist(tmp_path / "wl.yaml"), load_config(),
                 fda=_client(Fda(), store)).run()
    assert rep.proposed == [] and rep.filtered.get("already-approved:")


class _Quote:
    def __init__(self, bid: str, ask: str) -> None:
        self.bid, self.ask = Decimal(bid), Decimal(ask)


class OneSidedBroker:
    name = "fake"

    def get_quote(self, ticker: str) -> Any:
        return _Quote("20", "0")  # after hours: no ask


def test_market_cap_uses_one_sided_quote_and_ceiling(tmp_path: Path, store: SQLiteStore,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    from biotradeai.config import load_config
    from biotradeai.watchlist import scout as scout_mod
    from biotradeai.watchlist.model import Watchlist
    from tests.unit.test_watchlist import Mock, _scout
    monkeypatch.setattr(scout_mod, "_shares_outstanding", lambda ed, cik: Decimal(300_000_000))
    wl = Watchlist(tmp_path / "wl.yaml")
    rep = _scout(Mock(), store, wl, load_config(), broker=OneSidedBroker()).run()
    assert rep.proposed == [] and rep.filtered.get("market-cap")  # 300M x $20 = $6B > $5B
    wl2 = Watchlist(tmp_path / "wl2.yaml")
    rep2 = _scout(Mock(), SQLiteStore(tmp_path / "b.db"), wl2, load_config(),
                  broker=OneSidedBroker(), max_market_cap=Decimal(10) ** 10).run()
    assert rep2.proposed == ["ONCO:ONC-101"]
    assert wl2.get("ONCO:ONC-101").card["market"]["market_cap"] == "6000000000"
