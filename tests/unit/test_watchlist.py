"""Watchlist: model + life cycle, agent-managed events, SEC index, scout, follow loop.

Offline: a mock server stands in for SEC and ClinicalTrials.gov (real hostnames, so the
network allowlist is exercised too). Synthetic company "Onco Example Therapeutics".
"""

from __future__ import annotations

import copy
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

import chromadb
import httpx
import pytest

from biotradeai.config import GuardrailsConfig, load_config
from biotradeai.data.clinicaltrials import ClinicalTrialsClient
from biotradeai.data.edgar import EdgarClient
from biotradeai.data.secindex import (
    SecDirectory,
    business_days,
    parse_form_index,
    submission_text,
)
from biotradeai.data.signals import CatalystMention, OutcomeKind, Precision, When
from biotradeai.paths import REPO_ROOT
from biotradeai.rag.embeddings import HashingEmbedder
from biotradeai.rag.index import VectorIndex
from biotradeai.rag.ingest import trial_results_lines
from biotradeai.rag.ontology import Entity, OntologyError, load_ontology, upsert_entity
from biotradeai.rag.sync import DocumentSync
from biotradeai.store.sqlite_store import SQLiteStore
from biotradeai.watchlist.catalysts import (
    agent_catalysts,
    agent_outcomes,
    record_catalyst,
    record_outcome,
    tradable_date,
)
from biotradeai.watchlist.follow import Follower
from biotradeai.watchlist.model import (
    Stage,
    Status,
    WatchEntry,
    Watchlist,
    WatchlistError,
    quarter_end_after,
)
from biotradeai.watchlist.scout import Scout, norm_company

UA = "BioTradeAI tests ops@example.com"
TODAY = date(2026, 11, 20)
CIK = "0009000001"
NOW = datetime(2026, 11, 20, 15, tzinfo=UTC)


@pytest.fixture(scope="module")
def cfg() -> GuardrailsConfig:
    return load_config()


@pytest.fixture
def store(tmp_path: Path) -> SQLiteStore:
    return SQLiteStore(tmp_path / "a.db")


def _entry(**kw: Any) -> WatchEntry:
    base: dict[str, Any] = {"id": "ONCO:ONC-101", "cik": CIK, "ticker": "ONCO",
                            "company": "Onco Example Therapeutics", "entity_id":
                            "ENT-ONCO-ONC101", "drug_names": ["ONC-101", "oncotinib"],
                            "trial_ids": ["NCT09000001"], "proposed_at": NOW}
    return WatchEntry.model_validate({**base, **kw})


def _approved(tmp_path: Path, **kw: Any) -> tuple[Watchlist, WatchEntry]:
    wl = Watchlist(tmp_path / "wl.yaml")
    entry = _entry(**kw)
    wl.add_proposal(entry)
    e = wl.approve(entry.id, by="tester", now=NOW)
    return wl, e


# ------------------------------------------------------------------ model


def test_duplicates_and_rejections_are_suppressed(tmp_path: Path) -> None:
    wl = Watchlist(tmp_path / "wl.yaml")
    assert wl.add_proposal(_entry())
    assert not wl.add_proposal(_entry(id="ONCO:oncotinib", drug_names=["Oncotinib"]))
    wl.reject("ONCO:ONC-101", by="me", reason="too risky")
    assert not wl.add_proposal(_entry())  # rejected stays quiet
    assert wl.add_proposal(_entry(), material_change="new catalyst type: PDUFA")
    e = wl.get("onco:onc-101")
    assert e.status is Status.PROPOSED and e.card["previously"]["status"] == "rejected"
    assert [h["to"] for h in e.history] == ["proposed", "rejected", "proposed"]


def test_transitions_are_enforced_and_saved(tmp_path: Path) -> None:
    wl, e = _approved(tmp_path)
    assert e.checkin_due == (NOW + timedelta(days=365)).date()
    with pytest.raises(WatchlistError):
        wl.reject(e.id, by="me", reason="x")  # approved -> rejected not allowed
    wl.pause(e.id, reason="test")
    wl.approve(e.id, by="me")
    wl.save()
    again = Watchlist(tmp_path / "wl.yaml").get(e.id)
    assert again.status is Status.APPROVED and len(again.history) == 4


@pytest.mark.parametrize(("launch", "review"), [(date(2027, 1, 10), date(2027, 6, 30)),
                                                (date(2027, 3, 31), date(2027, 6, 30)),
                                                (date(2027, 11, 5), date(2028, 3, 31))])
def test_post_launch_review_is_first_full_quarter(launch: date, review: date) -> None:
    assert quarter_end_after(launch) == review


def test_upsert_entity_merges_and_refuses_collisions(tmp_path: Path) -> None:
    p = tmp_path / "syn.yaml"
    p.write_text("# header kept\nentities: {}\n", encoding="utf-8")
    upsert_entity(Entity("ENT-ONCO-ONC101", "ONCO", ("ONC-101",)), synonyms=p)
    upsert_entity(Entity("ENT-ONCO-ONC101", "ONCO", ("oncotinib",)), synonyms=p)
    assert load_ontology(p).entities["ENT-ONCO-ONC101"].names == ("ONC-101", "oncotinib")
    assert p.read_text(encoding="utf-8").startswith("# header kept")
    with pytest.raises(OntologyError):
        upsert_entity(Entity("ENT-XYZ-1", "XYZ", ("ONC 101",)), synonyms=p)


# ------------------------------------------------------------------ catalyst events


def _m(start: date, end: date, prec: Precision, event: str = "PDUFA") -> CatalystMention:
    return CatalystMention(event, When(start, end, prec, "x"), ("ONC-101",), "sentence")


def test_event_life(tmp_path: Path, store: SQLiteStore) -> None:
    _, e = _approved(tmp_path)
    d1, d2 = date(2027, 3, 15), date(2027, 6, 15)
    rec = lambda m, url="u1", filed=TODAY: record_catalyst(  # noqa: E731
        store, e, m, url=url, filed=filed, today=TODAY)
    assert rec(_m(d1, d1, Precision.DAY)).kind == "new"
    assert rec(_m(d1, d1, Precision.DAY)).kind == "same"
    ch = rec(_m(d2, d2, Precision.DAY), filed=TODAY + timedelta(days=1))
    assert ch.kind == "changed" and "2027-03-15" in ch.detail
    ev = store.get_event(e.entity_id, "PDUFA")
    assert ev is not None and ev["date"] == "2027-06-15"
    assert json.loads(ev["previous_dates"])[0]["date"] == "2027-03-15"
    assert rec(_m(d1, d1, Precision.DAY), filed=TODAY).kind == "same"  # older source ignored
    conflict = rec(_m(d1, d1, Precision.DAY), url="u2", filed=TODAY + timedelta(days=1))
    assert conflict.kind == "conflict"
    assert tradable_date(store.get_event(e.entity_id, "PDUFA") or {}, TODAY) is None
    assert rec(_m(d2, d2, Precision.DAY), url="u3",
               filed=TODAY + timedelta(days=2)).kind == "changed"  # newer source settles it
    assert rec(_m(date(2026, 1, 1), date(2026, 1, 1), Precision.DAY)).kind == "stale"
    assert record_outcome(store, e, OutcomeKind.CRL, url="u4", sentence="CRL",
                          filed=TODAY) == "PDUFA"
    assert rec(_m(d2, d2, Precision.DAY), filed=TODAY + timedelta(days=9)).kind == "new"


@pytest.mark.parametrize(("prec", "end", "ok"), [
    (Precision.DAY, date(2027, 3, 15), True), (Precision.QUARTER, date(2027, 3, 31), True),
    (Precision.HALF, date(2027, 6, 30), False), (Precision.YEAR, date(2027, 12, 31), False),
    (Precision.DAY, date(2026, 11, 1), False)])
def test_tradable_dates(tmp_path: Path, store: SQLiteStore, prec: Precision, end: date,
                        ok: bool) -> None:
    _, e = _approved(tmp_path)
    store.put_event({"entity_id": e.entity_id, "event_type": "PHASE3_READOUT", "trial_id": "",
                     "cik": CIK, "ticker": "ONCO", "drug": "ONC-101",
                     "date": end.isoformat() if prec is Precision.DAY else None,
                     "window_start": end.isoformat(), "window_end": end.isoformat(),
                     "precision": prec.value, "status": "upcoming", "outcome": None,
                     "source_url": "u", "source_sentence": "s", "source_filed": "2026-11-01"})
    ev = store.get_event(e.entity_id, "PHASE3_READOUT") or {}
    assert (tradable_date(ev, TODAY) is not None) is ok


def test_agent_catalysts_and_outcomes(tmp_path: Path, store: SQLiteStore) -> None:
    wl, e = _approved(tmp_path)
    d = date(2027, 3, 15)
    record_catalyst(store, e, _m(d, d, Precision.DAY), url="u", filed=TODAY, today=TODAY)
    cats = agent_catalysts(store, wl, TODAY)
    assert [(c.ticker, c.drug, c.event, c.date) for c in cats] == [
        ("ONCO", "ONC-101", "PDUFA", d)]
    wl.pause(e.id, reason="test")
    assert agent_catalysts(store, wl, TODAY) == []  # paused: no new buys
    record_outcome(store, e, OutcomeKind.CRL, url="u2", sentence="s", filed=TODAY)
    assert [o.result.value for o in agent_outcomes(store, wl)] == ["crl"]  # sells still see it


# ------------------------------------------------------------------ SEC index


def _index(rows: list[tuple[str, str, str, str]]) -> str:
    head = ("Description:           Daily Index of EDGAR Dissemination Feed by Form Type\n\n"
            "Form Type   Company Name                                                  CIK"
            "         Date Filed  File Name\n" + "-" * 140 + "\n")
    return head + "".join(f"{f:<17}{c:<62}{int(k):<12}{d:<12}{p}\n" for f, c, k, d, p in rows)


def test_parse_form_index() -> None:
    rows = parse_form_index(_index([
        ("8-K", "ONCO EXAMPLE THERAPEUTICS, INC.", CIK, "20261119",
         "edgar/data/9000001/0009000001-26-000101.txt"),
        ("SC 13G/A", "SOME FUND  LP", "0001234567", "20261119",
         "edgar/data/1234567/0001234567-26-000001.txt")]))
    assert [(r.form, r.cik, r.filed) for r in rows] == [
        ("8-K", CIK, date(2026, 11, 19)), ("SC 13G/A", "0001234567", date(2026, 11, 19))]
    assert rows[0].accession == "0009000001-26-000101"
    assert rows[0].url.endswith("0009000001-26-000101.txt")


def test_business_days_and_submission_text() -> None:
    assert business_days(date(2026, 11, 20), date(2026, 11, 23)) == [
        date(2026, 11, 20), date(2026, 11, 23)]
    raw = "<html><p>Hello</p>\nbegin 644 logo.jpg\nM9876\nend\n<p>World.</p></html>"
    assert submission_text(raw).split() == ["Hello", "World."]


# ------------------------------------------------------------------ mock SEC + CT.gov


EIGHT_K = ("<DOCUMENT><TYPE>EX-99.1<TEXT><html><p>Onco Example Therapeutics announced that "
           "the FDA accepted its NDA for ONC-101. The FDA assigned a PDUFA target action date "
           "of March 15, 2027 for ONC-101. Topline data from the pivotal Phase 3 trial of "
           "ONC-202 are expected in the second quarter of 2027.</p></html></TEXT></DOCUMENT>")


class Mock:
    def __init__(self) -> None:
        self.index: dict[str, str] = {}
        self.archive: dict[str, str] = {}
        self.study = json.loads((REPO_ROOT / "data" / "fixtures" / "ctgov" /
                                 "NCT09000001.json").read_text(encoding="utf-8"))
        self.study["protocolSection"]["statusModule"]["primaryCompletionDateStruct"] = {
            "date": "2027-03", "type": "ESTIMATED"}
        self.facts = json.loads((REPO_ROOT / "data" / "fixtures" / "edgar" / "companyfacts" /
                                 "ONCO.json").read_text(encoding="utf-8"))
        self.paths: list[str] = []

    COMPANIES: ClassVar[dict[str, tuple[str, str, str]]] = {
        CIK: ("Onco Example Therapeutics, Inc.", "2836", "ONCO"),
        "0009000002": ("Viri Example Bio, Inc.", "2836", "VRDX")}

    def submissions(self, cik: str) -> dict[str, Any]:
        name, sic, ticker = self.COMPANIES.get(cik, ("Car Co", "3711", "CARX"))
        return {"cik": cik, "name": name, "sic": sic, "sicDescription": "",
                "tickers": [ticker], "exchanges": ["Nasdaq"],
                "filings": {"recent": {k: [] for k in ("accessionNumber", "form", "filingDate",
                                                        "reportDate", "primaryDocument")}}}

    def __call__(self, req: httpx.Request) -> httpx.Response:
        p = req.url.path
        self.paths.append(f"{req.url.host}{p}")
        if req.url.host == "clinicaltrials.gov":
            if p.endswith("/studies"):
                return httpx.Response(200, json={"studies": [self.study]})
            return httpx.Response(200, json=self.study)
        if p.startswith("/Archives/edgar/daily-index/"):
            day = p.rsplit(".", 2)[-2]
            return (httpx.Response(200, text=self.index[day]) if day in self.index
                    else httpx.Response(404))
        if p.startswith("/Archives/edgar/data/"):
            return httpx.Response(200, text=self.archive.get(p, "<p>nothing.</p>"))
        if p.startswith("/submissions/CIK"):
            return httpx.Response(200, json=self.submissions(p[16:26]))
        if p.startswith("/api/xbrl/companyfacts/"):
            return httpx.Response(200, json=self.facts)
        if p == "/files/company_tickers_exchange.json":
            return httpx.Response(200, json={"fields": ["cik", "name", "ticker", "exchange"],
                                             "data": [[9000001, "Onco Example Therapeutics, "
                                                       "Inc.", "ONCO", "Nasdaq"],
                                                      [9000002, "Viri Example Bio, Inc.",
                                                       "VRDX", "Nasdaq"]]})
        return httpx.Response(404)


def _clients(mock: Mock) -> tuple[EdgarClient, ClinicalTrialsClient]:
    t = httpx.MockTransport(mock)
    return EdgarClient(UA, transport=t, backoff_s=0), ClinicalTrialsClient(transport=t,
                                                                           backoff_s=0)


def _scout(mock: Mock, store: SQLiteStore, wl: Watchlist, cfg: GuardrailsConfig,
           **kw: Any) -> Scout:
    ed, ct = _clients(mock)
    return Scout(edgar=ed, ctgov=ct, store=store, watchlist=wl, cfg=cfg, today=TODAY,
                 sleep=lambda s: None, **kw)


def _with_8k(mock: Mock) -> None:
    mock.index["20261119"] = _index([
        ("8-K", "ONCO EXAMPLE THERAPEUTICS, INC.", CIK, "20261119",
         "edgar/data/9000001/0009000001-26-000101.txt"),
        ("8-K", "CAR CO", "0001234567", "20261119",
         "edgar/data/1234567/0001234567-26-000001.txt")])
    mock.archive["/Archives/edgar/data/9000001/0009000001-26-000101.txt"] = EIGHT_K


def test_company_directory_caches(store: SQLiteStore) -> None:
    mock = Mock()
    ed, _ = _clients(mock)
    d = SecDirectory(ed, store, now=NOW)
    assert d.company(CIK).is_biotech and d.company(CIK).is_listed
    assert sum("/submissions/" in x for x in mock.paths) == 1  # cached after the first call


def test_scout_proposes_with_card_and_never_duplicates(tmp_path: Path, store: SQLiteStore,
                                                       cfg: GuardrailsConfig) -> None:
    mock = Mock()
    _with_8k(mock)
    wl = Watchlist(tmp_path / "wl.yaml")
    rep = _scout(mock, store, wl, cfg).run()
    assert "ONCO:ONC-101" in rep.proposed
    e = wl.get("ONCO:ONC-101")
    c = e.card
    assert c["sources"] == ["SEC 8-K", "ClinicalTrials.gov"]  # both sources agree
    assert c["runway"]["months"] == 12.0
    assert c["catalysts"][0]["event"] == "PDUFA" and c["catalysts"][0]["precision"] == "day"
    assert "March 15, 2027" in c["catalysts"][0]["sentence"]
    assert c["drug"]["trials"][0]["nct"] == "NCT09000001"
    assert "exact name" in c["match"]
    # the non-biotech filer was never downloaded
    assert not any("1234567/0001234567" in x for x in mock.paths)
    # second run: same signals, nothing new is proposed
    store.kv_set("scout.last_index_day", "2026-11-18")
    rep2 = _scout(mock, store, Watchlist(tmp_path / "wl.yaml"), cfg).run()
    assert "ONCO:ONC-101" not in rep2.proposed
    assert len(Watchlist(tmp_path / "wl.yaml").entries) == len(wl.entries)


def test_scout_respects_soft_cap(tmp_path: Path, store: SQLiteStore,
                                 cfg: GuardrailsConfig) -> None:
    mock = Mock()
    _with_8k(mock)
    wl, _ = _approved(tmp_path, id="X:1", drug_names=["XX-1"], entity_id="ENT-X-1")
    rep = _scout(mock, store, wl, cfg, soft_cap=1).run()
    assert rep.cap_reached and rep.proposed == []


def test_scout_filters_short_runway(tmp_path: Path, store: SQLiteStore,
                                    cfg: GuardrailsConfig) -> None:
    mock = Mock()
    _with_8k(mock)
    mock.facts = json.loads((REPO_ROOT / "data" / "fixtures" / "edgar" / "companyfacts" /
                             "DILU.json").read_text(encoding="utf-8"))
    rep = _scout(mock, store, Watchlist(tmp_path / "wl.yaml"), cfg).run()
    assert rep.proposed == [] and rep.filtered.get("runway")


def test_norm_company() -> None:
    assert norm_company("Onco Example Therapeutics, Inc.") == "onco example therapeutics"
    assert norm_company("BigPharma Holdings PLC") == "bigpharma"


# ------------------------------------------------------------------ follow loop


def _follower(mock: Mock, store: SQLiteStore, wl: Watchlist, cfg: GuardrailsConfig,
              today: date = TODAY) -> Follower:
    ed, ct = _clients(mock)
    emb = HashingEmbedder()
    sync = DocumentSync(store=store, cfg=cfg, index=VectorIndex(
        "t", client=chromadb.EphemeralClient()), embedder=emb, edgar=ed, ctgov=ct,
        pubmed=None, sleep=lambda s: None)
    return Follower(directory=SecDirectory(ed, store, now=NOW), ctgov=ct, store=store,
                    watchlist=wl, sync=sync, today=today, sleep=lambda s: None)


def _filing(mock: Mock, form: str, text: str, day: str = "20261120") -> None:
    path = f"edgar/data/9000001/0009000001-26-{len(mock.archive) + 200:06d}.txt"
    mock.index.setdefault(day, _index([]))
    mock.index[day] += f"{form:<17}{'ONCO EXAMPLE':<62}{9000001:<12}{day:<12}{path}\n"
    mock.archive[f"/Archives/{path}"] = text


def test_follow_records_events_outcomes_and_launch(tmp_path: Path, store: SQLiteStore,
                                                  cfg: GuardrailsConfig) -> None:
    mock = Mock()
    wl, e = _approved(tmp_path)
    _filing(mock, "8-K", "<p>The FDA assigned a PDUFA target action date of April 2, 2027 "
                         "for ONC-101.</p>")
    rep = _follower(mock, store, wl, cfg).run()
    assert any("PDUFA new" in x for x in rep.events)
    assert rep.docs.get("sec-8k") == 1 and rep.docs.get("ctgov") == 1
    assert agent_catalysts(store, wl, TODAY)[0].date == date(2027, 4, 2)
    # approval, then launch, in later filings
    _filing(mock, "8-K", "<p>The FDA approved ONC-101 (Oncora) for adults.</p>", "20261123")
    _filing(mock, "8-K", "<p>ONC-101 is now commercially available in the United States."
                         "</p>", "20261124")
    rep = _follower(mock, store, wl, cfg, today=date(2026, 11, 24)).run()
    e = wl.get(e.id)
    assert e.stage is Stage.LAUNCHED and e.launched_on == date(2026, 11, 24)
    assert e.review_after == date(2027, 3, 31)
    assert agent_catalysts(store, wl, date(2026, 11, 24)) == []  # PDUFA occurred
    # 10-Q filed after the first full quarter post-launch -> pause for re-approval
    _filing(mock, "10-Q", "<p>quarterly report</p>", "20270512")
    rep = _follower(mock, store, wl, cfg, today=date(2027, 5, 12)).run()
    assert wl.get(e.id).status is Status.PAUSED
    assert "first full quarter after launch" in (wl.get(e.id).pause_reason or "")


@pytest.mark.parametrize(("form", "text", "status", "why"), [
    ("8-K", "<p>Onco entered into a definitive merger agreement with BigPharma.</p>",
     Status.PAUSED, "acquisition"),
    ("8-K", "<p>Onco signed an exclusive license agreement for ONC-101 with PartnerCo.</p>",
     Status.PAUSED, "license"),
    ("8-K", "<p>The FDA placed ONC-101 on a full clinical hold.</p>", Status.PAUSED,
     "clinical hold"),
    ("25-NSE", "<p>notice</p>", Status.PAUSED, "delisting"),
    ("8-K", "<p>The Company discontinued development of ONC-101.</p>", Status.RETIRED,
     "program ended"),
])
def test_follow_material_changes(tmp_path: Path, store: SQLiteStore, cfg: GuardrailsConfig,
                                 form: str, text: str, status: Status, why: str) -> None:
    mock = Mock()
    wl, e = _approved(tmp_path)
    _filing(mock, form, text)
    _follower(mock, store, wl, cfg).run()
    got = wl.get(e.id)
    assert got.status is status
    assert why in (got.pause_reason or got.retire_reason or "")


def test_follow_pauses_on_stopped_trial_and_checkin(tmp_path: Path, store: SQLiteStore,
                                                    cfg: GuardrailsConfig) -> None:
    mock = Mock()
    mock.study["protocolSection"]["statusModule"]["overallStatus"] = "TERMINATED"
    wl, e = _approved(tmp_path)
    _follower(mock, store, wl, cfg).run()
    assert "terminated" in (wl.get(e.id).pause_reason or "")
    mock2 = Mock()
    wl2, e2 = _approved(tmp_path / "b")
    _follower(mock2, store, wl2, cfg, today=TODAY + timedelta(days=366)).run()
    assert "yearly check-in" in (wl2.get(e2.id).pause_reason or "")


def test_follow_reads_only_followed_companies(tmp_path: Path, store: SQLiteStore,
                                              cfg: GuardrailsConfig) -> None:
    mock = Mock()
    _with_8k(mock)  # 2026-11-19: one followed company, one other
    wl, _ = _approved(tmp_path)
    store.kv_set("follow.last_index_day", "2026-11-18")
    _follower(mock, store, wl, cfg).run()
    assert not any("1234567/0001234567" in x for x in mock.paths)


def test_trial_results_lines() -> None:
    raw = {"protocolSection": {"outcomesModule": {"primaryOutcomes": [
        {"measure": "Overall response rate", "timeFrame": "24 weeks"}]}},
        "resultsSection": {"outcomeMeasuresModule": {"outcomeMeasures": [
            {"type": "PRIMARY", "title": "ORR", "timeFrame": "24 weeks",
             "analyses": [{"pValue": "0.001", "statisticalMethod": "CMH"}]},
            {"type": "SECONDARY", "title": "OS"}]}}, "hasResults": True}
    lines = trial_results_lines(copy.deepcopy(raw))
    assert lines[0] == "Primary endpoint: Overall response rate (time frame: 24 weeks)."
    assert lines[1] == "Posted primary outcome result: ORR (time frame: 24 weeks); p=0.001 (CMH)."
    assert len(lines) == 2


def test_scout_filters_profitable_companies(tmp_path: Path, store: SQLiteStore,
                                            cfg: GuardrailsConfig) -> None:
    mock = Mock()
    _with_8k(mock)
    mock.facts = json.loads((REPO_ROOT / "data" / "fixtures" / "edgar" / "companyfacts" /
                             "PROF.json").read_text(encoding="utf-8"))
    rep = _scout(mock, store, Watchlist(tmp_path / "wl.yaml"), cfg).run()
    assert rep.proposed == [] and rep.filtered.get("profitable:")


def test_ctgov_names_exclude_comparators_and_clean_aliases(
        tmp_path: Path, store: SQLiteStore, cfg: GuardrailsConfig) -> None:
    mock = Mock()  # CT.gov signal only
    mock.study["protocolSection"]["armsInterventionsModule"]["interventions"] = [
        {"type": "DRUG", "name": "ONC-101 (oncotinib) 100 mg"},
        {"type": "DRUG", "name": "Pembrolizumab"}, {"type": "DRUG", "name": "Placebo"}]
    wl = Watchlist(tmp_path / "wl.yaml")
    rep = _scout(mock, store, wl, cfg).run()
    assert rep.proposed == ["ONCO:ONC-101"]
    e = wl.get("ONCO:ONC-101")
    assert e.drug_names == ["ONC-101", "oncotinib"]  # no comparator, no dose
    assert e.card["drug"]["trials"][0]["with"] == ["Pembrolizumab"]
    assert e.card["fit"]["has_tradable_date"] is False  # CT.gov dates never tradable


def test_clean_names_and_tidy_ids() -> None:
    from biotradeai.watchlist.scout import clean_names
    assert clean_names(["ZL-1109 (VRDN-003)"]) == ["ZL-1109", "VRDN-003"]
    assert clean_names(["Sacituzumab govitecan-hziy (SG)"]) == ["Sacituzumab govitecan-hziy"]
    assert clean_names(["Ritlecitinib 100 mg", "ritlecitinib"]) == ["Ritlecitinib"]


def test_numbers_are_not_drug_names_and_missing_runway_is_filtered(
        tmp_path: Path, store: SQLiteStore, cfg: GuardrailsConfig) -> None:
    from biotradeai.watchlist.scout import clean_names
    assert clean_names(["PCV21 vaccine", "515"]) == ["PCV21 vaccine"]
    mock = Mock()
    mock.facts = {"cik": 9000001, "entityName": "Foreign Co", "facts": {}}
    rep = _scout(mock, store, Watchlist(tmp_path / "wl.yaml"), cfg).run()
    assert rep.proposed == [] and rep.filtered.get("runway-unavailable")


def test_licensed_drug_flags_licensee_and_proposes_originator(
        tmp_path: Path, store: SQLiteStore, cfg: GuardrailsConfig) -> None:
    mock = Mock()  # Onco sponsors the trial of "ON-1109 (VRDX-003)": VRDX is another ticker
    mock.study["protocolSection"]["armsInterventionsModule"]["interventions"] = [
        {"type": "DRUG", "name": "ON-1109 (VRDX-003)"}, {"type": "DRUG", "name": "Placebo"}]
    wl = Watchlist(tmp_path / "wl.yaml")
    rep = _scout(mock, store, wl, cfg).run()
    assert set(rep.proposed) == {"ONCO:ON-1109", "VRDX:VRDX-003"}
    lic = wl.get("ONCO:ON-1109").card["licensing"]
    assert lic["role"] == "likely licensee" and lic["ticker"] == "VRDX"
    assert any("likely a licensee" in f for f in wl.get("ONCO:ON-1109").card["red_flags"])
    orig = wl.get("VRDX:VRDX-003")
    assert orig.card["licensing"]["role"] == "likely originator"
    assert orig.drug_names[0] == "VRDX-003" and "originator by drug code" in orig.card["match"]
    assert orig.card["drug"]["trials"][0]["nct"] == "NCT09000001"


def test_own_ticker_code_is_not_licensing(tmp_path: Path, store: SQLiteStore,
                                          cfg: GuardrailsConfig) -> None:
    mock = Mock()
    mock.study["protocolSection"]["armsInterventionsModule"]["interventions"] = [
        {"type": "DRUG", "name": "oncotinib (ONCO-101)"}]
    wl = Watchlist(tmp_path / "wl.yaml")
    rep = _scout(mock, store, wl, cfg).run()
    assert rep.proposed == ["ONCO:ONCO-101"]  # own code preferred as the ID
    assert wl.get("ONCO:ONCO-101").card["licensing"] is None


# ------------------------------------------------------------------ review regressions


def test_today_without_index_is_retried(tmp_path: Path, store: SQLiteStore,
                                        cfg: GuardrailsConfig) -> None:
    mock = Mock()
    wl, _ = _approved(tmp_path)
    store.kv_set("follow.last_index_day", "2026-11-19")
    store.kv_set("sync.initial.ENT-ONCO-ONC101", "done")
    _follower(mock, store, wl, cfg).run()  # 2026-11-20 index not published yet
    assert store.kv_get("follow.last_index_day") == "2026-11-19"
    _filing(mock, "8-K", "<p>The FDA assigned a PDUFA target action date of April 2, 2027 "
                         "for ONC-101.</p>")  # published later that day
    rep = _follower(mock, store, wl, cfg).run()
    assert any("PDUFA new" in x for x in rep.events)
    assert store.kv_get("follow.last_index_day") == "2026-11-20"


def test_unnamed_mention_not_credited_when_filing_names_other_drug(
        tmp_path: Path, store: SQLiteStore, cfg: GuardrailsConfig) -> None:
    mock = Mock()
    wl, e = _approved(tmp_path)
    _filing(mock, "8-K", "<p>ONC-300 for solid tumors had its NDA accepted. Our partner "
                         "confirmed timelines. The FDA set a PDUFA target action date of May 1, "
                         "2027.</p>")
    _follower(mock, store, wl, cfg).run()
    assert store.list_events(entity_id=e.entity_id) == []


def test_sync_error_is_recorded_and_retried(tmp_path: Path, store: SQLiteStore,
                                           cfg: GuardrailsConfig) -> None:
    class DownSec(Mock):
        def __call__(self, req: httpx.Request) -> httpx.Response:
            if req.url.path.startswith("/submissions/"):
                return httpx.Response(503)
            return super().__call__(req)

    wl, e = _approved(tmp_path)
    rep = _follower(DownSec(), store, wl, cfg).run()  # must not raise
    assert any("SEC filing list" in x for x in rep.errors)
    assert store.kv_get(f"sync.initial.{e.entity_id}") is None  # retried next run


def test_entity_ids_always_valid_and_upsert_validates(tmp_path: Path) -> None:
    from biotradeai.rag.ontology import ENTITY_ID
    from biotradeai.watchlist.model import entity_id_for
    for tk, drug in [("BRK.B", "x-1"), ("ABCDEFGHIJ", "Sacituzumab govitecan-hziy extra long"),
                     ("ONCO", "ONC-101")]:
        assert ENTITY_ID.match(entity_id_for(tk, drug)), (tk, drug)
    p = tmp_path / "syn.yaml"
    with pytest.raises(OntologyError):
        upsert_entity(Entity("ENT-BRK.B-X1", "BRK.B", ("X-1",)), synonyms=p)
    assert not p.exists()  # nothing written


def test_originator_does_not_inherit_licensee_catalysts(
        tmp_path: Path, store: SQLiteStore, cfg: GuardrailsConfig) -> None:
    mock = Mock()
    _with_8k(mock)  # Onco 8-K: PDUFA for ONC-101
    mock.study["protocolSection"]["armsInterventionsModule"]["interventions"] = [
        {"type": "DRUG", "name": "ONC-101 (VRDX-003)"}]
    wl = Watchlist(tmp_path / "wl.yaml")
    _scout(mock, store, wl, cfg).run()
    orig = wl.get("VRDX:VRDX-003")
    assert orig.card["catalysts"] == [] and orig.card["sources"] == ["ClinicalTrials.gov"]
    assert orig.card["licensing"]["licensee_catalysts"][0]["event"] == "PDUFA"


def test_rejected_pair_can_be_approved_by_human(tmp_path: Path) -> None:
    wl = Watchlist(tmp_path / "wl.yaml")
    wl.add_proposal(_entry())
    wl.reject("ONCO:ONC-101", by="me", reason="oops")
    assert wl.approve("ONCO:ONC-101", by="me").status is Status.APPROVED


def test_clean_names_drops_doses_and_formulations() -> None:
    from biotradeai.watchlist.scout import clean_names
    assert clean_names(["4D-150 IVT", "3E10 vg/eye"]) == ["4D-150"]
    assert clean_names(["zodasiran Injection"]) == ["zodasiran"]
    assert clean_names(["PCV21 vaccine", "515"]) == ["PCV21 vaccine"]
