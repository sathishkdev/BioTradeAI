"""Operator app: jobs and schedules, agent host, approve-later trades, settings, web UI."""

from __future__ import annotations

import ast
import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from biotradeai.agents.base_rates import BaseRates, load_base_rates
from biotradeai.brokers.approval import ApprovalAuthority
from biotradeai.brokers.fake_broker import FakeBroker
from biotradeai.config import GuardrailsConfig, load_config
from biotradeai.data.catalysts import Catalyst
from biotradeai.operator import jobs, settings
from biotradeai.operator.api import COOKIE, _csrf, create_app
from biotradeai.operator.db import OperatorDB
from biotradeai.operator.host import AgentHost
from biotradeai.operator.trades import TradeApprovals, TradeError, make_defer_sink
from biotradeai.paper.runner import DEFER, PaperRunner
from biotradeai.paths import REPO_ROOT
from biotradeai.scenario_fixtures import ScenarioFixture, load_fixture
from biotradeai.schemas import CatalystState, ResearchInputs
from biotradeai.store.sqlite_store import SQLiteStore
from biotradeai.watchlist.model import WatchEntry, Watchlist

FIX = REPO_ROOT / "data" / "fixtures" / "scenarios" / "onco.yaml"
SECRET = b"k" * 32
CAT = Catalyst("ONCO", "ONC-101", "PHASE3_READOUT", date(2027, 1, 15), "https://example.com/c")
LOCAL = datetime(2026, 11, 23, 6, 30).astimezone()  # a Monday


@pytest.fixture(scope="module")
def cfg() -> GuardrailsConfig:
    return load_config()


@pytest.fixture(scope="module")
def rates() -> BaseRates:
    return load_base_rates()


@pytest.fixture
def db(tmp_path: Path) -> OperatorDB:
    return OperatorDB(SQLiteStore(tmp_path / "op.db"))


# ------------------------------------------------------------------ jobs and schedules


def test_argv_is_built_only_from_validated_settings() -> None:
    scout = jobs.job("scout")
    assert scout.argv({"spread_minutes": "30", "max": 5, "with_quotes": False}) == [
        "watchlist", "scout", "--spread-minutes", "30", "--max", "5",
        "--max-market-cap-billions", "5.0"]
    assert jobs.job("paper").argv({"broker": "fake", "approvals": "queue"}) == [
        "paper", "run", "--broker", "fake", "--approve-later"]
    with pytest.raises(jobs.JobError):
        scout.argv({"spread_minutes": "999"})
    with pytest.raises(jobs.JobError):
        jobs.job("paper").argv({"broker": "robinhood; rm -rf /"})
    with pytest.raises(jobs.JobError):
        jobs.job("nope")


def test_schedule_fires_once_in_its_window(db: OperatorDB) -> None:
    jobs.save_schedule(db, "follow", enabled=True, days=[0, 1, 2, 3, 4], at_time="06:00",
                       params={"spread_minutes": 10}, by="t")
    s = jobs.schedule_for(db, "follow")
    slot = jobs.due_fire(s, LOCAL)
    assert slot is not None and slot.hour == 6
    db.mark_fired("follow", slot)
    s = jobs.schedule_for(db, "follow")
    assert jobs.due_fire(s, LOCAL) is None  # already fired today
    assert jobs.due_fire(s, LOCAL + timedelta(hours=3)) is None  # past the grace window
    nf = jobs.next_fire(s, LOCAL)
    assert nf is not None and nf.date() == date(2026, 11, 24) and nf.hour == 6
    sat = jobs.schedule_for(db, "follow")
    assert jobs.due_fire(sat, LOCAL + timedelta(days=5)) is None  # Saturday: not scheduled
    events = [e.event_type for e in db.store.read_audit()]
    assert "operator.schedule_changed" in events


def test_schedule_rejects_bad_input(db: OperatorDB) -> None:
    with pytest.raises(jobs.JobError):
        jobs.save_schedule(db, "follow", enabled=True, days=[], at_time="06:00", params={},
                           by="t")
    with pytest.raises(jobs.JobError):
        jobs.save_schedule(db, "follow", enabled=True, days=[0], at_time="25:00", params={},
                           by="t")


# ------------------------------------------------------------------ agent host


class FakeProc:
    _next = 1000

    def __init__(self, code: int | None = None) -> None:
        FakeProc._next += 1
        self.pid = FakeProc._next
        self.code = code
        self.terminated = False

    def poll(self) -> int | None:
        return self.code

    def terminate(self) -> None:
        self.terminated = True
        self.code = -15

    def kill(self) -> None:
        self.code = -9


def test_host_fires_schedule_runs_and_records_result(db: OperatorDB, tmp_path: Path) -> None:
    started: list[list[str]] = []
    procs: list[FakeProc] = []

    def spawner(argv: list[str], log: Path) -> FakeProc:
        started.append(argv)
        log.write_text("line one\nboom: something failed\n", encoding="utf-8")
        p = FakeProc()
        procs.append(p)
        return p

    jobs.save_schedule(db, "follow", enabled=True, days=[0], at_time="06:00",
                       params={"spread_minutes": 0}, by="t")
    host = AgentHost(db, spawner=spawner, now_local=lambda: LOCAL, log_root=tmp_path / "logs")
    host.claim()
    host.tick()
    assert started == [["watchlist", "follow", "--spread-minutes", "0"]]
    run = db.runs(job="follow")[0]
    assert run.status == "running" and run.trigger == "schedule"
    host.tick()  # still running, schedule already fired: nothing new starts
    assert len(started) == 1
    procs[0].code = 2
    host.tick()
    done = db.get_run(run.id)
    assert done is not None and done.status == "failed" and done.exit_code == 2
    assert "boom" in (done.error or "")
    events = [e.event_type for e in db.store.read_audit()]
    assert {"operator.host_started", "operator.run_started", "operator.run_finished"} <= set(
        events)


def test_one_run_per_job_and_cancel(db: OperatorDB, tmp_path: Path) -> None:
    procs: list[FakeProc] = []

    def spawner(argv: list[str], log: Path) -> FakeProc:
        procs.append(FakeProc())
        return procs[-1]

    host = AgentHost(db, spawner=spawner, now_local=lambda: LOCAL, log_root=tmp_path)
    host.claim()
    rid = jobs.run_now(db, "scout", by="t")
    with pytest.raises(jobs.JobError):
        jobs.run_now(db, "scout", by="t")  # already queued
    host.tick()
    assert db.get_run(rid).status == "running"  # type: ignore[union-attr]
    assert jobs.cancel(db, rid, by="t")
    host.tick()
    r = db.get_run(rid)
    assert r is not None and r.status == "cancelled" and procs[0].terminated
    q = jobs.run_now(db, "follow", by="t")
    assert jobs.cancel(db, q, by="t")
    host.tick()
    assert db.get_run(q).status == "cancelled"  # type: ignore[union-attr]


def test_host_restart_fails_orphaned_runs(db: OperatorDB) -> None:
    rid = db.enqueue("scout", trigger="manual", by="t", params={}, argv=["watchlist", "scout"])
    db.mark_started(rid, pid=999_999, log_path="x.log")
    AgentHost(db, spawner=lambda a, p: FakeProc()).claim()
    r = db.get_run(rid)
    assert r is not None and r.status == "failed" and "restarted" in (r.error or "")


# ------------------------------------------------------------------ approve-later trades


class StubGatherer:
    def __init__(self, fx: ScenarioFixture) -> None:
        self.fx = fx

    def gather(self, c: Catalyst, *, now: datetime,
               run_id: str) -> tuple[CatalystState, ResearchInputs]:
        return self.fx.catalyst.model_copy(update={"run_id": run_id}), self.fx.inputs


class Rig:
    def __init__(self, cfg: GuardrailsConfig, rates: BaseRates, db: OperatorDB) -> None:
        self.fx = load_fixture(FIX, cfg)
        self.broker = FakeBroker(quotes={"ONCO": (Decimal("9.98"), Decimal("10.02"))})
        self.saver = InMemorySaver()
        self.auth = ApprovalAuthority(SECRET, ttl=timedelta(minutes=15))
        self.db, self.cfg = db, cfg
        llm = self.fx.fake_llm()
        self.runner = PaperRunner(
            cfg=cfg, store=db.store, broker=self.broker, gatherer=StubGatherer(self.fx),
            rates=rates, generator=llm, critic=llm, authority=self.auth,
            approver=lambda o, p: DEFER, checkpointer=self.saver,
            on_defer=make_defer_sink(db, broker="fake", cfg=cfg), has_pending=db.has_pending,
            clock=lambda: self.fx.catalyst.as_of)
        self.trades = TradeApprovals(db, cfg=cfg, broker_factory=lambda name: self.broker,
                                     authority=self.auth, checkpointer=self.saver)


def test_deferred_entry_waits_then_ui_approval_places_it(
        cfg: GuardrailsConfig, rates: BaseRates, db: OperatorDB) -> None:
    rig = Rig(cfg, rates, db)
    rep = rig.runner.cycle([CAT], [])
    assert rep.orders[0]["status"] == "pending_approval" and rig.broker.orders == {}
    [p] = db.pending()
    assert p.kind == "entry" and p.ticker == "ONCO" and p.broker == "fake"
    assert db.has_pending("ONCO")
    with pytest.raises(TradeError, match="type 'ONCO'"):
        rig.trades.approve(p.id, typed="yes", operator="t")
    res = rig.trades.approve(p.id, typed="onco", operator="t")
    assert res["status"] == "placed" and len(rig.broker.orders) == 1
    assert db.get_pending(p.id).status == "approved"  # type: ignore[union-attr]
    assert db.store.load_sell_state("ONCO") is not None  # exit rules now manage it
    events = [e.event_type for e in db.store.read_audit()]
    assert "approval.deferred" in events and "approval.granted" in events
    assert "execution.placed" in events
    with pytest.raises(TradeError, match="already approved"):
        rig.trades.approve(p.id, typed="ONCO", operator="t")
    assert db.store.verify_audit_chain()


def test_pending_ticker_is_not_re_evaluated(cfg: GuardrailsConfig, rates: BaseRates,
                                            db: OperatorDB) -> None:
    rig = Rig(cfg, rates, db)
    rig.runner.cycle([CAT], [])
    rig.runner._evaluated_today = lambda t, d: False  # type: ignore[method-assign]
    rep = rig.runner.cycle([CAT], [])
    assert ("ONCO", "an order is waiting for your approval") in rep.skipped


def test_price_drift_blocks_and_reject_closes_the_run(
        cfg: GuardrailsConfig, rates: BaseRates, db: OperatorDB) -> None:
    rig = Rig(cfg, rates, db)
    rig.runner.cycle([CAT], [])
    [p] = db.pending()
    rig.broker.quotes["ONCO"] = (Decimal("8.50"), Decimal("8.60"))  # -15%
    with pytest.raises(TradeError, match="price moved"):
        rig.trades.approve(p.id, typed="ONCO", operator="t")
    assert db.get_pending(p.id).status == "pending"  # type: ignore[union-attr]
    with pytest.raises(TradeError, match="reason"):
        rig.trades.reject(p.id, reason=" ", operator="t")
    rig.trades.reject(p.id, reason="price ran away", operator="t")
    assert db.get_pending(p.id).status == "rejected"  # type: ignore[union-attr]
    responses = [e.payload for e in db.store.read_audit() if e.event_type == "hitl.response"]
    assert responses and responses[-1]["approved"] is False
    assert rig.broker.orders == {}


def test_expired_and_killswitch_block_approval(cfg: GuardrailsConfig, rates: BaseRates,
                                               db: OperatorDB) -> None:
    rig = Rig(cfg, rates, db)
    rig.runner.cycle([CAT], [])
    [p] = db.pending()
    db.store.set_killswitch(True, "test", changed_by="t")
    with pytest.raises(TradeError, match="kill-switch"):
        rig.trades.approve(p.id, typed="ONCO", operator="t")
    db.store.set_killswitch(False, None, changed_by="t")
    pid = db.add_pending(kind="exit", run_id="r", ticker="ONCO", order=p.order, pack={},
                         broker="fake", mode="paper",
                         expires_at=datetime.now(UTC) - timedelta(minutes=1))
    with pytest.raises(TradeError, match="expired"):
        rig.trades.approve(pid, typed="ONCO", operator="t")
    assert db.get_pending(pid).status == "expired"  # type: ignore[union-attr]


# ------------------------------------------------------------------ settings


def _cfg_copy(tmp_path: Path) -> Path:
    p = tmp_path / "guardrails.yaml"
    p.write_text((REPO_ROOT / "config" / "guardrails.yaml").read_text(encoding="utf-8"),
                 encoding="utf-8")
    return p


def test_guardrail_edit_keeps_comments_and_validates(tmp_path: Path, db: OperatorDB) -> None:
    p = _cfg_copy(tmp_path)
    f = settings.update_guardrail("sell.trailing_stop_pct", "0.12", by="t", store=db.store,
                                  file=p)
    assert f.value == 0.12
    text = p.read_text(encoding="utf-8")
    assert "trailing_stop_pct: 0.12" in text and "# PLACEHOLDER" in text
    assert load_config(p).sell.trailing_stop_pct == 0.12
    with pytest.raises(settings.SettingsError):
        settings.update_guardrail("max_position_pct", "0.10", by="t", store=db.store, file=p)
    with pytest.raises(settings.SettingsError):
        settings.update_guardrail("mode", "live", by="t", store=db.store, file=p)
    with pytest.raises(settings.SettingsError):  # fatal runway must stay below min runway
        settings.update_guardrail("fatal_runway_months", "9", by="t", store=db.store, file=p)
    assert load_config(p).max_position_pct == 0.05  # unchanged after refusals
    changes = [e.payload for e in db.store.read_audit() if e.event_type == "config.changed"]
    assert changes == [{"setting": "sell.trailing_stop_pct", "before": 0.10, "after": 0.12,
                        "by": "t"}]


def test_secret_replace_is_write_only(tmp_path: Path, db: OperatorDB,
                                      monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text("# keys\nALPACA_API_KEY=old\nOTHER=1\n", encoding="utf-8")
    monkeypatch.setenv("ALPACA_API_KEY", "old")
    settings.replace_secret("ALPACA_API_KEY", "new-value-123", by="t", store=db.store,
                            file=env)
    assert env.read_text(encoding="utf-8") == "# keys\nALPACA_API_KEY=new-value-123\nOTHER=1\n"
    log = json.dumps([e.payload for e in db.store.read_audit()])
    assert "new-value-123" not in log and "ALPACA_API_KEY" in log
    with pytest.raises(settings.SettingsError):
        settings.replace_secret("PATH", "x", by="t", store=db.store, file=env)
    with pytest.raises(settings.SettingsError):
        settings.replace_secret("APPROVAL_HMAC_SECRET", "short", by="t", store=db.store,
                                file=env)


def test_base_rates_review_toggle(tmp_path: Path, db: OperatorDB) -> None:
    p = tmp_path / "br.yaml"
    p.write_text((REPO_ROOT / "data" / "base_rates.yaml").read_text(encoding="utf-8"),
                 encoding="utf-8")
    settings.set_base_rates_reviewed(True, by="t", store=db.store, file=p)
    assert load_base_rates(p).reviewed is True


# ------------------------------------------------------------------ web UI


@pytest.fixture
def client(tmp_path: Path, db: OperatorDB) -> TestClient:
    wl = Watchlist(tmp_path / "wl.yaml")
    wl.entries.append(WatchEntry(id="ONCO:ONC-101", cik="0009000001", ticker="ONCO",
                                 company="Onco Example", entity_id="ENT-ONCO-ONC101",
                                 drug_names=["ONC-101"], proposed_at=datetime.now(UTC)))
    wl.save()
    c = TestClient(create_app("tok", db=db, watchlist_path=tmp_path / "wl.yaml"))
    return c


def test_ui_requires_session_and_form_token(client: TestClient) -> None:
    assert client.get("/").status_code == 401
    r = client.get("/auth?t=wrong", follow_redirects=False)
    assert r.status_code == 403
    r = client.get("/auth?t=tok", follow_redirects=False)
    assert r.status_code == 303 and COOKIE in r.headers.get("set-cookie", "")
    client.cookies.set(COOKIE, "tok")
    assert client.post("/agents/scout/run", data={}).status_code == 403  # no csrf
    r = client.post("/agents/scout/run", data={"csrf": _csrf("tok")},
                    headers={"origin": "https://evil.example"})
    assert r.status_code == 403
    assert "default-src 'self'" in client.get("/").headers["content-security-policy"]


def test_ui_pages_render_and_actions_work(client: TestClient, db: OperatorDB) -> None:
    client.cookies.set(COOKIE, "tok")
    for path in ("/", "/agents", "/agents/paper", "/trades", "/watchlist",
                 "/watchlist/ONCO:ONC-101", "/positions", "/catalysts", "/activity",
                 "/activity/run/none", "/safety", "/settings", "/api/status"):
        assert client.get(path).status_code == 200, path
    tok = {"csrf": _csrf("tok")}
    r = client.post("/agents/follow/run", data=tok, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/runs/")
    assert db.runs(job="follow")[0].status == "queued"
    assert client.get(r.headers["location"].split("?")[0]).status_code == 200
    r = client.post("/agents/follow/schedule", data={**tok, "enabled": "on", "days": ["0", "2"],
                                                    "at_time": "07:15", "spread_minutes": "5"},
                    follow_redirects=False)
    s = jobs.schedule_for(db, "follow")
    assert s.enabled and s.days == (0, 2) and s.at_time == "07:15"
    r = client.post("/watchlist/ONCO:ONC-101/reject", data={**tok, "note": "too early"},
                    follow_redirects=False)
    assert "ok=" in r.headers["location"]
    assert any(e.event_type == "watchlist.rejected" and e.payload["source"] == "ui"
               for e in db.store.read_audit())
    r = client.post("/safety/reset", data={**tok, "typed": "nope"}, follow_redirects=False)
    assert "err=" in r.headers["location"]


def test_ui_trade_approval_end_to_end(cfg: GuardrailsConfig, rates: BaseRates, db: OperatorDB,
                                      tmp_path: Path) -> None:
    rig = Rig(cfg, rates, db)
    rig.runner.cycle([CAT], [])
    [p] = db.pending()
    c = TestClient(create_app("tok", db=db, trades=rig.trades,
                              watchlist_path=tmp_path / "wl.yaml"))
    c.cookies.set(COOKIE, "tok")
    page = c.get("/trades").text
    assert "Fake broker · dry run" in page and "BUY" in page and 'data-match="ONCO"' in page
    r = c.post(f"/trades/{p.id}/approve", data={"csrf": _csrf("tok"), "typed": "ONCO"},
               follow_redirects=False)
    assert "ok=" in r.headers["location"] and len(rig.broker.orders) == 1


# ------------------------------------------------------------------ architecture


def test_web_layer_never_imports_brokers() -> None:
    src = REPO_ROOT / "src" / "biotradeai" / "operator"
    for name in ("api.py", "views.py", "jobs.py", "host.py", "db.py", "desktop.py"):
        tree = ast.parse((src / name).read_text(encoding="utf-8"))
        mods: list[Any] = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        mods += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        assert not any(m and m.startswith("biotradeai.brokers") for m in mods), name


def test_header_shows_where_orders_go_with_broker_link(client: TestClient,
                                                        db: OperatorDB) -> None:
    client.cookies.set(COOKIE, "tok")
    page = client.get("/positions").text
    assert "Orders go to Alpaca · PAPER" in page
    assert 'href="https://app.alpaca.markets/paper/dashboard/overview"' in page
    assert 'target="_blank" rel="noopener noreferrer"' in page
    jobs.save_schedule(db, "paper", enabled=False, days=[0], at_time="10:00",
                       params={"broker": "fake", "approvals": "reject"}, by="t")
    page = client.get("/").text
    assert "Orders go to Fake broker · dry run" in page and "alpaca.markets" not in page


def test_research_page_shows_the_agent_trace(client: TestClient, db: OperatorDB) -> None:
    client.cookies.set(COOKIE, "tok")
    rid = "paper-ONCO-20260105T150000"
    a = db.store.append_audit
    a("agent.start", {"model": "fake:m", "live": True,
                      "tools": ["search_evidence", "fetch_trial", "finish"]}, run_id=rid,
      ticker="ONCO")
    a("agent.turn", {"round": 0, "turn": 1, "text": "need a trial record",
                     "tool_calls": [{"id": "1", "name": "fetch_trial",
                                     "arguments": {"nct_id": "NCT01234567"}}]},
      run_id=rid, ticker="ONCO")
    a("agent.tool", {"round": 0, "tool": "fetch_trial", "nct_id": "NCT01234567",
                     "result": "added NCT01234567"}, run_id=rid, ticker="ONCO")
    a("agent.revise", {"round": 1, "revision": 1, "feedback": [
        {"node": "success", "reason": "excerpt is about safety"}]}, run_id=rid, ticker="ONCO")
    a("agent.score", {"round": 1, "stop_reason": "finished", "summary": "Tier-1 record found",
                      "confidence_before": 0.699, "confidence": 0.78, "tiers": [1, 2, 3],
                      "evidence_status": "OK"}, run_id=rid, ticker="ONCO")
    a("supervisor.decision", {"decision": "SKIP", "reasons": ["ev_min: 0.12 < 0.2"],
                              "research_ev": 0.12, "risk_ev": 0.12}, run_id=rid, ticker="ONCO")
    html = client.get(f"/activity/run/{rid}").text
    for s in ("Research agent", "fetch_trial(NCT01234567)", "added NCT01234567",
              "critic rejected 1 change", "Tier-1 record found", "tier 1: 1", "78%",
              "from 70%"):
        assert s in html, s
