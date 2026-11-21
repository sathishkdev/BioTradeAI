"""Operator app: local web server behind the desktop window.

Security model (local, single user):
* listens on 127.0.0.1 only;
* a random session token is created per launch; the window opens ``/auth?t=<token>``,
  which sets an HttpOnly, SameSite=Strict cookie. Every other request needs it;
* every POST also carries a CSRF field derived from the token, and a cross-origin
  ``Origin`` header is refused;
* secrets are never rendered; every action is audit-logged with ``source: ui``.
"""

from __future__ import annotations

import contextlib
import getpass
import hashlib
import hmac
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from biotradeai.config import ConfigError, load_config
from biotradeai.operator import jobs, safety, settings, views
from biotradeai.operator import watchlist_ops as wl_ops
from biotradeai.operator.db import OperatorDB
from biotradeai.operator.host import host_status, read_log, start_detached
from biotradeai.operator.trades import TradeApprovals, TradeError
from biotradeai.watchlist.model import Watchlist, WatchlistError

HERE = Path(__file__).parent / "ui"
COOKIE = "btai_session"
PRESENTATION_KEY = "ui.presentation"



def _csrf(token: str) -> str:
    return hmac.new(token.encode(), b"csrf", hashlib.sha256).hexdigest()[:32]


def _local(ts: str | None, fmt: str = "%b %d, %H:%M") -> str:
    if not ts:
        return "—"
    try:
        d = datetime.fromisoformat(ts)
    except ValueError:
        return ts
    if d.tzinfo is None:
        d = d.replace(tzinfo=UTC)
    return d.astimezone().strftime(fmt)


def _ago(ts: str | None) -> str:
    if not ts:
        return "never"
    d = datetime.fromisoformat(ts)
    if d.tzinfo is None:
        d = d.replace(tzinfo=UTC)
    s = int((datetime.now(UTC) - d).total_seconds())
    if s < 0:
        return "in " + _span(-s)
    return _span(s) + " ago"


def _span(s: int) -> str:
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60} min"
    if s < 86400:
        return f"{s // 3600} h"
    return f"{s // 86400} d"


_EVENTS = {"PDUFA": "PDUFA date", "PHASE3_READOUT": "Phase 3 readout",
           "PHASE2_READOUT": "Phase 2 readout", "ADCOM": "Advisory committee"}


def _event_name(v: str | None) -> str:
    if not v:
        return "—"
    return _EVENTS.get(v.upper(), v.replace("_", " ").capitalize())


def _dur(seconds: float | None) -> str:
    return "—" if seconds is None else _span(int(seconds))


_HEADERS = [
    (b"cache-control", b"no-store"),
    (b"x-frame-options", b"DENY"),
    (b"content-security-policy",
     b"default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:"),
]


class Guard:
    """Session cookie, CSRF and origin checks (pure ASGI: the form body is replayed)."""

    def __init__(self, app: Any, *, token: str, csrf: str) -> None:
        self.app, self.token, self.csrf = app, token, csrf

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        req = Request(scope)
        if scope["path"] == "/auth":
            if hmac.compare_digest(req.query_params.get("t", ""), self.token):
                resp: Response = RedirectResponse("/", status_code=303)
                resp.set_cookie(COOKIE, self.token, httponly=True, samesite="strict")
            else:
                resp = HTMLResponse("Invalid link. Open the app with `biotradeai ui`.", 403)
            await resp(scope, receive, send)
            return
        if not hmac.compare_digest(req.cookies.get(COOKIE, ""), self.token):
            await HTMLResponse("Session expired. Open the app again with `biotradeai ui`.",
                               401)(scope, receive, send)
            return
        if scope["method"] == "POST":
            origin = req.headers.get("origin")
            if origin and urlparse(origin).hostname not in ("127.0.0.1", "localhost"):
                await HTMLResponse("cross-origin request refused", 403)(scope, receive, send)
                return
            body = b""
            while True:
                msg = await receive()
                body += msg.get("body", b"")
                if not msg.get("more_body"):
                    break
            form = parse_qs(body.decode("utf-8", errors="replace"))
            if not hmac.compare_digest((form.get("csrf") or [""])[0], self.csrf):
                await HTMLResponse("missing or invalid form token; reload the page",
                                   403)(scope, receive, send)
                return
            sent = False

            async def replay() -> dict[str, Any]:
                nonlocal sent
                if not sent:
                    sent = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return dict(await receive())

            receive = replay

        async def send_with_headers(msg: dict[str, Any]) -> None:
            if msg["type"] == "http.response.start":
                msg = {**msg, "headers": [*msg.get("headers", []), *_HEADERS]}
            await send(msg)

        await self.app(scope, receive, send_with_headers)


def create_app(token: str, *, db: OperatorDB | None = None,
               trades: TradeApprovals | None = None,
               watchlist_path: Path | None = None) -> FastAPI:
    app = FastAPI(title="BioTradeAI Operator", docs_url=None, redoc_url=None,
                  openapi_url=None)
    tpl = Jinja2Templates(directory=str(HERE / "templates"))
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")
    state: dict[str, Any] = {"db": db, "trades": trades}
    operator = getpass.getuser()
    csrf = _csrf(token)

    def odb() -> OperatorDB:
        if state["db"] is None:
            state["db"] = OperatorDB()
        d: OperatorDB = state["db"]
        return d

    def wl() -> Watchlist:
        return Watchlist(watchlist_path)

    def approvals() -> TradeApprovals:
        if state["trades"] is None:
            state["trades"] = TradeApprovals(odb())
        t: TradeApprovals = state["trades"]
        return t

    def presenting() -> bool:
        return odb().store.kv_get(PRESENTATION_KEY) == "1"

    env = tpl.env
    # Templates are read once per app start, so they always match the running code
    # (editing files while the app runs can't mix old code with new templates).
    env.auto_reload = False
    env.filters["local"] = _local
    env.filters["ago"] = _ago
    env.filters["dur"] = _dur
    env.filters["pct"] = lambda v, d=0: "—" if v is None else f"{float(v):.{d}%}"
    env.filters["num"] = lambda v, d=2: "—" if v is None else f"{float(v):,.{d}f}"
    env.filters["pretty"] = lambda v: json.dumps(v, indent=2, default=str, sort_keys=True)
    env.filters["days"] = jobs.describe_days
    env.filters["money"] = lambda v: ("•••" if presenting() else
                                      ("—" if v in (None, "") else f"${float(v):,.2f}"))
    env.globals["weekdays"] = jobs.WEEKDAYS
    env.filters["event"] = _event_name

    # ------------------------------------------------------------------ security

    app.add_middleware(Guard, token=token, csrf=csrf)

    def page(request: Request, name: str, active: str, **ctx: Any) -> HTMLResponse:
        d = odb()
        cfg_mode = "?"
        with contextlib.suppress(ConfigError):
            cfg_mode = load_config().mode.value
        ks = d.store.get_killswitch()
        pending = len([p for p in d.pending() if not p.expired])
        w = wl()
        base = {"active": active, "csrf": csrf, "mode": cfg_mode, "ks": ks,
                "host": host_status(d), "badge_trades": pending,
                "badge_watch": sum(1 for e in w.entries if e.status.value in
                                   ("proposed", "paused")),
                "presenting": presenting(), "operator": operator,
                "broker": views.current_destination(d),
                "ok": request.query_params.get("ok"), "err": request.query_params.get("err")}
        return tpl.TemplateResponse(request, name, {**base, **ctx})

    def back(path: str, *, ok: str | None = None, err: str | None = None) -> RedirectResponse:
        sep = "&" if "?" in path else "?"
        if ok:
            path += f"{sep}ok={quote(ok)}"
        elif err:
            path += f"{sep}err={quote(err)}"
        return RedirectResponse(path, status_code=303)

    # ------------------------------------------------------------------ overview

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request) -> HTMLResponse:
        return page(request, "overview.html", "overview", o=views.overview(odb(), wl()))

    @app.get("/api/status")
    def status() -> JSONResponse:
        d = odb()
        running = [{"id": r.id, "job": jobs.JOBS[r.job].title if r.job in jobs.JOBS else r.job,
                    "status": r.status, "started": r.started_at}
                   for r in d.runs(status=("queued", "running"), limit=10)]
        w = wl()
        return JSONResponse({
            "host": host_status(d).running, "killswitch": d.store.get_killswitch().tripped,
            "trades": len([p for p in d.pending() if not p.expired]),
            "watch": sum(1 for e in w.entries if e.status.value in ("proposed", "paused")),
            "running": running})

    # ------------------------------------------------------------------ agents

    @app.get("/agents", response_class=HTMLResponse)
    def agents(request: Request) -> HTMLResponse:
        d = odb()
        return page(request, "agents.html", "agents", jobs_=views.job_views(d),
                    runs=d.runs(limit=40), titles={k: v.title for k, v in jobs.JOBS.items()})

    @app.get("/agents/{job_id}", response_class=HTMLResponse)
    def agent(request: Request, job_id: str) -> Response:
        if job_id not in jobs.JOBS:
            return back("/agents", err="unknown agent")
        d = odb()
        s = jobs.schedule_for(d, job_id)
        nf = jobs.next_fire(s, datetime.now().astimezone())
        return page(request, "agent.html", "agents", job=jobs.JOBS[job_id], s=s, next_fire=nf,
                    runs=d.runs(job=job_id, limit=25))

    @app.post("/agents/{job_id}/run")
    async def agent_run(request: Request, job_id: str) -> Response:
        try:
            rid = jobs.run_now(odb(), job_id, by=f"ui:{operator}")
        except jobs.JobError as exc:
            return back(f"/agents/{job_id}", err=str(exc))
        msg = "Queued." if host_status(odb()).running else (
            "Queued, but the agent host is not running: start it to run this.")
        return back(f"/runs/{rid}", ok=msg)

    @app.post("/agents/{job_id}/schedule")
    async def agent_schedule(request: Request, job_id: str) -> Response:
        form = await request.form()
        try:
            d = jobs.job(job_id)
            params = {p.name: form.get(p.name) for p in d.params}
            days = [int(str(x)) for x in form.getlist("days")]
            jobs.save_schedule(odb(), job_id, enabled=form.get("enabled") == "on", days=days,
                               at_time=str(form.get("at_time", "")), params=params,
                               by=f"ui:{operator}")
        except (jobs.JobError, ValueError) as exc:
            return back(f"/agents/{job_id}", err=str(exc))
        return back(f"/agents/{job_id}", ok="Saved.")

    @app.post("/runs/{run_id}/cancel")
    async def run_cancel(run_id: int) -> Response:
        ok = jobs.cancel(odb(), run_id, by=f"ui:{operator}")
        return back(f"/runs/{run_id}", ok="Stopping…" if ok else None,
                    err=None if ok else "This run already finished.")

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_page(request: Request, run_id: int) -> Response:
        r = odb().get_run(run_id)
        if r is None:
            return back("/agents", err="no such run")
        return page(request, "run.html", "agents", r=r, job=jobs.JOBS.get(r.job),
                    log=read_log(r.log_path))

    @app.get("/api/runs/{run_id}")
    def run_json(run_id: int) -> JSONResponse:
        r = odb().get_run(run_id)
        if r is None:
            return JSONResponse({"error": "not found"}, 404)
        return JSONResponse({"status": r.status, "active": r.active, "log": read_log(r.log_path),
                             "duration": _dur(r.duration_s), "error": r.error})

    @app.post("/host/start")
    async def host_start() -> Response:
        if host_status(odb()).running:
            return back("/agents", ok="The agent host is already running.")
        start_detached()
        return back("/agents", ok="Starting the agent host…")

    # ------------------------------------------------------------------ watchlist

    @app.get("/watchlist", response_class=HTMLResponse)
    def watchlist(request: Request, tab: str = "proposed") -> HTMLResponse:
        tabs = views.watchlist_tabs(wl())
        return page(request, "watchlist.html", "watchlist", tabs=tabs,
                    tab=tab if tab in tabs else "proposed")

    @app.get("/watchlist/{entry_id}", response_class=HTMLResponse)
    def watch_entry(request: Request, entry_id: str) -> Response:
        try:
            e = wl().get(entry_id)
        except WatchlistError as exc:
            return back("/watchlist", err=str(exc))
        events = [ev for ev in odb().store.list_events(entity_id=e.entity_id)]
        return page(request, "watch_entry.html", "watchlist", e=e, events=events)

    @app.post("/watchlist/{entry_id}/{action}")
    async def watch_action(request: Request, entry_id: str, action: str) -> Response:
        form = await request.form()
        note = str(form.get("note", "")).strip()
        by = operator
        try:
            if action == "approve":
                e = wl_ops.approve(entry_id, by=by, note=note or None, source="ui", wl=wl(),
                                   store=odb().store)
                msg = f"Approved {e.id}. The next follow run collects its documents."
            elif action == "reject":
                e = wl_ops.reject(entry_id, by=by, reason=note, source="ui", wl=wl(),
                                  store=odb().store)
                msg = f"Rejected {e.id}."
            elif action == "retire":
                e = wl_ops.retire(entry_id, by=by, reason=note, source="ui", wl=wl(),
                                  store=odb().store)
                msg = f"Retired {e.id}."
            elif action == "resync":
                e = wl_ops.resync(entry_id, source="ui", wl=wl(), store=odb().store)
                msg = "The next follow run re-reads its older filings."
            else:
                return back(f"/watchlist/{entry_id}", err="unknown action")
        except Exception as exc:
            return back(f"/watchlist/{entry_id}", err=str(exc))
        return back(f"/watchlist/{quote(e.id)}", ok=msg)

    # ------------------------------------------------------------------ trades

    @app.get("/trades", response_class=HTMLResponse)
    def trades_page(request: Request) -> HTMLResponse:
        d = odb()
        return page(request, "trades.html", "trades", cards=views.trade_cards(d),
                    history=views.trade_history(d))

    @app.post("/trades/{pending_id}/approve")
    async def trade_approve(pending_id: int, typed: str = Form("")) -> Response:
        try:
            res = approvals().approve(pending_id, typed=typed, operator=operator)
        except (TradeError, Exception) as exc:
            return back("/trades", err=str(exc))
        if res.get("status") in ("placed", "shadow_reviewed"):
            return back("/trades", ok=f"Order {res['status'].replace('_', ' ')}.")
        return back("/trades", err="Not placed: " + "; ".join(res.get("reasons") or ["?"]))

    @app.post("/trades/{pending_id}/reject")
    async def trade_reject(pending_id: int, reason: str = Form("")) -> Response:
        try:
            approvals().reject(pending_id, reason=reason, operator=operator)
        except (TradeError, Exception) as exc:
            return back("/trades", err=str(exc))
        return back("/trades", ok="Rejected.")

    # ------------------------------------------------------------------ positions etc.

    @app.get("/positions", response_class=HTMLResponse)
    def positions(request: Request) -> HTMLResponse:
        return page(request, "positions.html", "positions", rows=views.positions(odb()))

    @app.get("/catalysts", response_class=HTMLResponse)
    def catalysts(request: Request) -> HTMLResponse:
        return page(request, "catalysts.html", "catalysts",
                    rows=views.catalyst_rows(odb(), wl()))

    @app.get("/activity", response_class=HTMLResponse)
    def activity(request: Request, prefix: str = "", ticker: str = "", run: str = "",
                 before: int | None = None, hide_tests: int = 1) -> HTMLResponse:
        rows = views.recent_audit(odb(), prefix=prefix, ticker=ticker, run_id=run,
                                  limit=100, before_id=before)
        if hide_tests and not run:
            rows = [r for r in rows if not (r["run_id"] or "").startswith(
                ("backtest-", "fixture-"))]
        return page(request, "activity.html", "activity", rows=rows, prefix=prefix,
                    ticker=ticker, run=run, hide_tests=hide_tests,
                    chain=request.query_params.get("chain"))

    @app.get("/activity/run/{run_id}", response_class=HTMLResponse)
    def research_run(request: Request, run_id: str) -> HTMLResponse:
        rows = list(reversed(views.recent_audit(odb(), run_id=run_id, limit=500)))
        by = {r["event"]: r["payload"] for r in rows}
        return page(request, "research.html", "activity", run_id=run_id, rows=rows,
                    decision=by.get("supervisor.decision"), final=by.get("tot.finalize"),
                    risk=by.get("risk.verdict"), ingest=by.get("graph.ingest"),
                    agent=views.agent_trace(rows),
                    ticker=rows[0]["ticker"] if rows else "")

    @app.post("/activity/verify")
    async def verify() -> Response:
        ok = odb().store.verify_audit_chain()
        return back(f"/activity?chain={'ok' if ok else 'broken'}")

    # ------------------------------------------------------------------ safety

    @app.get("/safety", response_class=HTMLResponse)
    def safety_page(request: Request) -> HTMLResponse:
        d = odb()
        cfg = load_config()
        breaches = views.recent_audit(d, prefix="guardrail.breach", limit=20)
        ks_events = views.recent_audit(d, prefix="killswitch.", limit=10)
        return page(request, "safety.html", "safety", gate=safety.gate(d.store, cfg),
                    cfg=cfg, breaches=breaches, ks_events=ks_events,
                    ledger=views.ledger_today(d), api_errors=d.store.get_api_errors())

    @app.post("/safety/trip")
    async def ks_trip() -> Response:
        r = safety.trip(odb().store, load_config(), by=operator)
        msg = (f"Kill-switch tripped. Cancelled {r.cancelled} open order(s)." if
               r.broker_reached else "Kill-switch tripped (broker not reachable: check it "
               "for open orders).")
        return back("/safety", ok=msg)

    @app.post("/safety/reset")
    async def ks_reset(typed: str = Form("")) -> Response:
        try:
            safety.reset(odb().store, load_config(), by=operator, typed=typed.strip())
        except PermissionError as exc:
            return back("/safety", err=str(exc))
        return back("/safety", ok="Kill-switch reset. Trading can resume.")

    # ------------------------------------------------------------------ settings

    @app.get("/settings", response_class=HTMLResponse)
    def settings_page(request: Request) -> HTMLResponse:
        fields = settings.guardrail_fields()
        groups: dict[str, list[settings.Field]] = {}
        for f in fields:
            groups.setdefault(f.group, []).append(f)
        try:
            rates: dict[str, Any] | None = settings.base_rates_summary()
            rates_err = None
        except Exception as exc:
            rates, rates_err = None, str(exc)
        from biotradeai.operator.desktop import autostart_enabled

        return page(request, "settings.html", "settings", groups=groups, rates=rates,
                    rates_err=rates_err, secrets=settings.secrets_status(),
                    autostart=autostart_enabled())

    @app.post("/settings/guardrail")
    async def set_guardrail(path: str = Form(...), value: str = Form("")) -> Response:
        try:
            f = settings.update_guardrail(path, value, by=f"ui:{operator}", store=odb().store)
        except settings.SettingsError as exc:
            return back(f"/settings#g-{path}", err=f"{path}: {exc}")
        return back(f"/settings#g-{path}", ok=f"{path} = {f.value}. Agents use it from "
                                               "their next run.")

    @app.post("/settings/base-rates")
    async def set_reviewed(reviewed: str = Form("")) -> Response:
        try:
            settings.set_base_rates_reviewed(reviewed == "1", by=f"ui:{operator}",
                                             store=odb().store)
        except (settings.SettingsError, Exception) as exc:
            return back("/settings#base-rates", err=str(exc))
        return back("/settings#base-rates", ok="Saved.")

    @app.post("/settings/secret")
    async def set_secret(var: str = Form(...), value: str = Form("")) -> Response:
        try:
            settings.replace_secret(var, value, by=f"ui:{operator}", store=odb().store)
        except settings.SettingsError as exc:
            return back("/settings#secrets", err=str(exc))
        return back("/settings#secrets", ok=f"{var} replaced. New runs use it.")

    @app.post("/settings/presentation")
    async def set_presentation(on: str = Form("")) -> Response:
        odb().store.kv_set(PRESENTATION_KEY, "1" if on == "1" else "0")
        return back("/settings", ok="Presentation mode " + ("on." if on == "1" else "off."))

    @app.post("/settings/autostart")
    async def set_autostart(on: str = Form("")) -> Response:
        from biotradeai.operator.desktop import set_autostart as _set

        try:
            _set(on == "1")
        except RuntimeError as exc:
            return back("/settings", err=str(exc))
        odb().store.append_audit("operator.autostart", {"enabled": on == "1",
                                                        "by": f"ui:{operator}"})
        return back("/settings", ok="Agent host " + ("starts at login." if on == "1" else
                                                     "no longer starts at login."))

    return app
