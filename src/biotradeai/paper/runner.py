"""One paper-trading cycle (Phase 8): exits first, then new entries.

    cycle():
      1. mode must be ``paper``; drawdown check (may trip the kill-switch)
      2. sell cycle: for every broker position, build events (quote, human-entered
         outcomes, newly scheduled catalysts, new SEC filings), run the sell machine,
         ask the human to approve each exit, execute through ``execution.py``
      3. buy cycle: for each upcoming catalyst not held and not already evaluated
         today, gather live inputs, run the buy graph, ask the human, execute
      4. independent post-trade compliance check; any breach trips the kill-switch

Run it on a schedule (e.g. Windows Task Scheduler / cron) once or a few times a day.
Every step is audit-logged; ``paper/review.py`` summarises a day.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any, Protocol

from biotradeai.agents.base_rates import BaseRates
from biotradeai.agents.llm import LLMClient
from biotradeai.brokers.approval import ApprovalAuthority
from biotradeai.brokers.base import Broker
from biotradeai.brokers.execution import ExecStatus, ExecutionService
from biotradeai.config import GuardrailsConfig, Mode
from biotradeai.data.catalysts import Catalyst, upcoming
from biotradeai.data.http import FetchError
from biotradeai.data.outcomes import Outcome
from biotradeai.events import (
    CatalystOutcome,
    CatalystScheduled,
    CompletionDateChange,
    FilingEvent,
    PriceTick,
    SellEvent,
)
from biotradeai.graph.buy_graph import BuyGraphDeps, build_buy_graph, resume_run, start_run
from biotradeai.graph.sell_graph import Exit, SellEngine, SellPosition, SellState
from biotradeai.guardrails import rules
from biotradeai.guardrails.killswitch import KillSwitch
from biotradeai.paper.gather import EVENT_MAP, GatherError, exposure
from biotradeai.schemas import ApprovalToken, CatalystState, ResearchInputs, StagedOrder
from biotradeai.store.sqlite_store import SQLiteStore

# (order, evidence pack) -> True if the human approved, False if rejected, or DEFER to
# park it for a later decision (approve-later queue; the operator UI decides it).
DEFER = "defer"
Approver = Callable[[StagedOrder, dict[str, Any]], bool | str]
# (kind, order, evidence pack, run_id) -> pending id. Called when the approver defers.
DeferSink = Callable[[str, StagedOrder, dict[str, Any], str], int]
OUTCOME_EARLY_DAYS = 14  # a readout may be announced up to this long before the listed date


class Gatherer(Protocol):
    def gather(self, c: Catalyst, *, now: datetime,
               run_id: str) -> tuple[CatalystState, ResearchInputs]: ...


class TrialSource(Protocol):
    """Completion-date changes for the trials behind a held ticker (time-stop)."""

    def poll(self, ticker: str) -> list[CompletionDateChange]: ...


class FilingSource(Protocol):
    def poll(self, ticker: str, *, since: date) -> list[FilingEvent]: ...


@dataclass
class CycleReport:
    started_at: datetime
    evaluated: list[tuple[str, str]] = field(default_factory=list)  # (ticker, decision)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (ticker, reason)
    orders: list[dict[str, Any]] = field(default_factory=list)
    exits: list[dict[str, Any]] = field(default_factory=list)
    breaches: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    halted: str | None = None


class PaperRunner:
    def __init__(self, *, cfg: GuardrailsConfig, store: SQLiteStore, broker: Broker,
                 gatherer: Gatherer, rates: BaseRates, generator: LLMClient,
                 critic: LLMClient, authority: ApprovalAuthority, approver: Approver,
                 checkpointer: Any, filings: FilingSource | None = None,
                 trials: TrialSource | None = None,
                 on_defer: DeferSink | None = None,
                 has_pending: Callable[[str], bool] = lambda ticker: False,
                 clock: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None:
        self.cfg, self.store, self.broker = cfg, store, broker
        self._gatherer, self._rates = gatherer, rates
        self._gen, self._crit = generator, critic
        self._authority, self._approver = authority, approver
        self._saver, self._filings, self._clock = checkpointer, filings, clock
        self._trials = trials
        self._on_defer, self._has_pending = on_defer, has_pending
        self._exec = ExecutionService(broker, store, cfg, authority)
        self.killswitch: KillSwitch = self._exec.killswitch
        self._last_exec: dict[str, Any] | None = None

    # ------------------------------------------------------------------ helpers

    def _audit(self, event: str, payload: dict[str, Any], *, ticker: str | None = None,
               run_id: str | None = None) -> None:
        self.store.append_audit(event, payload, ticker=ticker, run_id=run_id)

    def _ask(self, kind: str, order: StagedOrder, pack: dict[str, Any],
             run_id: str) -> bool | None:
        """True/False = decided now; None = parked in the approve-later queue."""
        ans = self._approver(order, pack)
        if ans == DEFER:
            if self._on_defer is None:
                raise RuntimeError("approver deferred but no approve-later queue is wired")
            pid = self._on_defer(kind, order, pack, run_id)
            self._audit("approval.deferred", {"order": order.model_dump(mode="json"),
                                              "pending_id": pid, "kind": kind},
                        ticker=order.ticker, run_id=run_id)
            return None
        return bool(ans)

    def _approve_and_execute(self, order: StagedOrder, pack: dict[str, Any],
                             run_id: str) -> dict[str, Any]:
        ans = self._ask("exit", order, pack, run_id)
        if ans is None:
            return {"status": "pending_approval", "reasons": []}
        if not ans:
            self._audit("approval.rejected", {"order": order.model_dump(mode="json")},
                        ticker=order.ticker, run_id=run_id)
            return {"status": "human_rejected", "reasons": []}
        token = self._authority.issue(order)
        self._audit("approval.granted", {"order": order.model_dump(mode="json"),
                                         "expiry": token.expiry.isoformat()},
                    ticker=order.ticker, run_id=run_id)
        return self._run_exec(order, token, run_id)

    def _run_exec(self, order: StagedOrder, token: ApprovalToken, run_id: str) -> dict[str, Any]:
        r = self._exec.execute(order, token, run_id=run_id)
        return {"status": r.status.value, "reasons": list(r.reasons), "order_id": r.order_id}

    def _graph_exec(self, order: StagedOrder, token: ApprovalToken,
                    run_id: str) -> dict[str, Any]:
        """Executor handed to the buy graph; remembers the result for the cycle report."""
        self._last_exec = self._run_exec(order, token, run_id)
        return self._last_exec

    def _evaluated_today(self, ticker: str, today: date) -> bool:
        # run ids embed the cycle's own clock: paper-<TICKER>-<YYYYmmdd>T<HHMMSS>
        prefix = f"paper-{ticker}-{today:%Y%m%d}T"
        return any(e.event_type == "supervisor.decision" and (e.run_id or "").startswith(prefix)
                   for e in self.store.read_audit(ticker=ticker))

    # ------------------------------------------------------------------ cycle

    def cycle(self, catalysts: list[Catalyst], outcomes: list[Outcome]) -> CycleReport:
        now = self._clock()
        rep = CycleReport(now)
        if self.cfg.mode is not Mode.PAPER:
            raise RuntimeError(f"paper runner needs mode: paper (config says "
                               f"{self.cfg.mode.value})")
        self._audit("paper.cycle", {"started_at": now.isoformat(), "broker": self.broker.name,
                                    "catalysts": len(catalysts)})
        acct = self.broker.get_account()
        if self.killswitch.check_drawdown(start_equity=acct.last_equity, equity=acct.equity):
            rep.halted = "kill-switch tripped on daily drawdown"
        elif self.killswitch.tripped:
            rep.halted = f"kill-switch is tripped: {self.store.get_killswitch().reason}"
        if rep.halted:
            self._audit("paper.halted", {"reason": rep.halted})
            return rep
        self._sell_cycle(now, catalysts, outcomes, rep)
        self._buy_cycle(now, catalysts, rep)
        if rep.breaches:
            self.killswitch.trip("post-trade compliance breach: " + "; ".join(rep.breaches))
            rep.halted = "kill-switch tripped on a compliance breach"
        self._audit("paper.cycle_done", {"broker": self.broker.name,
                                         "evaluated": rep.evaluated, "skipped": rep.skipped,
                                         "orders": len(rep.orders), "exits": len(rep.exits),
                                         "breaches": rep.breaches, "errors": rep.errors[:20],
                                         "halted": rep.halted})
        return rep

    # ------------------------------------------------------------------ buys

    def _buy_cycle(self, now: datetime, catalysts: list[Catalyst], rep: CycleReport) -> None:
        held = set(exposure(self.broker).open_tickers)
        for c in upcoming(catalysts, today=now.date()):
            if c.ticker in held:
                continue
            if self._has_pending(c.ticker):
                rep.skipped.append((c.ticker, "an order is waiting for your approval"))
                continue
            if self._evaluated_today(c.ticker, now.date()):
                rep.skipped.append((c.ticker, "already evaluated today"))
                continue
            run_id = f"paper-{c.ticker}-{now:%Y%m%dT%H%M%S}"
            try:
                state, inputs = self._gatherer.gather(c, now=now, run_id=run_id)
            except GatherError as exc:
                rep.skipped.append((c.ticker, str(exc)))
                self._audit("paper.skip", {"catalyst": f"{c.event} {c.date}",
                                           "reason": str(exc)}, ticker=c.ticker, run_id=run_id)
                continue
            self._last_exec = None
            deps = BuyGraphDeps(self.cfg, self._rates, self._gen, self._crit, self.store,
                                executor=self._graph_exec, clock=lambda: now)
            app = build_buy_graph(deps, self._saver)
            out = start_run(app, state, inputs)
            rep.evaluated.append((c.ticker, out.state.decision.value if out.state.decision
                                  else "NONE"))
            order = out.state.staged_order
            if not out.awaiting_approval or order is None:
                continue
            pack = {"kind": "entry", "ticker": c.ticker, "catalyst": f"{c.event} {c.date}",
                    "event": c.event, "catalyst_date": c.date.isoformat(),
                    "drug": c.drug, "ev": out.state.risk_ev,
                    "confidence": inputs.evidence.confidence,
                    "runway_months": inputs.runway.months,
                    "sources": [s.url for s in inputs.evidence.sources][:5],
                    "checks": (out.risk or {}).get("checks", [])}
            ans = self._ask("entry", order, pack, run_id)
            if ans is None:  # parked: the graph stays at its human interrupt (checkpointed)
                rep.orders.append({"ticker": c.ticker, "side": "buy", "qty": order.qty,
                                   "limit": str(order.limit_price),
                                   "status": "pending_approval", "reasons": []})
                continue
            approved = ans
            token = self._authority.issue(order) if approved else None
            if token is not None:
                self._audit("approval.granted", {"order": order.model_dump(mode="json"),
                                                 "expiry": token.expiry.isoformat()},
                            ticker=c.ticker, run_id=run_id)
            resume_run(app, run_id, approved=approved, operator="paper-cli", token=token)
            res = self._last_exec or {"status": "human_rejected", "reasons": []}
            rep.orders.append({"ticker": c.ticker, "side": "buy", "qty": order.qty,
                               "limit": str(order.limit_price), **res})
            if res["status"] == ExecStatus.PLACED.value:
                held.add(c.ticker)
                record_entry(self.store, order, event=c.event, catalyst_date=c.date,
                             opened_on=self._clock().date())
                rep.breaches += self._post_trade_check(order, inputs, held)

    def _post_trade_check(self, order: StagedOrder, inputs: ResearchInputs,
                          held: set[str]) -> list[str]:
        """Independent re-check of what was just placed (a breach here = a bug)."""
        out = []
        if not rules.check_order_type(order, self.cfg).ok:
            out.append(f"{order.ticker}: non-limit order placed")
        lim = rules.effective_limits(self.cfg)
        cap = inputs.portfolio.equity * Decimal(str(lim.position_pct))
        if (order.limit_price or Decimal(0)) * order.qty + inputs.portfolio.existing_value > cap:
            out.append(f"{order.ticker}: order exceeds the {lim.position_pct:.0%} cap")
        if len(held) > lim.max_positions:
            out.append(f"{len(held)} positions/open buys > max {lim.max_positions}")
        for b in out:
            self._audit("guardrail.breach", {"breach": b}, ticker=order.ticker)
        return out

    # ------------------------------------------------------------------ sells

    def _position_state(self, ticker: str, qty: int, avg: Decimal,
                        catalysts: list[Catalyst]) -> SellPosition | None:
        saved = self.store.load_sell_state(ticker)
        if saved is not None:
            return SellPosition.model_validate(saved)
        mine = sorted((c for c in catalysts if c.ticker == ticker), key=lambda c: c.date)
        if not mine or avg <= 0:
            return None
        c = mine[0]
        return SellPosition(ticker=ticker, qty=qty, entry_price=avg,
                            catalyst_event=EVENT_MAP.get(c.event, c.event.lower()),
                            catalyst_date=c.date)

    def _sell_cycle(self, now: datetime, catalysts: list[Catalyst], outcomes: list[Outcome],
                    rep: CycleReport) -> None:
        open_sells: dict[str, int] = {}
        for o in self.broker.get_orders(open_only=True):
            if o.side.value == "sell":
                open_sells[o.ticker] = open_sells.get(o.ticker, 0) + o.qty
        engine = SellEngine(self.cfg, self.store)
        for p in self.broker.get_positions():
            if p.qty <= 0:
                continue
            if self._has_pending(p.ticker):
                rep.skipped.append((p.ticker, "an exit is waiting for your approval"))
                continue
            pos = self._position_state(p.ticker, p.qty, p.avg_entry_price, catalysts)
            if pos is None:
                rep.skipped.append((p.ticker, "position has no catalyst in catalysts.csv"))
                self._audit("paper.unmanaged_position", {"qty": p.qty}, ticker=p.ticker)
                continue
            free = p.qty - open_sells.get(p.ticker, 0)  # shares not already being sold
            if pos.state is SellState.CLOSED or free <= 0:
                if free > 0:
                    rep.skipped.append((p.ticker, f"exit staged earlier but {free} shares "
                                                  "still held: review manually"))
                    self._audit("paper.exit_unfilled", {"held": p.qty, "free": free},
                                ticker=p.ticker)
                continue
            pos = pos.model_copy(update={"qty": free})
            run_id = f"paper-sell-{p.ticker}-{now:%Y%m%dT%H%M%S}"
            # The fresh quote is applied first so any exit is priced at today's bid.
            exits: list[Exit] = []
            q = self.broker.get_quote(p.ticker)
            if q.bid > 0 and q.ask > 0:
                tick = PriceTick(ticker=p.ticker, ts=now, bid=q.bid, ask=q.ask)
                pos, exits = engine.process(pos, [tick], run_id=run_id)
            _, more = engine.process(pos, self._events(pos, now, catalysts, outcomes, rep.errors),
                                     run_id=run_id)
            exits += more
            for ex in exits:
                entry: dict[str, Any] = {"ticker": p.ticker, "trigger": ex.trigger.value,
                                         "qty": ex.qty, "reason": ex.reason}
                if ex.order is None:
                    entry["status"] = "needs_quote"
                else:
                    pack = {"kind": "exit", "ticker": p.ticker, "trigger": ex.trigger.value,
                            "reason": ex.reason, "held": p.qty}
                    entry |= self._approve_and_execute(ex.order, pack,
                                                       run_id=f"paper-exit-{p.ticker}")
                    entry["limit"] = str(ex.order.limit_price)
                rep.exits.append(entry)

    def _events(self, pos: SellPosition, now: datetime, catalysts: list[Catalyst],
                outcomes: list[Outcome], errors: list[str]) -> list[SellEvent]:
        evs: list[SellEvent] = []
        # A window event's date is the window END, so a readout can come much earlier:
        # anything after the entry counts. Unknown entry -> a margin before the date.
        earliest = pos.opened_on or pos.catalyst_date - timedelta(days=OUTCOME_EARLY_DAYS)
        for o in outcomes:
            if o.ticker == pos.ticker and earliest <= o.date <= now.date():
                evs.append(CatalystOutcome(ticker=pos.ticker, ts=now, result=o.result,
                                           source_url=o.source_url))
        for c in catalysts:
            if c.ticker == pos.ticker and c.date > pos.catalyst_date:
                evs.append(CatalystScheduled(ticker=pos.ticker, ts=now, event=c.event,
                                             catalyst_date=c.date))
        if self._trials is not None:
            try:
                evs += self._trials.poll(pos.ticker)
            except FetchError as exc:
                errors.append(f"{pos.ticker} trials: {exc}"[:200])
        if self._filings is not None:
            try:
                evs += self._filings.poll(pos.ticker, since=now.date() - timedelta(days=7))
            except FetchError as exc:
                errors.append(f"{pos.ticker} filings: {exc}"[:200])
        return evs


def record_entry(store: SQLiteStore, order: StagedOrder, *, event: str, catalyst_date: date,
                 opened_on: date) -> None:
    """After a buy is placed: the position and its sell state machine (HOLDING)."""
    if order.limit_price is None:  # pragma: no cover - limit orders only
        return
    pos = SellPosition(ticker=order.ticker, qty=order.qty, entry_price=order.limit_price,
                       catalyst_event=EVENT_MAP.get(event, event.lower()),
                       catalyst_date=catalyst_date, opened_on=opened_on)
    store.save_sell_state(order.ticker, pos.model_dump(mode="json"))
    store.upsert_position(order.ticker, order.qty, str(order.limit_price),
                          catalyst_date=catalyst_date.isoformat())
