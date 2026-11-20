"""Gather live inputs for one catalyst: RAG evidence, SEC runway, broker quote/account.

This is the data layer for paper trading. It reads from the broker (quote, account,
positions, open orders) but never places anything. Account identifiers never leave
the broker adapter.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

from biotradeai.brokers.base import Broker
from biotradeai.config import GuardrailsConfig
from biotradeai.data.catalysts import Catalyst
from biotradeai.data.edgar import EdgarClient
from biotradeai.data.pollers import FilingPoller
from biotradeai.data.runway import RunwayResult, compute_runway
from biotradeai.events import FilingEvent
from biotradeai.inputs import inputs_from_retrieval, runway_snapshot
from biotradeai.rag.retrieve import Retriever
from biotradeai.schemas import CatalystState, OrderSide, ResearchInputs
from biotradeai.store.sqlite_store import SQLiteStore

# catalysts.csv event -> base-rate event. Others have no base rates yet and are skipped.
EVENT_MAP = {"PHASE3_READOUT": "phase3_readout", "PDUFA": "pdufa"}
QUERY = "{drug} clinical trial results primary endpoint safety clinical hold regulatory"


class GatherError(RuntimeError):
    """This catalyst cannot be evaluated right now (reason in the message)."""


RunwayProvider = Callable[[str, datetime], RunwayResult]


def edgar_runway(edgar: EdgarClient, cfg: GuardrailsConfig) -> RunwayProvider:
    """Runway from SEC XBRL company facts, point-in-time at ``now``."""

    def provide(ticker: str, now: datetime) -> RunwayResult:
        cik = edgar.ticker_to_cik(ticker)
        return compute_runway(edgar.company_facts(cik),
                              lookback_quarters=cfg.runway.burn_lookback_quarters,
                              as_of=now.date())
    return provide


@dataclass(frozen=True)
class Exposure:
    open_tickers: list[str]  # held positions + tickers with open buy orders
    existing_value: dict[str, Decimal]  # market value + open buy notional, per ticker


def exposure(broker: Broker) -> Exposure:
    held = {p.ticker: p.market_value for p in broker.get_positions() if p.qty > 0}
    value = dict(held)
    for o in broker.get_orders(open_only=True):
        if o.side is OrderSide.BUY and o.limit_price is not None:
            value[o.ticker] = value.get(o.ticker, Decimal(0)) + o.limit_price * o.qty
    return Exposure(sorted(value), value)


class LiveGatherer:
    def __init__(self, *, retriever: Retriever, runway: RunwayProvider, broker: Broker,
                 cfg: GuardrailsConfig) -> None:
        self._retriever = retriever
        self._runway = runway
        self._broker = broker
        self._cfg = cfg

    def gather(self, c: Catalyst, *, now: datetime,
               run_id: str) -> tuple[CatalystState, ResearchInputs]:
        event = EVENT_MAP.get(c.event)
        if event is None:
            raise GatherError(f"no base rates for event {c.event}; supported: "
                              f"{sorted(EVENT_MAP)}")
        try:
            r = self._retriever.retrieve(QUERY.format(drug=c.drug), ticker=c.ticker,
                                         drug=c.drug, as_of=now)
        except ValueError as exc:  # unknown drug name (OntologyError) etc.
            raise GatherError(f"evidence: {exc}") from exc
        try:
            rw = self._runway(c.ticker, now)
        except Exception as exc:
            raise GatherError(f"runway: {exc}") from exc
        q = self._broker.get_quote(c.ticker)
        if q.bid <= 0 or q.ask <= 0:
            raise GatherError(f"no usable quote (bid {q.bid}, ask {q.ask}); market closed?")
        acct = self._broker.get_account()
        ex = exposure(self._broker)
        try:
            inputs = inputs_from_retrieval(
                r, runway=runway_snapshot(rw, observed_at=now), bid=q.bid, ask=q.ask,
                quote_at=q.timestamp, equity=acct.equity, open_tickers=ex.open_tickers,
                existing_value=ex.existing_value.get(c.ticker, Decimal(0)), now=now)
        except ValueError as exc:  # pydantic ValidationError is a ValueError
            raise GatherError(f"inputs failed validation: {exc}") from exc
        state = CatalystState(run_id=run_id, ticker=c.ticker, entity_id=r.entity_id,
                              catalyst_event=event, catalyst_date=c.date, as_of=now)
        return state, inputs


class EdgarFilings:
    """FilingSource for the sell cycle: new SEC filings per ticker (each reported once)."""

    def __init__(self, edgar: EdgarClient, store: SQLiteStore) -> None:
        self._edgar = edgar
        self._poller = FilingPoller(edgar, store)

    def poll(self, ticker: str, *, since: date) -> list[FilingEvent]:
        return self._poller.poll(ticker, self._edgar.ticker_to_cik(ticker), since=since)
