"""Quantitative Risk Agent. Fully deterministic; has VETO power only.

It never reads the Research Agent's EV to make its decision: it recomputes EV from the
scenario probabilities and returns with ``guardrails.rules.compute_ev``. The Research EV
is only compared against it; a gap larger than ``tot.ev_match_tolerance`` is itself a
veto (something in the pipeline is inconsistent, so nothing trades).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal

from biotradeai.config import GuardrailsConfig
from biotradeai.guardrails import rules
from biotradeai.guardrails.rules import RuleResult
from biotradeai.schemas import (
    CatalystState,
    EvidenceStatus,
    OrderSide,
    ResearchInputs,
    ScenarioSet,
)

CENT = Decimal("0.01")


@dataclass(frozen=True)
class RiskVerdict:
    veto: bool
    risk_ev: float
    research_ev: float | None
    ev_mismatch: bool
    checks: tuple[RuleResult, ...]
    max_qty: int
    limit_price: Decimal

    @property
    def reasons(self) -> list[str]:
        return [f"{c.rule}: {c.reason}" for c in self.checks if not c.ok]


def marketable_limit(inputs: ResearchInputs) -> Decimal:
    """Buy limit at the current ask, rounded up to the cent (still a LIMIT order)."""
    return inputs.market.ask.quantize(CENT, rounding=ROUND_HALF_UP)


def assess(state: CatalystState, scenarios: ScenarioSet, inputs: ResearchInputs, *,
           cfg: GuardrailsConfig, now: datetime) -> RiskVerdict:
    risk_ev = rules.compute_ev(scenarios)
    checks: list[RuleResult] = []

    mismatch = state.research_ev is None or abs(state.research_ev - risk_ev) > \
        cfg.tot.ev_match_tolerance
    checks.append(RuleResult(
        "ev_match", not mismatch,
        f"research EV {state.research_ev if state.research_ev is not None else 'missing'} vs "
        f"risk EV {risk_ev:+.6f} (tolerance {cfg.tot.ev_match_tolerance})"))
    checks.append(rules.check_ev(risk_ev, cfg))
    checks.append(rules.check_runway(inputs.runway.months, cfg))
    checks.append(RuleResult(
        "evidence", inputs.evidence.status is EvidenceStatus.OK,
        f"evidence {inputs.evidence.status.value}"
        + (f": {inputs.evidence.inconclusive_reason}" if inputs.evidence.inconclusive_reason
           else "")))
    checks.append(rules.check_confidence(inputs.evidence.confidence, cfg))
    checks.append(rules.check_spread(bid=inputs.market.bid, ask=inputs.market.ask, cfg=cfg))
    checks.append(rules.check_concurrency(open_tickers=set(inputs.portfolio.open_tickers),
                                          ticker=state.ticker, side=OrderSide.BUY, cfg=cfg))
    limit = marketable_limit(inputs)
    qty = rules.max_position_qty(inputs.portfolio.equity, limit,
                                 inputs.portfolio.existing_value, cfg)
    checks.append(RuleResult("position_room", qty > 0,
                             f"{qty} shares fit under the position cap at {limit}"))
    for what, ts in (("quote", inputs.market.observed_at),
                     ("runway", inputs.runway.observed_at),
                     ("portfolio", inputs.portfolio.observed_at),
                     ("evidence", inputs.evidence_observed_at)):
        checks.append(rules.check_ttl(ts, now=now, cfg=cfg, what=what))
    checks.append(RuleResult("catalyst_ahead", state.catalyst_date >= now.date(),
                             f"catalyst {state.catalyst_date} vs today {now.date()}"))
    return RiskVerdict(veto=not rules.all_ok(checks), risk_ev=risk_ev,
                       research_ev=state.research_ev, ev_mismatch=mismatch,
                       checks=tuple(checks), max_qty=qty, limit_price=limit)
