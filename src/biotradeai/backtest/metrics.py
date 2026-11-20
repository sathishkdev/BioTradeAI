"""Backtest metrics (Phase 7). Every metric is recomputed here from raw state, NOT
taken from the agents' own verdicts, so a bug in the Risk Agent shows up as a breach.

Targets (plan section 8, Phase 7):
    Brier score             <= 0.18
    groundedness            >= 0.98
    guardrail breaches      == 0
    escalation accuracy     == 1.00   (INCONCLUSIVE on every Tier-1 vs Tier-2 conflict)
    latency per ticker      <= 45 s   (excluding human wait)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from biotradeai.agents.base_rates import BaseRates
from biotradeai.agents.research import is_llm_adjusted
from biotradeai.backtest.harness import CaseResult
from biotradeai.config import GuardrailsConfig
from biotradeai.guardrails import rules
from biotradeai.schemas import Decision, EvidenceStatus, OrderType

BRIER_TARGET = 0.18
GROUNDEDNESS_TARGET = 0.98
LATENCY_TARGET_S = 45.0


def brier(pairs: list[tuple[float, int]]) -> float | None:
    if not pairs:
        return None
    return math.fsum((p - o) ** 2 for p, o in pairs) / len(pairs)


def success_probability(res: CaseResult) -> float | None:
    """P(clinical success) at depth 1 after the Critic, or None if no tree was built."""
    for n in res.state.tot_nodes:
        if n.depth == 1 and n.label == "success":
            return n.probability
    return None


@dataclass
class Grounding:
    total: int = 0
    grounded: int = 0
    ungrounded: list[str] = field(default_factory=list)

    def add(self, ok: bool, what: str) -> None:
        self.total += 1
        self.grounded += ok
        if not ok:
            self.ungrounded.append(what)


def grounding(res: CaseResult) -> Grounding:
    g = Grounding()
    cid = res.case.id
    bundle = {s.sha256 for s in res.inputs.evidence.sources}
    for key, rc in res.retrieval.claims.items():
        g.add(rc.evidence.sha256 in bundle, f"{cid}: claim {key}")
    for n in res.state.tot_nodes:
        if is_llm_adjusted(n):  # code rebalancing is arithmetic, not a claim
            g.add(bool(n.evidence_ids) and set(n.evidence_ids) <= bundle,
                  f"{cid}: adjustment on {n.node_id}")
    if res.state.runway_months is not None:
        g.add(bool(res.inputs.runway.source_accns), f"{cid}: runway figure")
    return g


def breaches(res: CaseResult, cfg: GuardrailsConfig) -> list[str]:
    """Independent re-check of every guardrail for anything the system would buy."""
    s, inp, cid = res.state, res.inputs, res.case.id
    out: list[str] = []
    if inp.evidence.status is EvidenceStatus.INCONCLUSIVE and \
            s.decision is not Decision.INCONCLUSIVE:
        out.append(f"{cid}: inconclusive evidence routed to {s.decision}")
    if s.decision is not Decision.BUY_CANDIDATE:
        return out
    o = s.staged_order
    if o is None or s.scenarios is None:
        return [*out, f"{cid}: BUY_CANDIDATE without order/scenarios"]
    ev = rules.compute_ev(s.scenarios)
    if s.risk_ev is None or abs(ev - s.risk_ev) > cfg.tot.ev_match_tolerance:
        out.append(f"{cid}: recorded EV {s.risk_ev} != recomputed {ev}")
    if ev < cfg.min_ev:
        out.append(f"{cid}: EV {ev:.4f} < min {cfg.min_ev}")
    if inp.runway.months < cfg.min_runway_months:
        out.append(f"{cid}: runway {inp.runway.months:.2f} < min")
    if inp.evidence.confidence < cfg.min_confidence:
        out.append(f"{cid}: confidence {inp.evidence.confidence} < min")
    if o.order_type is not OrderType.LIMIT or o.limit_price is None:
        out.append(f"{cid}: non-limit order")
    else:
        cap = inp.portfolio.equity * Decimal(str(rules.effective_limits(cfg).position_pct))
        if o.limit_price * o.qty + inp.portfolio.existing_value > cap:
            out.append(f"{cid}: order exceeds position cap")
    if not rules.check_spread(bid=inp.market.bid, ask=inp.market.ask, cfg=cfg).ok:
        out.append(f"{cid}: spread over limit")
    return out


@dataclass
class Metrics:
    n_cases: int
    decisions: dict[str, int]
    brier: float | None
    brier_baseline: float | None
    brier_n: int
    groundedness: float
    ungrounded: list[str]
    breaches: list[str]
    escalation_accuracy: float | None
    false_escalations: list[str]
    latency_mean_s: float
    latency_max_s: float
    per_case: list[dict[str, Any]]

    def checks(self) -> dict[str, bool | None]:
        return {
            "brier": None if self.brier is None else self.brier <= BRIER_TARGET,
            "groundedness": self.groundedness >= GROUNDEDNESS_TARGET,
            "guardrail_compliance": not self.breaches,
            "escalation_accuracy": None if self.escalation_accuracy is None
            else self.escalation_accuracy == 1.0,
            "latency": self.latency_max_s <= LATENCY_TARGET_S,
        }


def compute_metrics(results: list[CaseResult], cfg: GuardrailsConfig,
                    rates: BaseRates) -> Metrics:
    decisions: dict[str, int] = {}
    pairs: list[tuple[float, int]] = []
    base_pairs: list[tuple[float, int]] = []
    g_total = Grounding()
    all_breaches: list[str] = []
    conflicts, escalated, false_esc = 0, 0, []
    per_case = []
    for r in results:
        d = r.state.decision.value if r.state.decision else "NONE"
        decisions[d] = decisions.get(d, 0) + 1
        outcome = 1 if r.case.outcome == "success" else 0
        p = success_probability(r)
        if p is not None:
            pairs.append((p, outcome))
            base_pairs.append((rates.event(r.case.catalyst_event).clinical["success"].p, outcome))
        g = grounding(r)
        g_total.total += g.total
        g_total.grounded += g.grounded
        g_total.ungrounded += g.ungrounded
        all_breaches += breaches(r, cfg)
        inconclusive = r.state.decision is Decision.INCONCLUSIVE
        if r.case.expect_conflict:
            conflicts += 1
            escalated += inconclusive
        elif inconclusive:
            false_esc.append(f"{r.case.id}: {r.inputs.evidence.inconclusive_reason}")
        per_case.append({
            "id": r.case.id, "ticker": r.case.ticker, "decision": d,
            "outcome": r.case.outcome, "p_success": p,
            "confidence": r.inputs.evidence.confidence, "runway_months": r.inputs.runway.months,
            "risk_ev": r.state.risk_ev, "veto_reasons": list(r.state.veto_reasons),
            "inconclusive_reason": r.inputs.evidence.inconclusive_reason,
            "sources": len(r.inputs.evidence.sources), "latency_s": round(r.latency_s, 3),
        })
    lat = [r.latency_s for r in results] or [0.0]
    return Metrics(
        n_cases=len(results), decisions=decisions, brier=brier(pairs),
        brier_baseline=brier(base_pairs), brier_n=len(pairs),
        groundedness=(g_total.grounded / g_total.total) if g_total.total else 1.0,
        ungrounded=g_total.ungrounded, breaches=all_breaches,
        escalation_accuracy=(escalated / conflicts) if conflicts else None,
        false_escalations=false_esc,
        latency_mean_s=math.fsum(lat) / len(lat), latency_max_s=max(lat), per_case=per_case)
