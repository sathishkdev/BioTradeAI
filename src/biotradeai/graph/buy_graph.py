"""Supervisor: the buy-side LangGraph state machine.

    ingest -> generate <-> critique (one round per ToT depth) -> finalize -> risk
           -> skip | stage -> hitl (interrupt, human) -> execute
    ingest -> inconclusive   (evidence INCONCLUSIVE: human review, no ToT)

The Supervisor only routes. It cannot generate hypotheses (that is
``agents/research.py``), alter risk math (``agents/risk.py`` / ``guardrails/rules.py``)
or call brokers: execution goes through an injected ``executor`` callable that the CLI
wires to ``brokers/execution.py``. This module does not import ``biotradeai.brokers``
(enforced by tests/unit/test_architecture.py).

Every transition re-validates the state with the Pydantic contracts, and every step is
written to the append-only audit log. State is checkpointed by LangGraph (SQLite) under
``thread_id = run_id`` so a run waiting for a human survives a restart.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from biotradeai.agents import research, risk, tot
from biotradeai.agents.base_rates import BaseRates
from biotradeai.agents.llm import LLMClient
from biotradeai.config import GuardrailsConfig
from biotradeai.paths import data_dir
from biotradeai.schemas import (
    ApprovalToken,
    CatalystState,
    Decision,
    EvidenceStatus,
    OrderSide,
    OrderType,
    ResearchInputs,
    StagedOrder,
    ToTNode,
)
from biotradeai.store.sqlite_store import SQLiteStore

# (order, token, run_id) -> result summary. Wired to ExecutionService by the CLI.
Executor = Callable[[StagedOrder, ApprovalToken, str], dict[str, Any]]


class BuyState(TypedDict, total=False):
    catalyst: dict[str, Any]  # CatalystState (JSON)
    inputs: dict[str, Any]  # ResearchInputs (JSON)
    depth: int  # last ToT depth generated
    draft: list[dict[str, Any]]  # generator output awaiting the critic
    risk: dict[str, Any]
    approval: dict[str, Any]
    execution: dict[str, Any]


@dataclass
class BuyGraphDeps:
    cfg: GuardrailsConfig
    rates: BaseRates
    generator_llm: LLMClient
    critic_llm: LLMClient
    store: SQLiteStore
    executor: Executor | None = None
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)


def _cat(state: BuyState) -> CatalystState:
    return CatalystState.model_validate(state["catalyst"])


def _inp(state: BuyState) -> ResearchInputs:
    return ResearchInputs.model_validate(state["inputs"])


def _dump(c: CatalystState) -> dict[str, Any]:
    # Round-trip through the contract: a node can never emit an invalid state.
    return CatalystState.model_validate(c.model_dump()).model_dump(mode="json")


def build_buy_graph(deps: BuyGraphDeps, checkpointer: BaseCheckpointSaver[Any]) -> Any:
    cfg, store = deps.cfg, deps.store
    depth_limit = research.max_depth(cfg)

    def audit(event: str, c: CatalystState, payload: dict[str, Any]) -> None:
        store.append_audit(event, payload, run_id=c.run_id, ticker=c.ticker)

    # ------------------------------------------------------------------ nodes

    def ingest(state: BuyState) -> dict[str, Any]:
        c, inp = _cat(state), _inp(state)
        if inp.evidence.ticker != c.ticker or inp.evidence.entity_id != c.entity_id:
            raise ValueError("inputs belong to a different ticker/entity (semantic bleed)")
        c = c.model_copy(update={"evidence": inp.evidence, "runway_months": inp.runway.months})
        audit("graph.ingest", c, {
            "catalyst": f"{c.catalyst_event} {c.catalyst_date}",
            "evidence_status": inp.evidence.status.value,
            "confidence": inp.evidence.confidence,
            "source_hashes": [x.sha256 for x in inp.evidence.sources],
            "claims": inp.claims, "runway_months": inp.runway.months,
            "runway_source_accns": inp.runway.source_accns,
            "base_rates_reviewed": deps.rates.reviewed,
        })
        return {"catalyst": _dump(c), "depth": 0}

    def route_ingest(state: BuyState) -> str:
        return "inconclusive" if _inp(state).evidence.status is EvidenceStatus.INCONCLUSIVE \
            else "generate"

    def generate(state: BuyState) -> dict[str, Any]:
        c, inp = _cat(state), _inp(state)
        depth = state.get("depth", 0) + 1
        draft, trace = research.generate_level(deps.generator_llm, c, inp, depth, cfg=cfg,
                                               rates=deps.rates)
        audit("tot.generate", c, {**trace.as_dict(),
                                  "draft": [n.model_dump(mode="json") for n in draft]})
        return {"depth": depth, "draft": [n.model_dump(mode="json") for n in draft]}

    def critique(state: BuyState) -> dict[str, Any]:
        c, inp, depth = _cat(state), _inp(state), state["depth"]
        draft = [ToTNode.model_validate(d) for d in state["draft"]]
        nodes, trace = research.critique_level(deps.critic_llm, c, inp, depth, draft, cfg=cfg)
        audit("tot.critique", c, {**trace.as_dict(),
                                  "nodes": [n.model_dump(mode="json") for n in nodes]})
        for n in nodes:
            if n.pruned:
                audit("tot.prune", c, {"node": n.node_id, "reason": n.prune_reason,
                                       "probability": n.probability})
        kept = [n for n in c.tot_nodes if n.depth != depth]
        c = c.model_copy(update={"tot_nodes": [*kept, *nodes]})
        return {"catalyst": _dump(c), "draft": []}

    def route_critique(state: BuyState) -> str:
        c = _cat(state)
        more = state["depth"] < depth_limit and tot.frontier(c.tot_nodes, state["depth"])
        return "generate" if more else "finalize"

    def finalize(state: BuyState) -> dict[str, Any]:
        c = _cat(state)
        scenarios, ev = research.finalize(list(c.tot_nodes), cfg)
        c = c.model_copy(update={"scenarios": scenarios, "research_ev": ev})
        audit("tot.finalize", c, {"scenarios": scenarios.model_dump(mode="json"),
                                  "research_ev": ev,
                                  "residual_return": cfg.tot.residual_scenario_return})
        return {"catalyst": _dump(c)}

    def assess(state: BuyState) -> dict[str, Any]:
        c, inp = _cat(state), _inp(state)
        if c.scenarios is None:
            raise ValueError("risk step reached without scenarios")
        v = risk.assess(c, c.scenarios, inp, cfg=cfg, now=deps.clock())
        c = c.model_copy(update={"risk_ev": v.risk_ev, "veto_reasons": v.reasons})
        summary = {"veto": v.veto, "risk_ev": v.risk_ev, "research_ev": v.research_ev,
                   "ev_mismatch": v.ev_mismatch, "max_qty": v.max_qty,
                   "limit_price": str(v.limit_price),
                   "checks": [{"rule": r.rule, "ok": r.ok, "reason": r.reason}
                              for r in v.checks]}
        if v.ev_mismatch:
            audit("risk.ev_mismatch", c, {"research_ev": v.research_ev, "risk_ev": v.risk_ev,
                                          "tolerance": cfg.tot.ev_match_tolerance})
        audit("risk.verdict", c, summary)
        return {"catalyst": _dump(c), "risk": summary}

    def route_risk(state: BuyState) -> str:
        return "skip" if state["risk"]["veto"] else "stage"

    def decide(c: CatalystState, decision: Decision, why: list[str]) -> dict[str, Any]:
        c = c.model_copy(update={"decision": decision})
        audit("supervisor.decision", c, {"decision": decision.value, "reasons": why,
                                         "research_ev": c.research_ev, "risk_ev": c.risk_ev})
        return {"catalyst": _dump(c)}

    def inconclusive(state: BuyState) -> dict[str, Any]:
        c, inp = _cat(state), _inp(state)
        return decide(c, Decision.INCONCLUSIVE,
                      [inp.evidence.inconclusive_reason or "evidence inconclusive"])

    def skip(state: BuyState) -> dict[str, Any]:
        c = _cat(state)
        return decide(c, Decision.SKIP, list(c.veto_reasons))

    def stage(state: BuyState) -> dict[str, Any]:
        c, r = _cat(state), state["risk"]
        order = StagedOrder(
            ticker=c.ticker, side=OrderSide.BUY, qty=r["max_qty"], order_type=OrderType.LIMIT,
            limit_price=r["limit_price"], ev=c.risk_ev,
            confidence=c.evidence.confidence if c.evidence else None,
            runway_months=c.runway_months,
            source_hashes=[x.sha256 for x in c.evidence.sources] if c.evidence else [])
        c = c.model_copy(update={"staged_order": order})
        out = decide(c, Decision.BUY_CANDIDATE, [])
        audit("hitl.request", c, {"order": order.model_dump(mode="json")})
        return out

    def hitl(state: BuyState) -> dict[str, Any]:
        c = _cat(state)
        answer = interrupt({"run_id": c.run_id,
                            "order": c.staged_order.model_dump(mode="json")
                            if c.staged_order else None})
        if not isinstance(answer, dict):
            answer = {"approved": False, "operator": "unknown", "note": "malformed resume"}
        approval = {"approved": bool(answer.get("approved")),
                    "operator": str(answer.get("operator", "unknown")),
                    "token": answer.get("token")}
        audit("hitl.response", c, {"approved": approval["approved"],
                                   "operator": approval["operator"],
                                   "has_token": approval["token"] is not None})
        return {"approval": approval}

    def route_hitl(state: BuyState) -> str:
        a = state.get("approval") or {}
        return "execute" if a.get("approved") and a.get("token") else END

    def execute(state: BuyState) -> dict[str, Any]:
        c = _cat(state)
        if deps.executor is None or c.staged_order is None:
            result: dict[str, Any] = {"status": "not_executed",
                                      "reasons": ["no executor configured"]}
        else:
            token = ApprovalToken.model_validate(state["approval"]["token"])
            result = deps.executor(c.staged_order, token, c.run_id)
        audit("execution.result", c, result)
        return {"execution": result}

    # ------------------------------------------------------------------ wiring

    g = StateGraph(BuyState)
    g.add_node("ingest", ingest)
    g.add_node("generate", generate)
    g.add_node("critique", critique)
    g.add_node("finalize", finalize)
    g.add_node("risk", assess)
    g.add_node("inconclusive", inconclusive)
    g.add_node("skip", skip)
    g.add_node("stage", stage)
    g.add_node("hitl", hitl)
    g.add_node("execute", execute)
    g.add_edge(START, "ingest")
    g.add_conditional_edges("ingest", route_ingest, ["generate", "inconclusive"])
    g.add_edge("generate", "critique")
    g.add_conditional_edges("critique", route_critique, ["generate", "finalize"])
    g.add_edge("finalize", "risk")
    g.add_conditional_edges("risk", route_risk, ["skip", "stage"])
    g.add_edge("stage", "hitl")
    g.add_conditional_edges("hitl", route_hitl, ["execute", END])
    for terminal in ("inconclusive", "skip", "execute"):
        g.add_edge(terminal, END)
    return g.compile(checkpointer=checkpointer)


# --------------------------------------------------------------------------- running


@dataclass(frozen=True)
class RunOutcome:
    state: CatalystState
    awaiting_approval: bool
    risk: dict[str, Any] | None
    execution: dict[str, Any] | None


def _thread(run_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": run_id}}


def _outcome(app: Any, run_id: str) -> RunOutcome:
    snap = app.get_state(_thread(run_id))
    v = snap.values
    return RunOutcome(state=CatalystState.model_validate(v["catalyst"]),
                      awaiting_approval="hitl" in (snap.next or ()),
                      risk=v.get("risk"), execution=v.get("execution"))


def start_run(app: Any, catalyst: CatalystState, inputs: ResearchInputs) -> RunOutcome:
    app.invoke({"catalyst": catalyst.model_dump(mode="json"),
                "inputs": inputs.model_dump(mode="json")}, _thread(catalyst.run_id))
    return _outcome(app, catalyst.run_id)


def resume_run(app: Any, run_id: str, *, approved: bool, operator: str,
               token: ApprovalToken | None) -> RunOutcome:
    app.invoke(Command(resume={"approved": approved, "operator": operator,
                               "token": token.model_dump(mode="json") if token else None}),
               _thread(run_id))
    return _outcome(app, run_id)


def get_outcome(app: Any, run_id: str) -> RunOutcome:
    return _outcome(app, run_id)


@contextmanager
def sqlite_checkpointer(path: Path | None = None) -> Iterator[SqliteSaver]:
    """LangGraph checkpoints in ``data/checkpoints.db`` (git-ignored)."""
    p = path or (data_dir() / "checkpoints.db")
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), check_same_thread=False)
    try:
        yield SqliteSaver(conn)
    finally:
        conn.close()
