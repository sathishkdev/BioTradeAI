"""Supervisor: the buy-side LangGraph state machine.

    ingest -> agent <-> tools (research agent: the LLM picks tools until it finishes)
           -> score (code re-scores all evidence found)
           -> inconclusive            (evidence still INCONCLUSIVE: no ToT)
           -> generate <-> critique   (one round per ToT depth)
                 critique -> revise -> agent ...   (Critic rejected nudges: search again,
                                                     then regenerate that depth; capped)
           -> finalize -> risk -> skip | stage -> hitl (interrupt, human) -> execute

With no research agent configured (``agent_llm`` / ``toolbox`` unset, or
``agent.enabled: false``) ``agent`` and ``score`` pass straight through and the graph
behaves exactly as the single-search version did.

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
from biotradeai.agents import research_agent as ra
from biotradeai.agents.base_rates import BaseRates
from biotradeai.agents.llm import ChatTurn, LLMClient, LLMError, ToolLLMClient
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
    agent: dict[str, Any]  # research agent working memory (ra.AgentRun)
    feedback: list[dict[str, Any]]  # Critic rejections for the depth being revised
    revisions: int  # revision rounds used
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
    agent_llm: ToolLLMClient | None = None  # research agent (tool loop); None = off
    toolbox: ra.ToolboxFactory | None = None  # tools for one catalyst; None = off


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

    # ------------------------------------------------------------------ research agent

    acfg = cfg.agent
    boxes: dict[str, ra.Toolbox | None] = {}

    def box_for(c: CatalystState) -> ra.Toolbox | None:
        if deps.toolbox is None or deps.agent_llm is None or not acfg.enabled:
            return None
        if c.run_id not in boxes:
            boxes[c.run_id] = deps.toolbox(c)
        return boxes[c.run_id]

    def agent(state: BuyState) -> dict[str, Any]:
        """One model turn of the research agent (the LLM chooses the next step)."""
        c = _cat(state)
        run = ra.AgentRun.from_dict(state.get("agent"))
        box = box_for(c)
        llm = deps.agent_llm
        if box is None or llm is None:
            return {"agent": ra.AgentRun(done=True, stop_reason="agent off").as_dict()}
        if run.ran and run.done:  # finished in the tools step: go on to scoring
            return {}
        if not run.ran:  # first visit: seed with the same baseline search as before
            base = box.baseline()
            run.add_found(base)
            run.baseline_n = len(run.found)
            run.ran, run.done = True, False
            run.messages = [ChatTurn("user", ra.first_prompt(
                c, box.drug_names, ra.format_chunks(run, base))).as_dict()]
            audit("agent.start", c, {"model": f"{llm.provider}:{llm.model}", "live": box.live,
                                     "tools": [t.name for t in ra.tools_for(box)],
                                     "baseline": [run.handle(b.sha256) for b in base]})
        if run.turns >= acfg.max_turns or run.calls >= acfg.max_tool_calls:
            run.done, run.stop_reason = True, "limit reached"
            return {"agent": run.as_dict()}
        try:
            resp = llm.converse(run.turn_objs(), tools=ra.tools_for(box), system=ra.SYSTEM,
                                temperature=0.0, max_tokens=acfg.max_tokens_per_turn)
        except LLMError as exc:
            run.done, run.stop_reason = True, f"model error: {str(exc)[:160]}"
            audit("agent.error", c, {"round": run.round, "error": str(exc)[:300]})
            return {"agent": run.as_dict()}
        run.turns += 1
        run.messages.append(ChatTurn("assistant", resp.text, resp.tool_calls).as_dict())
        audit("agent.turn", c, {"round": run.round, "turn": run.turns, "text": resp.text[:600],
                                "tool_calls": [t.as_dict() for t in resp.tool_calls],
                                "usage": resp.usage})
        if not resp.tool_calls:  # answered in words: treat as done
            run.done, run.stop_reason = True, "finished"
            run.summary = run.summary or resp.text.strip()[:1200]
        return {"agent": run.as_dict()}

    def route_agent(state: BuyState) -> str:
        return "score" if ra.AgentRun.from_dict(state.get("agent")).done else "tools"

    def tools(state: BuyState) -> dict[str, Any]:
        """Run the tool calls from the last model turn. Code checks every call."""
        c = _cat(state)
        run = ra.AgentRun.from_dict(state.get("agent"))
        box = box_for(c)
        last = ChatTurn.from_dict(run.messages[-1])
        for call in last.tool_calls:
            if box is None or run.calls >= acfg.max_tool_calls:
                reply = "error: tool-call limit reached; call finish"
                out = ra.ToolOutcome(reply, {"tool": call.name, "error": "limit"})
                run.done, run.stop_reason = True, "limit reached"
            else:
                out = ra.run_tool(box, call, run)
                run.calls += 1
                run.total_calls += 1
            run.messages.append(ChatTurn("tool", out.reply, tool_call_id=call.id,
                                         name=call.name).as_dict())
            audit("agent.tool", c, {"round": run.round, **out.audit})
            if out.finished:
                run.done, run.stop_reason = True, "finished"
        return {"agent": run.as_dict()}

    def score(state: BuyState) -> dict[str, Any]:
        """Re-score everything the agent found, in code (the agent cannot set it)."""
        c, inp = _cat(state), _inp(state)
        run = ra.AgentRun.from_dict(state.get("agent"))
        box = box_for(c)
        if box is None or not run.ran:
            return {}
        before = inp.evidence.confidence
        upd = box.score([s for s, _ in run.found_pairs()], run.baseline_n)
        inp = ra.update_inputs(inp, upd)
        c = c.model_copy(update={"evidence": inp.evidence})
        audit("agent.score", c, {
            "round": run.round, "stop_reason": run.stop_reason, "tool_calls": run.calls,
            "found": len(run.found), "summary": run.summary,
            "key_evidence": run.key_evidence,
            "confidence_before": before, "confidence": upd.confidence,
            # "authority" alone would be masked by the audit redactor (looks like "auth")
            "source_weight": upd.authority, "density": upd.density,
            "evidence_status": inp.evidence.status.value,
            "inconclusive_reason": inp.evidence.inconclusive_reason,
            "source_hashes": [x.sha256 for x in inp.evidence.sources],
            "tiers": [int(x.tier) for x in inp.evidence.sources],
            "claims": inp.claims})
        return {"catalyst": _dump(c), "inputs": inp.model_dump(mode="json")}

    def route_score(state: BuyState) -> str:
        return "inconclusive" if _inp(state).evidence.status is EvidenceStatus.INCONCLUSIVE \
            else "generate"

    def revise(state: BuyState) -> dict[str, Any]:
        """The Critic rejected nudges: send the agent back to look for better evidence."""
        c = _cat(state)
        run = ra.AgentRun.from_dict(state.get("agent"))
        fb = state.get("feedback") or []
        run.round += 1
        run.turns = run.calls = 0
        run.done, run.stop_reason = False, ""
        run.messages.append(ChatTurn("user", ra.revise_prompt(fb)).as_dict())
        n = state.get("revisions", 0) + 1
        audit("agent.revise", c, {"round": run.round, "revision": n, "feedback": fb})
        # Regenerate the same depth after the new search (generate adds 1).
        return {"agent": run.as_dict(), "revisions": n, "depth": state["depth"] - 1}

    # ------------------------------------------------------------------ tree of thoughts

    def generate(state: BuyState) -> dict[str, Any]:
        c, inp = _cat(state), _inp(state)
        depth = state.get("depth", 0) + 1
        refs = {e.sha256: e.ref for e in research.evidence_catalog(inp)}
        notes = ra.research_notes(ra.AgentRun.from_dict(state.get("agent")), refs)
        fb = [f for f in state.get("feedback") or [] if f.get("depth") == depth]
        draft, trace = research.generate_level(deps.generator_llm, c, inp, depth, cfg=cfg,
                                               rates=deps.rates, notes=notes,
                                               feedback=fb or None)
        audit("tot.generate", c, {**trace.as_dict(), "revision": bool(fb),
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
        # Only a Critic's reasoned "reject" is feedback worth a new search. A missing or
        # unreadable verdict is a model failure, and an uncited nudge is the Generator's
        # problem: neither is fixed by looking for more evidence.
        proposed = {n.node_id: n.adjustment for n in draft if research.is_llm_adjusted(n)}
        feedback = []
        for n in nodes:
            why = n.justification.removeprefix(research.REJECTED)
            if (n.node_id in proposed and n.justification.startswith(research.REJECTED)
                    and why not in research.NOT_FEEDBACK):
                feedback.append({"depth": depth, "node": n.node_id,
                                 "proposed": proposed[n.node_id], "reason": why})
        kept = [n for n in c.tot_nodes if n.depth != depth]
        c = c.model_copy(update={"tot_nodes": [*kept, *nodes]})
        return {"catalyst": _dump(c), "draft": [], "feedback": feedback}

    def route_critique(state: BuyState) -> str:
        c = _cat(state)
        run = ra.AgentRun.from_dict(state.get("agent"))
        if (state.get("feedback") and run.ran and box_for(c) is not None
                and state.get("revisions", 0) < acfg.max_revisions):
            return "revise"
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
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_node("score", score)
    g.add_node("revise", revise)
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
    g.add_edge("ingest", "agent")
    g.add_conditional_edges("agent", route_agent, ["tools", "score"])
    g.add_edge("tools", "agent")
    g.add_conditional_edges("score", route_score, ["generate", "inconclusive"])
    g.add_edge("revise", "agent")
    g.add_edge("generate", "critique")
    g.add_conditional_edges("critique", route_critique, ["revise", "generate", "finalize"])
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
    agent: dict[str, Any] | None = None  # research agent working memory, if it ran


# LangGraph stops a run after this many steps (its default is 25). The research agent's
# loop is bounded by agent.max_turns / max_tool_calls / max_revisions; this is only the
# outer safety net, well above what those limits allow.
RECURSION_LIMIT = 250


def _thread(run_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": run_id}, "recursion_limit": RECURSION_LIMIT}


def _outcome(app: Any, run_id: str) -> RunOutcome:
    snap = app.get_state(_thread(run_id))
    v = snap.values
    return RunOutcome(state=CatalystState.model_validate(v["catalyst"]),
                      awaiting_approval="hitl" in (snap.next or ()),
                      risk=v.get("risk"), execution=v.get("execution"), agent=v.get("agent"))


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
