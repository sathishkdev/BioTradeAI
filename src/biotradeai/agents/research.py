"""Clinical Research Agent: Tree-of-Thoughts Generator and Critic.

Depth 1 = clinical outcome (success / partial / failure, all three always enumerated)
Depth 2 = regulatory reaction, conditional on the clinical outcome
Depth 3 = financing / dilution, bucketed by runway (computed in code)

What the LLM may do: propose a bounded adjustment (+/- ``tot.max_prob_adjustment``) to a
base-rate probability, with a justification and cited evidence ids. Nothing else.
What code does: base rates, returns, the bounded projection that keeps siblings summing
to 1, fatal-flaw pruning, the beam, scenario flattening and EV (``agents/tot.py``).

Generator and Critic use separate prompts and are separate LangGraph nodes. The Critic
can only make the tree MORE conservative: it can reject an adjustment (back to the base
rate) and it applies the deterministic fatal-flaw prune. It never adds an adjustment.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from biotradeai.agents import tot
from biotradeai.agents.base_rates import BaseRates, Outcome
from biotradeai.agents.llm import LLMClient, LLMError, LLMMessage
from biotradeai.config import GuardrailsConfig
from biotradeai.guardrails import rules
from biotradeai.schemas import CatalystState, ResearchInputs, ScenarioSet, ToTNode

MAX_EXCERPT_CHARS = 500
LEVEL_NAMES = {1: "clinical outcome", 2: "regulatory reaction", 3: "financing / dilution"}
MAX_DEPTH = len(LEVEL_NAMES)


# --------------------------------------------------------------------------- evidence


@dataclass(frozen=True)
class EvidenceItem:
    ref: str  # short id shown to the LLM, e.g. "E1"
    sha256: str
    tier: int
    published_at: str
    url: str
    excerpt: str


def evidence_catalog(inputs: ResearchInputs) -> list[EvidenceItem]:
    out = []
    for i, s in enumerate(inputs.evidence.sources, start=1):
        text = inputs.excerpts.get(s.sha256, "")
        out.append(EvidenceItem(f"E{i}", s.sha256, int(s.tier), s.published_at.date().isoformat(),
                                s.url, " ".join(text.split())[:MAX_EXCERPT_CHARS]))
    return out


# --------------------------------------------------------------------------- candidates


@dataclass(frozen=True)
class Candidate:
    node_id: str
    parent_id: str | None
    depth: int
    label: str
    base_rate: float
    step_return: float
    description: str


def _table(rates: BaseRates, event: str, depth: int, parent: ToTNode | None,
           runway_months: float) -> dict[str, Outcome]:
    ev = rates.event(event)
    if depth == 1:
        return ev.clinical
    if parent is None:
        raise tot.TreeError(f"depth {depth} needs a parent")
    if depth == 2:
        return ev.regulatory[parent.node_id.split("/")[0]]
    if depth == 3:
        return rates.financing_for(runway_months)
    raise tot.TreeError(f"no base-rate level for depth {depth}")


def candidates_for_depth(nodes: list[ToTNode], depth: int, *, rates: BaseRates, event: str,
                         runway_months: float) -> list[Candidate]:
    parents: list[ToTNode | None] = [None]
    if depth > 1:
        parents = [*tot.frontier(nodes, depth - 1)]
    out = []
    for p in parents:
        for label, o in _table(rates, event, depth, p, runway_months).items():
            nid = label if p is None else f"{p.node_id}/{label}"
            out.append(Candidate(nid, None if p is None else p.node_id, depth, label, o.p,
                                 o.return_, o.description))
    return out


# --------------------------------------------------------------------------- LLM plumbing


def extract_json(text: str) -> Any | None:
    """First JSON object in a reply (tolerates ```json fences and chatter)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class _Loose(BaseModel):
    model_config = ConfigDict(extra="ignore")


class _Adjustment(_Loose):
    node: str
    adjustment: float
    justification: str = ""
    evidence: list[str] = []


class _GeneratorReply(_Loose):
    adjustments: list[_Adjustment]


class _Review(_Loose):
    node: str
    verdict: str
    reason: str = ""


class _CriticReply(_Loose):
    reviews: list[_Review]


GENERATOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"adjustments": {"type": "array", "items": {
        "type": "object",
        "properties": {"node": {"type": "string"}, "adjustment": {"type": "number"},
                       "justification": {"type": "string"},
                       "evidence": {"type": "array", "items": {"type": "string"}}},
        "required": ["node", "adjustment", "justification", "evidence"]}}},
    "required": ["adjustments"],
}
CRITIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"reviews": {"type": "array", "items": {
        "type": "object",
        "properties": {"node": {"type": "string"},
                       "verdict": {"type": "string", "enum": ["accept", "reject"]},
                       "reason": {"type": "string"}},
        "required": ["node", "verdict", "reason"]}}},
    "required": ["reviews"],
}

GENERATOR_SYSTEM = """\
You are the GENERATOR in a Tree-of-Thoughts analysis of a biotech catalyst.
Each candidate outcome already has a base-rate probability from industry data.
Your only job: propose small adjustments to those probabilities where the evidence
clearly supports it.
Rules:
- Each adjustment is a number between -{d} and +{d}. 0 keeps the base rate. Prefer 0.
- Every non-zero adjustment MUST cite evidence ids from the list (e.g. "E2") and give a
  one-sentence justification that the cited text supports.
- Do not compute expected values, returns, position sizes or runway; code does that.
- Probabilities are re-balanced by code so siblings still sum to 1.
Reply with JSON only:
{{"adjustments": [{{"node": "<node id>", "adjustment": 0.0, "justification": "...",
  "evidence": ["E1"]}}]}}"""

CRITIC_SYSTEM = """\
You are the CRITIC in a Tree-of-Thoughts analysis of a biotech catalyst.
You review probability adjustments proposed by a separate generator.
For each proposed adjustment answer "accept" or "reject".
Reject when the cited evidence text does not support the justification, overstates it,
is about a different drug or company, or when the same evidence is double counted.
You cannot propose new adjustments or change their size.
Reply with JSON only:
{"reviews": [{"node": "<node id>", "verdict": "accept", "reason": "..."}]}"""


def _header(state: CatalystState, inputs: ResearchInputs, depth: int) -> str:
    claims = ", ".join(f"{k}={v}" for k, v in sorted(inputs.claims.items())) or "none"
    return (f"Ticker: {state.ticker}\nDrug entity: {state.entity_id}\n"
            f"Catalyst: {state.catalyst_event} on {state.catalyst_date.isoformat()}\n"
            f"Cash runway (computed in code): {inputs.runway.months:.1f} months\n"
            f"Resolved source claims: {claims}\n"
            f"Depth: {depth} ({LEVEL_NAMES[depth]})\n")


def _evidence_block(catalog: list[EvidenceItem]) -> str:
    if not catalog:
        return "Evidence: none\n"
    lines = [f"[{e.ref}] tier {e.tier}, {e.published_at}: {e.excerpt}" for e in catalog]
    return "Evidence:\n" + "\n".join(lines) + "\n"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _call(llm: LLMClient, system: str, prompt: str,
          schema: dict[str, Any]) -> tuple[str | None, str | None]:
    try:
        r = llm.complete([LLMMessage("user", prompt)], system=system, temperature=0.0,
                         max_tokens=1200, json_schema=schema)
    except LLMError as exc:
        return None, f"llm error: {exc}"
    return r.text, None


# --------------------------------------------------------------------------- generator


@dataclass
class LevelTrace:
    """What gets audit-logged for one Generator or Critic step."""

    role: str
    depth: int
    llm: str
    prompt_sha256: str
    reply: str | None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"role": self.role, "depth": self.depth, "llm": self.llm,
                "prompt_sha256": self.prompt_sha256,
                "reply": (self.reply or "")[:4000], "notes": self.notes}


BASE_RATE = "base rate"
REBALANCED = "base rate (moved only by rebalancing so siblings sum to 1)"
REJECTED = "critic rejected: "
NO_EVIDENCE = "no valid evidence cited"
NO_VERDICT = "critic gave no verdict"
# Rejections that are not the Critic's judgement of the evidence (see buy_graph.critique).
NOT_FEEDBACK = frozenset({NO_EVIDENCE, NO_VERDICT})


def is_llm_adjusted(n: ToTNode) -> bool:
    """True when the probability moved because the LLM proposed it (and it still stands).
    Moves caused only by code rebalancing, or by a rejected proposal, are not claims."""
    return abs(n.adjustment) > 1e-12 and not n.justification.startswith((BASE_RATE, REJECTED))


def _make_nodes(cands: list[Candidate], adj: dict[str, float], just: dict[str, str],
                evid: dict[str, list[str]], max_adjustment: float) -> list[ToTNode]:
    """Project each sibling group so it sums to 1 within the allowed band."""
    out: list[ToTNode] = []
    groups: dict[str | None, list[Candidate]] = {}
    for c in cands:
        groups.setdefault(c.parent_id, []).append(c)
    for group in groups.values():
        probs = tot.bounded_projection([c.base_rate for c in group],
                                       [adj.get(c.node_id, 0.0) for c in group], max_adjustment)
        for c, p in zip(group, probs, strict=True):
            out.append(ToTNode(
                node_id=c.node_id, parent_id=c.parent_id, depth=c.depth, label=c.label,
                base_rate=c.base_rate, adjustment=max(-1.0, min(1.0, p - c.base_rate)),
                probability=p, step_return=c.step_return,
                justification=just.get(c.node_id, BASE_RATE if abs(p - c.base_rate) < 1e-12
                                       else REBALANCED),
                evidence_ids=evid.get(c.node_id, [])))
    return out


def generate_level(llm: LLMClient, state: CatalystState, inputs: ResearchInputs, depth: int,
                   *, cfg: GuardrailsConfig, rates: BaseRates, notes: str | None = None,
                   feedback: list[dict[str, Any]] | None = None
                   ) -> tuple[list[ToTNode], LevelTrace]:
    """Generator step: candidate children for every surviving node at ``depth - 1``.

    ``notes``: the research agent's summary (context, never citable evidence).
    ``feedback``: the Critic's reasons for rejecting this depth's previous proposals,
    when the research agent was sent back for better evidence (a revision round).
    """
    d = cfg.tot.max_prob_adjustment
    cands = candidates_for_depth(list(state.tot_nodes), depth, rates=rates,
                                 event=state.catalyst_event, runway_months=inputs.runway.months)
    catalog = evidence_catalog(inputs)
    ref_to_sha = {e.ref: e.sha256 for e in catalog}
    lines = [f"- node {c.node_id}: {c.description}; base rate {c.base_rate:.3f}"
             + (f" (given {c.parent_id})" if c.parent_id else "") for c in cands]
    prompt = (_header(state, inputs, depth) + _evidence_block(catalog)
              + "Candidates:\n" + "\n".join(lines) + "\n")
    if notes:
        prompt += ("Research notes (summary by the research agent; context only, cite E ids "
                   f"from the evidence list, never these notes):\n{notes}\n")
    if feedback:
        prompt += ("A critic rejected earlier proposals at this depth:\n"
                   + "\n".join(f"- {f['node']}: {f['reason']}" for f in feedback)
                   + "\nRe-propose only where the evidence above now supports it.\n")
    reply, err = _call(llm, GENERATOR_SYSTEM.format(d=d), prompt, GENERATOR_SCHEMA)
    trace = LevelTrace("generator", depth, f"{llm.provider}:{llm.model}", _sha(prompt), reply)
    if err:
        trace.notes.append(err)

    adj: dict[str, float] = {}
    just: dict[str, str] = {}
    evid: dict[str, list[str]] = {}
    parsed: _GeneratorReply | None = None
    if reply is not None:
        raw = extract_json(reply)
        try:
            parsed = _GeneratorReply.model_validate(raw) if raw is not None else None
        except ValidationError:
            parsed = None
        if parsed is None:
            trace.notes.append("unparseable generator reply: using base rates")
    known = {c.node_id for c in cands}
    for a in parsed.adjustments if parsed else []:
        if a.node not in known:
            trace.notes.append(f"ignored adjustment for unknown node {a.node!r}")
            continue
        value = a.adjustment
        if abs(value) > d:
            trace.notes.append(f"{a.node}: adjustment {value:+.3f} clipped to +/-{d}")
            value = max(-d, min(d, value))
        refs = [r.strip().strip("[]") for r in a.evidence]
        bad = [r for r in refs if r not in ref_to_sha]
        if bad:
            trace.notes.append(f"{a.node}: dropped unknown evidence ids {bad}")
        adj[a.node] = value
        just[a.node] = a.justification.strip()[:500] or "no justification given"
        evid[a.node] = sorted({ref_to_sha[r] for r in refs if r in ref_to_sha})
    return _make_nodes(cands, adj, just, evid, d), trace


# --------------------------------------------------------------------------- critic


def fatal_flaws(inputs: ResearchInputs, cfg: GuardrailsConfig) -> list[str]:
    """Deterministic thesis killers. Any of them collapses every non-failure branch."""
    reasons = []
    if inputs.claims.get("clinical_hold") == "active":
        reasons.append("active clinical hold")
    if rules.is_fatal_runway(inputs.runway.months, cfg):
        reasons.append(f"runway {inputs.runway.months:.2f} mo < fatal "
                       f"{cfg.fatal_runway_months} mo")
    return reasons


def _with(n: ToTNode, **kw: Any) -> ToTNode:
    return n.model_copy(update=kw)


def critique_level(llm: LLMClient, state: CatalystState, inputs: ResearchInputs, depth: int,
                   draft: list[ToTNode], *,
                   cfg: GuardrailsConfig) -> tuple[list[ToTNode], LevelTrace]:
    """Critic step for the ``draft`` nodes produced by ``generate_level`` at ``depth``."""
    d = cfg.tot.max_prob_adjustment
    adjusted = [n for n in draft if is_llm_adjusted(n)]
    catalog = {e.sha256: e for e in evidence_catalog(inputs)}
    trace = LevelTrace("critic", depth, f"{llm.provider}:{llm.model}", "", None)

    rejected: dict[str, str] = {}
    # Groundedness rule (deterministic): no evidence, no adjustment.
    for n in adjusted:
        if not n.evidence_ids:
            rejected[n.node_id] = NO_EVIDENCE
    to_review = [n for n in adjusted if n.node_id not in rejected]
    if to_review:
        lines = []
        for n in to_review:
            cites = "; ".join(f"[{catalog[s].ref}] {catalog[s].excerpt}" for s in n.evidence_ids
                              if s in catalog)
            lines.append(f"- node {n.node_id}: base {n.base_rate:.3f}, proposed "
                         f"{n.adjustment:+.3f}. Justification: {n.justification}. "
                         f"Cited: {cites}")
        prompt = _header(state, inputs, depth) + "Proposed adjustments:\n" + "\n".join(lines)
        reply, err = _call(llm, CRITIC_SYSTEM, prompt, CRITIC_SCHEMA)
        trace = LevelTrace("critic", depth, f"{llm.provider}:{llm.model}", _sha(prompt), reply)
        verdicts: dict[str, tuple[str, str]] = {}
        if err:
            trace.notes.append(err)
        raw = extract_json(reply) if reply is not None else None
        try:
            parsed = _CriticReply.model_validate(raw) if raw is not None else None
        except ValidationError:
            parsed = None
        if parsed is None:
            trace.notes.append("no usable critic reply: every adjustment rejected")
        else:
            verdicts = {r.node: (r.verdict.strip().lower(), r.reason) for r in parsed.reviews}
        for n in to_review:
            v, why = verdicts.get(n.node_id, ("missing", NO_VERDICT))
            if v != "accept":
                rejected[n.node_id] = why or v

    # Re-project each sibling group with only the accepted adjustments.
    groups: dict[str | None, list[ToTNode]] = {}
    for n in draft:
        groups.setdefault(n.parent_id, []).append(n)
    out: list[ToTNode] = []
    for group in groups.values():
        proposed = [n.adjustment if is_llm_adjusted(n) and n.node_id not in rejected
                    else 0.0 for n in group]
        probs = tot.bounded_projection([n.base_rate for n in group], proposed, d)
        for n, p in zip(group, probs, strict=True):
            upd: dict[str, Any] = {"probability": p,
                                   "adjustment": max(-1.0, min(1.0, p - n.base_rate))}
            if n.node_id in rejected:
                upd["justification"] = f"{REJECTED}{rejected[n.node_id]}"[:500]
                upd["evidence_ids"] = []
            elif not is_llm_adjusted(n):
                upd["justification"] = BASE_RATE if abs(p - n.base_rate) < 1e-12 else REBALANCED
            out.append(_with(n, **upd))
    trace.notes += [f"rejected {k}: {v}" for k, v in sorted(rejected.items())]

    if depth == 1:
        flaws = fatal_flaws(inputs, cfg)
        if flaws:
            why = "fatal flaw: " + "; ".join(flaws)
            out = [_with(n, pruned=True, prune_reason=why) if n.label != "failure" else n
                   for n in out]
            trace.notes.append(why)
        alive = sorted((n for n in out if not n.pruned),
                       key=lambda n: (-n.probability, n.label))
        keep = {n.node_id for n in alive[:cfg.tot.beam_width]}
        out = [_with(n, pruned=True, prune_reason=f"outside beam (width {cfg.tot.beam_width})")
               if not n.pruned and n.node_id not in keep else n for n in out]
    return out, trace


# --------------------------------------------------------------------------- finalize


def max_depth(cfg: GuardrailsConfig) -> int:
    return min(cfg.tot.depth, MAX_DEPTH)


def finalize(nodes: list[ToTNode], cfg: GuardrailsConfig) -> tuple[ScenarioSet, float]:
    """Flatten into a ScenarioSet and compute the Research Agent's own EV."""
    scenarios = tot.build_scenarios(nodes, residual_return=cfg.tot.residual_scenario_return,
                                    tolerance=cfg.tot.prob_sum_tolerance)
    return scenarios, tree_ev(nodes, cfg)


def tree_ev(nodes: list[ToTNode], cfg: GuardrailsConfig) -> float:
    return tot.tree_ev(nodes, cfg.tot.residual_scenario_return)
