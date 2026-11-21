"""Research agent: a bounded, tool-using LLM loop that gathers evidence for one catalyst.

This is the one place in BioTradeAI where an LLM decides its own next step. It chooses
which searches to run, which trial records or abstracts to fetch, and when it has
enough evidence. Everything around it stays a fixed workflow:

* **Scope is fixed by code.** Every tool is locked to the ticker and drug entity being
  evaluated; the model cannot name a different company.
* **Read-only to the world.** Fetch tools reach only allowlisted official APIs through
  the existing rate limits and budgets, and add documents to the evidence store through
  the normal ingest pipeline (chunk, tag, claims, embed). Text the model reads is
  always a stored chunk with a hash; it never sees raw web pages.
* **Hard limits.** ``agent.max_tool_calls`` and ``agent.max_turns`` per round.
* **No authority over outcomes.** Confidence is recomputed in code over everything
  found (``rag.retrieve.score_evidence``); the baseline search stays in the pool so
  nothing weak can be hidden. The Generator's nudges stay capped, the Critic reviews
  them, the Risk agent can veto, and a human approves every order.
* **Catalyst dates are read-only.** A disagreeing date found here is reported, never
  written.

The loop itself runs as two LangGraph nodes (``agent`` <-> ``tools`` in
``graph/buy_graph.py``) so every model turn and every tool call is a checkpointed,
audited step. This module holds the parts that need no I/O: tool definitions, prompts,
and how each tool call is checked and turned into a reply for the model.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from biotradeai.agents.llm import ChatTurn, ToolCall, ToolSpec
from biotradeai.schemas import CatalystState, EvidenceBundle, ResearchInputs

NCT = re.compile(r"\bNCT\d{8}\b")
EXCERPT_FOR_AGENT = 420
TIER_NAMES = {1: "regulatory", 2: "SEC", 3: "literature/press"}


# --------------------------------------------------------------------------- toolbox


@dataclass(frozen=True)
class FoundChunk:
    sha256: str
    similarity: float
    tier: int
    published_on: str
    doc_type: str
    title: str
    text: str


@dataclass(frozen=True)
class TrialBrief:
    nct_id: str
    title: str
    phases: tuple[str, ...]
    status: str
    sponsor: str


@dataclass(frozen=True)
class EvidenceUpdate:
    """Evidence re-scored in code over everything the agent found."""

    evidence: EvidenceBundle
    excerpts: dict[str, str]
    claims: dict[str, str]
    authority: float
    density: float
    confidence: float


class Toolbox(Protocol):
    """What the agent may do. Implemented with real I/O in ``paper/agent_tools.py``.

    ``live`` is False in backtests and when the drug is not on the watchlist: then only
    ``search`` (the local index, point-in-time) is offered.
    """

    live: bool
    drug_names: tuple[str, ...]

    def baseline(self) -> list[FoundChunk]: ...
    def search(self, query: str) -> list[FoundChunk]: ...
    def known_trials(self) -> list[str]: ...
    def list_trials(self) -> list[TrialBrief]: ...
    def fetch_trial(self, nct_id: str) -> str: ...
    def fetch_pubmed(self, query: str) -> str: ...
    def score(self, shas: Sequence[str], baseline_n: int) -> EvidenceUpdate:
        """Re-score in code. ``shas`` are all chunks found, the first ``baseline_n``
        being the baseline search. Relevance is measured against the fixed decision
        question, never the agent's own query wording."""
        ...


ToolboxFactory = Callable[[CatalystState], "Toolbox | None"]


# --------------------------------------------------------------------------- tools

SEARCH = ToolSpec(
    "search_evidence",
    "Search the stored evidence for this drug (SEC risk factors and 8-Ks, "
    "ClinicalTrials.gov records, PubMed abstracts). Results become evidence you can "
    "cite as C1, C2, ... Only results returned by this tool count as evidence.",
    {"type": "object", "properties": {"query": {
        "type": "string", "description": "What to look for, in plain words"}},
     "required": ["query"]})
LIST_TRIALS = ToolSpec(
    "list_trials",
    "List ClinicalTrials.gov studies whose intervention is this drug. Use it to find "
    "NCT ids for fetch_trial.",
    {"type": "object", "properties": {}})
FETCH_TRIAL = ToolSpec(
    "fetch_trial",
    "Fetch one ClinicalTrials.gov record (Tier 1, the most authoritative source) and "
    "add it to the evidence store. Then use search_evidence to read it.",
    {"type": "object", "properties": {"nct_id": {
        "type": "string", "description": "e.g. NCT01234567, from list_trials or evidence"}},
     "required": ["nct_id"]})
FETCH_PUBMED = ToolSpec(
    "fetch_pubmed",
    "Fetch recent PubMed abstracts that name this drug, optionally narrowed by extra "
    "words, and add them to the evidence store (Tier 3). Then use search_evidence.",
    {"type": "object", "properties": {"query": {
        "type": "string", "description": "Optional extra words, e.g. 'phase 3 topline'"}}})
FINISH = ToolSpec(
    "finish",
    "Stop searching. Give a short factual summary and the chunk ids that matter most.",
    {"type": "object", "properties": {
        "summary": {"type": "string", "description": "2-4 sentences, facts only"},
        "key_evidence": {"type": "array", "items": {"type": "string"},
                         "description": "chunk ids such as C2"}},
     "required": ["summary", "key_evidence"]})


def tools_for(box: Toolbox) -> list[ToolSpec]:
    return [SEARCH, LIST_TRIALS, FETCH_TRIAL, FETCH_PUBMED, FINISH] if box.live \
        else [SEARCH, FINISH]


SYSTEM = """\
You are the RESEARCH AGENT for one biotech catalyst. Your job is to gather the evidence
a later analysis will rely on: facts about this drug's trial results, safety, regulatory
status and the catalyst itself.

How to work:
- Start from the baseline results you are given. Look for what is missing, especially
  authoritative sources: a ClinicalTrials.gov record (tier 1) beats an SEC filing
  (tier 2), which beats literature or press (tier 3).
- Use search_evidence with focused queries. Use the fetch tools only when the store
  clearly lacks a source; then search again to read what was added.
- Report negative or contradicting evidence as readily as positive evidence.
- You are limited to this ticker and drug. You cannot trade, change any date, or
  compute probabilities, returns or position sizes; code does that.
- When you have enough, or nothing more is useful, call finish. Prefer few calls."""


def first_prompt(c: CatalystState, names: Sequence[str], baseline: str) -> str:
    return (f"Ticker: {c.ticker}\nDrug entity: {c.entity_id} (names: {', '.join(names)})\n"
            f"Catalyst: {c.catalyst_event} on {c.catalyst_date.isoformat()}\n\n"
            f"Baseline search results:\n{baseline}\n\n"
            "Gather any missing evidence, then call finish.")


def revise_prompt(feedback: Sequence[dict[str, Any]]) -> str:
    lines = [f"- {f['node']} (proposed {f['proposed']:+.3f}): {f['reason']}" for f in feedback]
    return ("An independent critic rejected these proposed probability changes because the "
            "cited evidence did not support them:\n" + "\n".join(lines) + "\n\n"
            "Look for evidence that settles these points either way (supporting OR "
            "contradicting). If none exists, say so. Then call finish.")


# --------------------------------------------------------------------------- run state


@dataclass
class AgentRun:
    """The agent's working memory, stored in graph state as JSON between steps."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    found: list[list[Any]] = field(default_factory=list)  # [[sha256, similarity], ...]
    handles: dict[str, str] = field(default_factory=dict)  # sha256 -> "C1"
    known_ncts: list[str] = field(default_factory=list)
    baseline_n: int = 0  # the first N found chunks are the baseline search
    turns: int = 0  # this round
    calls: int = 0  # this round
    total_calls: int = 0
    round: int = 0
    done: bool = True
    ran: bool = False
    summary: str = ""
    key_evidence: list[str] = field(default_factory=list)
    stop_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> AgentRun:
        return cls(**(d or {}))

    def turn_objs(self) -> list[ChatTurn]:
        return [ChatTurn.from_dict(m) for m in self.messages]

    def handle(self, sha: str) -> str:
        if sha not in self.handles:
            self.handles[sha] = f"C{len(self.handles) + 1}"
        return self.handles[sha]

    def add_found(self, chunks: Sequence[FoundChunk]) -> None:
        best = {s: float(v) for s, v in self.found}
        order = [s for s, _ in self.found]
        for ch in chunks:
            if ch.sha256 not in best:
                order.append(ch.sha256)
                best[ch.sha256] = ch.similarity
            else:
                best[ch.sha256] = max(best[ch.sha256], ch.similarity)
            self.handle(ch.sha256)
            for n in NCT.findall(ch.text):
                if n not in self.known_ncts:
                    self.known_ncts.append(n)
        self.found = [[s, best[s]] for s in order]

    def found_pairs(self) -> list[tuple[str, float]]:
        return [(str(s), float(v)) for s, v in self.found]


def format_chunks(run: AgentRun, chunks: Sequence[FoundChunk]) -> str:
    if not chunks:
        return "no matching evidence"
    out = []
    for ch in chunks:
        text = " ".join(ch.text.split())[:EXCERPT_FOR_AGENT]
        out.append(f"[{run.handle(ch.sha256)}] tier {ch.tier} ({TIER_NAMES.get(ch.tier, '?')})"
                   f" · {ch.doc_type} · {ch.published_on} · similarity {ch.similarity:.2f}\n"
                   f"  {ch.title[:90]}\n  {text}")
    return "\n".join(out)


# --------------------------------------------------------------------------- one tool call


@dataclass(frozen=True)
class ToolOutcome:
    reply: str  # what the model reads
    audit: dict[str, Any]  # what the audit log records
    finished: bool = False


def run_tool(box: Toolbox, call: ToolCall, run: AgentRun) -> ToolOutcome:
    """Check and run one tool call. Errors become replies the model can react to."""
    name, args = call.name, call.arguments
    offered = {t.name for t in tools_for(box)}
    if name not in offered:
        return ToolOutcome(f"error: tool {name!r} is not available here",
                           {"tool": name, "error": "not offered"})
    try:
        if name == "search_evidence":
            q = str(args.get("query", "")).strip()[:300]
            if not q:
                return ToolOutcome("error: give a query", {"tool": name, "error": "empty"})
            hits = box.search(q)
            run.add_found(hits)
            return ToolOutcome(format_chunks(run, hits), {
                "tool": name, "query": q, "results": len(hits),
                "chunks": [run.handle(h.sha256) for h in hits]})
        if name == "list_trials":
            trials = box.list_trials()
            for t in trials:
                if t.nct_id not in run.known_ncts:
                    run.known_ncts.append(t.nct_id)
            text = "\n".join(f"{t.nct_id} · {'/'.join(t.phases) or 'phase n/a'} · {t.status}"
                             f" · {t.sponsor} · {t.title[:90]}" for t in trials)
            return ToolOutcome(text or "no trials list this drug as an intervention",
                               {"tool": name, "trials": [t.nct_id for t in trials]})
        if name == "fetch_trial":
            nct = str(args.get("nct_id", "")).strip().upper()
            known = set(run.known_ncts) | set(box.known_trials())
            if nct not in known:
                return ToolOutcome(
                    f"error: {nct or 'that id'} is not a known trial for this drug; call "
                    "list_trials or use an NCT id that appears in the evidence",
                    {"tool": name, "nct_id": nct, "error": "unknown trial"})
            msg = box.fetch_trial(nct)
            return ToolOutcome(msg, {"tool": name, "nct_id": nct, "result": msg[:200]})
        if name == "fetch_pubmed":
            q = str(args.get("query", "") or "").strip()[:200]
            msg = box.fetch_pubmed(q)
            return ToolOutcome(msg, {"tool": name, "query": q, "result": msg[:200]})
        # finish
        summary = str(args.get("summary", "") or "").strip()[:1200]
        if not summary:  # some models put their summary in the text, not the arguments
            said = [m.get("content", "") for m in run.messages if m.get("role") == "assistant"]
            summary = next((s.strip()[:1200] for s in reversed(said) if s.strip()), "")
        by_handle = {v: k for k, v in run.handles.items()}
        keys = [str(k).strip().strip("[]") for k in args.get("key_evidence") or []]
        run.summary = summary or "no summary given"
        run.key_evidence = [k for k in keys if k in by_handle]
        return ToolOutcome("ok", {"tool": name, "summary": run.summary,
                                  "key_evidence": run.key_evidence}, finished=True)
    except Exception as exc:  # network, validation: the model is told, the run goes on
        return ToolOutcome(f"error: {str(exc)[:200]}", {"tool": name, "error": str(exc)[:300]})


def research_notes(run: AgentRun | None, refs: dict[str, str]) -> str | None:
    """The agent's summary, shown to the Generator as notes (never as evidence).

    The agent cites chunks as C1, C2 ...; the Generator sees evidence as E1, E2 ...
    (``refs``: sha256 -> E ref). Handles are rewritten, and ones that did not make it
    into the evidence set are marked so the Generator cannot cite them.
    """
    if run is None or not run.ran or not run.summary:
        return None
    to_e = {h: refs.get(sha, "not in evidence") for sha, h in run.handles.items()}
    text = re.sub(r"\bC\d+\b", lambda m: to_e.get(m.group(0), m.group(0)), run.summary)
    keys = [to_e[k] for k in run.key_evidence if to_e.get(k, "").startswith("E")]
    return text + (f" Key evidence: {', '.join(keys)}." if keys else "")


def update_inputs(inputs: ResearchInputs, upd: EvidenceUpdate) -> ResearchInputs:
    return ResearchInputs.model_validate({
        **inputs.model_dump(mode="json"),
        "evidence": upd.evidence.model_dump(mode="json"),
        "excerpts": upd.excerpts, "claims": upd.claims})
