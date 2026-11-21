"""Research agent: tool calling, tool checks, the agent <-> tools loop in the buy graph,
code-only re-scoring, limits, Critic feedback rounds and the real toolbox. All offline."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import chromadb
import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from biotradeai.agents import research_agent as ra
from biotradeai.agents.base_rates import BaseRates, load_base_rates
from biotradeai.agents.llm import (
    AnthropicClient,
    ChatTurn,
    FakeLLMClient,
    LLMError,
    LLMMessage,
    OllamaClient,
    ToolCall,
    ToolResponse,
    ToolSpec,
)
from biotradeai.config import GuardrailsConfig, load_config
from biotradeai.data.clinicaltrials import TrialRecord
from biotradeai.graph.buy_graph import BuyGraphDeps, build_buy_graph, start_run
from biotradeai.paper.agent_tools import EvidenceToolbox
from biotradeai.paths import REPO_ROOT
from biotradeai.rag.embeddings import HashingEmbedder
from biotradeai.rag.index import VectorIndex
from biotradeai.rag.ingest import Document, ingest_documents
from biotradeai.rag.ontology import Entity, Ontology
from biotradeai.rag.retrieve import Retriever
from biotradeai.scenario_fixtures import load_fixture
from biotradeai.schemas import CatalystState, Decision, EvidenceStatus, SourceTier
from biotradeai.store.sqlite_store import SQLiteStore

FIX = REPO_ROOT / "data" / "fixtures" / "scenarios"
CFG = load_config()


@pytest.fixture(scope="module")
def rates() -> BaseRates:
    return load_base_rates()


@pytest.fixture
def store(tmp_path: Any) -> SQLiteStore:
    return SQLiteStore(tmp_path / "a.db")


# ------------------------------------------------------------------ LLM tool calling


def test_fake_client_scripts_tool_calls() -> None:
    llm = FakeLLMClient(tool_replies=[{"text": "look", "tool_calls": [
        {"name": "search_evidence", "arguments": {"query": "x"}}]}])
    r = llm.converse([ChatTurn("user", "hi")], tools=[ra.SEARCH])
    assert r.text == "look" and r.tool_calls[0].name == "search_evidence"
    assert r.tool_calls[0].arguments == {"query": "x"}
    assert llm.converse([ChatTurn("user", "hi")], tools=[ra.SEARCH]).tool_calls == ()


def test_chat_turn_round_trips_through_json() -> None:
    t = ChatTurn("assistant", "t", (ToolCall("c1", "finish", {"summary": "s"}),))
    assert ChatTurn.from_dict(json.loads(json.dumps(t.as_dict()))) == t


def test_anthropic_tool_use_translation() -> None:
    seen: dict[str, Any] = {}

    def create(**kw: Any) -> Any:
        seen.update(kw)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="searching"),
                     SimpleNamespace(type="tool_use", id="tu1", name="search_evidence",
                                     input={"query": "phase 3"})],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5))

    client = AnthropicClient("m", client=SimpleNamespace(messages=SimpleNamespace(create=create)))
    turns = [ChatTurn("user", "go"),
             ChatTurn("assistant", "", (ToolCall("a", "search_evidence", {"query": "q"}),
                                        ToolCall("b", "list_trials", {}))),
             ChatTurn("tool", "r1", tool_call_id="a", name="search_evidence"),
             ChatTurn("tool", "r2", tool_call_id="b", name="list_trials")]
    r = client.converse(turns, tools=[ra.SEARCH], system="sys")
    assert r.text == "searching" and r.tool_calls == (
        ToolCall("tu1", "search_evidence", {"query": "phase 3"}),)
    assert seen["tools"][0]["input_schema"]["required"] == ["query"]
    msgs = seen["messages"]
    assert msgs[1]["content"][0]["type"] == "tool_use"
    # both tool results travel back in ONE user message
    assert msgs[2]["role"] == "user" and [b["tool_use_id"] for b in msgs[2]["content"]] == \
        ["a", "b"]


def test_ollama_tool_call_translation() -> None:
    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return httpx.Response(200, json={"message": {"content": "", "tool_calls": [
            {"function": {"name": "fetch_trial", "arguments": {"nct_id": "NCT01234567"}}}]}})

    c = OllamaClient("m", host="http://ollama.test", transport=httpx.MockTransport(handler))
    r = c.converse([ChatTurn("user", "go")], tools=[ra.FETCH_TRIAL])
    assert r.tool_calls[0].name == "fetch_trial"
    assert r.tool_calls[0].arguments == {"nct_id": "NCT01234567"}
    assert seen[0]["tools"][0]["function"]["name"] == "fetch_trial"


def test_tool_turns_are_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_API_SECRET", "supersecretvalue123")
    llm = FakeLLMClient()
    llm.converse([ChatTurn("tool", "key supersecretvalue123", tool_call_id="x")], tools=[])
    assert "supersecretvalue123" not in llm.tool_calls_seen[0][0].content


# ------------------------------------------------------------------ fake toolbox


def _chunk(sha: str, sim: float, tier: int = 2, text: str = "evidence text") -> ra.FoundChunk:
    return ra.FoundChunk(sha, sim, tier, "2025-01-01", "10-Q", "title", text)


@dataclass
class FakeBox:
    """Toolbox double. ``score`` returns ``upgraded`` once any search ran."""

    base: list[ra.FoundChunk]
    baseline_update: ra.EvidenceUpdate
    upgraded: ra.EvidenceUpdate | None = None
    live: bool = True
    drug_names: tuple[str, ...] = ("ONC-101",)
    searches: list[str] = field(default_factory=list)
    fetched: list[str] = field(default_factory=list)
    scored: list[tuple[list[str], int]] = field(default_factory=list)

    def baseline(self) -> list[ra.FoundChunk]:
        return list(self.base)

    def search(self, query: str) -> list[ra.FoundChunk]:
        self.searches.append(query)
        return [_chunk("f" * 64, 0.9, tier=1, text="NCT07654321 trial record")]

    def known_trials(self) -> list[str]:
        return []

    def list_trials(self) -> list[ra.TrialBrief]:
        return [ra.TrialBrief("NCT01234567", "A trial", ("PHASE3",), "COMPLETED", "Onco")]

    def fetch_trial(self, nct_id: str) -> str:
        self.fetched.append(nct_id)
        return f"added {nct_id}"

    def fetch_pubmed(self, query: str) -> str:
        return "no abstracts found"

    def score(self, shas: Sequence[str], baseline_n: int) -> ra.EvidenceUpdate:
        self.scored.append((list(shas), baseline_n))
        return self.upgraded if (self.searches and self.upgraded) else self.baseline_update


def _update_from(inputs: Any, *, confidence: float | None = None,
                 status: EvidenceStatus | None = None) -> ra.EvidenceUpdate:
    ev = inputs.evidence.model_copy(update={
        k: v for k, v in (("confidence", confidence), ("status", status),
                          ("inconclusive_reason", None if status is EvidenceStatus.OK
                           else inputs.evidence.inconclusive_reason))
        if v is not None or k == "inconclusive_reason"})
    return ra.EvidenceUpdate(ev, dict(inputs.excerpts), dict(inputs.claims), 0.9, 1.0,
                             ev.confidence)


def _graph(rates: BaseRates, store: SQLiteStore, box: Any, agent: Any, llm: Any = None,
           cfg: GuardrailsConfig = CFG) -> Any:
    fx = load_fixture(FIX / "onco.yaml", cfg)
    deps = BuyGraphDeps(cfg, rates, llm or fx.fake_llm(), llm or fx.fake_llm(), store,
                        executor=lambda *a: {}, clock=lambda: fx.catalyst.as_of,
                        agent_llm=agent, toolbox=(lambda c: box) if box is not None else None)
    return fx, build_buy_graph(deps, InMemorySaver())


def _events(store: SQLiteStore, run_id: str) -> list[str]:
    return [e.event_type for e in store.read_audit(run_id=run_id, limit=2000)]


# ------------------------------------------------------------------ run_tool checks


def test_search_only_box_refuses_fetch_tools() -> None:
    box = FakeBox([], None, live=False)  # type: ignore[arg-type]
    assert [t.name for t in ra.tools_for(box)] == ["search_evidence", "finish"]
    out = ra.run_tool(box, ToolCall("1", "fetch_trial", {"nct_id": "NCT01234567"}),
                      ra.AgentRun())
    assert out.reply.startswith("error") and box.fetched == []


def test_fetch_trial_needs_a_known_trial_id() -> None:
    box = FakeBox([], None)  # type: ignore[arg-type]
    run = ra.AgentRun()
    out = ra.run_tool(box, ToolCall("1", "fetch_trial", {"nct_id": "NCT09999999"}), run)
    assert "not a known trial" in out.reply and box.fetched == []
    ra.run_tool(box, ToolCall("2", "list_trials", {}), run)
    out = ra.run_tool(box, ToolCall("3", "fetch_trial", {"nct_id": "nct01234567"}), run)
    assert box.fetched == ["NCT01234567"] and "added" in out.reply
    # an NCT id that appears in found evidence is also allowed
    ra.run_tool(box, ToolCall("4", "search_evidence", {"query": "trial"}), run)
    assert "NCT07654321" in run.known_ncts


def test_finish_keeps_only_real_handles() -> None:
    run = ra.AgentRun()
    run.add_found([_chunk("a" * 64, 0.5)])
    out = ra.run_tool(FakeBox([], None),  # type: ignore[arg-type]
                      ToolCall("1", "finish", {"summary": "s", "key_evidence": ["C1", "C9"]}),
                      run)
    assert out.finished and run.key_evidence == ["C1"]


def test_found_keeps_order_and_best_similarity() -> None:
    run = ra.AgentRun()
    run.add_found([_chunk("a" * 64, 0.5), _chunk("b" * 64, 0.4)])
    run.add_found([_chunk("b" * 64, 0.8), _chunk("c" * 64, 0.3)])
    assert run.found_pairs() == [("a" * 64, 0.5), ("b" * 64, 0.8), ("c" * 64, 0.3)]
    assert run.handles == {"a" * 64: "C1", "b" * 64: "C2", "c" * 64: "C3"}


def test_research_notes_rewrite_handles_to_evidence_refs() -> None:
    run = ra.AgentRun(ran=True, summary="C2 shows the endpoint was met; C1 is old.")
    run.handles = {"a" * 64: "C1", "b" * 64: "C2"}
    run.key_evidence = ["C2"]
    notes = ra.research_notes(run, {"b" * 64: "E3"})
    assert notes is not None and notes.startswith("E3 shows")
    assert "not in evidence is old" in notes and notes.endswith("Key evidence: E3.")


# ------------------------------------------------------------------ graph loop


def _scripted_agent() -> FakeLLMClient:
    return FakeLLMClient(tool_replies=[
        {"text": "need tier 1", "tool_calls": [{"name": "list_trials", "arguments": {}}]},
        {"tool_calls": [{"name": "fetch_trial", "arguments": {"nct_id": "NCT01234567"}}]},
        {"tool_calls": [{"name": "search_evidence", "arguments": {"query": "ONC-101 topline"}}]},
        {"tool_calls": [{"name": "finish", "arguments": {
            "summary": "C1 trial record found.", "key_evidence": ["C1"]}}]},
    ])


def test_agent_loop_runs_tools_and_code_rescores(rates: BaseRates, store: SQLiteStore) -> None:
    fx = load_fixture(FIX / "onco.yaml", CFG)
    base = _update_from(fx.inputs)
    up = _update_from(fx.inputs, confidence=0.95)
    box = FakeBox([_chunk(s.sha256, 0.6) for s in fx.inputs.evidence.sources], base, up)
    agent = _scripted_agent()
    gen = fx.fake_llm()
    fx, app = _graph(rates, store, box, agent, llm=gen)
    out = start_run(app, fx.catalyst, fx.inputs)
    assert box.fetched == ["NCT01234567"] and box.searches == ["ONC-101 topline"]
    # baseline chunks first, the agent's find after: nothing it found can push them out
    shas, n = box.scored[-1]
    assert n == len(box.base) and shas[:n] == [b.sha256 for b in box.base]
    assert out.state.evidence is not None and out.state.evidence.confidence == 0.95
    assert out.agent is not None and out.agent["summary"] == "C1 trial record found."
    ev = _events(store, fx.catalyst.run_id)
    for name in ("agent.start", "agent.turn", "agent.tool", "agent.score", "tot.generate"):
        assert name in ev, name
    assert ev.count("agent.tool") == 4
    sc = next(e.payload for e in store.read_audit(run_id=fx.catalyst.run_id)
          if e.event_type == "agent.score")
    assert sc["confidence_before"] == fx.inputs.evidence.confidence and sc["confidence"] == 0.95
    # the Generator saw the agent's notes as context, with C handles rewritten to E refs
    gen_prompts = [m[-1].content for m in gen.calls if "Candidates:" in m[-1].content]
    assert gen_prompts and all("Research notes" in p for p in gen_prompts)
    assert "E1 trial record found. Key evidence: E1." in gen_prompts[0]


def test_agent_rescues_inconclusive_evidence(rates: BaseRates, store: SQLiteStore) -> None:
    fx = load_fixture(FIX / "onco.yaml", CFG)
    weak = fx.inputs.model_copy(update={"evidence": fx.inputs.evidence.model_copy(update={
        "confidence": 0.6, "status": EvidenceStatus.INCONCLUSIVE,
        "inconclusive_reason": "confidence 0.600 < min 0.7"})})
    box = FakeBox([], _update_from(weak), _update_from(fx.inputs))
    deps = BuyGraphDeps(CFG, rates, fx.fake_llm(), fx.fake_llm(), store,
                        executor=lambda *a: {}, clock=lambda: fx.catalyst.as_of,
                        agent_llm=_scripted_agent(), toolbox=lambda c: box)
    out = start_run(build_buy_graph(deps, InMemorySaver()), fx.catalyst, weak)
    assert out.state.decision is Decision.BUY_CANDIDATE and out.awaiting_approval


def test_without_a_search_inconclusive_stays_inconclusive(rates: BaseRates,
                                                          store: SQLiteStore) -> None:
    fx = load_fixture(FIX / "onco.yaml", CFG)
    weak = fx.inputs.model_copy(update={"evidence": fx.inputs.evidence.model_copy(update={
        "confidence": 0.6, "status": EvidenceStatus.INCONCLUSIVE,
        "inconclusive_reason": "confidence 0.600 < min 0.7"})})
    box = FakeBox([], _update_from(weak), _update_from(fx.inputs))
    lazy = FakeLLMClient(tool_replies=[{"tool_calls": [{"name": "finish", "arguments": {
        "summary": "nothing more", "key_evidence": []}}]}])
    gen = fx.fake_llm()
    deps = BuyGraphDeps(CFG, rates, gen, gen, store, executor=lambda *a: {},
                        clock=lambda: fx.catalyst.as_of, agent_llm=lazy,
                        toolbox=lambda c: box)
    out = start_run(build_buy_graph(deps, InMemorySaver()), fx.catalyst, weak)
    assert out.state.decision is Decision.INCONCLUSIVE and gen.calls == []


def test_agent_off_behaves_like_before(rates: BaseRates, store: SQLiteStore) -> None:
    fx, app = _graph(rates, store, None, None)
    out = start_run(app, fx.catalyst, fx.inputs)
    assert out.state.decision is Decision.BUY_CANDIDATE
    assert out.state.risk_ev == pytest.approx(0.29755216)
    assert not any(e.startswith("agent.") for e in _events(store, fx.catalyst.run_id))


def test_disabled_in_config_means_off(rates: BaseRates, store: SQLiteStore) -> None:
    cfg = CFG.model_copy(update={"agent": CFG.agent.model_copy(update={"enabled": False})})
    fx = load_fixture(FIX / "onco.yaml", cfg)
    box = FakeBox([], _update_from(fx.inputs))
    agent = _scripted_agent()
    _, app = _graph(rates, store, box, agent, cfg=cfg)
    start_run(app, fx.catalyst, fx.inputs)
    assert agent.tool_calls_seen == [] and box.scored == []


def test_tool_call_limit_stops_a_runaway_agent(rates: BaseRates, store: SQLiteStore) -> None:
    fx = load_fixture(FIX / "onco.yaml", CFG)
    box = FakeBox([], _update_from(fx.inputs))
    forever = FakeLLMClient(tool_replies=lambda turns: {"tool_calls": [
        {"name": "search_evidence", "arguments": {"query": f"q{len(turns)}"}}]})
    _, app = _graph(rates, store, box, forever)
    out = start_run(app, fx.catalyst, fx.inputs)
    assert len(box.searches) == CFG.agent.max_tool_calls
    assert out.agent is not None and out.agent["stop_reason"] == "limit reached"
    assert out.state.decision is Decision.BUY_CANDIDATE  # the run still completes


def test_model_error_falls_back_to_baseline(rates: BaseRates, store: SQLiteStore) -> None:
    fx = load_fixture(FIX / "onco.yaml", CFG)
    box = FakeBox([], _update_from(fx.inputs))

    def boom(turns: Any) -> Any:
        raise LLMError("model down")

    broken = FakeLLMClient(tool_replies=boom)
    _, app = _graph(rates, store, box, broken)
    out = start_run(app, fx.catalyst, fx.inputs)
    assert out.state.decision is Decision.BUY_CANDIDATE
    assert "agent.error" in _events(store, fx.catalyst.run_id)


# ------------------------------------------------------------------ Critic feedback loop


def _generator_critic(critic_accepts_after: int) -> tuple[FakeLLMClient, list[str]]:
    """Generator always nudges depth-1 success citing E1; Critic rejects until N reviews."""
    prompts: list[str] = []
    reviews = [0]

    def reply(msgs: Sequence[LLMMessage]) -> str:
        text = msgs[-1].content
        prompts.append(text)
        depth = int(re.search(r"^Depth: (\d+)", text, re.M).group(1))  # type: ignore[union-attr]
        if "Proposed adjustments:" in text:
            reviews[0] += 1
            ok = reviews[0] > critic_accepts_after
            return json.dumps({"reviews": [{"node": "success",
                                            "verdict": "accept" if ok else "reject",
                                            "reason": "excerpt does not mention efficacy"}]})
        if depth == 1:
            return json.dumps({"adjustments": [{"node": "success", "adjustment": 0.05,
                                                "justification": "positive data",
                                                "evidence": ["E1"]}]})
        return json.dumps({"adjustments": []})

    return FakeLLMClient(reply), prompts


def test_critic_rejection_sends_agent_back_then_regenerates(rates: BaseRates,
                                                            store: SQLiteStore) -> None:
    fx = load_fixture(FIX / "onco.yaml", CFG)
    box = FakeBox([], _update_from(fx.inputs))
    agent = FakeLLMClient(tool_replies=lambda turns: (
        {"tool_calls": [{"name": "search_evidence", "arguments": {"query": "efficacy"}}]}
        if turns[-1].role == "user" else
        {"tool_calls": [{"name": "finish", "arguments": {"summary": "done",
                                                          "key_evidence": []}}]}))
    llm, prompts = _generator_critic(critic_accepts_after=1)
    _, app = _graph(rates, store, box, agent, llm=llm)
    out = start_run(app, fx.catalyst, fx.inputs)
    ev = _events(store, fx.catalyst.run_id)
    assert ev.count("agent.revise") == 1  # one rejection, then accepted
    # the agent was told why, and searched again
    assert any("critic rejected" in t.content for turns in agent.tool_calls_seen
               for t in turns if t.role == "user")
    assert box.searches == ["efficacy", "efficacy"]
    # depth 1 was generated twice; the second time with the Critic's reason in the prompt
    gens = [p for p in prompts if "Candidates:" in p and "Depth: 1" in p]
    assert len(gens) == 2 and "excerpt does not mention efficacy" in gens[1]
    success = next(n for n in out.state.tot_nodes if n.node_id == "success")
    assert success.adjustment > 0  # accepted on the second round


def test_revision_rounds_are_capped(rates: BaseRates, store: SQLiteStore) -> None:
    fx = load_fixture(FIX / "onco.yaml", CFG)
    box = FakeBox([], _update_from(fx.inputs))
    agent = FakeLLMClient(tool_replies=lambda turns: {"tool_calls": [
        {"name": "finish", "arguments": {"summary": "none", "key_evidence": []}}]})
    llm, _ = _generator_critic(critic_accepts_after=99)
    _, app = _graph(rates, store, box, agent, llm=llm)
    out = start_run(app, fx.catalyst, fx.inputs)
    assert _events(store, fx.catalyst.run_id).count("agent.revise") == CFG.agent.max_revisions
    success = next(n for n in out.state.tot_nodes if n.node_id == "success")
    assert success.probability == pytest.approx(success.base_rate)  # never accepted


# ------------------------------------------------------------------ real toolbox


ONTO = Ontology({"ENT-ONCO-101": Entity("ENT-ONCO-101", "ONCO", ("ONC-101",)),
                 "ENT-OTHR-1": Entity("ENT-OTHR-1", "OTHR", ("OT-1",))})
EMB = HashingEmbedder()


def _doc(text: str, tier: int, *, ticker: str = "ONCO", entity: str = "ENT-ONCO-101",
         when: str = "2025-01-01") -> Document:
    return Document(ticker=ticker, tier=SourceTier(tier),
                    published_at=datetime.fromisoformat(when).replace(tzinfo=UTC),
                    url="https://e.org/x", title="t", text=text, doc_type="test",
                    entity_id=entity)


class StubSync:
    def __init__(self, index: VectorIndex, store: SQLiteStore, trial: TrialRecord) -> None:
        self.index, self.store, self.pubmed = index, store, None
        self.ctgov = SimpleNamespace(get_study=lambda nct: trial,
                                     search=lambda **kw: iter([trial]))
        self.added: list[str] = []

    def add(self, e: Any, doc: Document, source: str, rep: Any) -> bool:
        self.added.append(doc.url)
        ingest_documents([doc], index=self.index, embedder=EMB, ontology=ONTO, cfg=CFG.rag)
        return True


def _trial(interventions: tuple[str, ...], title: str = "ONC-101 versus placebo"
           ) -> TrialRecord:
    return TrialRecord("NCT01234567", title, "Onco Inc", "COMPLETED",
                       ("PHASE3",), ("cancer",), interventions, None, None, None,
                       datetime(2025, 6, 1, tzinfo=UTC), raw={"protocolSection": {
                           "outcomesModule": {"primaryOutcomes": [{
                               "measure": "ONC-101 clinical trial primary endpoint results "
                                          "and safety"}]}}})


def _real_box(store: SQLiteStore, trial: TrialRecord) -> tuple[EvidenceToolbox, StubSync]:
    idx = VectorIndex(f"t-{uuid.uuid4().hex[:12]}", client=chromadb.EphemeralClient())
    ingest_documents([
        _doc("ONC-101 risk factors: the trial may fail to meet its primary endpoint.", 2),
        _doc("ONC-101 was well tolerated in a small open-label study.", 3),
        _doc("OT-1 met its primary endpoint.", 1, ticker="OTHR", entity="ENT-OTHR-1"),
    ], index=idx, embedder=EMB, ontology=ONTO, cfg=CFG.rag)
    sync = StubSync(idx, store, trial)
    entry = SimpleNamespace(id="ONCO:ONC-101", trial_ids=[])
    box = EvidenceToolbox(retriever=Retriever(idx, EMB, ONTO, CFG), cfg=CFG, ticker="ONCO",
                          entity_id="ENT-ONCO-101", drug_names=("ONC-101",),
                          as_of=datetime(2026, 1, 1, tzinfo=UTC),
                          sync=sync, entry=entry, run_id="r1")  # type: ignore[arg-type]
    return box, sync


def test_toolbox_fetch_then_search_raises_confidence_in_code(store: SQLiteStore) -> None:
    box, _sync = _real_box(store, _trial(("Drug: ONC-101",)))
    run = ra.AgentRun()
    run.add_found(box.baseline())
    run.baseline_n = len(run.found)
    before = box.score([x for x, _ in run.found_pairs()], run.baseline_n)
    assert all(s.entity_id == "ENT-ONCO-101" for s in before.evidence.sources)
    assert "added NCT01234567" in box.fetch_trial("NCT01234567")
    run.add_found(box.search("ONC-101 versus placebo ClinicalTrials.gov record"))
    after = box.score([x for x, _ in run.found_pairs()], run.baseline_n)
    tiers = [int(s.tier) for s in after.evidence.sources]
    assert 1 in tiers and after.authority > before.authority
    assert all(s.ticker == "ONCO" for s in after.evidence.sources)  # no bleed from OTHR
    assert any(e.event_type == "agent.fetch" and e.run_id == "r1"
               for e in store.read_audit())


def test_toolbox_refuses_a_trial_for_another_drug(store: SQLiteStore) -> None:
    box, sync = _real_box(store, _trial(("Drug: OT-1",), title="OT-1 versus placebo"))
    msg = box.fetch_trial("NCT01234567")
    assert msg.startswith("refused") and sync.added == []
    assert box.list_trials() == []  # listed only when the intervention names this drug


def test_toolbox_without_watchlist_entry_is_search_only(store: SQLiteStore) -> None:
    box, _ = _real_box(store, _trial(("Drug: ONC-101",)))
    off = EvidenceToolbox(retriever=box._r, cfg=CFG, ticker="ONCO", entity_id="ENT-ONCO-101",
                          drug_names=("ONC-101",), as_of=None, sync=None, entry=None)
    assert not off.live and [t.name for t in ra.tools_for(off)] == ["search_evidence", "finish"]
    with pytest.raises(PermissionError):
        off.fetch_trial("NCT01234567")


def test_tool_spec_shapes_are_valid_json_schema() -> None:
    for t in (ra.SEARCH, ra.LIST_TRIALS, ra.FETCH_TRIAL, ra.FETCH_PUBMED, ra.FINISH):
        assert isinstance(t, ToolSpec) and t.parameters["type"] == "object"
        assert set(t.parameters.get("required", [])) <= set(t.parameters["properties"])


def test_tool_response_type_is_frozen() -> None:
    r = ToolResponse("t", (), "fake", "m")
    with pytest.raises(AttributeError):
        r.text = "x"  # type: ignore[misc]


def test_catalyst_state_unchanged_by_agent_fields() -> None:
    # The agent's memory lives in graph state, never in the CatalystState contract.
    assert "agent" not in CatalystState.model_fields


def test_agent_gets_no_extra_turn_after_finish(rates: BaseRates, store: SQLiteStore) -> None:
    fx = load_fixture(FIX / "onco.yaml", CFG)
    box = FakeBox([], _update_from(fx.inputs))
    agent = FakeLLMClient(tool_replies=[{"text": "I am done here.", "tool_calls": [
        {"name": "finish", "arguments": {}}]}])
    _, app = _graph(rates, store, box, agent)
    out = start_run(app, fx.catalyst, fx.inputs)
    assert len(agent.tool_calls_seen) == 1  # one model call, then straight to scoring
    assert out.agent is not None and out.agent["summary"] == "I am done here."


def test_missing_critic_verdict_is_not_feedback(rates: BaseRates, store: SQLiteStore) -> None:
    fx = load_fixture(FIX / "onco.yaml", CFG)
    box = FakeBox([], _update_from(fx.inputs))
    agent = FakeLLMClient(tool_replies=lambda t: {"tool_calls": [
        {"name": "finish", "arguments": {"summary": "s", "key_evidence": []}}]})

    def reply(msgs: Sequence[LLMMessage]) -> str:
        if "Proposed adjustments:" in msgs[-1].content:
            return "I cannot decide"  # unusable: every nudge counts as rejected
        return json.dumps({"adjustments": [{"node": "success", "adjustment": 0.05,
                                            "justification": "x", "evidence": ["E1"]}]})

    _, app = _graph(rates, store, box, agent, llm=FakeLLMClient(reply))
    start_run(app, fx.catalyst, fx.inputs)
    assert "agent.revise" not in _events(store, fx.catalyst.run_id)


def test_scoring_ignores_the_agents_query_wording(store: SQLiteStore) -> None:
    """A chunk the agent finds with a query of its own is scored against the decision
    question: unrelated authoritative text cannot lift confidence."""
    box, _ = _real_box(store, _trial(("Drug: ONC-101",)))
    run = ra.AgentRun()
    run.add_found(box.baseline())
    run.baseline_n = len(run.found)
    shas = [x for x, _ in run.found_pairs()]
    before = box.score(shas, run.baseline_n)
    # the agent "found" the baseline chunks again with a high similarity of its own
    run.add_found([ra.FoundChunk(s, 0.99, 1, "2025-01-01", "x", "t", "") for s in shas])
    again = box.score([x for x, _ in run.found_pairs()], run.baseline_n)
    assert again.confidence == before.confidence
