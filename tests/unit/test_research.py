"""Research Agent (Generator + Critic) and base-rate loading."""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

from biotradeai.agents import research
from biotradeai.agents.base_rates import (
    BaseRateError,
    BaseRates,
    load_base_rates,
    parse_base_rates,
)
from biotradeai.agents.llm import FakeLLMClient, LLMError, LLMMessage, LLMResponse
from biotradeai.config import GuardrailsConfig, load_config
from biotradeai.paths import REPO_ROOT
from biotradeai.scenario_fixtures import ScenarioFixture, load_fixture
from biotradeai.schemas import ResearchInputs, ToTNode

FIX = REPO_ROOT / "data" / "fixtures" / "scenarios"


@pytest.fixture(scope="module")
def cfg() -> GuardrailsConfig:
    return load_config()


@pytest.fixture(scope="module")
def rates() -> BaseRates:
    return load_base_rates()


@pytest.fixture
def onco(cfg: GuardrailsConfig) -> ScenarioFixture:
    return load_fixture(FIX / "onco.yaml", cfg)


def _llm(*replies: Any) -> FakeLLMClient:
    return FakeLLMClient([r if isinstance(r, str) else json.dumps(r) for r in replies])


def _gen(cfg: GuardrailsConfig, rates: BaseRates, fx: ScenarioFixture, reply: Any,
         inputs: ResearchInputs | None = None) -> tuple[list[ToTNode], research.LevelTrace]:
    return research.generate_level(_llm(reply), fx.catalyst, inputs or fx.inputs, 1, cfg=cfg,
                                   rates=rates)


def _by(nodes: list[ToTNode]) -> dict[str, ToTNode]:
    return {n.node_id: n for n in nodes}


# ------------------------------------------------------------------ base rates


def test_real_base_rates_load_and_cite_sources(rates: BaseRates) -> None:
    assert rates.sources and rates.sources[0].url.startswith("https://")
    for ev in rates.events.values():
        assert set(ev.clinical) == {"success", "partial", "failure"}


def _raw() -> dict[str, Any]:
    out: dict[str, Any] = yaml.safe_load(
        (REPO_ROOT / "data" / "base_rates.yaml").read_text(encoding="utf-8"))
    return out


def test_table_not_summing_to_one_rejected() -> None:
    raw = _raw()
    raw["events"]["pdufa"]["clinical"]["success"]["p"] = 0.9
    with pytest.raises(BaseRateError):
        parse_base_rates(raw)


def test_missing_clinical_outcome_rejected() -> None:
    raw = _raw()
    del raw["events"]["pdufa"]["clinical"]["partial"]
    with pytest.raises(BaseRateError):
        parse_base_rates(raw)


def test_financing_buckets_must_increase() -> None:
    raw = _raw()
    raw["financing"][0]["runway_below"] = 30
    with pytest.raises(BaseRateError):
        parse_base_rates(raw)


@pytest.mark.parametrize(("months", "raise_p"),
                         [(11.99, 0.70), (12.0, 0.40), (23.99, 0.40), (24.0, 0.15)])
def test_financing_bucket_boundaries(rates: BaseRates, months: float, raise_p: float) -> None:
    assert rates.financing_for(months)["dilutive_raise"].p == raise_p


def test_unknown_event_rejected(rates: BaseRates) -> None:
    with pytest.raises(BaseRateError):
        rates.event("earnings")


# ------------------------------------------------------------------ generator


def test_depth1_enumerates_all_three_outcomes(cfg: GuardrailsConfig, rates: BaseRates,
                                              onco: ScenarioFixture) -> None:
    nodes, _ = _gen(cfg, rates, onco, {"adjustments": []})
    assert {n.label for n in nodes} == {"success", "partial", "failure"}
    assert sum(n.probability for n in nodes) == pytest.approx(1.0)


def test_unparseable_reply_falls_back_to_base_rates(cfg: GuardrailsConfig, rates: BaseRates,
                                                    onco: ScenarioFixture) -> None:
    nodes, trace = _gen(cfg, rates, onco, "I think it will go up!")
    assert all(n.adjustment == pytest.approx(0) for n in nodes)
    assert any("unparseable" in x for x in trace.notes)


def test_llm_error_falls_back_to_base_rates(cfg: GuardrailsConfig, rates: BaseRates,
                                            onco: ScenarioFixture) -> None:
    class Broken:
        provider, model = "fake", "broken"

        def complete(self, messages: Sequence[LLMMessage], **_: Any) -> LLMResponse:
            raise LLMError("down")

    nodes, trace = research.generate_level(Broken(), onco.catalyst, onco.inputs, 1, cfg=cfg,
                                           rates=rates)
    assert all(n.adjustment == pytest.approx(0) for n in nodes)
    assert any("llm error" in x for x in trace.notes)


def test_adjustment_clipped_to_max(cfg: GuardrailsConfig, rates: BaseRates,
                                   onco: ScenarioFixture) -> None:
    reply = {"adjustments": [{"node": "success", "adjustment": 0.6, "justification": "x",
                              "evidence": ["E1"]}]}
    nodes, trace = _gen(cfg, rates, onco, reply)
    s = _by(nodes)["success"]
    assert s.probability <= s.base_rate + cfg.tot.max_prob_adjustment + 1e-12
    assert any("clipped" in x for x in trace.notes)
    assert sum(n.probability for n in nodes) == pytest.approx(1.0)


def test_unknown_node_and_evidence_are_dropped(cfg: GuardrailsConfig, rates: BaseRates,
                                               onco: ScenarioFixture) -> None:
    reply = {"adjustments": [
        {"node": "moonshot", "adjustment": 0.1, "justification": "x", "evidence": ["E1"]},
        {"node": "success", "adjustment": 0.1, "justification": "x", "evidence": ["E9", "E1"]},
    ]}
    nodes, trace = _gen(cfg, rates, onco, reply)
    assert "moonshot" not in _by(nodes)
    assert _by(nodes)["success"].evidence_ids == [onco.inputs.evidence.sources[0].sha256]
    assert any("unknown node" in x for x in trace.notes)
    assert any("E9" in x for x in trace.notes)


def test_prompt_carries_no_return_assumptions(cfg: GuardrailsConfig, rates: BaseRates,
                                              onco: ScenarioFixture) -> None:
    llm = _llm({"adjustments": []})
    research.generate_level(llm, onco.catalyst, onco.inputs, 1, cfg=cfg, rates=rates)
    prompt = llm.calls[0][-1].content
    assert "base rate 0.500" in prompt and "[E1]" in prompt
    assert "return" not in prompt.lower()  # returns are code-only, never LLM inputs


# ------------------------------------------------------------------ critic


def _draft(cfg: GuardrailsConfig, rates: BaseRates, fx: ScenarioFixture,
           inputs: ResearchInputs | None = None) -> list[ToTNode]:
    reply = {"adjustments": [
        {"node": "success", "adjustment": 0.15, "justification": "j", "evidence": ["E1"]},
        {"node": "failure", "adjustment": -0.15, "justification": "j", "evidence": []}]}
    return _gen(cfg, rates, fx, reply, inputs)[0]


def test_rebalancing_side_effects_are_not_treated_as_llm_claims(
        cfg: GuardrailsConfig, rates: BaseRates, onco: ScenarioFixture) -> None:
    """pdufa: failure base 0.094 cannot drop by 0.15, so code moves 'partial' to rebalance.
    That move is arithmetic, not a claim: it must not be 'rejected' or need evidence."""
    cat = onco.catalyst.model_copy(update={"catalyst_event": "pdufa"})
    reply = {"adjustments": [
        {"node": "success", "adjustment": 0.15, "justification": "j", "evidence": ["E1"]},
        {"node": "failure", "adjustment": -0.15, "justification": "j", "evidence": ["E1"]}]}
    draft, _ = research.generate_level(_llm(reply), cat, onco.inputs, 1, cfg=cfg, rates=rates)
    partial = _by(draft)["partial"]
    assert abs(partial.adjustment) > 0 and not research.is_llm_adjusted(partial)
    assert partial.justification == research.REBALANCED
    critic = _llm({"reviews": [{"node": "success", "verdict": "accept", "reason": "ok"},
                               {"node": "failure", "verdict": "accept", "reason": "ok"}]})
    nodes, _ = research.critique_level(critic, cat, onco.inputs, 1, draft, cfg=cfg)
    assert "partial" not in critic.calls[0][-1].content
    assert not _by(nodes)["partial"].justification.startswith(research.REJECTED)
    assert sum(n.probability for n in nodes) == pytest.approx(1.0)


def test_unevidenced_adjustment_rejected_without_asking_llm(
        cfg: GuardrailsConfig, rates: BaseRates, onco: ScenarioFixture) -> None:
    critic = _llm({"reviews": [{"node": "success", "verdict": "accept", "reason": "ok"}]})
    nodes, _ = research.critique_level(critic, onco.catalyst, onco.inputs, 1,
                                       _draft(cfg, rates, onco), cfg=cfg)
    f = _by(nodes)["failure"]
    assert "no valid evidence" in f.justification
    assert "failure" not in critic.calls[0][-1].content.split("Proposed adjustments:")[1]
    assert sum(n.probability for n in nodes) == pytest.approx(1.0)


@pytest.mark.parametrize("reply", [
    {"reviews": [{"node": "success", "verdict": "reject", "reason": "overstated"}]},
    {"reviews": []},  # missing verdict = reject
    "not json",       # unusable reply = reject
])
def test_critic_rejection_restores_base_rate(cfg: GuardrailsConfig, rates: BaseRates,
                                             onco: ScenarioFixture, reply: Any) -> None:
    nodes, _ = research.critique_level(_llm(reply), onco.catalyst, onco.inputs, 1,
                                       _draft(cfg, rates, onco), cfg=cfg)
    alive = [n for n in nodes if not n.pruned]
    for n in nodes:
        assert n.probability == pytest.approx(n.base_rate)
    assert alive and all(n.evidence_ids == [] for n in nodes)


def test_critic_never_increases_any_adjustment(cfg: GuardrailsConfig, rates: BaseRates,
                                               onco: ScenarioFixture) -> None:
    draft = _draft(cfg, rates, onco)
    critic = _llm({"reviews": [{"node": "success", "verdict": "accept", "reason": "ok"}]})
    nodes, _ = research.critique_level(critic, onco.catalyst, onco.inputs, 1, draft, cfg=cfg)
    # The critic can only pull the tree back toward base rates, never further away.
    assert sum(abs(n.adjustment) for n in nodes) <= sum(abs(n.adjustment) for n in draft) + 1e-9
    assert all(abs(n.adjustment) <= cfg.tot.max_prob_adjustment + 1e-12 for n in nodes)


def test_beam_keeps_top_two_and_prunes_rest(cfg: GuardrailsConfig, rates: BaseRates,
                                            onco: ScenarioFixture) -> None:
    nodes, _ = research.critique_level(_llm(), onco.catalyst, onco.inputs, 1,
                                       _gen(cfg, rates, onco, {"adjustments": []})[0], cfg=cfg)
    by = _by(nodes)
    assert not by["success"].pruned and not by["failure"].pruned
    assert by["partial"].pruned and "beam" in (by["partial"].prune_reason or "")


def _with_inputs(fx: ScenarioFixture, **upd: Any) -> ResearchInputs:
    data = copy.deepcopy(fx.inputs.model_dump())
    for k, v in upd.items():
        if k == "months":
            data["runway"]["months"] = v
        else:
            data[k] = v
    return ResearchInputs.model_validate(data)


@pytest.mark.parametrize(("months", "fatal"), [(2.99, True), (3.0, False), (3.01, False)])
def test_fatal_runway_prune_boundary(cfg: GuardrailsConfig, rates: BaseRates,
                                     onco: ScenarioFixture, months: float, fatal: bool) -> None:
    inp = _with_inputs(onco, months=months)
    draft = _gen(cfg, rates, onco, {"adjustments": []}, inp)[0]
    nodes, _ = research.critique_level(_llm(), onco.catalyst, inp, 1, draft, cfg=cfg)
    by = _by(nodes)
    assert by["success"].pruned is fatal
    assert not by["failure"].pruned
    if fatal:
        assert "fatal flaw" in (by["success"].prune_reason or "")


def test_active_clinical_hold_collapses_non_failure_branches(
        cfg: GuardrailsConfig, rates: BaseRates, onco: ScenarioFixture) -> None:
    inp = _with_inputs(onco, claims={"clinical_hold": "active"})
    draft = _gen(cfg, rates, onco, {"adjustments": []}, inp)[0]
    nodes, _ = research.critique_level(_llm(), onco.catalyst, inp, 1, draft, cfg=cfg)
    by = _by(nodes)
    assert by["success"].pruned and by["partial"].pruned and not by["failure"].pruned
    assert "clinical hold" in (by["success"].prune_reason or "")


def test_fixture_files_parse(cfg: GuardrailsConfig) -> None:
    for p in sorted(Path(FIX).glob("*.yaml")):
        fx = load_fixture(p, cfg)
        assert fx.inputs.evidence.ticker == fx.catalyst.ticker
