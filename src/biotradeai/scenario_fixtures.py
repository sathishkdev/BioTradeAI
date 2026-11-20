"""Offline end-to-end scenarios (``data/fixtures/scenarios/*.yaml``).

A fixture describes one ticker/catalyst: source texts, cash and burn, a quote, the
portfolio, and optionally scripted LLM replies. Claims and conflicts are extracted with
the real RAG code (``rag.claims`` + ``rag.retrieve.resolve_claims``) and runway is
computed in code, so the fixtures exercise the same logic as a live run.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from biotradeai.agents.llm import FakeLLMClient, LLMMessage
from biotradeai.config import GuardrailsConfig
from biotradeai.rag.claims import extract_claims
from biotradeai.rag.index import Chunk
from biotradeai.rag.retrieve import resolve_claims
from biotradeai.schemas import (
    CatalystState,
    EvidenceBundle,
    EvidenceStatus,
    MarketSnapshot,
    PortfolioSnapshot,
    ResearchInputs,
    RunwaySnapshot,
    SourceRecord,
    SourceTier,
)

MONTHS_PER_QUARTER = 3


@dataclass(frozen=True)
class ScenarioFixture:
    name: str
    catalyst: CatalystState
    inputs: ResearchInputs
    generator_replies: dict[int, Any]
    critic_replies: dict[int, Any]

    def fake_llm(self) -> FakeLLMClient:
        """One scripted client that answers generator and critic prompts by depth."""

        def reply(msgs: Sequence[LLMMessage]) -> str:
            text = msgs[-1].content
            m = re.search(r"^Depth: (\d+)", text, re.M)
            depth = int(m.group(1)) if m else 0
            if "Proposed adjustments:" in text:
                return json.dumps(self.critic_replies.get(depth, {"reviews": []}))
            return json.dumps(self.generator_replies.get(depth, {"adjustments": []}))

        return FakeLLMClient(reply, model=f"fixture-{self.name}")


def runway_months(cash: Decimal, quarterly_burn: Decimal) -> float:
    if quarterly_burn <= 0:
        return 999.0  # not burning: effectively unlimited
    return float(cash / quarterly_burn * MONTHS_PER_QUARTER)


def load_fixture(path: Path, cfg: GuardrailsConfig) -> ScenarioFixture:
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    run = raw["run"]
    as_of = datetime.fromisoformat(str(run["as_of"]))
    ticker, eid = str(run["ticker"]), str(run["entity_id"])

    chunks: list[Chunk] = []
    excerpts: dict[str, str] = {}
    for i, s in enumerate(raw["evidence"]["sources"]):
        text = " ".join(str(s["text"]).split())
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        published = datetime.fromisoformat(str(s["published_at"]))
        chunks.append(Chunk(chunk_id=f"{path.stem}-{i}", text=text, ticker=ticker,
                            entity_id=eid, tier=SourceTier(int(s["tier"])),
                            published_at=published, sha256=sha, doc_sha256=sha,
                            url=str(s["url"]), title=str(s.get("title", "")),
                            doc_type=str(s.get("doc_type", "")), claims=extract_claims(text)))
        excerpts[sha] = text
    resolved, conflicts = resolve_claims(chunks)
    confidence = float(raw["evidence"]["confidence"])
    reasons = [f"unresolved {c.key} conflict: {c.note}" for c in conflicts if not c.resolved]
    if confidence < cfg.min_confidence:
        reasons.append(f"confidence {confidence:.3f} < min {cfg.min_confidence}")
    bundle = EvidenceBundle(
        ticker=ticker, entity_id=eid,
        sources=[SourceRecord(tier=c.tier, published_at=c.published_at, sha256=c.sha256,
                              url=c.url, ticker=ticker, entity_id=eid) for c in chunks],
        confidence=confidence,
        status=EvidenceStatus.INCONCLUSIVE if reasons else EvidenceStatus.OK,
        inconclusive_reason="; ".join(reasons) or None)

    rw = raw["runway"]
    cash, burn = Decimal(str(rw["cash"])), Decimal(str(rw["quarterly_burn"]))
    mk, pf = raw["market"], raw["portfolio"]
    inputs = ResearchInputs(
        evidence=bundle, evidence_observed_at=as_of, excerpts=excerpts,
        claims={k: v.value for k, v in resolved.items()},
        runway=RunwaySnapshot(months=runway_months(cash, burn), cash=cash,
                              quarterly_burn=burn, observed_at=as_of,
                              source_accns=[str(a) for a in rw.get("source_accns", [])]),
        market=MarketSnapshot(bid=Decimal(str(mk["bid"])), ask=Decimal(str(mk["ask"])),
                              observed_at=as_of),
        portfolio=PortfolioSnapshot(equity=Decimal(str(pf["equity"])),
                                    open_tickers=list(pf.get("open_tickers", [])),
                                    existing_value=Decimal(str(pf.get("existing_value", 0))),
                                    observed_at=as_of),
    )
    catalyst = CatalystState(run_id=str(run["run_id"]), ticker=ticker, entity_id=eid,
                             catalyst_event=str(run["catalyst_event"]),
                             catalyst_date=date.fromisoformat(str(run["catalyst_date"])),
                             as_of=as_of)
    llm = raw.get("fake_llm") or {}
    return ScenarioFixture(
        name=path.stem, catalyst=catalyst, inputs=inputs,
        generator_replies={int(k): v for k, v in (llm.get("generator") or {}).items()},
        critic_replies={int(k): v for k, v in (llm.get("critic") or {}).items()})
