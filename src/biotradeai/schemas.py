"""Pydantic v2 contracts passed between agents, the supervisor and the brokers.

These are the *only* shapes allowed across agent boundaries. Validation here is
structural (types, ranges, invariants such as "probabilities sum to 1"). Risk
policy (EV >= min_ev, runway >= min_runway_months, ...) lives in
``guardrails/rules.py`` so it can be unit-tested against the config.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from enum import IntEnum, StrEnum
from typing import Annotated, Self

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

TICKER_PATTERN = r"^[A-Z][A-Z0-9.\-]{0,9}$"
SHA256_PATTERN = r"^[0-9a-f]{64}$"

Ticker = Annotated[str, Field(pattern=TICKER_PATTERN)]
Sha256 = Annotated[str, Field(pattern=SHA256_PATTERN)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
# Long-only equity: a position can lose at most 100%.
Return = Annotated[float, Field(ge=-1.0)]


def _non_blank(v: str) -> str:
    if not v.strip():
        raise ValueError("must not be blank")
    return v


NonBlank = Annotated[str, AfterValidator(_non_blank)]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- sources


class SourceTier(IntEnum):
    """Lower number = more authoritative."""

    REGULATORY = 1  # FDA, ClinicalTrials.gov
    SEC = 2  # EDGAR filings
    LITERATURE = 3  # PubMed, press releases, news


class SourceRecord(_Contract):
    tier: SourceTier
    published_at: AwareDatetime
    sha256: Sha256
    url: NonBlank
    ticker: Ticker
    entity_id: NonBlank


class EvidenceStatus(StrEnum):
    OK = "OK"
    INCONCLUSIVE = "INCONCLUSIVE"


class EvidenceBundle(_Contract):
    ticker: Ticker
    entity_id: NonBlank
    sources: list[SourceRecord]
    confidence: Probability
    status: EvidenceStatus
    inconclusive_reason: str | None = None

    @model_validator(mode="after")
    def _no_semantic_bleed(self) -> Self:
        for s in self.sources:
            if s.ticker != self.ticker or s.entity_id != self.entity_id:
                raise ValueError(
                    f"source {s.sha256[:12]} belongs to {s.ticker}/{s.entity_id}, "
                    f"not {self.ticker}/{self.entity_id} (semantic bleed)"
                )
        if self.status is EvidenceStatus.INCONCLUSIVE and not self.inconclusive_reason:
            raise ValueError("INCONCLUSIVE evidence must state inconclusive_reason")
        return self


# --------------------------------------------------------------------------- ToT


class ToTNode(_Contract):
    """One thought in the Tree of Thoughts, with base-rate anchoring logged."""

    node_id: NonBlank
    parent_id: str | None
    depth: int = Field(ge=1)
    label: NonBlank
    base_rate: Probability
    adjustment: float = Field(ge=-1.0, le=1.0)
    probability: Probability  # conditional on the parent
    justification: str
    evidence_ids: list[Sha256] = Field(default_factory=list)
    pruned: bool = False
    prune_reason: str | None = None

    @model_validator(mode="after")
    def _prune_needs_reason(self) -> Self:
        if self.pruned and not self.prune_reason:
            raise ValueError("pruned node must state prune_reason")
        return self


class Scenario(_Contract):
    """A terminal outcome with its *joint* probability and expected return."""

    depth: int = Field(ge=1)
    label: NonBlank
    probability: Probability
    expected_return: Return
    evidence_ids: list[Sha256] = Field(default_factory=list)
    is_residual: bool = False


class ScenarioSet(_Contract):
    """The full outcome space. Probabilities MUST sum to 1 (within ``tolerance``).

    EV is computed over ALL outcomes (plan correction #2). Pruned branches are
    collapsed into a residual scenario rather than dropped, so the set still sums
    to 1. ``tolerance`` is passed in from ``config.tot.prob_sum_tolerance``.
    """

    scenarios: list[Scenario] = Field(min_length=2)
    tolerance: float = Field(gt=0.0, le=0.01)

    @model_validator(mode="after")
    def _probabilities_sum_to_one(self) -> Self:
        total = sum(s.probability for s in self.scenarios)
        if abs(total - 1.0) > self.tolerance:
            raise ValueError(
                f"scenario probabilities sum to {total:.6f}, expected 1.0 ± {self.tolerance}"
            )
        labels = [s.label for s in self.scenarios]
        if len(set(labels)) != len(labels):
            raise ValueError("scenario labels must be unique")
        return self


# --------------------------------------------------------------------------- orders


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    # Non-limit types exist only so upstream requests can be represented and then
    # *rejected* by the guardrail / broker adapter.
    LIMIT = "limit"
    MARKET = "market"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(StrEnum):
    DAY = "day"
    GTC = "gtc"


Price = Annotated[Decimal, Field(gt=Decimal(0), max_digits=12, decimal_places=4)]


class StagedOrder(_Contract):
    ticker: Ticker
    side: OrderSide
    qty: int = Field(gt=0)
    order_type: OrderType
    limit_price: Price | None
    time_in_force: TimeInForce = TimeInForce.DAY
    extended_hours: bool = False
    # Evidence pack shown to the human approver.
    ev: float | None = None
    confidence: Probability | None = None
    runway_months: float | None = None
    source_hashes: list[Sha256] = Field(default_factory=list)

    @model_validator(mode="after")
    def _limit_needs_price(self) -> Self:
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit order requires limit_price")
        return self


class ApprovalToken(_Contract):
    """Human approval for exactly one order. Signed with HMAC-SHA256 in Phase 2."""

    ticker: Ticker
    side: OrderSide
    qty: int = Field(gt=0)
    limit_price: Price
    expiry: AwareDatetime
    nonce: Annotated[str, Field(min_length=16, max_length=128)]
    signature: Annotated[str, Field(pattern=SHA256_PATTERN)]

    def signing_payload(self) -> bytes:
        """Canonical bytes that the signature covers (everything except itself)."""
        body = {
            "ticker": self.ticker,
            "side": self.side.value,
            "qty": self.qty,
            "limit_price": format(self.limit_price.normalize(), "f"),
            "expiry": self.expiry.isoformat(),
            "nonce": self.nonce,
        }
        return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


# --------------------------------------------------------------------------- graph state


class Decision(StrEnum):
    BUY_CANDIDATE = "BUY_CANDIDATE"
    SKIP = "SKIP"
    INCONCLUSIVE = "INCONCLUSIVE"


class CatalystState(_Contract):
    """State carried through the buy graph for one ticker/catalyst."""

    run_id: NonBlank
    ticker: Ticker
    entity_id: NonBlank
    catalyst_event: NonBlank
    catalyst_date: date
    as_of: datetime
    evidence: EvidenceBundle | None = None
    tot_nodes: list[ToTNode] = Field(default_factory=list)
    scenarios: ScenarioSet | None = None
    research_ev: float | None = None
    risk_ev: float | None = None
    runway_months: float | None = None
    decision: Decision | None = None
    veto_reasons: list[str] = Field(default_factory=list)
    staged_order: StagedOrder | None = None
