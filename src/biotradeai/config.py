"""Load and validate ``config/guardrails.yaml``.

Every risk threshold in the system is read from here. The models use
``extra="forbid"`` and have no defaults, so a missing, misspelled or extra key
fails fast at startup instead of silently falling back to a magic number.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from biotradeai.paths import config_path

# Governance plan: "Hard caps enforce <= 5% portfolio equity per ticker".
# Config may be stricter, never looser. This is a policy ceiling, not a tunable.
HARD_CEILING_POSITION_PCT = 0.05


class ConfigError(RuntimeError):
    """Raised when guardrails.yaml is missing or invalid."""


class Mode(StrEnum):
    PAPER = "paper"
    SHADOW = "shadow"
    LIVE = "live"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=False)


class OrdersConfig(_Strict):
    # Only "limit" is a legal value. Listing "market" or "stop" fails validation.
    allowed_types: list[Literal["limit"]] = Field(min_length=1)
    allow_extended_hours: bool
    approval_ttl_minutes: float = Field(gt=0.0, le=60.0)
    # Approve-later queue (operator UI): a staged order waits this long for a human,
    # and is blocked at approval if the price moved more than this since staging.
    pending_approval_hours: float = Field(gt=0.0, le=8.0)
    max_price_drift_pct: float = Field(gt=0.0, le=0.2)


class RunwayConfig(_Strict):
    burn_lookback_quarters: int = Field(ge=1, le=8)


class RagConfig(_Strict):
    chunk_tokens: int = Field(ge=300, le=500)
    chunk_overlap_tokens: int = Field(ge=0, le=100)
    top_k: int = Field(ge=3, le=5)
    hybrid: bool
    max_pool: int = Field(ge=10, le=100_000)
    min_similarity: float = Field(ge=-1.0, le=1.0)
    min_supporting_chunks: int = Field(ge=1, le=5)
    recall_target: float = Field(gt=0.0, le=1.0)
    tier_weights: dict[int, float]

    @model_validator(mode="after")
    def _checks(self) -> Self:
        if self.chunk_overlap_tokens >= self.chunk_tokens:
            raise ValueError("rag.chunk_overlap_tokens must be < chunk_tokens")
        w = self.tier_weights
        if set(w) != {1, 2, 3}:
            raise ValueError("rag.tier_weights needs exactly tiers 1, 2, 3")
        if not (1.0 >= w[1] >= w[2] >= w[3] > 0.0):
            raise ValueError("rag.tier_weights must satisfy 1 >= t1 >= t2 >= t3 > 0")
        return self


class ToTConfig(_Strict):
    depth: int = Field(ge=1, le=5)
    beam_width: int = Field(ge=1, le=5)
    # Return assigned to the collapsed/pruned residual scenario. Long-only, so the
    # worst case is -100%; it must be a loss (<= 0) to be conservative.
    residual_scenario_return: float = Field(ge=-1.0, le=0.0)
    max_prob_adjustment: float = Field(gt=0.0, le=0.5)
    prob_sum_tolerance: float = Field(gt=0.0, le=0.01)
    # Research EV vs Risk EV: a larger gap is logged and blocks the trade.
    ev_match_tolerance: float = Field(gt=0.0, le=0.01)


class SellConfig(_Strict):
    profit_target_pct: float = Field(gt=0.0)
    scale_out_fraction: float = Field(gt=0.0, le=1.0)
    trailing_stop_pct: float = Field(gt=0.0, lt=1.0)
    completion_delay_months: float = Field(gt=0.0)


class LiveGateConfig(_Strict):
    min_paper_days: int = Field(ge=0)
    pilot_position_pct: float = Field(gt=0.0, le=HARD_CEILING_POSITION_PCT)
    pilot_max_positions: int = Field(ge=1)


class GuardrailsConfig(_Strict):
    mode: Mode
    min_ev: float = Field(ge=0.0)
    min_runway_months: float = Field(gt=0.0)
    fatal_runway_months: float = Field(gt=0.0)
    max_position_pct: float = Field(gt=0.0, le=HARD_CEILING_POSITION_PCT)
    max_concurrent_positions: int = Field(ge=1)
    min_confidence: float = Field(ge=0.0, le=1.0)
    max_spread_pct: float = Field(gt=0.0, lt=1.0)
    ttl_hours: float = Field(gt=0.0)
    daily_drawdown_limit: float = Field(gt=0.0, lt=1.0)
    max_api_errors: int = Field(ge=1)
    runway: RunwayConfig
    rag: RagConfig
    orders: OrdersConfig
    tot: ToTConfig
    sell: SellConfig
    live_gate: LiveGateConfig

    @model_validator(mode="after")
    def _cross_field_checks(self) -> Self:
        if self.fatal_runway_months >= self.min_runway_months:
            raise ValueError(
                "fatal_runway_months (ToT prune) must be < min_runway_months (Risk veto)"
            )
        if self.live_gate.pilot_position_pct > self.max_position_pct:
            raise ValueError("live_gate.pilot_position_pct must be <= max_position_pct")
        if self.live_gate.pilot_max_positions > self.max_concurrent_positions:
            raise ValueError("live_gate.pilot_max_positions must be <= max_concurrent_positions")
        return self


def parse_config(raw: Any) -> GuardrailsConfig:
    """Validate an already-parsed YAML mapping."""
    if not isinstance(raw, dict):
        raise ConfigError("guardrails config must be a YAML mapping at the top level")
    try:
        return GuardrailsConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid guardrails config:\n{exc}") from exc


def load_config(path: Path | None = None) -> GuardrailsConfig:
    """Load guardrails.yaml (default location, or ``BIOTRADEAI_CONFIG``)."""
    p = path or config_path()
    if not p.is_file():
        raise ConfigError(f"guardrails config not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        try:
            raw = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ConfigError(f"guardrails config is not valid YAML: {exc}") from exc
    return parse_config(raw)
