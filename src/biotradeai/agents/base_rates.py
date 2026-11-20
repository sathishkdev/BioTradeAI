"""Loader for ``data/base_rates.yaml`` (anchors for every ToT probability and return).

Every outcome table must sum to exactly 1 (within 1e-9) so that, before any LLM
adjustment, the tree is already a complete probability space.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from biotradeai.paths import REPO_ROOT

CLINICAL_LABELS = ("success", "partial", "failure")


class BaseRateError(RuntimeError):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Outcome(_Strict):
    p: float = Field(ge=0.0, le=1.0)
    # Long-only: a step can lose at most 100%.
    return_: float = Field(alias="return", ge=-1.0)
    description: str


def _sums_to_one(table: dict[str, Outcome], where: str) -> None:
    if len(table) < 2:
        raise ValueError(f"{where}: needs at least two outcomes")
    total = math.fsum(o.p for o in table.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"{where}: probabilities sum to {total}, expected 1")


class EventRates(_Strict):
    description: str
    clinical: dict[str, Outcome]
    regulatory: dict[str, dict[str, Outcome]]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if set(self.clinical) != set(CLINICAL_LABELS):
            raise ValueError(f"clinical outcomes must be exactly {CLINICAL_LABELS}")
        _sums_to_one(self.clinical, "clinical")
        if set(self.regulatory) != set(CLINICAL_LABELS):
            raise ValueError(f"regulatory tables needed for each of {CLINICAL_LABELS}")
        for k, t in self.regulatory.items():
            _sums_to_one(t, f"regulatory.{k}")
        return self


class FinancingBucket(_Strict):
    runway_below: float | None = Field(gt=0.0)
    outcomes: dict[str, Outcome]

    @model_validator(mode="after")
    def _check(self) -> Self:
        _sums_to_one(self.outcomes, f"financing<{self.runway_below}")
        return self


class Source(_Strict):
    id: str
    citation: str
    url: str


class BaseRates(_Strict):
    reviewed: bool
    sources: list[Source] = Field(min_length=1)
    events: dict[str, EventRates] = Field(min_length=1)
    financing: list[FinancingBucket] = Field(min_length=1)

    @model_validator(mode="after")
    def _buckets(self) -> Self:
        bounds = [b.runway_below for b in self.financing]
        if bounds[-1] is not None or any(b is None for b in bounds[:-1]):
            raise ValueError("only the last financing bucket may (and must) be unbounded")
        finite = [b for b in bounds if b is not None]
        if finite != sorted(finite) or len(set(finite)) != len(finite):
            raise ValueError("financing buckets must be strictly increasing")
        return self

    def event(self, name: str) -> EventRates:
        try:
            return self.events[name]
        except KeyError:
            raise BaseRateError(
                f"no base rates for catalyst event {name!r}; known: {sorted(self.events)}"
            ) from None

    def financing_for(self, runway_months: float) -> dict[str, Outcome]:
        for b in self.financing:
            if b.runway_below is None or runway_months < b.runway_below:
                return b.outcomes
        raise BaseRateError("no financing bucket matched")  # pragma: no cover


def base_rates_path() -> Path:
    return REPO_ROOT / "data" / "base_rates.yaml"


def parse_base_rates(raw: Any) -> BaseRates:
    try:
        return BaseRates.model_validate(raw)
    except ValidationError as exc:
        raise BaseRateError(f"invalid base rates:\n{exc}") from exc


def load_base_rates(path: Path | None = None) -> BaseRates:
    p = path or base_rates_path()
    if not p.is_file():
        raise BaseRateError(f"base rates file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        try:
            raw = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise BaseRateError(f"base rates file is not valid YAML: {exc}") from exc
    return parse_base_rates(raw)
