"""Market and filing events that drive the deterministic sell state machine (Phase 6).

Pollers (``data/pollers.py``) and replay fixtures produce these; ``graph/sell_graph.py``
consumes them. They are plain validated contracts with no behaviour.
"""

from __future__ import annotations

import re
from datetime import date
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from biotradeai.schemas import Price, Ticker


class _Event(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ticker: Ticker
    ts: AwareDatetime


class PriceTick(_Event):
    kind: Literal["price"] = "price"
    bid: Price
    ask: Price


class ReadoutResult(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"  # primary endpoint missed / safety stop
    CRL = "crl"  # FDA complete response letter
    MIXED = "mixed"  # ambiguous: treated like a failure (sell 100%)


class CatalystOutcome(_Event):
    kind: Literal["outcome"] = "outcome"
    result: ReadoutResult
    source_url: str


class FilingEvent(_Event):
    kind: Literal["filing"] = "filing"
    form: str
    accession: str
    dilutive: bool
    reason: str
    url: str


class CompletionDateChange(_Event):
    kind: Literal["completion_date"] = "completion_date"
    nct_id: str
    old_date: date | None
    new_date: date


class CatalystScheduled(_Event):
    """A NEXT readout for the same ticker was scheduled (e.g. Phase 3 after Phase 2)."""

    kind: Literal["catalyst_scheduled"] = "catalyst_scheduled"
    event: str
    catalyst_date: date


SellEvent = Annotated[
    PriceTick | CatalystOutcome | FilingEvent | CompletionDateChange | CatalystScheduled,
    Field(discriminator="kind"),
]


# --------------------------------------------------------------------------- dilution


# Registration statements and prospectus supplements for new shares.
DILUTIVE_FORMS = re.compile(r"^(S-3|S-3ASR|S-3/A|S-1|S-1/A|424B[1-5])$")
WATCHED_FORMS = re.compile(r"^(8-K|8-K/A)$|" + DILUTIVE_FORMS.pattern)

_OFFERING = re.compile(
    r"\b(underwritten|registered direct|public|follow-on|secondary|confidentially marketed)"
    r"\s+(public\s+)?offering\b"
    r"|\bat[- ]the[- ]market\s+(offering|program|sales|equity)"
    r"|\bprivate placement\b"
    r"|\bpre-funded warrants?\b"
    r"|\b(sale|issuance|issue and sell|offer and sell)\s+(of\s+)?(up to\s+)?"
    r"([\d,.]+\s+(million\s+)?)?(shares|its common stock|our common stock|ordinary shares)",
    re.I,
)


def classify_filing(form: str, text: str | None = None) -> tuple[bool, str]:
    """(dilutive?, reason). Deterministic; an 8-K needs text that names an offering."""
    f = form.strip().upper()
    if DILUTIVE_FORMS.match(f):
        return True, f"form {f} registers or prices new shares"
    if f in ("8-K", "8-K/A"):
        m = _OFFERING.search(text or "")
        if m:
            return True, f"8-K mentions an equity offering: {m.group(0)!r}"
        return False, "8-K without offering language"
    return False, f"form {f} is not dilution-related"


__all__ = [
    "CatalystOutcome", "CatalystScheduled", "CompletionDateChange", "FilingEvent",
    "PriceTick", "ReadoutResult", "SellEvent", "classify_filing",
]
