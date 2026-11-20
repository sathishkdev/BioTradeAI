"""Replay fixtures for the sell state machine (``data/fixtures/sell/*.yaml``).

A fixture holds one position, an ordered event stream and the exits it MUST produce.
Filing events carry the raw form and (for 8-Ks) text, and are classified by the same
code the live poller uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from pydantic import TypeAdapter

from biotradeai.events import SellEvent, classify_filing
from biotradeai.graph.sell_graph import SellPosition, Trigger

_EVENTS: TypeAdapter[list[SellEvent]] = TypeAdapter(list[SellEvent])


@dataclass(frozen=True)
class ExpectedExit:
    trigger: Trigger
    qty: int
    limit_price: Decimal | None


@dataclass(frozen=True)
class SellReplay:
    name: str
    description: str
    position: SellPosition
    events: list[SellEvent]
    expected: list[ExpectedExit]


def _event(raw: dict[str, Any]) -> dict[str, Any]:
    e = dict(raw)
    if e.get("kind") == "filing":
        text = e.pop("text", None)
        dilutive, reason = classify_filing(str(e["form"]), text)
        e.setdefault("dilutive", dilutive)
        e.setdefault("reason", reason)
        e.setdefault("url", f"https://www.sec.gov/fixture/{e['accession']}")
    return e


def load_replay(path: Path) -> SellReplay:
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    pos = SellPosition.model_validate(raw["position"])
    events = _EVENTS.validate_python(
        [_event({"ticker": pos.ticker, **e}) for e in raw["events"]])
    expected = [ExpectedExit(Trigger(x["trigger"]), int(x["qty"]),
                             Decimal(str(x["limit"])) if x.get("limit") is not None else None)
                for x in raw.get("expected_exits") or []]
    return SellReplay(path.stem, str(raw.get("description", "")), pos, events, expected)
