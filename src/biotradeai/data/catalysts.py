"""Loader for the human-curated ``data/catalysts.csv``.

This file, not ClinicalTrials.gov, is the source of truth for PDUFA / readout dates
(plan correction #3). The human maintains it; code only reads and validates it.

Columns: ticker, drug, event, date, source_url
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from biotradeai.paths import data_dir

COLUMNS = ("ticker", "drug", "event", "date", "source_url")
EVENTS = frozenset({"PDUFA", "PHASE3_READOUT", "PHASE2_READOUT", "ADCOM", "OTHER"})
_TICKER = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


class CatalystFileError(ValueError):
    pass


@dataclass(frozen=True)
class Catalyst:
    ticker: str
    drug: str
    event: str
    date: date
    source_url: str


def default_path() -> Path:
    return data_dir() / "catalysts.csv"


def load_catalysts(path: Path | None = None) -> list[Catalyst]:
    p = path or default_path()
    with p.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise CatalystFileError(
                f"{p.name}: header must be {','.join(COLUMNS)}, got {reader.fieldnames}"
            )
        out: list[Catalyst] = []
        seen: set[tuple[str, str, str, date]] = set()
        for lineno, row in enumerate(reader, start=2):
            vals = {k: (row.get(k) or "").strip() for k in COLUMNS}
            if not any(vals.values()):
                continue
            where = f"{p.name}:{lineno}"
            if not _TICKER.match(vals["ticker"]):
                raise CatalystFileError(f"{where}: bad ticker {vals['ticker']!r}")
            if not vals["drug"]:
                raise CatalystFileError(f"{where}: drug is required")
            event = vals["event"].upper()
            if event not in EVENTS:
                raise CatalystFileError(f"{where}: event must be one of {sorted(EVENTS)}")
            try:
                d = date.fromisoformat(vals["date"])
            except ValueError:
                raise CatalystFileError(f"{where}: date must be YYYY-MM-DD") from None
            if not vals["source_url"].startswith(("https://", "http://")):
                raise CatalystFileError(f"{where}: source_url must be an http(s) URL")
            key = (vals["ticker"], vals["drug"], event, d)
            if key in seen:
                raise CatalystFileError(f"{where}: duplicate row")
            seen.add(key)
            out.append(Catalyst(vals["ticker"], vals["drug"], event, d, vals["source_url"]))
    return out


def upcoming(catalysts: list[Catalyst], *, today: date) -> list[Catalyst]:
    return sorted((c for c in catalysts if c.date >= today), key=lambda c: (c.date, c.ticker))
