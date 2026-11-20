"""Loader for the human-curated ``data/outcomes.csv``: what actually happened at a catalyst.

Readout results drive the post-catalyst sell triggers. Like catalysts.csv, a human
enters them (from the company press release / FDA letter); code never infers them.

Columns: ticker, drug, date, result, source_url   (result: positive|negative|crl|mixed)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from biotradeai.events import ReadoutResult
from biotradeai.paths import data_dir

COLUMNS = ("ticker", "drug", "date", "result", "source_url")


class OutcomeFileError(ValueError):
    pass


@dataclass(frozen=True)
class Outcome:
    ticker: str
    drug: str
    date: date
    result: ReadoutResult
    source_url: str


def default_path() -> Path:
    return data_dir() / "outcomes.csv"


def load_outcomes(path: Path | None = None) -> list[Outcome]:
    p = path or default_path()
    if not p.exists():
        return []
    out: list[Outcome] = []
    with p.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise OutcomeFileError(f"{p.name}: header must be {','.join(COLUMNS)}")
        for lineno, row in enumerate(reader, start=2):
            v = {k: (row.get(k) or "").strip() for k in COLUMNS}
            if not any(v.values()):
                continue
            try:
                out.append(Outcome(v["ticker"].upper(), v["drug"], date.fromisoformat(v["date"]),
                                   ReadoutResult(v["result"].lower()), v["source_url"]))
            except ValueError as exc:
                raise OutcomeFileError(f"{p.name}:{lineno}: {exc}") from exc
            if not v["source_url"].startswith(("https://", "http://")):
                raise OutcomeFileError(f"{p.name}:{lineno}: source_url must be http(s)")
    return out
