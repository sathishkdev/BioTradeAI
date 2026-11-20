"""Generate synthetic SEC companyfacts JSON fixtures for runway tests.

The output mimics the real ``/api/xbrl/companyfacts/CIK##########.json`` shape so the
parser is exercised on realistic structure. The *expected* runway values are NOT
computed here — they are worked out by hand in
``data/fixtures/edgar/expected_runway.yaml``.

Run:  python scripts/build_edgar_fixtures.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

OUT = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "edgar" / "companyfacts"
M = 1_000_000

CASH = "CashAndCashEquivalentsAtCarryingValue"
CASH_R = "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"
MS_C = "MarketableSecuritiesCurrent"
STI = "ShortTermInvestments"
MS_NC = "MarketableSecuritiesNoncurrent"
OCF = "NetCashProvidedByUsedInOperatingActivities"

# fact = (tag, start or None, end, value in $M)
Fact = tuple[str, str | None, str, float]


def filing(form: str, fy: int, fp: str, filed: str, seq: int, cik: int,
           facts: list[Fact]) -> list[dict[str, Any]]:
    accn = f"{cik:010d}-{filed[2:4]}-{seq:06d}"
    rows = []
    for tag, start, end, val in facts:
        row: dict[str, Any] = {"end": end, "val": int(val * M), "accn": accn, "fy": fy,
                               "fp": fp, "form": form, "filed": filed}
        if start:
            row = {"start": start, **row}
        rows.append({"tag": tag, **row})
    return rows


SPECS: dict[str, dict[str, Any]] = {
    # 1. Classic 10-Q YTD cash flows + 10-K. Plan's $ONCO: $40M cash, $10M/quarter.
    "ONCO": {"cik": 9000001, "name": "Onco Example Therapeutics", "filings": [
        ("10-Q", 2025, "Q1", "2025-05-08", [(CASH, None, "2025-03-31", 60),
                                            (OCF, "2025-01-01", "2025-03-31", -10)]),
        ("10-Q", 2025, "Q2", "2025-08-07", [(CASH, None, "2025-06-30", 55),
                                            (OCF, "2025-01-01", "2025-06-30", -20)]),
        ("10-Q", 2025, "Q3", "2025-11-06", [(CASH, None, "2025-09-30", 50),
                                            (OCF, "2025-01-01", "2025-09-30", -30)]),
        ("10-K", 2025, "FY", "2026-03-02", [(CASH, None, "2025-12-31", 25),
                                            (STI, None, "2025-12-31", 15),
                                            (CASH, None, "2024-12-31", 70),
                                            (OCF, "2025-01-01", "2025-12-31", -40)]),
    ]},
    # 2. Dilution trap: one month of cash.
    "DILU": {"cik": 9000002, "name": "Dilution Example Bio", "filings": [
        ("10-Q", 2026, "Q1", "2026-05-12", [(CASH, None, "2026-03-31", 3),
                                            (OCF, "2026-01-01", "2026-03-31", -9)]),
    ]},
    # 3. Restatement: 10-Q/A restates Q3 cash and YTD OCF.
    "RSTA": {"cik": 9000003, "name": "Restated Example Pharma", "filings": [
        ("10-Q", 2025, "Q1", "2025-05-09", [(CASH, None, "2025-03-31", 80),
                                            (OCF, "2025-01-01", "2025-03-31", -15)]),
        ("10-Q", 2025, "Q2", "2025-08-08", [(CASH, None, "2025-06-30", 65),
                                            (OCF, "2025-01-01", "2025-06-30", -30)]),
        ("10-Q", 2025, "Q3", "2025-11-07", [(CASH, None, "2025-09-30", 50),
                                            (OCF, "2025-01-01", "2025-09-30", -45)]),
        ("10-Q/A", 2025, "Q3", "2025-12-15", [(CASH, None, "2025-09-30", 45),
                                              (OCF, "2025-01-01", "2025-09-30", -48)]),
    ]},
    # 4. Duplicates: comparatives repeated in later filings with identical values.
    "DUPE": {"cik": 9000004, "name": "Duplicate Facts Bio", "filings": [
        ("10-Q", 2025, "Q1", "2025-05-06", [(CASH, None, "2025-03-31", 80),
                                            (OCF, "2025-01-01", "2025-03-31", -5)]),
        ("10-Q", 2025, "Q2", "2025-08-05", [(CASH, None, "2025-06-30", 74),
                                            (OCF, "2025-01-01", "2025-06-30", -11)]),
        ("10-Q", 2025, "Q3", "2025-11-04", [(CASH, None, "2025-09-30", 67),
                                            (OCF, "2025-01-01", "2025-09-30", -17)]),
        ("10-K", 2025, "FY", "2026-02-25", [(CASH, None, "2025-12-31", 60),
                                            (OCF, "2025-01-01", "2025-12-31", -24)]),
        ("10-Q", 2026, "Q1", "2026-05-05", [(CASH, None, "2026-03-31", 54),
                                            (CASH, None, "2025-12-31", 60),
                                            (OCF, "2026-01-01", "2026-03-31", -6),
                                            (OCF, "2025-01-01", "2025-03-31", -5)]),
    ]},
    # 5. No marketable securities reported at all.
    "NOMS": {"cik": 9000005, "name": "No Securities Bio", "filings": [
        ("10-Q", 2025, "Q1", "2025-05-10", [(CASH, None, "2025-03-31", 36),
                                            (OCF, "2025-01-01", "2025-03-31", -4)]),
        ("10-Q", 2025, "Q2", "2025-08-09", [(CASH, None, "2025-06-30", 30),
                                            (OCF, "2025-01-01", "2025-06-30", -10)]),
    ]},
    # 6. Fallback cash tag + current and non-current securities; 3-month OCF facts.
    "FALB": {"cik": 9000006, "name": "Fallback Tag Therapeutics", "filings": [
        ("10-Q", 2025, "Q1", "2025-05-12", [(OCF, "2025-01-01", "2025-03-31", -7)]),
        ("10-Q", 2025, "Q2", "2025-08-11", [(OCF, "2025-04-01", "2025-06-30", -7)]),
        ("10-Q", 2025, "Q3", "2025-11-10", [(CASH_R, None, "2025-09-30", 20),
                                            (MS_C, None, "2025-09-30", 10),
                                            (MS_NC, None, "2025-09-30", 5),
                                            (OCF, "2025-07-01", "2025-09-30", -7)]),
    ]},
    # 7. Cash-flow positive company: not burning, runway infinite.
    "PROF": {"cik": 9000007, "name": "Profitable Example Bio", "filings": [
        ("10-Q", 2025, "Q1", "2025-05-07", [(CASH, None, "2025-03-31", 95),
                                            (OCF, "2025-01-01", "2025-03-31", 5)]),
        ("10-Q", 2025, "Q2", "2025-08-06", [(CASH, None, "2025-06-30", 100),
                                            (OCF, "2025-01-01", "2025-06-30", 8)]),
    ]},
    # 8. Annual data only (no 10-Qs yet).
    "ANNL": {"cik": 9000008, "name": "Annual Only Pharma", "filings": [
        ("10-K", 2025, "FY", "2026-03-20", [(CASH, None, "2025-12-31", 72),
                                            (OCF, "2025-01-01", "2025-12-31", -48)]),
    ]},
    # 9. No cash tags at all -> runway unavailable.
    "NOCF": {"cik": 9000009, "name": "No Cash Facts Inc", "filings": [
        ("10-Q", 2025, "Q1", "2025-05-15", [(OCF, "2025-01-01", "2025-03-31", -3)]),
    ]},
    # 10. Missing Q2 10-Q: Q3 cannot be derived; Q4 can (12M - 9M).
    "GAPQ": {"cik": 9000010, "name": "Gappy Quarters Bio", "filings": [
        ("10-Q", 2025, "Q1", "2025-05-14", [(CASH, None, "2025-03-31", 70),
                                            (OCF, "2025-01-01", "2025-03-31", -8)]),
        ("10-Q", 2025, "Q3", "2025-11-13", [(CASH, None, "2025-09-30", 54),
                                            (OCF, "2025-01-01", "2025-09-30", -27)]),
        ("10-K", 2025, "FY", "2026-03-10", [(CASH, None, "2025-12-31", 45),
                                            (OCF, "2025-01-01", "2025-12-31", -36)]),
    ]},
}


def build(ticker: str, spec: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seq, (form, fy, fp, filed, facts) in enumerate(spec["filings"], start=1):
        rows.extend(filing(form, fy, fp, filed, seq, spec["cik"], facts))
    by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        tag = r.pop("tag")
        by_tag[tag].append(r)
    return {
        "cik": spec["cik"],
        "entityName": spec["name"],
        "facts": {"us-gaap": {
            tag: {"label": tag, "description": "synthetic fixture", "units": {"USD": facts}}
            for tag, facts in sorted(by_tag.items())
        }},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ticker, spec in SPECS.items():
        path = OUT / f"{ticker}.json"
        path.write_text(json.dumps(build(ticker, spec), indent=1) + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(OUT.parents[3])}")


if __name__ == "__main__":
    main()
