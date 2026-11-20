"""Write data/fixtures/backtest/cases.yaml: 12 synthetic, labelled backtest cases.

Every case has documents published BEFORE its decision date (allowed) and one
"leak" document published AFTER it that states the real outcome. The point-in-time
harness must never return the leak. Outcomes are labels for the Brier score.

Synthetic data only proves the harness and metrics work. It says nothing about real
predictive skill; that needs real historical catalysts (Phase 8 / user-curated).

    python scripts/build_backtest_fixtures.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

OUT = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "backtest" / "cases.yaml"

STRONG = ("{drug} Phase 2 results posted on ClinicalTrials.gov. The randomized study met its "
          "primary endpoint, with objective response rate of 44% versus 17% for control. No new "
          "safety signals were observed and discontinuations were low. The Phase 3 trial uses "
          "the same primary endpoint, dose and patient population.")
WEAK = ("{drug} Phase 2 results posted on ClinicalTrials.gov. The randomized study did not meet "
        "its primary endpoint of progression-free survival. A post-hoc subgroup analysis showed "
        "a numerical trend. The Phase 3 trial enrolls a broader population with a new endpoint.")
TEN_Q = ("{drug} Phase 3 trial: topline data are expected in the first half of the year. "
         "Enrollment in the Phase 3 trial is complete. Safety monitoring of {drug} is ongoing "
         "and the data monitoring committee recommended the trial continue.")
HOLD_T1 = ("{drug} Phase 3 status update on ClinicalTrials.gov. The FDA placed the {drug} study "
           "on a full clinical hold after a serious hepatic adverse event. The clinical hold "
           "remains in place and dosing is paused.")
LIFT_T2 = ("In the quarter the FDA lifted the clinical hold on the {drug} Phase 3 study and "
           "dosing of {drug} resumed at all trial sites. Primary endpoint safety review is "
           "ongoing.")
LEAK_OK = "{drug} Phase 3 topline: the study met its primary endpoint with high significance."
LEAK_BAD = "{drug} Phase 3 topline: the study did not meet its primary endpoint."

# id, ticker, drug, event, evidence, outcome, runway months (cash/burn), conflict
CASES = [
    ("bt01", "BTAA", "AA-101", "phase3_readout", "strong", "success", 24, False),
    ("bt02", "BTAB", "AB-202", "phase3_readout", "strong", "success", 18, False),
    ("bt03", "BTAC", "AC-303", "phase3_readout", "strong", "failure", 15, False),
    ("bt04", "BTAD", "AD-404", "phase3_readout", "strong", "success", 30, False),
    ("bt05", "BTAE", "AE-505", "phase3_readout", "weak", "failure", 20, False),
    ("bt06", "BTAF", "AF-606", "phase3_readout", "weak", "failure", 14, False),
    ("bt07", "BTAG", "AG-707", "phase3_readout", "weak", "success", 26, False),
    ("bt08", "BTAH", "AH-808", "phase3_readout", "weak", "failure", 12, False),
    ("bt09", "BTAI", "AI-909", "phase3_readout", "strong", "success", 22, True),
    ("bt10", "BTAJ", "AJ-110", "phase3_readout", "weak", "failure", 16, True),
    ("bt11", "BTAK", "AK-211", "phase3_readout", "strong", "failure", 1, False),
    ("bt12", "BTAL", "AL-312", "pdufa", "strong", "success", 28, False),
]


def case(cid: str, ticker: str, drug: str, event: str, evidence: str, outcome: str,
         runway: int, conflict: bool, i: int) -> dict[str, Any]:
    month = 1 + i % 9
    decision = f"2026-{month:02d}-10T15:00:00+00:00"
    burn = 10_000_000
    docs: list[dict[str, Any]] = [
        {"tier": 1, "published_at": f"2025-{month + 2:02d}-01T00:00:00+00:00",
         "url": f"https://clinicaltrials.gov/study/NCT0999{i:04d}", "doc_type": "ctgov",
         "text": (STRONG if evidence == "strong" else WEAK).format(drug=drug)},
        {"tier": 2, "published_at": f"2025-{month + 3:02d}-15T00:00:00+00:00",
         "url": f"https://www.sec.gov/fixture/{ticker.lower()}-10q.htm", "doc_type": "10-Q",
         "text": TEN_Q.format(drug=drug)},
    ]
    if conflict:
        docs.append({"tier": 1, "published_at": f"2025-{month + 1:02d}-20T00:00:00+00:00",
                     "url": f"https://clinicaltrials.gov/study/NCT0998{i:04d}",
                     "doc_type": "ctgov", "text": HOLD_T1.format(drug=drug)})
        docs.append({"tier": 2, "published_at": f"2025-{month + 3:02d}-20T00:00:00+00:00",
                     "url": f"https://www.sec.gov/fixture/{ticker.lower()}-8k.htm",
                     "doc_type": "8-K", "text": LIFT_T2.format(drug=drug)})
    docs.append({"tier": 3, "published_at": f"2026-{month:02d}-25T00:00:00+00:00",
                 "url": f"https://example.com/{ticker.lower()}-topline", "doc_type": "press",
                 "leak": True,
                 "text": (LEAK_OK if outcome == "success" else LEAK_BAD).format(drug=drug)})
    return {
        "id": cid, "ticker": ticker, "drug": drug,
        "entity_id": f"ENT-{ticker}-{drug.replace('-', '')}",
        "catalyst_event": event, "catalyst_date": f"2026-{month:02d}-24",
        "decision_at": decision, "outcome": outcome, "expect_conflict": conflict,
        "runway": {"cash": runway * burn // 3, "quarterly_burn": burn,
                   "source_accns": [f"0009{i:06d}-25-000010"]},
        "market": {"bid": "9.99", "ask": "10.03"},
        "portfolio": {"equity": "100000", "open_tickers": [], "existing_value": "0"},
        "docs": docs,
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cases = [case(*c, i=i) for i, c in enumerate(CASES, start=1)]
    header = ("# GENERATED by scripts/build_backtest_fixtures.py - do not edit by hand.\n"
              "# Synthetic cases: they validate the harness, not real predictive skill.\n")
    OUT.write_text(header + yaml.safe_dump({"cases": cases}, sort_keys=False, width=100),
                   encoding="utf-8", newline="\n")
    print(f"wrote {len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()
