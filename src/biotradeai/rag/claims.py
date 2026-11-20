"""Deterministic extraction of a few decision-critical claims from text.

Claims are what conflict resolution compares across sources. They are deliberately
narrow and regex-based (no LLM) so that a conflict, e.g. one source saying a clinical
hold is active and another saying it was lifted, is detected the same way every time.

Claim keys and values:
  clinical_hold      "active" | "lifted"
  primary_endpoint   "met" | "not_met"
  pdufa_date         ISO date "YYYY-MM-DD"
  crl                "received"
"""

from __future__ import annotations

import re
from datetime import datetime

_MONTH = (r"(January|February|March|April|May|June|July|August|September|October|"
          r"November|December)")

_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    ("clinical_hold", "lifted", re.compile(
        r"clinical hold\b[^.]{0,60}\b(lifted|removed|released|resolved)"
        r"|(lifted|removed|released)\s+the\s+(partial\s+|full\s+)?clinical hold", re.I)),
    ("clinical_hold", "active", re.compile(
        r"(placed|put|imposed)\b[^.]{0,40}\b(partial |full )?clinical hold"
        r"|clinical hold (remains|is) in (place|effect)|is (currently )?on (a )?(partial |full )?"
        r"clinical hold", re.I)),
    ("primary_endpoint", "not_met", re.compile(
        r"(did not|failed to|does not) (meet|achieve) (its|the) primary (efficacy )?endpoint",
        re.I)),
    ("primary_endpoint", "met", re.compile(
        r"(met|achieved) (its|the) primary (efficacy )?endpoint", re.I)),
    ("crl", "received", re.compile(r"complete response letter|\bCRL\b", re.I)),
]

_PDUFA = re.compile(
    rf"PDUFA[^.]{{0,60}}?(?P<date>{_MONTH}\s+\d{{1,2}},\s+\d{{4}}|\d{{4}}-\d{{2}}-\d{{2}})",
    re.I,
)


def _parse_date(s: str) -> str | None:
    for fmt in ("%B %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return None


_SENTENCES = re.compile(r"(?<=[.!?])\s+")
# Risk-factor language is hypothetical ("we may receive a CRL"). Such sentences state no fact.
_HEDGE = re.compile(r"\b(may|might|could|would|if|whether|potential(ly)?|possible|"
                    r"cannot assure|no assurance|risk that)\b", re.I)


def extract_claims(text: str) -> dict[str, str]:
    """Return {claim_key: value} from factual (non-hedged) sentences. Where a key appears
    more than once, the last occurrence wins (later sentences usually describe the more
    recent state)."""
    hits: list[tuple[int, int, str, str]] = []
    for si, sent in enumerate(_SENTENCES.split(text)):
        if _HEDGE.search(sent):
            continue
        for key, value, pat in _PATTERNS:
            for m in pat.finditer(sent):
                hits.append((si, m.start(), key, value))
        for m in _PDUFA.finditer(sent):
            d = _parse_date(m.group("date"))
            if d:
                hits.append((si, m.start(), "pdufa_date", d))
    out: dict[str, str] = {}
    for _, _, key, value in sorted(hits):
        out[key] = value
    return out
