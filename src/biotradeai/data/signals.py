"""Deterministic text signals from SEC filings / press releases (no LLM).

* ``find_catalysts``: sentences announcing an upcoming PDUFA date, advisory committee,
  or Phase 2/3 topline readout, with the date or window parsed into exact dates.
* ``find_outcomes``: endpoint met / not met, CRL, FDA approval, commercial launch,
  program discontinued.
* ``find_material_changes``: acquisition/merger, licensing of the drug.
* ``drug_mentions``: pipeline codes (ABC-123) and INN-style generic names (-mab, -nib).

Every hit keeps the exact sentence, so a human (approval card, order approval) can see
precisely what the agent read. Hedged/hypothetical sentences are skipped where that
matters (outcomes), reusing the RAG claim extractor's hedge rule.
"""

from __future__ import annotations

import re
from calendar import monthrange
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from biotradeai.rag.claims import extract_claims

_SENTENCES = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")
_HEDGE = re.compile(r"\b(may|might|could|would|if|whether|potential(ly)?|possible|"
                    r"cannot assure|no assurance|risk that)\b", re.I)

_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december"]
_MONTH_RE = "(" + "|".join(m.capitalize() for m in _MONTHS) + ")"


class Precision(StrEnum):
    DAY = "day"
    MONTH = "month"
    QUARTER = "quarter"
    HALF = "half"
    YEAR = "year"


@dataclass(frozen=True)
class When:
    start: date
    end: date
    precision: Precision
    text: str

    @property
    def exact(self) -> date | None:
        return self.start if self.precision is Precision.DAY else None


def _qend(year: int, q: int) -> date:
    m = q * 3
    return date(year, m, monthrange(year, m)[1])


_ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "1st": 1, "2nd": 2, "3rd": 3,
        "4th": 4}
_PERIODS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(rf"\b{_MONTH_RE}\s+(\d{{1,2}}),\s+(20\d\d)\b"), "day"),
    (re.compile(r"\b(20\d\d)-(\d\d)-(\d\d)\b"), "iso"),
    (re.compile(r"\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\s+(?:of\s+)?(20\d\d)\b",
                re.I), "q_words"),
    (re.compile(r"\b(?:Q([1-4])|([1-4])Q)\s*[-']?\s*(20\d\d)\b"), "q"),
    (re.compile(r"\b(first|second|1st|2nd)\s+half\s+(?:of\s+)?(20\d\d)\b", re.I), "h_words"),
    (re.compile(r"\b(?:H([12])|([12])H)\s*[-']?\s*(20\d\d)\b"), "h"),
    (re.compile(r"\bmid[- ](20\d\d)\b", re.I), "mid"),
    (re.compile(r"\b(?:year[- ]end|end of)\s+(20\d\d)\b", re.I), "yearend"),
    (re.compile(rf"\b(?:in\s+)?{_MONTH_RE}\s+(20\d\d)\b"), "month"),
]


_NEW_CUE = re.compile(r"\b(to|until|now|new(ly)?|revised|is)\s+(?:the\s+)?$", re.I)
_OLD_CUE = re.compile(r"\b(from|previously|originally|prior|old)\s+(?:the\s+)?$", re.I)


def _pick(matches: list[re.Match[str]], text: str) -> re.Match[str]:
    """When a sentence states two dates ('moved from March 15 to June 15'), the new one
    is the date after 'to/until/now'; otherwise the one NOT after 'from'; else the last."""
    if len(matches) == 1:
        return matches[0]
    for m in matches:
        if _NEW_CUE.search(text[max(0, m.start() - 20): m.start()]):
            return m
    rest = [m for m in matches if not _OLD_CUE.search(text[max(0, m.start() - 20): m.start()])]
    return (rest or matches)[-1]


def parse_when(text: str) -> When | None:
    """The most precise date or window mentioned in ``text``. If a sentence gives two
    dates of the same precision (a change), the new date is chosen (``_pick``)."""
    for pat, kind in _PERIODS:
        found = list(pat.finditer(text))
        if not found:
            continue
        m = _pick(found, text)
        g = m.groups()
        try:
            if kind == "day":
                d = date(int(g[2]), _MONTHS.index(g[0].lower()) + 1, int(g[1]))
                return When(d, d, Precision.DAY, m.group(0))
            if kind == "iso":
                d = date(int(g[0]), int(g[1]), int(g[2]))
                return When(d, d, Precision.DAY, m.group(0))
            if kind in ("q_words", "q"):
                q = _ORD[g[0].lower()] if kind == "q_words" else int(g[0] or g[1])
                y = int(g[-1])
                return When(date(y, q * 3 - 2, 1), _qend(y, q), Precision.QUARTER, m.group(0))
            if kind in ("h_words", "h"):
                h = _ORD[g[0].lower()] if kind == "h_words" else int(g[0] or g[1])
                y = int(g[-1])
                return When(date(y, 1 if h == 1 else 7, 1),
                            date(y, 6, 30) if h == 1 else date(y, 12, 31),
                            Precision.HALF, m.group(0))
            if kind == "mid":
                y = int(g[0])
                return When(date(y, 4, 1), date(y, 9, 30), Precision.HALF, m.group(0))
            if kind == "yearend":
                y = int(g[0])
                return When(date(y, 10, 1), date(y, 12, 31), Precision.QUARTER, m.group(0))
            if kind == "month":
                y, mo = int(g[1]), _MONTHS.index(g[0].lower()) + 1
                return When(date(y, mo, 1), date(y, mo, monthrange(y, mo)[1]),
                            Precision.MONTH, m.group(0))
        except ValueError:
            continue
    return None


# --------------------------------------------------------------------------- drug names

_CODE = re.compile(r"\b([A-Z]{1,6}-?\d{2,6}[A-Z]?)\b")
_NOT_DRUG = re.compile(r"^(COVID-?19|SARS-?\d+|Q[1-4]|H[12]|FY\d+|NCT\d+|ISO\d+|CFR\d*|"
                       r"\d+[A-Z]?|[A-Z]{1,2}\d{1,2})$")
_INN = re.compile(r"\b([a-z]{3,}(?:mab|nib|tinib|ciclib|parib|lisib|tide|glutide|cel|gene|"
                  r"vec|stat|sertib|rafenib|zumab|ximab|umab|gliflozin|sentan|"
                  r"tamab|kinra|cept|ase))\b", re.I)


def drug_mentions(text: str) -> list[str]:
    """Candidate drug names in order of appearance (codes first-class, then INN names)."""
    out: list[str] = []
    for m in _CODE.finditer(text):
        code = m.group(1)
        if not _NOT_DRUG.match(code) and code not in out:
            out.append(code)
    for m in _INN.finditer(text):
        name = m.group(1).lower()
        if name not in out and name not in _NOT_INN and len(name) >= 6:
            out.append(name)
    return out


# Ordinary words that happen to end like drug names (-ase, -cept, -cel, -stat, -gene ...).
_NOT_INN = frozenset({
    "accept", "base", "bases", "cancel", "case", "cases", "cease", "chase", "concept",
    "concepts", "database", "decrease", "decreases", "disease", "diseases", "ease", "erase",
    "excel", "except", "increase", "increases", "intercept", "lease", "lowercase", "parcel",
    "phase", "phases", "please", "precept", "purchase", "purchases", "release", "releases",
    "showcase", "thermostat", "upcase"
})


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCES.split(" ".join(text.split())) if s.strip()]


# --------------------------------------------------------------------------- catalysts

_PDUFA = re.compile(r"\bPDUFA\b|target action date|goal date", re.I)
_ADCOM = re.compile(r"advisory committee", re.I)
_READOUT = re.compile(r"\b(top-?line|data|results|read-?out)\b", re.I)
_PH3 = re.compile(r"\bphase\s*(3|III)\b|\bpivotal\b|\bregistrational\b", re.I)
_PH2 = re.compile(r"\bphase\s*(2|II)(b)?\b", re.I)
_PAST = re.compile(r"\b(announced|reported|presented|released)\b[^.]{0,60}?"
                   r"\b(data|results)\b", re.I)


@dataclass(frozen=True)
class CatalystMention:
    event_type: str  # PDUFA | ADCOM | PHASE3_READOUT | PHASE2_READOUT
    when: When
    drugs: tuple[str, ...]
    sentence: str


# Safe-harbor / risk-factor boilerplate lists many programs and dates in one sentence;
# it states no fact about a specific catalyst, so it is never a source.
_BOILERPLATE = re.compile(
    r"forward-looking|risks? and uncertaint|including,? but not limited to|"
    r"include,? but (?:are )?not limited to|no assurance|cannot assure|could cause|"
    r"may disagree|safe harbor|actual results", re.I)


def find_catalysts(text: str) -> list[CatalystMention]:
    out: list[CatalystMention] = []
    seen: set[tuple[str, object, object, tuple[str, ...]]] = set()
    prev = ""
    for s in sentences(text):
        when = parse_when(s)
        if when is not None and _BOILERPLATE.search(s):
            prev = s
            continue
        if when is not None:
            kind = None
            if _PDUFA.search(s):
                kind = "PDUFA"
            elif _ADCOM.search(s):
                kind = "ADCOM"
            elif _READOUT.search(s) and not _PAST.search(s):
                kind = "PHASE3_READOUT" if _PH3.search(s) else (
                    "PHASE2_READOUT" if _PH2.search(s) else None)
            if kind:
                drugs = tuple(drug_mentions(s) or drug_mentions(prev))
                key = (kind, when.start, when.end, drugs)
                if key not in seen:  # press releases repeat the same date in headline/body
                    seen.add(key)
                    out.append(CatalystMention(kind, when, drugs, s))
        prev = s
    return out


# --------------------------------------------------------------------------- outcomes

class OutcomeKind(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    CRL = "crl"
    APPROVED = "approved"
    LAUNCHED = "launched"
    DISCONTINUED = "discontinued"


_APPROVED = re.compile(r"\b(FDA|Food and Drug Administration)\b[^.]{0,40}\b(approved|granted"
                       r"\s+(full\s+|accelerated\s+|traditional\s+)?approval)", re.I)
_LAUNCH = re.compile(r"\b(commercial(ly)?\s+launch(ed)?|launched\s+commercially|"
                     r"(is\s+)?now\s+(commercially\s+)?available\s+(in|to|for)\b)", re.I)
_DISCONTINUE = re.compile(r"\b(discontinu(e|ed|ing)|terminat(e|ed|ing)|halt(ed)?)\b[^.]{0,40}"
                          r"\b(development|program|programme|study|trial)\b", re.I)


@dataclass(frozen=True)
class OutcomeMention:
    kind: OutcomeKind
    drugs: tuple[str, ...]
    sentence: str


def find_outcomes(text: str) -> list[OutcomeMention]:
    out: list[OutcomeMention] = []
    for s in sentences(text):
        if _HEDGE.search(s):
            continue
        drugs = tuple(drug_mentions(s))
        claims = extract_claims(s)
        if claims.get("primary_endpoint") == "met":
            out.append(OutcomeMention(OutcomeKind.POSITIVE, drugs, s))
        elif claims.get("primary_endpoint") == "not_met":
            out.append(OutcomeMention(OutcomeKind.NEGATIVE, drugs, s))
        if claims.get("crl") == "received" and re.search(r"\breceived\b|\bissued\b", s, re.I):
            out.append(OutcomeMention(OutcomeKind.CRL, drugs, s))
        if _APPROVED.search(s):
            out.append(OutcomeMention(OutcomeKind.APPROVED, drugs, s))
        if _LAUNCH.search(s) and not re.search(r"\b(expect|plan|anticipate|prepar)", s, re.I):
            out.append(OutcomeMention(OutcomeKind.LAUNCHED, drugs, s))
        if _DISCONTINUE.search(s):
            out.append(OutcomeMention(OutcomeKind.DISCONTINUED, drugs, s))
    return out


# --------------------------------------------------------------------------- material change

_ACQUIRE = re.compile(r"definitive\s+(merger\s+)?agreement|agreement\s+and\s+plan\s+of\s+merger"
                      r"|tender\s+offer|(to\s+be|will\s+be|has\s+been)\s+acquired", re.I)
_LICENSE = re.compile(r"\b(exclusive\s+)?(worldwide\s+)?licens(e|ing)\s+agreement|"
                      r"\bsold\s+(the\s+)?(worldwide\s+)?rights\b|\bout-?licens", re.I)


@dataclass(frozen=True)
class ChangeMention:
    kind: str  # "acquisition" | "license"
    drugs: tuple[str, ...]
    sentence: str


def find_material_changes(text: str) -> list[ChangeMention]:
    out: list[ChangeMention] = []
    for s in sentences(text):
        if _HEDGE.search(s):
            continue
        if _ACQUIRE.search(s):
            out.append(ChangeMention("acquisition", tuple(drug_mentions(s)), s))
        elif _LICENSE.search(s):
            out.append(ChangeMention("license", tuple(drug_mentions(s)), s))
    return out


def mentions_any(sentence_drugs: tuple[str, ...], names: tuple[str, ...]) -> bool:
    want = {re.sub(r"[\s\-_.]", "", n).lower() for n in names}
    return any(re.sub(r"[\s\-_.]", "", d).lower() in want for d in sentence_drugs)
