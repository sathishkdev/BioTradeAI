"""Cash runway from XBRL companyfacts — pure, deterministic code (no LLM).

Definitions
-----------
* **Liquidity** = cash & equivalents + current marketable securities + non-current
  marketable securities, all at the latest balance-sheet date. Within each group the
  first tag in the fallback chain that has a value at that date is used (never summed,
  so overlapping tags cannot double count).
* **Discrete quarterly operating cash flow**: 10-Q cash-flow statements are
  year-to-date, so quarters are derived by differencing YTD facts with the same start
  date (6M - 3M = Q2, 9M - 6M = Q3, 12M - 9M = Q4). Reported 3-month facts are used
  directly.
* **Quarterly burn** = max(latest quarter's burn, mean burn over the last
  ``lookback_quarters``, 0). Taking the max is deliberately conservative for a veto.
  If no quarter can be derived, the latest annual operating cash flow / 4 is used.
* **Runway months** = liquidity / quarterly burn * 3; ``inf`` if not burning cash.

Data hygiene
------------
* Only 10-K/10-Q/20-F (+ amendments) facts are used.
* Duplicate facts for the same period (comparatives repeated in later filings,
  restatements) are resolved by latest ``filed`` date, then accession number — so a
  restatement always wins, deterministically.
* ``as_of`` restricts to facts filed on/before that date (point-in-time backtests).
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

from biotradeai.data.edgar import CompanyFacts, Fact

CASH_TAGS = (
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    "Cash",
)
SECURITIES_CURRENT_TAGS = (
    "MarketableSecuritiesCurrent",
    "ShortTermInvestments",
    "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
)
SECURITIES_NONCURRENT_TAGS = (
    "MarketableSecuritiesNoncurrent",
    "AvailableForSaleSecuritiesDebtSecuritiesNoncurrent",
)
OCF_TAGS = (
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
)
FILING_FORMS = frozenset({"10-K", "10-K/A", "10-Q", "10-Q/A", "10-KT", "20-F", "20-F/A"})

_DAYS_PER_QUARTER = 91.31
_PERIOD_SLACK_DAYS = 15  # tolerance when classifying a duration as N quarters
_MAX_QUARTER_GAP_DAYS = 100  # consecutive quarter ends further apart than this = gap
_STALE_BURN_DAYS = 10  # latest quarter ending this long before balance date = note


class RunwayUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class QuarterOCF:
    start: date
    end: date
    ocf: Decimal  # negative = cash burned
    method: str  # "reported" | "derived" | "annual/4"
    accns: tuple[str, ...]


@dataclass(frozen=True)
class RunwayResult:
    cik: str
    balance_sheet_date: date
    cash: Decimal
    securities_current: Decimal
    securities_noncurrent: Decimal
    quarterly_burn: Decimal
    runway_months: float
    quarters_used: tuple[QuarterOCF, ...]
    source_accns: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def liquidity(self) -> Decimal:
        return self.cash + self.securities_current + self.securities_noncurrent

    @property
    def burning(self) -> bool:
        return self.quarterly_burn > 0


# --------------------------------------------------------------------------- helpers


def _dedupe(facts: list[Fact], as_of: date | None,
            notes: list[str]) -> dict[tuple[date | None, date], Fact]:
    best: dict[tuple[date | None, date], Fact] = {}
    for f in facts:
        if f.form.upper() not in FILING_FORMS:
            continue
        if as_of is not None and f.filed > as_of:
            continue
        key = (f.start, f.end)
        cur = best.get(key)
        if cur is None or (f.filed, f.accn) > (cur.filed, cur.accn):
            if cur is not None and cur.val != f.val:
                older, newer = (cur, f)
                notes.append(
                    f"{f.tag} {f.start or ''}..{f.end}: {older.val} ({older.accn}) "
                    f"superseded by {newer.val} ({newer.accn})"
                )
            best[key] = f
        elif cur.val != f.val:
            notes.append(
                f"{f.tag} {f.start or ''}..{f.end}: {f.val} ({f.accn}) "
                f"superseded by {cur.val} ({cur.accn})"
            )
    return best


def _instants(cf: CompanyFacts, tag: str, as_of: date | None,
              notes: list[str]) -> dict[date, Fact]:
    return {end: f for (start, end), f in _dedupe(cf.get(tag), as_of, notes).items()
            if start is None}


def _pick(cf: CompanyFacts, tags: tuple[str, ...], on: date, as_of: date | None,
          notes: list[str]) -> Fact | None:
    for tag in tags:
        f = _instants(cf, tag, as_of, notes).get(on)
        if f is not None:
            return f
    return None


def _n_quarters(start: date, end: date) -> int | None:
    days = (end - start).days + 1
    n = round(days / _DAYS_PER_QUARTER)
    if n < 1 or n > 4 or abs(days - n * _DAYS_PER_QUARTER) > _PERIOD_SLACK_DAYS:
        return None
    return n


def discrete_quarters(cf: CompanyFacts, as_of: date | None = None,
                      notes: list[str] | None = None) -> list[QuarterOCF]:
    """All derivable discrete quarters of operating cash flow, oldest first."""
    notes = notes if notes is not None else []
    durations: dict[tuple[date, date], Fact] = {}
    for tag in OCF_TAGS:
        d = {(s, e): f for (s, e), f in _dedupe(cf.get(tag), as_of, notes).items()
             if s is not None}
        if d:
            durations = d
            break

    by_start: dict[date, list[tuple[int, Fact]]] = defaultdict(list)
    quarters: dict[date, QuarterOCF] = {}
    for (start, end), f in durations.items():
        n = _n_quarters(start, end)
        if n is None:
            continue
        by_start[start].append((n, f))
        if n == 1:
            quarters[end] = QuarterOCF(start, end, f.val, "reported", (f.accn,))

    for _start, items in by_start.items():
        items.sort(key=lambda t: t[0])
        for (n_prev, prev), (n_cur, cur) in itertools.pairwise(items):
            if n_cur != n_prev + 1 or cur.end in quarters:
                continue
            quarters[cur.end] = QuarterOCF(
                prev.end + timedelta(days=1), cur.end, cur.val - prev.val, "derived",
                (prev.accn, cur.accn),
            )
    return [quarters[k] for k in sorted(quarters)]


def _annual_fallback(cf: CompanyFacts, on: date, as_of: date | None,
                     notes: list[str]) -> QuarterOCF | None:
    for tag in OCF_TAGS:
        annual = [f for (s, e), f in _dedupe(cf.get(tag), as_of, notes).items()
                  if s is not None and e <= on and _n_quarters(s, e) == 4]
        if annual:
            f = max(annual, key=lambda x: x.end)
            return QuarterOCF(f.start or f.end, f.end, f.val / 4, "annual/4", (f.accn,))
    return None


# --------------------------------------------------------------------------- main


def compute_runway(cf: CompanyFacts, *, lookback_quarters: int,
                   as_of: date | None = None) -> RunwayResult:
    if lookback_quarters < 1:
        raise ValueError("lookback_quarters must be >= 1")
    notes: list[str] = []

    dates: set[date] = set()
    for tag in CASH_TAGS:
        dates.update(_instants(cf, tag, as_of, notes))
    if not dates:
        raise RunwayUnavailableError(f"CIK {cf.cik}: no cash facts (tags {CASH_TAGS})")
    bs_date = max(dates)

    cash_f = _pick(cf, CASH_TAGS, bs_date, as_of, notes)
    if cash_f is None:  # pragma: no cover - bs_date came from these tags
        raise RunwayUnavailableError("cash fact vanished")
    if cash_f.tag != CASH_TAGS[0]:
        notes.append(f"cash taken from fallback tag {cash_f.tag}")
    cur_f = _pick(cf, SECURITIES_CURRENT_TAGS, bs_date, as_of, notes)
    non_f = _pick(cf, SECURITIES_NONCURRENT_TAGS, bs_date, as_of, notes)
    if cur_f is None:
        notes.append(f"no current marketable securities at {bs_date}; treated as 0")
    if non_f is None:
        notes.append(f"no non-current marketable securities at {bs_date}; treated as 0")

    all_q = [q for q in discrete_quarters(cf, as_of, notes) if q.end <= bs_date]
    used = list(reversed(all_q))[:lookback_quarters]  # newest first
    if not used:
        fb = _annual_fallback(cf, bs_date, as_of, notes)
        if fb is None:
            raise RunwayUnavailableError(f"CIK {cf.cik}: no operating cash flow facts")
        notes.append("no discrete quarter derivable; used annual operating cash flow / 4")
        used = [fb]
    else:
        if len(used) < lookback_quarters:
            notes.append(f"only {len(used)} of {lookback_quarters} quarters available")
        for newer, older in itertools.pairwise(used):
            if (newer.end - older.end).days > _MAX_QUARTER_GAP_DAYS:
                notes.append(f"gap in quarterly data between {older.end} and {newer.end}")
    if (bs_date - used[0].end).days > _STALE_BURN_DAYS:
        notes.append(f"latest burn quarter ends {used[0].end}, before balance date {bs_date}")

    latest_burn = -used[0].ocf
    avg_burn = -sum((q.ocf for q in used), Decimal(0)) / Decimal(len(used))
    burn = max(latest_burn, avg_burn, Decimal(0))

    cash = cash_f.val
    sc = cur_f.val if cur_f else Decimal(0)
    snc = non_f.val if non_f else Decimal(0)
    liquidity = cash + sc + snc
    months = math.inf if burn == 0 else float(liquidity / burn * 3)

    accns = {cash_f.accn}
    accns.update(f.accn for f in (cur_f, non_f) if f is not None)
    for q in used:
        accns.update(q.accns)
    return RunwayResult(
        cik=cf.cik,
        balance_sheet_date=bs_date,
        cash=cash,
        securities_current=sc,
        securities_noncurrent=snc,
        quarterly_burn=burn,
        runway_months=months,
        quarters_used=tuple(used),
        source_accns=tuple(sorted(accns)),
        notes=tuple(dict.fromkeys(notes)),  # de-duplicate, keep order
    )
