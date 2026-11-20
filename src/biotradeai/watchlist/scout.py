"""Watchlist scout: proposes company+drug pairs with an evidence card. Never approves.

Signals (both free, official, incremental):
  1. SEC daily form index -> 8-Ks from listed biotech companies (industry codes 2834,
     2836, 8731) -> sentences announcing an upcoming PDUFA / Phase 3 / Phase 2 readout.
  2. ClinicalTrials.gov -> industry-sponsored Phase 3 trials whose primary completion is
     3-12 months out -> lead sponsor matched to a listed company by name.

Then, in code: merge by company + drug, drop pairs already on the watchlist (rejected or
retired pairs come back only with a material change), filter (runway, supported event,
date window, optional spread), rank, cap, and write proposals with a review card.

Load is kept low and flat: one index file per business day, company records cached for
30 days, only new filings read, every per-item step paced (``data/pacing.py``).
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

from biotradeai.brokers.base import Broker
from biotradeai.config import GuardrailsConfig
from biotradeai.data.clinicaltrials import ClinicalTrialsClient
from biotradeai.data.edgar import EdgarClient
from biotradeai.data.http import FetchError
from biotradeai.data.pacing import paced
from biotradeai.data.runway import RunwayUnavailableError, compute_runway
from biotradeai.data.secindex import Company, SecDirectory, business_days, submission_text
from biotradeai.data.signals import CatalystMention, find_catalysts
from biotradeai.guardrails import rules
from biotradeai.store.sqlite_store import SQLiteStore
from biotradeai.watchlist.catalysts import SUPPORTED
from biotradeai.watchlist.model import (
    DEFAULT_SOFT_CAP,
    Status,
    WatchEntry,
    Watchlist,
    drug_key,
    entity_id_for,
)

MIN_LEAD_DAYS = 14
MAX_LEAD_DAYS = 365
CT_WINDOW = (60, 365)  # days ahead for Phase 3 primary completion
INITIAL_LOOKBACK_DAYS = 7
MAX_LOOKBACK_DAYS = 14
LARGE_SPONSOR_TRIALS = 10  # many Phase 3s = diversified; one readout rarely moves it
_NON_DRUG = re.compile(r"placebo|standard of care|best supportive|vehicle|saline|"
                       r"observation|usual care", re.I)
_SUFFIX = re.compile(r"\b(incorporated|inc|corporation|corp|company|co|ltd|limited|plc|"
                     r"holdings?|group|sa|ag|nv|se|llc|lp|the)\b")
_GOING_CONCERN = re.compile(r"substantial doubt[^.]{0,80}going concern", re.I)
_ATM = re.compile(r"at[- ]the[- ]market", re.I)


def norm_company(name: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", name.lower())
    return " ".join(_SUFFIX.sub(" ", s).split())


@dataclass
class Candidate:
    cik: str
    company: Company
    names: list[str]
    sec: list[dict[str, Any]] = field(default_factory=list)  # catalyst mentions
    trials: list[dict[str, Any]] = field(default_factory=list)
    match: str = ""
    sponsor_trials: int = 0
    flags: list[str] = field(default_factory=list)
    filtered: str | None = None
    runway: dict[str, Any] | None = None
    market: dict[str, Any] | None = None
    score: float = 0.0

    @property
    def key(self) -> str:
        return drug_key(self.names[0])


@dataclass
class ScoutReport:
    started_at: datetime
    days_scanned: list[str] = field(default_factory=list)
    filings_read: int = 0
    trials_seen: int = 0
    candidates: int = 0
    filtered: dict[str, int] = field(default_factory=dict)
    proposed: list[str] = field(default_factory=list)
    cap_reached: bool = False
    errors: list[str] = field(default_factory=list)


class Scout:
    def __init__(self, *, edgar: EdgarClient, ctgov: ClinicalTrialsClient, store: SQLiteStore,
                 watchlist: Watchlist, cfg: GuardrailsConfig, today: date,
                 broker: Broker | None = None, spread_minutes: float = 0.0,
                 max_proposals: int = 10, soft_cap: int = DEFAULT_SOFT_CAP,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.edgar, self.ctgov, self.store, self.wl = edgar, ctgov, store, watchlist
        self.dir = SecDirectory(edgar, store)
        self.cfg, self.today, self.broker = cfg, today, broker
        self.window_s = spread_minutes * 60
        self.max_proposals, self.soft_cap, self._sleep = max_proposals, soft_cap, sleep
        self._cands: dict[tuple[str, str], Candidate] = {}

    # ------------------------------------------------------------------ helpers

    def _cand(self, company: Company, names: list[str]) -> Candidate:
        for c in self._cands.values():
            if c.cik == company.cik and {drug_key(n) for n in names} & \
                    {drug_key(n) for n in c.names}:
                for n in names:
                    if drug_key(n) not in {drug_key(x) for x in c.names}:
                        c.names.append(n)
                return c
        c = Candidate(company.cik, company, list(dict.fromkeys(names)))
        self._cands[(company.cik, c.key)] = c
        return c

    def _in_window(self, m: CatalystMention) -> bool:
        return (m.when.end >= self.today + timedelta(days=MIN_LEAD_DAYS)
                and m.when.start <= self.today + timedelta(days=MAX_LEAD_DAYS))

    # ------------------------------------------------------------------ signal 1: SEC

    def scan_sec(self, rep: ScoutReport) -> None:
        last = self.store.kv_get("scout.last_index_day")
        start = (date.fromisoformat(last) + timedelta(days=1) if last
                 else self.today - timedelta(days=INITIAL_LOOKBACK_DAYS))
        start = max(start, self.today - timedelta(days=MAX_LOOKBACK_DAYS))
        filings = []
        for d in business_days(start, self.today):
            try:
                day = self.dir.filings(d)
            except FetchError as exc:
                rep.errors.append(f"index {d}: {exc}")
                continue
            rep.days_scanned.append(d.isoformat())
            filings += [f for f in day if f.form in ("8-K", "8-K/A")]
            self.store.kv_set("scout.last_index_day", d.isoformat())
        for f in paced(filings, window_s=self.window_s, sleep=self._sleep):
            try:
                co = self.dir.company(f.cik)
                if not (co.is_biotech and co.is_listed):
                    continue
                text = submission_text(self.edgar.archive_text(f.path))
            except FetchError as exc:
                rep.errors.append(f"{f.cik}: {exc}"[:200])
                continue
            rep.filings_read += 1
            for m in find_catalysts(text):
                if not m.drugs or not self._in_window(m):
                    continue
                c = self._cand(co, list(m.drugs[:3]))
                c.sec.append({"event": m.event_type, "start": m.when.start.isoformat(),
                              "end": m.when.end.isoformat(), "precision": m.when.precision.value,
                              "when": m.when.text, "sentence": m.sentence[:500],
                              "url": f.url, "filed": f.filed.isoformat()})
                if _GOING_CONCERN.search(text) and "going-concern language in 8-K" not in c.flags:
                    c.flags.append("going-concern language in 8-K")
                if _ATM.search(text) and "at-the-market offering mentioned" not in c.flags:
                    c.flags.append("at-the-market offering mentioned")

    # ------------------------------------------------------------------ signal 2: CT.gov

    def scan_ctgov(self, rep: ScoutReport) -> None:
        lo = self.today + timedelta(days=CT_WINDOW[0])
        hi = self.today + timedelta(days=CT_WINDOW[1])
        adv = (f"AREA[Phase]PHASE3 AND AREA[LeadSponsorClass]INDUSTRY AND "
               f"AREA[PrimaryCompletionDate]RANGE[{lo.isoformat()},{hi.isoformat()}]")
        by_sponsor: dict[str, list[dict[str, Any]]] = {}
        try:
            for rec in self.ctgov.search(advanced=adv, statuses=(
                    "RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION"),
                    page_size=100, max_pages=10):
                rep.trials_seen += 1
                ps = rec.raw.get("protocolSection") or {}
                drugs = [i.get("name", "") for i in
                         (ps.get("armsInterventionsModule") or {}).get("interventions") or []
                         if str(i.get("type", "")).upper() in ("DRUG", "BIOLOGICAL", "GENETIC",
                                                                 "COMBINATION_PRODUCT")
                         and i.get("name") and not _NON_DRUG.search(i["name"])]
                if not drugs:
                    continue
                endpoint = ((ps.get("outcomesModule") or {}).get("primaryOutcomes")
                            or [{}])[0].get("measure", "")
                enroll = ((ps.get("designModule") or {}).get("enrollmentInfo") or {}).get(
                    "count")
                by_sponsor.setdefault(rec.sponsor, []).append({
                    "nct": rec.nct_id, "title": rec.title[:200], "status": rec.overall_status,
                    "drugs": drugs[:4], "endpoint": str(endpoint)[:200], "enrollment": enroll,
                    "conditions": list(rec.conditions[:3]),
                    "primary_completion": rec.primary_completion.raw
                    if rec.primary_completion else None})
        except FetchError as exc:
            rep.errors.append(f"ctgov search: {exc}"[:200])
            return
        if not by_sponsor:
            return
        try:
            rows = self.edgar.ticker_exchange_rows()
        except FetchError as exc:
            rep.errors.append(f"ticker list: {exc}"[:200])
            return
        names: dict[str, str] = {}
        for row_cik, name, _ticker, _exch in rows:
            names.setdefault(norm_company(name), row_cik)
        for sponsor, trials in paced(list(by_sponsor.items()), window_s=self.window_s,
                                     sleep=self._sleep):
            n = norm_company(sponsor)
            cik: str | None = names.get(n)
            how = "exact name"
            if cik is None:
                hits = {c for k, c in names.items() if k and (k.startswith(n + " ")
                                                              or n.startswith(k + " "))}
                cik, how = (hits.pop(), "partial name") if len(hits) == 1 else (None, "")
            if cik is None:
                continue
            try:
                co = self.dir.company(cik)
            except FetchError as exc:
                rep.errors.append(f"{cik}: {exc}"[:200])
                continue
            if not (co.is_biotech and co.is_listed):
                continue
            for t in trials:
                c = self._cand(co, t["drugs"][:2])
                c.trials.append(t)
                c.sponsor_trials = len(trials)
                c.match = f"CT.gov sponsor '{sponsor}' -> {co.name} ({how})"

    # ------------------------------------------------------------------ enrich + rank

    def _enrich(self, c: Candidate) -> None:
        try:
            r = compute_runway(self.edgar.company_facts(c.cik),
                               lookback_quarters=self.cfg.runway.burn_lookback_quarters,
                               as_of=self.today)
            c.runway = {"months": round(r.runway_months, 1), "cash": str(r.liquidity),
                        "quarterly_burn": str(r.quarterly_burn),
                        "balance_sheet_date": r.balance_sheet_date.isoformat()}
            if not rules.check_runway(r.runway_months, self.cfg).ok:
                c.filtered = f"runway {r.runway_months:.1f} mo < {self.cfg.min_runway_months}"
        except (RunwayUnavailableError, FetchError) as exc:
            c.runway = {"error": str(exc)[:160]}
            c.flags.append("runway could not be computed")
        try:
            subs = self.edgar.submissions(c.cik)
            recent = [f for f in subs.filings
                      if f.filing_date >= self.today - timedelta(days=90)
                      and (f.form.startswith("S-3") or f.form.startswith("424B"))]
            if recent:
                c.flags.append(f"{len(recent)} shelf/prospectus filing(s) in 90 days "
                               f"({recent[0].form} {recent[0].filing_date})")
        except FetchError:
            pass
        if self.broker is not None and c.company.ticker:
            try:
                q = self.broker.get_quote(c.company.ticker)
                spread = rules.spread_pct(q.bid, q.ask)
                c.market = {"bid": str(q.bid), "ask": str(q.ask),
                            "spread_pct": None if spread is None else round(spread, 4)}
                if spread is None or not rules.check_spread(bid=q.bid, ask=q.ask,
                                                            cfg=self.cfg).ok:
                    c.filtered = c.filtered or f"spread {spread} above limit or no quote"
                shares = _shares_outstanding(self.edgar, c.cik)
                if shares:
                    c.market["market_cap"] = str(int(shares * (q.bid + q.ask) / 2))
            except Exception as exc:
                c.market = {"error": str(exc)[:120]}

    @staticmethod
    def _score(c: Candidate) -> float:
        s = 0.0
        if c.sec and c.trials:
            s += 3  # two independent sources agree
        if any(x["precision"] == "day" for x in c.sec):
            s += 2
        if any(x["event"] == "PDUFA" for x in c.sec):
            s += 1
        if any(x["event"] in SUPPORTED for x in c.sec) or c.trials:
            s += 1
        if c.runway and float(c.runway.get("months", 0) or 0) >= 12:
            s += 1
        if c.sponsor_trials > LARGE_SPONSOR_TRIALS:
            s -= 3
        elif 0 < c.sponsor_trials <= 2:
            s += 1
        s -= 0.5 * len([f for f in c.flags if "runway could not" not in f])
        return s

    # ------------------------------------------------------------------ run

    def run(self) -> ScoutReport:
        rep = ScoutReport(datetime.now(UTC))
        self.scan_sec(rep)
        self.scan_ctgov(rep)
        rep.candidates = len(self._cands)
        if self.wl.active_count() >= self.soft_cap:
            rep.cap_reached = True
            self._audit(rep)
            return rep
        fresh: list[tuple[Candidate, str | None]] = []
        for c in self._cands.values():
            cur = self.wl.find(c.cik, c.names)
            material = None
            if cur is not None and cur.status in (Status.REJECTED, Status.RETIRED):
                seen = {x.get("event") for x in cur.card.get("catalysts", [])}
                new = {x["event"] for x in c.sec} - seen
                material = f"new catalyst type: {', '.join(sorted(new))}" if new else None
                if material is None:
                    rep.filtered["previously decided"] = rep.filtered.get(
                        "previously decided", 0) + 1
                    continue
            elif cur is not None:
                continue
            if not c.company.ticker:
                continue
            fresh.append((c, material))
        prelim = sorted(fresh, key=lambda t: -self._score(t[0]))[: self.max_proposals * 2]
        for c, _ in paced(prelim, window_s=self.window_s, sleep=self._sleep):
            self._enrich(c)
            c.score = self._score(c)
        room = min(self.max_proposals, self.soft_cap - self.wl.active_count())
        ranked = sorted((t for t in prelim if t[0].filtered is None), key=lambda t: -t[0].score)
        for c, _ in prelim:
            if c.filtered:
                k = c.filtered.split(" ")[0]
                rep.filtered[k] = rep.filtered.get(k, 0) + 1
        for c, material in ranked[: max(room, 0)]:
            entry = self._entry(c)
            if self.wl.add_proposal(entry, material_change=material):
                rep.proposed.append(entry.id)
                self.store.append_audit("watchlist.proposed", {
                    "id": entry.id, "score": entry.score, "material_change": material,
                    "sources": entry.card.get("sources")}, ticker=entry.ticker)
        self.wl.save()
        self._audit(rep)
        return rep

    def _audit(self, rep: ScoutReport) -> None:
        self.store.append_audit("watchlist.scout", {
            "days": rep.days_scanned, "filings_read": rep.filings_read,
            "trials_seen": rep.trials_seen, "candidates": rep.candidates,
            "filtered": rep.filtered, "proposed": rep.proposed, "cap_reached": rep.cap_reached,
            "errors": rep.errors[:20]})

    def _entry(self, c: Candidate) -> WatchEntry:
        ticker = c.company.ticker or c.cik
        primary = next((n for n in c.names if re.search(r"\d", n)), c.names[0])
        names = [primary, *[n for n in c.names if n != primary]]
        sources = (["SEC 8-K"] if c.sec else []) + (["ClinicalTrials.gov"] if c.trials else [])
        card = {
            "sources": sources,
            "company": {"name": c.company.name, "ticker": ticker, "cik": c.cik,
                        "sic": c.company.sic, "exchanges": list(c.company.exchanges)},
            "match": c.match or "named in the company's own 8-K",
            "runway": c.runway, "market": c.market,
            "drug": {"names": names, "trials": c.trials[:5]},
            "catalysts": c.sec[:5],
            "ctgov_note": "ClinicalTrials.gov dates are sponsor estimates, never used as "
                          "catalyst dates" if c.trials else None,
            "concentration": {"sponsor_phase3_trials_in_window": c.sponsor_trials},
            "red_flags": c.flags,
            "fit": {"supported_event": any(x["event"] in SUPPORTED for x in c.sec)
                    or bool(c.trials),
                    "events": sorted({x["event"] for x in c.sec})},
        }
        return WatchEntry(id=f"{ticker}:{primary}", cik=c.cik, ticker=ticker,
                          company=c.company.name, entity_id=entity_id_for(ticker, primary),
                          drug_names=names, trial_ids=[t["nct"] for t in c.trials][:5],
                          score=round(c.score, 2), proposed_at=datetime.now(UTC), card=card)


def _shares_outstanding(edgar: EdgarClient, cik: str) -> Decimal | None:
    """dei:EntityCommonStockSharesOutstanding, latest value (from the cached facts call)."""
    try:
        raw = edgar.company_facts(cik).raw
    except FetchError:
        return None
    units = (((raw.get("facts") or {}).get("dei") or {}).get(
        "EntityCommonStockSharesOutstanding") or {}).get("units") or {}
    vals = units.get("shares") or []
    if not vals:
        return None
    latest = max(vals, key=lambda v: (str(v.get("end", "")), str(v.get("filed", ""))))
    return Decimal(str(latest["val"]))
