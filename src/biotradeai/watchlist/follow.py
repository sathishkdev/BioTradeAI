"""Daily follow loop for APPROVED watchlist drugs. Read-only toward the outside world.

For each approved company+drug:
  * first run after approval: initial document sync (``rag/sync.py``)
  * new SEC filings (daily index, not per-company polling):
      8-K   -> ingest; catalyst dates; outcomes; approval/launch; material changes
      10-Q/K-> ingest risk factors; post-launch quarter check
      25    -> delisting notice -> pause
  * ClinicalTrials.gov trials: terminated/withdrawn/suspended -> pause; new results ingested
  * PubMed refresh on a rolling weekly slice
  * life-cycle rules: yearly check-in, approved-but-not-launched after ~6 months,
    first full quarter after launch reported -> pause for re-approval

Pauses never retire silently: the user re-approves or retires. Only "program ended"
(company says development is discontinued) retires automatically.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta

from biotradeai.data.clinicaltrials import ClinicalTrialsClient
from biotradeai.data.http import FetchError
from biotradeai.data.pacing import paced, rolling_batch
from biotradeai.data.secindex import IndexEntry, SecDirectory, business_days, day_done
from biotradeai.data.signals import (
    OutcomeKind,
    drug_mentions,
    find_catalysts,
    find_material_changes,
    find_outcomes,
    mentions_any,
    sentences,
)
from biotradeai.rag.claims import extract_claims
from biotradeai.rag.sync import DocumentSync, SyncReport
from biotradeai.store.sqlite_store import SQLiteStore
from biotradeai.watchlist.catalysts import record_catalyst, record_outcome
from biotradeai.watchlist.model import (
    LAUNCH_GRACE_DAYS,
    Stage,
    Status,
    WatchEntry,
    Watchlist,
    quarter_end_after,
)

DELISTING_FORMS = frozenset({"25", "25-NSE"})
PERIODIC_FORMS = frozenset({"10-Q", "10-K", "10-Q/A", "10-K/A"})
EIGHT_K_FORMS = frozenset({"8-K", "8-K/A"})
STOPPED_TRIAL = frozenset({"TERMINATED", "WITHDRAWN", "SUSPENDED"})
PUBMED_PARTS = 7  # each drug's PubMed refresh once a week
MAX_LOOKBACK_DAYS = 7


@dataclass
class FollowReport:
    entries: int = 0
    filings: int = 0
    events: list[str] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)
    paused: list[str] = field(default_factory=list)
    retired: list[str] = field(default_factory=list)
    stage: list[str] = field(default_factory=list)
    docs: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def add_docs(self, r: SyncReport) -> None:
        for k, v in r.added.items():
            self.docs[k] = self.docs.get(k, 0) + v
        self.errors += r.errors


class Follower:
    def __init__(self, *, directory: SecDirectory, ctgov: ClinicalTrialsClient,
                 store: SQLiteStore, watchlist: Watchlist, sync: DocumentSync, today: date,
                 spread_minutes: float = 0.0, sleep: Callable[[float], None] = time.sleep
                 ) -> None:
        self.dir, self.ctgov, self.store, self.wl, self.sync = (directory, ctgov, store,
                                                                watchlist, sync)
        self.today, self.window_s, self._sleep = today, spread_minutes * 60, sleep

    # ------------------------------------------------------------------ helpers

    def _pause(self, e: WatchEntry, reason: str, rep: FollowReport) -> None:
        if e.status is Status.APPROVED:
            self.wl.pause(e.id, reason=reason)
            rep.paused.append(f"{e.id}: {reason}")
            self.store.append_audit("watchlist.paused", {"id": e.id, "reason": reason},
                                    ticker=e.ticker)

    def _retire(self, e: WatchEntry, reason: str, rep: FollowReport) -> None:
        if e.status in (Status.APPROVED, Status.PAUSED):
            self.wl.retire(e.id, reason=reason)
            rep.retired.append(f"{e.id}: {reason}")
            self.store.append_audit("watchlist.retired", {"id": e.id, "reason": reason},
                                    ticker=e.ticker)

    # ------------------------------------------------------------------ filings

    def on_8k(self, entries: list[WatchEntry], f: IndexEntry, rep: FollowReport) -> None:
        r = SyncReport(f.accession)
        text = self.sync.sec_8k(entries[0], path=f.path, filed=f.filed, rep=r)
        rep.add_docs(r)
        if text:
            self.read_8k(entries, f, text, rep)

    def read_8k(self, entries: list[WatchEntry], f: IndexEntry, text: str,
                rep: FollowReport, *, history: bool = False) -> None:
        """Apply one 8-K's signals. ``history=True`` (filings from before the approval,
        read by the first sync) records catalyst DATES only: old outcomes, deals and holds
        were part of what the human saw when approving, so they don't pause or close
        anything retroactively."""
        all_drugs = drug_mentions(text)
        for e in entries:
            if e.status is not Status.APPROVED:
                continue
            names = tuple(e.drug_names)
            # A mention that names no drug is credited to this drug only when this is the
            # company's only followed drug AND the filing names no other drug at all.
            single = len(entries) == 1 and all(mentions_any((d,), names) for d in all_drugs)
            for m in find_catalysts(text):
                if mentions_any(m.drugs, names) or (single and not m.drugs):
                    ch = record_catalyst(self.store, e, m, url=f.url, filed=f.filed,
                                         today=self.today)
                    if ch.kind in ("new", "changed", "conflict"):
                        rep.events.append(f"{e.id} {ch.event_type} {ch.kind}: {ch.detail}")
                        self.store.append_audit(f"event.{ch.kind}", {
                            "id": e.id, "event": ch.event_type, "detail": ch.detail,
                            "source": f.url, "sentence": m.sentence[:400]}, ticker=e.ticker)
            if history:
                continue
            for o in find_outcomes(text):
                if not (mentions_any(o.drugs, names) or (single and not o.drugs)):
                    continue
                self._apply_outcome(e, o.kind, o.sentence, f, rep)
            for c in find_material_changes(text):
                if c.kind == "acquisition" or mentions_any(c.drugs, names):
                    self._pause(e, f"material change ({c.kind}): {c.sentence[:160]}", rep)
            for s in sentences(text):
                if extract_claims(s).get("clinical_hold") == "active" and \
                        mentions_any(tuple(_names_in(s, names)), names):
                    self._pause(e, f"clinical hold: {s[:160]}", rep)

    def _apply_outcome(self, e: WatchEntry, kind: OutcomeKind, sentence: str, f: IndexEntry,
                       rep: FollowReport) -> None:
        etype = record_outcome(self.store, e, kind, url=f.url, sentence=sentence,
                               filed=f.filed)
        if etype:
            rep.outcomes.append(f"{e.id} {etype}: {kind.value}")
        self.store.append_audit("watchlist.outcome", {"id": e.id, "kind": kind.value,
                                                      "event": etype, "source": f.url,
                                                      "sentence": sentence[:400]},
                                ticker=e.ticker)
        if kind is OutcomeKind.APPROVED and e.stage is Stage.DEVELOPMENT:
            e.stage, e.fda_approved_on = Stage.FDA_APPROVED, f.filed
            rep.stage.append(f"{e.id}: FDA approved {f.filed}")
        elif kind is OutcomeKind.LAUNCHED and e.stage is not Stage.LAUNCHED:
            e.stage, e.launched_on = Stage.LAUNCHED, f.filed
            e.review_after = quarter_end_after(f.filed)
            rep.stage.append(f"{e.id}: launched {f.filed}; review after {e.review_after}")
        elif kind is OutcomeKind.DISCONTINUED:
            self._retire(e, f"program ended: {sentence[:160]}", rep)

    def on_periodic(self, entries: list[WatchEntry], f: IndexEntry, rep: FollowReport) -> None:
        for e in entries:
            if (e.status is Status.APPROVED and e.stage is Stage.LAUNCHED and e.review_after
                    and f.filed > e.review_after):
                self._pause(e, f"first full quarter after launch reported ({f.form} filed "
                               f"{f.filed}); re-approve to keep following", rep)

    # ------------------------------------------------------------------ run

    def run(self) -> FollowReport:
        rep = FollowReport()
        approved = self.wl.with_status(Status.APPROVED)
        rep.entries = len(approved)
        by_cik: dict[str, list[WatchEntry]] = {}
        for e in approved:
            by_cik.setdefault(e.cik, []).append(e)

        # 1. initial sync for newly approved drugs
        for e in approved:
            key = f"sync.initial.{e.entity_id}"
            if not self.store.kv_get(key):  # None or '' (resync requested)
                r = self.sync.initial(e, today=self.today)
                rep.add_docs(r)
                # Catalyst dates from the 8-Ks just loaded, oldest first, so a later
                # filing's date supersedes an earlier one (and stays in previous_dates).
                for filed, path, text in sorted(r.eight_ks, key=lambda x: x[0]):
                    hist = IndexEntry("8-K", e.company, e.cik, filed, path)
                    self.read_8k([e], hist, text, rep, history=True)
                if not r.errors:  # retried next run otherwise (already-added docs dedupe)
                    self.store.kv_set(key, self.today.isoformat())

        # 2. new SEC filings from the daily index
        last = self.store.kv_get("follow.last_index_day")
        start = (date.fromisoformat(last) + timedelta(days=1) if last
                 else self.today - timedelta(days=1))
        start = max(start, self.today - timedelta(days=MAX_LOOKBACK_DAYS))
        todo: list[IndexEntry] = []
        for d in business_days(start, self.today):
            try:
                day = self.dir.filings(d)
            except FetchError as exc:
                rep.errors.append(f"index {d}: {exc}"[:200])
                break  # keep the marker before this day so it is retried
            todo += [f for f in day or [] if f.cik in by_cik]
            if not day_done(d, self.today, day):
                break  # today's index isn't out yet: retry it next run
            self.store.kv_set("follow.last_index_day", d.isoformat())
        for f in paced(todo, window_s=self.window_s, sleep=self._sleep):
            rep.filings += 1
            ents = by_cik[f.cik]
            try:
                if f.form in EIGHT_K_FORMS:
                    self.on_8k(ents, f, rep)
                elif f.form in PERIODIC_FORMS:
                    self.on_periodic(ents, f, rep)
                elif f.form in DELISTING_FORMS:
                    for e in ents:
                        self._pause(e, f"delisting notice (Form {f.form}, {f.filed})", rep)
            except FetchError as exc:
                rep.errors.append(f"{f.accession}: {exc}"[:200])

        # 3. trials
        for e in paced([x for x in approved if x.status is Status.APPROVED],
                       window_s=self.window_s, sleep=self._sleep):
            for nct in e.trial_ids:
                r = SyncReport(e.id)
                try:
                    raw = self.sync.trial(e, nct, r)
                except FetchError as exc:
                    rep.errors.append(f"{nct}: {exc}"[:200])
                    continue
                rep.add_docs(r)
                status = str(((raw.get("protocolSection") or {}).get("statusModule") or {})
                             .get("overallStatus", "")).upper()
                if status in STOPPED_TRIAL:
                    self._pause(e, f"trial {nct} is {status.lower()}", rep)

        # 4. PubMed on a weekly rolling slice
        slice_ids = set(rolling_batch([e.entity_id for e in approved], day=self.today,
                                      parts=PUBMED_PARTS))
        for e in approved:
            if e.entity_id in slice_ids and e.status is Status.APPROVED:
                r = SyncReport(e.id)
                try:
                    self.sync.pubmed_refresh(e, r)
                except FetchError as exc:
                    r.errors.append(str(exc)[:200])
                rep.add_docs(r)

        # 5. time-based life-cycle rules
        for e in approved:
            if e.status is not Status.APPROVED:
                continue
            if e.checkin_due and self.today >= e.checkin_due:
                self._pause(e, "yearly check-in: still want to follow this drug?", rep)
            elif (e.stage is Stage.FDA_APPROVED and e.fda_approved_on
                  and self.today >= e.fda_approved_on + timedelta(days=LAUNCH_GRACE_DAYS)):
                self._pause(e, f"FDA approved {e.fda_approved_on} but no launch found "
                               "after ~6 months", rep)

        self.wl.save()
        self.store.append_audit("watchlist.follow", {
            "entries": rep.entries, "filings": rep.filings, "events": rep.events,
            "outcomes": rep.outcomes, "paused": rep.paused, "retired": rep.retired,
            "stage": rep.stage, "docs": rep.docs, "errors": rep.errors[:20]})
        return rep


def _names_in(sentence: str, names: tuple[str, ...]) -> list[str]:
    low = sentence.lower()
    return [n for n in names if n.lower() in low]
