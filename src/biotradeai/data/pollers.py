"""Pollers that turn EDGAR filings and CT.gov date changes into sell-machine events.

* ``FilingPoller``: new 8-K / S-3 / S-1 / 424B filings since a date. Each accession is
  reported once (``seen_filings`` table). 8-K text is fetched and classified in code.
* ``TrialDatePoller``: compares a trial's primary completion date with the last value
  recorded in ``evaluated_ncts`` and reports a change.

Both are deterministic and network-mockable (pass ``transport=`` to the clients).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time

from biotradeai.data.clinicaltrials import ClinicalTrialsClient
from biotradeai.data.edgar import EdgarClient
from biotradeai.events import WATCHED_FORMS, CompletionDateChange, FilingEvent, classify_filing
from biotradeai.rag.ingest import html_to_text
from biotradeai.store.sqlite_store import SQLiteStore


def _day_ts(d: date) -> datetime:
    return datetime.combine(d, time(0, 0), tzinfo=UTC)


class FilingPoller:
    def __init__(self, edgar: EdgarClient, store: SQLiteStore) -> None:
        self._edgar = edgar
        self._store = store

    def poll(self, ticker: str, cik: str, *, since: date) -> list[FilingEvent]:
        subs = self._edgar.submissions(cik)
        out: list[FilingEvent] = []
        for f in sorted(subs.filings, key=lambda f: (f.filing_date, f.accession_number)):
            form = f.form.strip().upper()
            if f.filing_date < since or not WATCHED_FORMS.match(form):
                continue
            if not self._store.mark_filing_seen(f.accession_number, ticker, form,
                                                f.filing_date.isoformat()):
                continue
            text = None
            if form.startswith("8-K"):
                text = html_to_text(self._edgar.fetch_document(cik, f))
            dilutive, reason = classify_filing(form, text)
            ev = FilingEvent(ticker=ticker, ts=_day_ts(f.filing_date), form=form,
                             accession=f.accession_number, dilutive=dilutive, reason=reason,
                             url=f.url(cik))
            self._store.append_audit("poll.filing", ev.model_dump(mode="json"), ticker=ticker)
            out.append(ev)
        return out


class TrialDatePoller:
    def __init__(self, ctgov: ClinicalTrialsClient, store: SQLiteStore) -> None:
        self._ctgov = ctgov
        self._store = store

    def poll(self, ticker: str, nct_id: str) -> list[CompletionDateChange]:
        rec = self._ctgov.get_study(nct_id)
        new = rec.primary_completion.value if rec.primary_completion else None
        prev_row = self._store.get_nct(nct_id)
        prev = (date.fromisoformat(prev_row["last_completion_date"])
                if prev_row and prev_row.get("last_completion_date") else None)
        self._store.record_nct(nct_id, ticker,
                               last_completion_date=new.isoformat() if new else None,
                               decision=prev_row.get("decision") if prev_row else None)
        if new is None or prev_row is None or new == prev:
            return []  # first sighting only records the baseline
        ev = CompletionDateChange(ticker=ticker, ts=rec.fetched_at, nct_id=nct_id,
                                  old_date=prev, new_date=new)
        self._store.append_audit("poll.completion_date", ev.model_dump(mode="json"),
                                 ticker=ticker)
        return [ev]
