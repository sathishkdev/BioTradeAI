"""Watchlist decisions shared by the CLI and the operator UI (same checks, same audit)."""

from __future__ import annotations

from biotradeai.rag.ontology import Entity, upsert_entity
from biotradeai.store.sqlite_store import SQLiteStore
from biotradeai.watchlist.model import WatchEntry, Watchlist, WatchlistError


def approve(entry_id: str, *, by: str, note: str | None, source: str,
            wl: Watchlist | None = None, store: SQLiteStore | None = None) -> WatchEntry:
    wl = wl or Watchlist()
    e = wl.approve(entry_id, by=by, note=note)  # raises if the transition is not allowed
    upsert_entity(Entity(e.entity_id, e.ticker, tuple(e.drug_names)))  # raises on collision
    wl.save()  # only now is anything written
    (store or SQLiteStore()).append_audit(
        "watchlist.approved", {"id": e.id, "names": e.drug_names, "note": note,
                               "source": source, "by": by}, ticker=e.ticker)
    return e


def reject(entry_id: str, *, by: str, reason: str, source: str,
           wl: Watchlist | None = None, store: SQLiteStore | None = None) -> WatchEntry:
    if not reason.strip():
        raise WatchlistError("give a short reason")
    wl = wl or Watchlist()
    e = wl.reject(entry_id, by=by, reason=reason)
    wl.save()
    (store or SQLiteStore()).append_audit(
        "watchlist.rejected", {"id": e.id, "reason": reason, "source": source, "by": by},
        ticker=e.ticker)
    return e


def retire(entry_id: str, *, by: str, reason: str, source: str,
           wl: Watchlist | None = None, store: SQLiteStore | None = None) -> WatchEntry:
    if not reason.strip():
        raise WatchlistError("give a short reason")
    wl = wl or Watchlist()
    e = wl.retire(entry_id, by=by, reason=reason)
    wl.save()
    (store or SQLiteStore()).append_audit(
        "watchlist.retired", {"id": e.id, "reason": reason, "by": "human", "source": source},
        ticker=e.ticker)
    return e


def resync(entry_id: str, *, source: str, wl: Watchlist | None = None,
           store: SQLiteStore | None = None) -> WatchEntry:
    """Re-run the first sync on the next follow (documents are never duplicated)."""
    e = (wl or Watchlist()).get(entry_id)
    s = store or SQLiteStore()
    s.kv_set(f"sync.initial.{e.entity_id}", "")  # empty = not done
    s.append_audit("watchlist.resync", {"id": e.id, "source": source}, ticker=e.ticker)
    return e
