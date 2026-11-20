"""Entity ontology: map every name a drug goes by to one ``Entity_ID``.

``data/synonyms.yaml`` (human-owned) lists, per entity, its ticker, current names
(pipeline code, generic, brand) and ``previous_codes`` (renamed programs). Matching is
case- and separator-insensitive: "ONC-101", "onc 101" and "ONC101" are the same name.

An LLM may *propose* a mapping (``propose``), which only appends to
``data/synonym_proposals.yaml``. Only ``approve_proposal`` (a human CLI command) writes
to ``synonyms.yaml``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from biotradeai.paths import data_dir

ENTITY_ID = re.compile(r"^ENT-[A-Z0-9][A-Z0-9-]{1,40}$")
TICKER = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


class OntologyError(ValueError):
    pass


def normalize(name: str) -> str:
    """Case-fold and drop every non-alphanumeric character."""
    return re.sub(r"[^0-9a-z]", "", name.casefold())


def company_entity(ticker: str) -> str:
    """Entity used for company-wide text that names no specific drug."""
    return f"ENT-{ticker.upper().replace('.', '-')}-COMPANY"


@dataclass(frozen=True)
class Entity:
    entity_id: str
    ticker: str
    names: tuple[str, ...]
    previous_codes: tuple[str, ...] = ()

    @property
    def all_names(self) -> tuple[str, ...]:
        return self.names + self.previous_codes


def _name_pattern(name: str) -> re.Pattern[str]:
    # Allow any run of space/hyphen/underscore/dot between alphanumeric runs.
    parts = re.findall(r"[0-9A-Za-z]+", name)
    body = r"[\s\-_.]*".join(re.escape(p) for p in parts)
    return re.compile(rf"(?<![0-9A-Za-z]){body}(?![0-9A-Za-z])", re.IGNORECASE)


@dataclass
class Ontology:
    entities: dict[str, Entity]
    _index: dict[str, str] = field(default_factory=dict, repr=False)
    _patterns: list[tuple[re.Pattern[str], str]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        for e in self.entities.values():
            for n in e.all_names:
                key = normalize(n)
                if not key:
                    raise OntologyError(f"{e.entity_id}: empty name {n!r}")
                other = self._index.get(key)
                if other is not None and other != e.entity_id:
                    raise OntologyError(f"name {n!r} maps to both {other} and {e.entity_id}")
                self._index[key] = e.entity_id
                self._patterns.append((_name_pattern(n), e.entity_id))
        # Longest names first so "ONC-101 XR" wins over "ONC-101".
        self._patterns.sort(key=lambda t: -len(t[0].pattern))

    # ------------------------------------------------------------------ lookup

    def resolve(self, name: str) -> str | None:
        return self._index.get(normalize(name))

    def require(self, name: str) -> Entity:
        eid = self.resolve(name)
        if eid is None:
            raise OntologyError(f"unknown drug name {name!r}; add it to data/synonyms.yaml")
        return self.entities[eid]

    def entities_for(self, ticker: str) -> list[Entity]:
        return [e for e in self.entities.values() if e.ticker == ticker.upper()]

    def find_mentions(self, text: str, ticker: str | None = None) -> set[str]:
        """Entity ids whose names appear in ``text`` (optionally only for ``ticker``)."""
        found = {eid for pat, eid in self._patterns if pat.search(text)}
        if ticker is not None:
            found = {eid for eid in found if self.entities[eid].ticker == ticker.upper()}
        return found


# --------------------------------------------------------------------------- file io


def default_path() -> Path:
    return data_dir() / "synonyms.yaml"


def proposals_path() -> Path:
    return data_dir() / "synonym_proposals.yaml"


def _parse(raw: Any, src: str) -> Ontology:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict) or not isinstance(raw.get("entities", {}) or {}, dict):
        raise OntologyError(f"{src}: expected a mapping with an 'entities' mapping")
    out: dict[str, Entity] = {}
    for eid, body in (raw.get("entities") or {}).items():
        if not ENTITY_ID.match(str(eid)):
            raise OntologyError(f"{src}: bad entity id {eid!r} (expected ENT-...)")
        if not isinstance(body, dict):
            raise OntologyError(f"{src}: {eid} must be a mapping")
        ticker = str(body.get("ticker", ""))
        if not TICKER.match(ticker):
            raise OntologyError(f"{src}: {eid} has bad ticker {ticker!r}")
        names = tuple(str(n) for n in body.get("names") or ())
        if not names:
            raise OntologyError(f"{src}: {eid} needs at least one name")
        prev = tuple(str(n) for n in body.get("previous_codes") or ())
        out[str(eid)] = Entity(str(eid), ticker, names, prev)
    return Ontology(out)


def load_ontology(path: Path | None = None) -> Ontology:
    p = path or default_path()
    if not p.exists():
        return Ontology({})
    return _parse(yaml.safe_load(p.read_text(encoding="utf-8")), p.name)


def _dump(entities: dict[str, Entity]) -> dict[str, Any]:
    body: dict[str, Any] = {}
    for eid in sorted(entities):
        e = entities[eid]
        item: dict[str, Any] = {"ticker": e.ticker, "names": list(e.names)}
        if e.previous_codes:
            item["previous_codes"] = list(e.previous_codes)
        body[eid] = item
    return {"entities": body}


# --------------------------------------------------------------------------- proposals


def propose(name: str, entity_id: str, *, ticker: str, kind: str, rationale: str,
            source: str, path: Path | None = None) -> dict[str, Any]:
    """Queue a mapping for human review. Never edits synonyms.yaml."""
    if kind not in {"name", "previous_code"}:
        raise OntologyError("kind must be 'name' or 'previous_code'")
    if not ENTITY_ID.match(entity_id) or not TICKER.match(ticker):
        raise OntologyError("bad entity id or ticker")
    p = path or proposals_path()
    items: list[dict[str, Any]] = []
    if p.exists():
        items = list((yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("proposals")
                     or [])
    item = {"name": name, "entity_id": entity_id, "ticker": ticker, "kind": kind,
            "rationale": rationale, "source": source,
            "proposed_at": datetime.now(UTC).isoformat(), "status": "pending"}
    items.append(item)
    p.write_text(yaml.safe_dump({"proposals": items}, sort_keys=False), encoding="utf-8")
    return item


def list_proposals(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or proposals_path()
    if not p.exists():
        return []
    return list((yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("proposals") or [])


def approve_proposal(index: int, *, approver: str, proposals: Path | None = None,
                     synonyms: Path | None = None) -> Entity:
    """Human-only: merge proposal ``index`` into synonyms.yaml (validated before writing)."""
    pp = proposals or proposals_path()
    sp = synonyms or default_path()
    items = list_proposals(pp)
    if not 0 <= index < len(items) or items[index].get("status") != "pending":
        raise OntologyError(f"no pending proposal #{index}")
    prop = items[index]
    onto = load_ontology(sp)
    ents = dict(onto.entities)
    cur = ents.get(prop["entity_id"])
    if cur is None:
        cur = Entity(prop["entity_id"], prop["ticker"], (prop["name"],))
    elif cur.ticker != prop["ticker"]:
        raise OntologyError("proposal ticker does not match existing entity")
    elif prop["kind"] == "name":
        cur = Entity(cur.entity_id, cur.ticker, (*cur.names, prop["name"]), cur.previous_codes)
    else:
        cur = Entity(cur.entity_id, cur.ticker, cur.names, (*cur.previous_codes, prop["name"]))
    ents[cur.entity_id] = cur
    _write_synonyms(ents, sp)
    items[index] = {**prop, "status": "approved", "approved_by": approver,
                    "approved_at": datetime.now(UTC).isoformat()}
    pp.write_text(yaml.safe_dump({"proposals": items}, sort_keys=False), encoding="utf-8")
    return cur


def _write_synonyms(ents: dict[str, Entity], path: Path) -> None:
    Ontology(ents)  # raises if any name collides with another entity
    header = ""
    if path.exists():  # keep the human-written comment header
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        header = "".join(ln for ln in lines[: next((i for i, ln in enumerate(lines)
                                                   if not ln.startswith("#")), len(lines))])
    path.write_text(header + yaml.safe_dump(_dump(ents), sort_keys=False), encoding="utf-8",
                    newline="\n")


def upsert_entity(entity: Entity, *, synonyms: Path | None = None) -> Entity:
    """Add or extend one entity in synonyms.yaml. Called only from a HUMAN approval
    (``watchlist approve``), never by an agent on its own. Names are merged, never
    removed; a collision with another entity raises OntologyError and writes nothing."""
    sp = synonyms or default_path()
    ents = dict(load_ontology(sp).entities)
    cur = ents.get(entity.entity_id)
    if cur is not None and cur.ticker != entity.ticker:
        raise OntologyError(f"{entity.entity_id} already belongs to {cur.ticker}")
    if cur is not None:
        names = tuple(dict.fromkeys((*cur.names, *entity.names)))
        prev = tuple(dict.fromkeys((*cur.previous_codes, *entity.previous_codes)))
        entity = Entity(entity.entity_id, entity.ticker, names, prev)
    ents[entity.entity_id] = entity
    _write_synonyms(ents, sp)
    return entity
