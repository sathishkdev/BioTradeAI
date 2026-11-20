"""RAG and ontology CLI commands."""

from __future__ import annotations

import argparse
import getpass
from datetime import UTC, datetime
from pathlib import Path

from biotradeai.config import load_config
from biotradeai.rag.embeddings import make_embedder
from biotradeai.rag.evaluate import run_eval
from biotradeai.rag.index import VectorIndex
from biotradeai.rag.ingest import ingest_documents, load_local_document
from biotradeai.rag.ontology import (
    approve_proposal,
    list_proposals,
    load_ontology,
    propose,
)
from biotradeai.rag.retrieve import Retriever
from biotradeai.schemas import SourceTier


def _date(s: str) -> datetime:
    d = datetime.fromisoformat(s)
    return d if d.tzinfo else d.replace(tzinfo=UTC)


def cmd_rag_eval(args: argparse.Namespace) -> int:
    cfg = load_config()
    emb = make_embedder(args.embedder)
    rep = run_eval(emb, cfg)
    print(f"embedder {rep.embedder}  queries {rep.n_queries}  top_k {rep.top_k}")
    print(f"recall@{rep.top_k}: {rep.mean_recall:.3f}  (target {cfg.rag.recall_target})")
    print(f"semantic bleed: {rep.total_bleed} chunk(s) (target 0)")
    for q in rep.per_query:
        if q.recall < 1.0:
            print(f"  miss {q.query_id}: recall {q.recall:.2f} missing {list(q.missing)}")
    ok = rep.passed(cfg.rag.recall_target)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_rag_ingest(args: argparse.Namespace) -> int:
    cfg = load_config()
    onto = load_ontology()
    eid = onto.require(args.drug).entity_id if args.drug else None
    doc = load_local_document(Path(args.path), ticker=args.ticker.upper(),
                              tier=SourceTier(args.tier), published_at=_date(args.date),
                              url=args.url, doc_type=args.doc_type, entity_id=eid)
    emb = make_embedder(args.embedder)
    chunks = ingest_documents([doc], index=VectorIndex(emb.name), embedder=emb,
                              ontology=onto, cfg=cfg.rag)
    by_entity: dict[str, int] = {}
    for c in chunks:
        by_entity[c.entity_id] = by_entity.get(c.entity_id, 0) + 1
    print(f"ingested {len(chunks)} chunk(s) from {args.path}: {by_entity}")
    return 0


def cmd_rag_query(args: argparse.Namespace) -> int:
    cfg = load_config()
    emb = make_embedder(args.embedder)
    r = Retriever(VectorIndex(emb.name), emb, load_ontology(), cfg).retrieve(
        args.question, ticker=args.ticker, drug=args.drug,
        as_of=_date(args.as_of) if args.as_of else None)
    print(f"{r.ticker} / {r.entity_id}   status {r.bundle.status.value}   "
          f"confidence {r.confidence:.3f} (authority {r.authority:.3f} x density "
          f"{r.density:.2f})")
    for h in r.hits:
        mark = "*" if h in r.relevant else " "
        print(f" {mark} sim {h.similarity:.3f}  T{int(h.chunk.tier)}  "
              f"{h.chunk.published_at.date()}  {h.chunk.url}")
        print(f"      {h.chunk.text[:160]}...")
    for k, c in r.claims.items():
        print(f"  claim {k} = {c.value}  ({c.rule}, tier {int(c.evidence.tier)})")
    for cf in r.conflicts:
        print(f"  conflict {cf.key}: {'resolved' if cf.resolved else 'UNRESOLVED'} - {cf.note}")
    for reason in r.reasons:
        print(f"  reason: {reason}")
    return 0


def cmd_ontology(args: argparse.Namespace) -> int:
    if args.action == "resolve":
        e = load_ontology().require(args.name)
        print(f"{args.name} -> {e.entity_id} ({e.ticker}) names={list(e.names)} "
              f"previous={list(e.previous_codes)}")
    elif args.action == "list":
        for e in load_ontology().entities.values():
            print(f"{e.entity_id:<20} {e.ticker:<6} {', '.join(e.all_names)}")
    elif args.action == "proposals":
        for i, p in enumerate(list_proposals()):
            print(f"#{i} [{p['status']}] {p['name']} -> {p['entity_id']} ({p['kind']}) "
                  f"from {p['source']}: {p['rationale']}")
    elif args.action == "propose":
        propose(args.name, args.entity, ticker=args.ticker.upper(), kind=args.kind,
                rationale=args.rationale or "", source=f"human:{getpass.getuser()}")
        print("proposal queued; approve with: python -m biotradeai ontology approve --index <n>")
    elif args.action == "approve":
        e = approve_proposal(args.index, approver=getpass.getuser())
        print(f"approved -> {e.entity_id}: {', '.join(e.all_names)}")
    return 0


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    rag = sub.add_parser("rag", help="retrieval: eval | ingest | query")
    rsub = rag.add_subparsers(dest="rag_cmd", required=True)
    ev = rsub.add_parser("eval", help="recall/bleed on the labeled 30-drug fixture set")
    ev.add_argument("--embedder", help="st:<model> | ollama:<model> | hash")
    ev.set_defaults(fn=cmd_rag_eval)
    ing = rsub.add_parser("ingest", help="ingest a local PDF/HTML/TXT document")
    ing.add_argument("path")
    ing.add_argument("--ticker", required=True)
    ing.add_argument("--tier", type=int, choices=[1, 2, 3], required=True)
    ing.add_argument("--date", required=True, help="published date YYYY-MM-DD")
    ing.add_argument("--url", required=True, help="where the document came from")
    ing.add_argument("--doc-type", default="local")
    ing.add_argument("--drug", help="tag every chunk with this drug (else auto-detect)")
    ing.add_argument("--embedder")
    ing.set_defaults(fn=cmd_rag_ingest)
    q = rsub.add_parser("query", help="retrieve evidence for a ticker + drug")
    q.add_argument("ticker")
    q.add_argument("drug")
    q.add_argument("question")
    q.add_argument("--as-of")
    q.add_argument("--embedder")
    q.set_defaults(fn=cmd_rag_query)

    o = sub.add_parser("ontology", help="resolve | list | proposals | propose | approve")
    o.add_argument("action", choices=["resolve", "list", "proposals", "propose", "approve"])
    o.add_argument("name", nargs="?")
    o.add_argument("--entity")
    o.add_argument("--ticker")
    o.add_argument("--kind", default="name", choices=["name", "previous_code"])
    o.add_argument("--rationale")
    o.add_argument("--index", type=int)
    o.set_defaults(fn=cmd_ontology)
