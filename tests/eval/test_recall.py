"""Phase 4 acceptance: recall@top_k >= rag.recall_target with the real embedder, zero bleed.

Runs by default when the embedding model is already cached locally (no network needed);
otherwise skipped. Download it once with:  python -m biotradeai rag eval
"""

from __future__ import annotations

import os

import chromadb
import pytest

from biotradeai.config import load_config
from biotradeai.rag.embeddings import DEFAULT_EMBEDDER, EmbeddingError, make_embedder
from biotradeai.rag.evaluate import run_eval

CFG = load_config()


@pytest.fixture(scope="module")
def embedder():  # type: ignore[no-untyped-def]
    spec = os.environ.get("BIOTRADEAI_EMBEDDER") or DEFAULT_EMBEDDER
    try:
        return make_embedder(spec, local_files_only=True)
    except EmbeddingError as exc:
        pytest.skip(f"embedding model not cached ({exc}); run: python -m biotradeai rag eval")


def test_recall_target_and_zero_bleed(embedder) -> None:  # type: ignore[no-untyped-def]
    rep = run_eval(embedder, CFG, client=chromadb.EphemeralClient())
    misses = [q for q in rep.per_query if q.recall < 1.0]
    assert rep.total_bleed == 0
    assert rep.mean_recall >= CFG.rag.recall_target, (
        f"recall@{rep.top_k} {rep.mean_recall:.3f} < {CFG.rag.recall_target}; "
        f"misses: {[(q.query_id, q.missing) for q in misses]}")
    renamed = next(q for q in rep.per_query if q.query_id == "q-renamed")
    assert renamed.recall == 1.0
