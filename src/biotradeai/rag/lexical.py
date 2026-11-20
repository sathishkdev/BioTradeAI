"""BM25 keyword scoring and Reciprocal Rank Fusion (hybrid retrieval).

Runs only over the already pre-filtered pool (one ticker + one entity), so it can never
widen what is retrievable; it only changes ranking.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

_STOP = frozenset([
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from", "has", "have",
    "in", "is", "it", "its", "no", "not", "of", "on", "or", "our", "that", "the", "this",
    "to", "was", "we", "were", "what", "which", "with",
])
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60  # standard constant from Cormack et al. 2009


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


def bm25_scores(query: str, docs: Sequence[str]) -> list[float]:
    toks = [tokenize(d) for d in docs]
    n = len(toks)
    if n == 0:
        return []
    avgdl = sum(len(t) for t in toks) / n or 1.0
    df: Counter[str] = Counter()
    for t in toks:
        df.update(set(t))
    q = tokenize(query)
    scores = []
    for t in toks:
        tf = Counter(t)
        s = 0.0
        for term in q:
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            f = tf[term]
            s += idf * f * (BM25_K1 + 1) / (f + BM25_K1 * (1 - BM25_B + BM25_B * len(t) / avgdl))
        scores.append(s)
    return scores


def rrf(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> list[str]:
    """Fuse several best-first id lists; ties broken by id for determinism."""
    score: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            score[item] = score.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(score, key=lambda i: (-score[i], i))
