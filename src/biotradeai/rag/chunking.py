"""Paragraph-aware chunking with token overlap.

Token counts are *estimated* as 1.3 tokens per whitespace word, which is close to the
BPE tokenizers used by embedding models for English prose. Using an estimate keeps
chunking deterministic and independent of the chosen embedder.

Algorithm: paragraphs (blank-line separated) are packed greedily into a chunk until the
next paragraph would push it past ``chunk_tokens``. Paragraphs larger than a whole chunk
are split on sentence boundaries, then on words. Each chunk after the first starts with
the last ``overlap_tokens`` worth of words of the previous chunk.
"""

from __future__ import annotations

import math
import re

TOKENS_PER_WORD = 1.3
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text.split()) * TOKENS_PER_WORD)


def _words_for(tokens: int) -> int:
    return max(1, math.floor(tokens / TOKENS_PER_WORD))


def paragraphs(text: str) -> list[str]:
    paras = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    return [" ".join(p.split()) for p in paras if p.strip()]


def _split_long(para: str, max_words: int) -> list[str]:
    if len(para.split()) <= max_words:
        return [para]
    out: list[str] = []
    cur: list[str] = []
    for sent in _SENTENCE.split(para):
        words = sent.split()
        if len(words) > max_words:  # a monster sentence: hard split on words
            if cur:
                out.append(" ".join(cur))
                cur = []
            for i in range(0, len(words), max_words):
                out.append(" ".join(words[i:i + max_words]))
            continue
        if len(cur) + len(words) > max_words and cur:
            out.append(" ".join(cur))
            cur = []
        cur.extend(words)
    if cur:
        out.append(" ".join(cur))
    return out


def chunk_text(text: str, *, chunk_tokens: int, overlap_tokens: int) -> list[str]:
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap must be smaller than chunk size")
    max_words = _words_for(chunk_tokens)
    overlap_words = _words_for(overlap_tokens) if overlap_tokens > 0 else 0
    units = [u for p in paragraphs(text) for u in _split_long(p, max_words - overlap_words)]

    chunks: list[str] = []
    cur: list[str] = []
    cur_words = 0
    for unit in units:
        n = len(unit.split())
        if cur and cur_words + n > max_words:
            chunks.append("\n\n".join(cur))
            tail = " ".join(chunks[-1].split()[-overlap_words:]) if overlap_words else ""
            cur = [tail] if tail else []
            cur_words = len(tail.split())
        cur.append(unit)
        cur_words += n
    if cur and (not chunks or cur_words > overlap_words):
        chunks.append("\n\n".join(cur))
    return chunks
