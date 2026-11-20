"""Turn data-layer results into the ``ResearchInputs`` contract the buy graph consumes.

Plain values in, contract out. Broker access (quotes, account) happens in the caller
(CLI / Phase 8 runner), never inside agents or the graph.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from decimal import Decimal

from biotradeai.data.runway import RunwayResult
from biotradeai.rag.index import Chunk
from biotradeai.rag.retrieve import RetrievalResult
from biotradeai.schemas import (
    MarketSnapshot,
    PortfolioSnapshot,
    ResearchInputs,
    RunwaySnapshot,
)

EXCERPT_CHARS = 1200


def runway_snapshot(r: RunwayResult, *, observed_at: datetime) -> RunwaySnapshot:
    return RunwaySnapshot(months=r.runway_months, cash=r.liquidity,
                          quarterly_burn=r.quarterly_burn, observed_at=observed_at,
                          source_accns=list(r.source_accns))


def excerpts_for(chunks: Iterable[Chunk], wanted: set[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for c in chunks:
        if c.sha256 in wanted and c.sha256 not in out:
            out[c.sha256] = c.text[:EXCERPT_CHARS]
    return out


def inputs_from_retrieval(result: RetrievalResult, *, runway: RunwaySnapshot,
                          bid: Decimal, ask: Decimal, quote_at: datetime,
                          equity: Decimal, open_tickers: list[str], existing_value: Decimal,
                          now: datetime | None = None) -> ResearchInputs:
    now = now or datetime.now(UTC)
    wanted = {s.sha256 for s in result.bundle.sources}
    return ResearchInputs(
        evidence=result.bundle, evidence_observed_at=now,
        excerpts=excerpts_for((h.chunk for h in result.hits), wanted),
        claims={k: v.value for k, v in result.claims.items()},
        runway=runway,
        market=MarketSnapshot(bid=bid, ask=ask, observed_at=quote_at),
        portfolio=PortfolioSnapshot(equity=equity, open_tickers=open_tickers,
                                    existing_value=existing_value, observed_at=now),
    )
