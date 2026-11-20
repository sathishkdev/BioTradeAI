"""Run every backtest case, compute metrics and write the report."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from biotradeai.agents.base_rates import BaseRates
from biotradeai.agents.llm import LLMClient
from biotradeai.backtest.harness import (
    FIXTURE,
    BacktestCase,
    CaseResult,
    PointInTimeViolationError,
    build_index,
    run_case,
)
from biotradeai.backtest.metrics import Metrics, compute_metrics
from biotradeai.backtest.report import write_report
from biotradeai.config import GuardrailsConfig
from biotradeai.rag.embeddings import Embedder
from biotradeai.rag.retrieve import Retriever
from biotradeai.store.sqlite_store import SQLiteStore

CAVEAT = ("Synthetic cases validate the harness and metrics only. They are not evidence of "
          "real predictive skill; that needs real historical catalysts.")


@dataclass(frozen=True)
class BacktestRun:
    metrics: Metrics
    results: list[CaseResult]
    leaks: list[str]
    report: Path | None


def run_backtest(cases: list[BacktestCase], *, cfg: GuardrailsConfig, rates: BaseRates,
                 embedder: Embedder, generator: LLMClient, critic: LLMClient,
                 store: SQLiteStore, out_dir: Path | None = None, write: bool = True,
                 cases_file: Path = FIXTURE, client: Any | None = None) -> BacktestRun:
    started = datetime.now(UTC)
    index, onto = build_index(cases, embedder=embedder, cfg=cfg, client=client)
    retriever = Retriever(index, embedder, onto, cfg)
    results: list[CaseResult] = []
    leaks: list[str] = []
    for c in cases:
        try:
            results.append(run_case(c, retriever=retriever, cfg=cfg, rates=rates,
                                    generator=generator, critic=critic, store=store))
        except PointInTimeViolationError as exc:
            leaks.append(str(exc))
    m = compute_metrics(results, cfg, rates)
    meta = {"started_at": started.isoformat(timespec="seconds"), "cases_file": str(cases_file),
            "generator": f"{generator.provider}:{generator.model}",
            "critic": f"{critic.provider}:{critic.model}", "embedder": embedder.name,
            "base_rates_reviewed": rates.reviewed, "leaks": len(leaks), "leak_detail": leaks,
            "caveat": CAVEAT}
    store.append_audit("backtest.run", {"meta": meta, "checks": m.checks(),
                                        "brier": m.brier, "groundedness": m.groundedness,
                                        "breaches": m.breaches})
    report = write_report(m, meta, out_dir) if write else None
    return BacktestRun(m, results, leaks, report)
