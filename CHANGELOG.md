# Changelog

All notable changes. Dates are UTC. Details and reasons: [docs/DECISIONS.md](docs/DECISIONS.md).

## [0.1.0] — 2026-11-20

First public version: everything except the Robinhood phases, which need the
owner's account.

### Added
- **Phase 0, bootstrap:**
  - strict guardrails config;
  - Pydantic contracts;
  - append-only, hash-chained SQLite audit log;
  - switchable LLMs (Anthropic / Ollama / fake).
- **Phase 1, data layer:**
  - ClinicalTrials.gov v2 and SEC EDGAR clients;
  - XBRL cash runway (point-in-time, restatement-safe);
  - freshness checks.
- **Phase 2, safety:**
  - guardrail rules as tested pure functions;
  - HMAC single-use approval tokens;
  - Alpaca paper broker;
  - kill-switch;
  - `execution.py` as the only order path.
- **Phase 4, RAG:**
  - drug ontology;
  - ingest with claim extraction;
  - strict ticker+drug+date retrieval, hybrid BM25 + vector ranking;
  - source-tier conflict resolution and confidence scoring.
- **Phase 5, agents:**
  - Tree-of-Thoughts Research Agent: bounded, evidence-cited LLM adjustments, with a
    critic that can only reject;
  - deterministic Risk Agent;
  - LangGraph supervisor with human approval.
- **Phase 6, selling:** deterministic sell state machine; EDGAR and CT.gov pollers.
- **Phase 7, backtest:**
  - point-in-time harness with leak documents;
  - Brier score, groundedness, breach, escalation and latency metrics;
  - reports.
- **Phase 8, paper trading:** Alpaca paper runner, daily review, live-gate counter.
- **Network guardrails:**
  - official-API host allowlist;
  - per-source polite rates (about half of published limits);
  - daily budgets, circuit breaker, `Retry-After` handling;
  - balanced pacing for bulk jobs.
- **Watchlist agent:**
  - scout that proposes company+drug pairs with evidence cards;
  - human approval;
  - automatic document sync;
  - daily follow loop with agent-managed catalyst events and life-cycle pauses;
  - licensing detection.
- Time-stop sell rule wired to watchlist trials.
- Docs: README with architecture diagrams, step-by-step `docs/SETUP.md`, decisions log.
- CI: ruff, strict mypy and offline pytest on Linux and Windows.

### Added later
- MIT license.

### Not yet
- Phases 3, 9 and 10 (Robinhood spike, shadow mode, live pilot).
- A backtest on real historical catalysts, with a stronger model.
