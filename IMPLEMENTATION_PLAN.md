# BioTradeAI — Implementation Plan (for Claude Code)

Read this whole file before writing any code. Work phase by phase. Do not start a phase until the previous phase's acceptance tests pass. Ask me before deviating from anything marked MUST.

## 1. What we are building

A multi-agent decision-support system for trading biotech equities around catalysts (FDA PDUFA dates, Phase 3 readouts). It researches a ticker, builds a Tree-of-Thoughts (ToT) scenario analysis, validates it with an independent risk check, and stages a limit order for human approval. Sell logic is a deterministic event-driven state machine, not an LLM.

Design source docs (in this repo or provided separately): "Biotech Catalyst Trading Agent – Architecture", "Capstone Architecture, Governance & Safety Plan". Where this file conflicts with those PDFs, this file wins (see section 3 for the corrections).

This is decision support, not financial advice. Real-money trading is gated behind a long paper phase.

## 2. Non-negotiables (MUST)

1. The LLM never enforces risk rules. Every numeric rule is a plain, unit-tested function that reads `config/guardrails.yaml`.
2. Only one module (`brokers/execution.py`) may call a broker's order-placement function. No agent or LLM tool binding may reach it directly.
3. Limit orders only. The adapter rejects market, stop, and stop-limit orders regardless of what upstream asks for.
4. Every state transition, thought trace, source hash, tool payload, and order is written to an append-only audit log (SQLite). No updates or deletes on that table.
5. Arithmetic (runway, burn, EV, position size) is computed in code, never by an LLM.
6. Default mode is `paper`. Live mode requires explicit multi-factor enablement (section 7).
7. Secrets only via environment variables / `.env` (git-ignored). Never log credentials, account numbers, or API keys. Redact account identifiers before any text reaches an LLM.
8. Target Windows 11 and Linux. Use `pathlib`, no shell scripts required for core workflows, no Unix-only APIs.
9. Commit early and often with small, tested commits.

## 3. Corrections to the PDF designs (apply these)

| # | PDF says | Implement this instead |
|---|---|---|
| 1 | Critic prunes at runway < 3 months; Risk Agent vetoes at < 6 months | Both stay, with different roles: `fatal_runway_months: 3` = prune the ToT branch immediately; `min_runway_months: 6` = Risk Agent veto. Defined once in config. |
| 2 | Composite EV computed over surviving branches only ($ONCO example sums to 90%) | EV is computed over ALL outcomes with probabilities summing to 1.0. Pruning only saves compute; a pruned branch is collapsed into a residual scenario with a conservative return from config (`residual_scenario_return`). A validator rejects any scenario set whose probabilities do not sum to 1.0 (±0.001). |
| 3 | PDUFA dates come from ClinicalTrials.gov | CT.gov has trial dates only (sponsor-reported, often stale). PDUFA/catalyst dates come from 8-K/press-release extraction and a manually curated `data/catalysts.csv` that the human maintains. Treat CT.gov dates as "claimed", not unconditional truth. |
| 4 | MCP as shared state manager | State lives in the LangGraph checkpointer (SQLite) and the app SQLite DB. MCP is only the tool boundary. |
| 5 | LLM-assigned branch probabilities | Anchor to base rates (phase/indication approval rates in `data/base_rates.yaml`); the LLM may adjust within a bounded band (`max_prob_adjustment`, default ±0.15). Log base rate, adjustment, and justification. |
| 6 | "Robinhood MCP" for live execution | Use Robinhood's official Trading MCP (section 7), behind the `Broker` interface. Alpaca paper trading is the default and the dry-run mode. |

## 4. Tech stack

- Python 3.11+, `uv` or `pip` + `pyproject.toml`
- LangGraph (Supervisor state machine, checkpointer, `interrupt` for HITL)
- Pydantic v2 (all schemas and inter-agent JSON contracts)
- SQLite (`sqlite3`/SQLAlchemy) for audit, positions, evaluated NCTs, kill-switch state
- ChromaDB (local persistent) for RAG
- `httpx` for HTTP clients, `tenacity` for retries
- `alpaca-py` for Alpaca; official `mcp` Python SDK (streamable HTTP client) for Robinhood
- `pytest`, `pytest-asyncio`, `hypothesis` (property tests for guardrails), `ruff`, `mypy`
- LLM access behind an `LLMClient` interface (provider configurable by env; default Anthropic). Agents depend on the interface, not a vendor SDK.
- Do NOT add CrewAI. Model Generator and Critic as separate LangGraph nodes with separate prompts.

## 5. Repository layout

```
BioTradeAI/
  README.md
  IMPLEMENTATION_PLAN.md
  pyproject.toml
  .gitignore            # .env, data/*.db, data/chroma/, __pycache__, .venv
  .env.example
  config/
    guardrails.yaml
  data/
    catalysts.csv       # human-curated: ticker, drug, event, date, source_url
    base_rates.yaml
    synonyms.yaml       # entity/drug synonym dictionary
    fixtures/           # offline sample filings, CT.gov JSON, scenario fixtures
  src/biotradeai/
    config.py           # loads + validates guardrails.yaml (pydantic)
    schemas.py          # CatalystState, SourceRecord, EvidenceBundle, ToTNode, Scenario, StagedOrder, ApprovalToken
    store/
      sqlite_store.py   # audit (append-only), positions, evaluated_ncts, killswitch
    data/
      clinicaltrials.py # CT.gov v2 client
      edgar.py          # submissions + XBRL companyfacts client (User-Agent required, <=10 req/s)
      runway.py         # cash, burn, runway from XBRL facts (pure code)
      freshness.py      # TTL checks
    rag/
      ontology.py       # Entity_ID mapper
      ingest.py         # chunking + metadata tagging
      retrieve.py       # metadata pre-filter + vector search + tier resolution + confidence
    guardrails/
      rules.py          # pure functions (position cap, concurrency, EV, runway, spread, drawdown, confidence, TTL)
      killswitch.py
    agents/
      research.py       # Clinical Research Agent (ToT generator)
      risk.py           # Quantitative Risk Agent (recomputes EV independently)
      llm.py            # LLMClient interface + implementations
    brokers/
      base.py           # Broker protocol
      alpaca_broker.py  # paper (default) and live
      robinhood_broker.py
      execution.py      # the ONLY place orders are placed; verifies approval token
      approval.py       # HMAC approval tokens
    graph/
      buy_graph.py      # Supervisor LangGraph: ingest -> research -> risk -> HITL -> execute
      sell_graph.py     # deterministic sell state machine
    cli.py              # evaluate, approve, positions, killswitch, replay
  tests/
    unit/ integration/ fixtures/
```

## 6. `config/guardrails.yaml` (single source of truth)

```yaml
mode: paper                  # paper | shadow | live
min_ev: 0.20
min_runway_months: 6         # Risk Agent veto
fatal_runway_months: 3       # ToT immediate prune
max_position_pct: 0.05       # of account equity (broker account, see section 7)
max_concurrent_positions: 3
min_confidence: 0.70
max_spread_pct: 0.015
ttl_hours: 24
daily_drawdown_limit: 0.03   # vs start-of-day account equity
max_api_errors: 3            # consecutive
orders:
  allowed_types: [limit]
  allow_extended_hours: false   # revisit after Robinhood spike
tot:
  depth: 3
  beam_width: 2
  residual_scenario_return: -0.70   # PLACEHOLDER, set deliberately
  max_prob_adjustment: 0.15
  prob_sum_tolerance: 0.001
sell:
  profit_target_pct: 0.30
  scale_out_fraction: 0.50
  trailing_stop_pct: 0.10          # PLACEHOLDER
  completion_delay_months: 6
live_gate:
  min_paper_days: 90
  pilot_position_pct: 0.01
  pilot_max_positions: 1
```

Rule: every threshold is read from this file. No magic numbers in code. Tests must assert behavior at, just below, and just above each threshold.

## 7. Brokers and modes

`Broker` protocol: `get_account()`, `get_quote()`, `get_order_book()`, `review_order()`, `place_order()`, `cancel_order()`, `get_positions()`, `get_orders()`. Agents never see this; only `execution.py` and the data layer do.

| Mode | Broker | Behavior |
|---|---|---|
| `paper` (default, dry run) | Alpaca paper trading | Orders actually placed on paper. Requires a valid CLI approval token. |
| `shadow` | Robinhood MCP | Reads and `review_order` (simulation) only. `place_order` raises. Log "would have placed" to audit. |
| `live` | Robinhood MCP (or Alpaca live later) | Limit orders only, with Robinhood trade approvals ON. |

### Alpaca (paper)
- Base URL `https://paper-api.alpaca.markets`; keys via `ALPACA_API_KEY` / `ALPACA_API_SECRET`.
- Used for pipeline validation. Do not treat paper fills as proof of live fill quality for illiquid biotech.

### Robinhood official Trading MCP
- Remote MCP server: `https://agent.robinhood.com/mcp/trading` (streamable HTTP; authenticates via Robinhood's flow).
- Orders can only be placed in a dedicated Robinhood MCP account funded by the user. This deposit is the hard loss cap.
- External-agent MCP accounts default to trade approvals OFF. At startup in `live` mode, call `get_trade_approval_setting`; if approvals are OFF, refuse to start.
- Docs list these tools (verify against `list_tools` at runtime; do not assume): `get_accounts`, `get_portfolio`, `get_equity_quotes`, `get_equity_price_book`, `get_equity_tradability`, `review_equity_order`, `place_equity_order`, `cancel_equity_order`, `get_equity_orders`, `get_equity_positions`, `get_sec_filing_facts`, `get_financials`, `get_earnings_calendar`, `get_trade_approvals`.
- The agent has read access to all the user's Robinhood accounts. The adapter MUST filter to the MCP account and strip account numbers before any data reaches an LLM or log.
- Robinhood supports market/stop orders; our adapter must still reject everything except limit.

### Approval gate
- `paper`/`shadow`: `approval.py` issues an HMAC-SHA256 token over `{ticker, side, qty, limit_price, expiry, nonce}` using `APPROVAL_HMAC_SECRET`. The CLI shows ticker, limit, qty, EV, confidence, runway, top sources, then waits for `approve`/`reject`. `execution.py` verifies the token, expiry, and single use (nonce stored in SQLite) before placing.
- `live`: Robinhood's in-app trade approval is the binding human gate (the agent cannot approve its own order). Still write the full evidence pack to the audit log and send a notification (CLI print / optional webhook).

### Live gate (all must be true)
1. `mode: live` in config AND `BIOTRADEAI_LIVE=1` in environment AND typed confirmation at startup.
2. Audit log shows at least `live_gate.min_paper_days` of paper operation with zero guardrail breaches.
3. Robinhood approvals verified ON.
4. Pilot sizing applies until the user edits config explicitly: `pilot_position_pct`, `pilot_max_positions`.

## 8. Phases (each has acceptance tests; do not skip)

### Phase 0 — Bootstrap (0.5 wk)
- Repo, `pyproject.toml`, ruff/mypy/pytest config, `.gitignore`, `.env.example`, `config/guardrails.yaml`.
- `config.py` loads YAML into a validated Pydantic model; fail fast on missing/invalid keys.
- `schemas.py`: `SourceRecord(tier, published_at, sha256, url, ticker, entity_id)`, `Scenario(depth, label, probability, expected_return, evidence_ids)`, `EvidenceBundle`, `CatalystState`, `StagedOrder`, `ApprovalToken`.
- `sqlite_store.py`: tables `audit_log` (append-only; enforce with triggers that raise on UPDATE/DELETE), `positions`, `evaluated_ncts`, `approval_nonces`, `killswitch`.
- Acceptance: config rejects bad values; audit triggers block update/delete (tested); schema validation rejects probabilities not summing to 1.0.

### Phase 1 — Data layer (1 wk, no LLM)
- `clinicaltrials.py` (CT.gov v2 `studies` endpoint), `edgar.py` (submissions + `companyfacts`, descriptive `User-Agent`, rate-limited), `freshness.py`.
- `runway.py`: cash (cash and equivalents + marketable securities), quarterly burn from operating cash flow, runway months. Handle missing tags, restatements, and duplicate facts deterministically.
- Fixtures for 10 tickers with hand-verified runway values.
- Acceptance: computed runway matches hand-verified values; stale records (older than TTL) are rejected; network clients are mockable and tests run offline.

### Phase 2 — Guardrails, brokers, kill-switch (1.5 wk)
- `rules.py` pure functions: position size cap, concurrency, EV threshold, runway veto and fatal prune, spread check, confidence check, TTL check, drawdown, order-type check.
- `killswitch.py`: trips on daily drawdown > limit or `max_api_errors` consecutive failures; cancels open orders, blocks new ones, persists state across restarts; only a human CLI command resets it.
- `Broker` protocol, `AlpacaBroker`, `execution.py`, `approval.py`, `RobinhoodBroker` (shadow-capable; `place_order` gated by mode).
- Acceptance: property tests (hypothesis) show no order can exceed 5% of equity or be non-limit; forged/expired/reused tokens rejected; killswitch survives restart; forced drawdown test cancels orders; shadow mode never calls place.

### Phase 3 — Robinhood spike (0.5 wk, read-only)
Goal: confirm assumptions the docs don't answer. Write findings to `docs/robinhood_spike.md`.
- Can a custom MCP client authenticate and keep a session without repeated manual login?
- With approvals ON, does `place_equity_order` from an external agent create a pending approval visible via `get_trade_approvals`?
- Are limit orders accepted pre-market/after-hours?
- What do `review_equity_order` warnings look like for thin biotech names?
- Cash vs limited-margin settlement implications for our trade frequency.
Do not place real orders in this phase.

### Phase 4 — RAG (1.5 wk)
- `ontology.py`: synonym dictionary (`data/synonyms.yaml`) maps pipeline codes, generic and brand names to one `Entity_ID`. An LLM may propose new mappings but a human approves them into the YAML.
- `ingest.py`: sources are EDGAR 10-K/10-Q Item 1A, FDA AdCom briefing docs, PubMed abstracts. Paragraph chunks of ~300–500 tokens with 50 overlap. Metadata: `ticker, entity_id, source_tier, published_at, sha256`.
- `retrieve.py`: hard metadata pre-filter on `ticker` AND `entity_id`, then vector similarity, top-k 3–5; tier resolution Tier 1 (FDA/CT.gov) > Tier 2 (SEC) > Tier 3 (PubMed/press); same-tier conflicts resolved by most recent date; confidence score from source authority and chunk density; below `min_confidence` or unresolved Tier 1 vs Tier 2 conflict returns `INCONCLUSIVE`.
- Build a labeled retrieval test set (30 drugs with known risk facts).
- Acceptance: recall target agreed with me before coding (propose one); a semantic-bleed test returns zero chunks from a different ticker/entity; renamed-drug test retrieves old-code risk factors.

### Phase 5 — Agents and ToT (2 wk)
- `LLMClient` interface plus a deterministic fake client for tests.
- Research Agent: depth 1 clinical outcomes (must enumerate success, partial, and failure/CRL so probabilities sum to 1), depth 2 regulatory reaction, depth 3 financial/dilution impact. Beam width 2 on clinical outcomes. Fatal-flaw prune: active clinical hold or runway below `fatal_runway_months` collapses the branch into the residual scenario.
- Risk Agent: mostly deterministic. Recomputes EV itself from scenario probabilities and returns, ignoring the Research Agent's own EV. Checks runway, portfolio caps, spread/liquidity, confidence, and freshness. Has veto only.
- Supervisor (`buy_graph.py`): LangGraph state machine with checkpointer; transitions validated by Pydantic JSON schemas; workers cannot call each other. Routes: BUY candidate → HITL, SKIP, INCONCLUSIVE (human review). Supervisor cannot generate hypotheses, alter risk math, or call brokers.
- Acceptance: `$ONCO` fixture (cash $40M, burn $10M/quarter) runs end to end with the failure branch included in EV; conflict fixture routes to INCONCLUSIVE; dilution-trap fixture (1 month of cash) routes to SKIP; EV recomputation mismatch between agents is logged and blocks the trade.

### Phase 6 — Sell state machine (1 wk, deterministic)
Triggers, evaluated by event, no LLM:
1. Pre-catalyst run-up: price reaches `profit_target_pct` before the event → sell.
2. Post-catalyst: positive readout → sell `scale_out_fraction`, trailing stop on the rest; failure/CRL → sell 100% (limit orders at a marketable price, not market orders).
3. Dilution invalidation: new Form S-3 or 8-K indicating a secondary offering → sell.
4. Time-stop: primary completion date pushed out more than `completion_delay_months` → free the capital.
Never hold between Phase 2 and Phase 3 readouts (treat as independent trades).
- Pollers for EDGAR filings and CT.gov date changes.
- Acceptance: replayed event streams from fixtures fire exactly the expected exits; sell orders also pass the order-type guardrail; all exits are audit-logged.

### Phase 7 — Backtest and evaluation (2 wk)
- Point-in-time harness: only documents with `published_at` before the decision date may be retrieved. Test that a future-dated document cannot be returned.
- Metrics: Brier score (target ≤ 0.18), groundedness (every clinical claim and runway figure traceable to a source hash; target ≥ 98%), guardrail compliance (0 breaches), escalation accuracy (100% INCONCLUSIVE on conflicting Tier 1/Tier 2), latency (≤ 45 s per ticker excluding human wait).
- Output a report to `reports/` for each run.

### Phase 8 — Alpaca paper trading (8–12 wk)
Run on 5–10 tickers from `data/catalysts.csv`. Daily review of the audit log. No live orders.

### Phase 9 — Robinhood shadow mode (4 wk)
Live data and `review_equity_order` only. Compare what would have been placed against paper decisions.

### Phase 10 — Live pilot
Only after the live gate in section 7. Small dedicated MCP-account deposit, approvals ON.

## 9. Working rules for Claude Code

- Plan first: before each phase, list the files you will create or change and the tests you will write. Then implement test-first.
- Run `ruff`, `mypy`, and `pytest` before every commit; do not commit failing tests.
- Never call a real broker `place_order` in tests; use fakes. Network-dependent tests must be marked `integration` and skipped by default.
- Do not invent API fields. Verify against the official docs or `list_tools` output; when unsure, write a small probe script and record the result.
- Ask me when a PLACEHOLDER value in config needs a real decision (`residual_scenario_return`, `trailing_stop_pct`, the extended-hours setting, retrieval recall target).
- Keep a `docs/DECISIONS.md` with a short entry for every design decision or deviation.
- Do not add dependencies without noting why in `docs/DECISIONS.md`.

## 10. Open questions for me (ask before Phase 3/5)

1. Which 5–10 tickers and catalysts seed `data/catalysts.csv`?
2. Which LLM provider and model for the Research Agent?
3. Dedicated MCP-account deposit size and a pilot loss limit.
4. Do I want extended-hours orders allowed (depends on the Phase 3 spike)?
