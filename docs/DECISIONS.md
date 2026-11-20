# Design decisions

Short entries, newest last. Each says what was decided and why.

## D-001 — Phase 0: package layout and tooling (2026-11-19)
`src/` layout, setuptools backend, `pyproject.toml` holds ruff/mypy/pytest config.
Tests marked `integration` (network, local Ollama, broker sandbox) are skipped by default.

## D-002 — Dependencies (2026-11-19)
Core: `pydantic` (contracts), `PyYAML` (guardrails), `python-dotenv` (`.env` secrets),
`httpx` (HTTP clients incl. Ollama), `tenacity` (retries, Phase 1).
Optional extras so nothing is installed before it's needed: `anthropic`, `graph`
(LangGraph), `rag` (ChromaDB), `brokers` (alpaca-py, mcp). Dev: pytest, pytest-asyncio,
hypothesis, ruff, mypy, types-PyYAML.
Ollama is reached via plain `httpx` against `/api/chat` rather than the `ollama` package,
which saves a dependency and lets tests use `httpx.MockTransport`.

## D-003 — 5% position cap is a hard ceiling in the config validator (2026-11-19)
The governance plan calls ≤5% equity per ticker a hard cap. `config.py` rejects
`max_position_pct > 0.05` (and requires pilot caps ≤ main caps). Config can be stricter,
never looser. Constant `HARD_CEILING_POSITION_PCT` is the one policy number in code.

## D-004 — `orders.allowed_types` may only contain `limit` (2026-11-19)
Typed as `list[Literal["limit"]]`, so listing `market`/`stop` fails at load. The broker
adapter rejects non-limit orders again in Phase 2 (defense in depth).

## D-005 — `residual_scenario_return` must be in [-1.0, 0.0] (2026-11-19)
A collapsed/pruned scenario is meant to be conservative, so it may not be a gain.
The actual value (-0.70) is still a PLACEHOLDER to decide before Phase 5.

## D-006 — ScenarioSet carries its tolerance explicitly (2026-11-19)
`ScenarioSet(scenarios=..., tolerance=cfg.tot.prob_sum_tolerance)`. Keeps schemas free
of config imports and avoids a hidden default tolerance. Scenarios in a set are terminal
outcomes with *joint* probabilities; `ToTNode.probability` is conditional on its parent.

## D-007 — Audit log is hash-chained in addition to triggers (2026-11-19)
Triggers block UPDATE/DELETE through SQLite. A SHA-256 chain (`prev_hash`, `entry_hash`)
makes tampering outside SQLite (dropping triggers, editing the file) detectable via
`verify_audit_chain()`. Payloads are redacted before hashing and writing.

## D-008 — LLM switcher built in Phase 0, not Phase 5 (2026-11-19)
The user wants to swap between local (Ollama) and hosted models, so `agents/llm.py`
landed early. Spec `<provider>:<model>`, per-role env overrides
(`BIOTRADEAI_LLM_RESEARCH`), default `anthropic:claude-sonnet-5-5` per the plan.
All prompts pass through `redact_text` before leaving the process.

## D-009 — Monetary values use `Decimal` (2026-11-19)
`limit_price` is `Decimal` (≤4 dp) so approval-token signatures are canonical and free of
float rounding (`4.5` and `4.5000` sign identically).
