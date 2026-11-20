# BioTradeAI

Multi-agent decision support for trading biotech equities around catalysts (FDA PDUFA
dates, Phase 3 readouts). A Research Agent builds a Tree-of-Thoughts scenario analysis, an
independent Risk Agent re-checks it with deterministic rules, and a limit order is staged
for **human approval**. Selling is a deterministic state machine, not an LLM.

> Decision support only — not financial advice. Default mode is `paper`.

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the full design and
[docs/DECISIONS.md](docs/DECISIONS.md) for design decisions.

## Status

| Phase | What | Status |
|---|---|---|
| 0 | Bootstrap: config, schemas, SQLite audit store, LLM switcher | done |
| 1 | Data layer (CT.gov, EDGAR, runway, catalysts.csv) | done |
| 2 | Guardrail rules, approval tokens, brokers, kill-switch, execution | done |
| 4 | RAG: ontology, ingest, hybrid retrieval, tier/conflict resolution | done |
| 5 | Agents + Tree of Thoughts + LangGraph supervisor | done |
| 6 | Sell state machine (deterministic), EDGAR/CT.gov pollers | done |
| 7 | Point-in-time backtest, metrics, reports | done |
| 8 | Alpaca paper trading: runner, daily review, live-gate counter (run needs your keys) | built |
| 3, 9, 10 | Robinhood spike, shadow mode, live pilot (needs your accounts) | last |

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,rag,graph]"
copy .env.example .env      # then edit .env
```

On Linux/macOS use `source .venv/bin/activate` and `cp`.

## Everyday commands

```powershell
python -m biotradeai config-check          # validate config/guardrails.yaml
python -m biotradeai init-db               # create data/biotradeai.db
python -m biotradeai audit-tail -n 20      # show audit log + verify hash chain
python -m biotradeai llm-ping --llm ollama:llama3.2:3b
python -m biotradeai runway ONCO --facts-file data\fixtures\edgar\companyfacts\ONCO.json
python -m biotradeai runway MRNA            # live SEC data (needs SEC_USER_AGENT)
python -m biotradeai trial NCT04368728      # live ClinicalTrials.gov
python -m biotradeai catalysts
python -m biotradeai account                      # Alpaca paper (needs keys)
python -m biotradeai positions
python -m biotradeai order ONCO --qty 100 --limit 10.00 --broker fake   # dry run
python -m biotradeai killswitch status            # trip | reset
python -m biotradeai rag eval                     # recall/bleed on 30-drug test set
python -m biotradeai rag ingest doc.pdf --ticker ONCO --tier 1 --date 2026-01-05 --url https://... --drug ONC-101
python -m biotradeai rag query ONCO ONC-101 "hepatic safety signals?"
python -m biotradeai ontology resolve oncotinib   # list | proposals | propose | approve
python -m biotradeai evaluate --fixture data\fixtures\scenarios\onco.yaml   # ToT -> risk -> approve
python -m biotradeai evaluate --fixture data\fixtures\scenarios\onco.yaml --llm ollama:llama3.2:3b
python -m biotradeai sell-replay                  # replay all sell-trigger fixtures
python -m biotradeai backtest                     # base-rate baseline -> reports/
python -m biotradeai backtest --llm ollama:llama3.2:3b
python -m biotradeai paper run                    # one cycle (needs keys); approve prompts
python -m biotradeai paper run --broker fake --auto-reject   # dry run
python -m biotradeai paper review                 # today's audit summary
python -m biotradeai paper gate                   # paper days / breaches for live gate
```

Quality gates (run before every commit):

```powershell
ruff check .
mypy
pytest                      # unit tests, offline
pytest -m integration       # needs network / a running Ollama
```

## Switching LLMs

Models are chosen with a spec string `<provider>:<model>`:

| Spec | Meaning |
|---|---|
| `ollama:llama3.2:3b` | local model served by Ollama (`OLLAMA_HOST`, default `http://localhost:11434`) |
| `anthropic:claude-sonnet-5-5` | Claude via API (`pip install -e ".[anthropic]"`, set `ANTHROPIC_API_KEY`) |
| `fake` | deterministic stub used by tests |

Resolution order for each agent: CLI `--llm` → `BIOTRADEAI_LLM_<ROLE>` (e.g.
`BIOTRADEAI_LLM_RESEARCH`) → `BIOTRADEAI_LLM` → default `anthropic:claude-sonnet-5-5`.
So you can run the Research Agent on Claude and everything else on a local model.

## Safety model (short version)

* Every numeric threshold lives in `config/guardrails.yaml`; the loader rejects missing,
  unknown or out-of-range keys, and caps position size at 5%.
* LLMs never do arithmetic or enforce rules — those are plain, tested Python functions.
* The audit log is append-only (SQLite triggers) and hash-chained (tamper-evident).
* Secrets and account identifiers are redacted before logging or sending to an LLM.
