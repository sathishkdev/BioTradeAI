# BioTradeAI — Step-by-step setup

This guide takes you from a fresh checkout to a daily **Alpaca paper-trading** run.
Each step ends with a check so you know it worked before moving on. Commands are for
Windows PowerShell; Linux/macOS equivalents are noted where they differ.

> Decision support only, not financial advice. Nothing here places a real-money order:
> `mode: paper` is the default, live trading is blocked in code, and every order needs
> your typed approval.

**Contents**
1. [Prerequisites](#1-prerequisites)
2. [Get the code and a Python environment](#2-get-the-code-and-a-python-environment)
3. [Check the offline install](#3-check-the-offline-install)
4. [Pick your LLM](#4-pick-your-llm)
5. [Create accounts and keys](#5-create-accounts-and-keys)
6. [Fill in `.env`](#6-fill-in-env)
7. [Verify each live connection](#7-verify-each-live-connection)
8. [Add your tickers and drugs](#8-add-your-tickers-and-drugs)
9. [Load documents into the knowledge base](#9-load-documents-into-the-knowledge-base)
10. [Dry run, then your first paper cycle](#10-dry-run-then-your-first-paper-cycle)
11. [Schedule the daily run](#11-schedule-the-daily-run)
12. [Daily routine](#12-daily-routine)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Prerequisites

| Need | Why | Get it |
|---|---|---|
| Python **3.11+** | runtime (tested on 3.14) | https://www.python.org/downloads/ |
| Git | clone / commit | https://git-scm.com/ |
| Ollama (optional) | free local LLM | https://ollama.com/download |
| ~2 GB disk | embedding model (~130 MB), Ollama model (~2 GB) | — |

Check:

```powershell
python --version
```

```powershell
git --version
```

## 2. Get the code and a Python environment

```powershell
git clone <your-repo-url> BioTradeAI
```

```powershell
cd BioTradeAI
```

Create and activate a virtual environment inside the repo:

```powershell
python -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

(Linux/macOS: `python3 -m venv .venv` then `source .venv/bin/activate`.)

> Already have a venv elsewhere (e.g. `..\.vaenv`)? Activate it by its absolute path
> instead, e.g. `C:\Users\<you>\source\repos\agentic-capstone\.vaenv\Scripts\Activate.ps1`,
> and always `cd` into the `BioTradeAI` folder before running commands.

If PowerShell refuses to run the activate script, allow local scripts once for your user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Install the package with the extras you need:

```powershell
pip install -e ".[dev,rag,graph]"
```

Add `anthropic` if you will use Claude: `pip install -e ".[dev,rag,graph,anthropic]"`.

## 3. Check the offline install

Nothing below needs keys or network (apart from a one-time model download in `rag eval`).

```powershell
python -m biotradeai config-check
```

```powershell
python -m biotradeai init-db
```

```powershell
pytest
```

Expected: `OK: guardrails valid, mode=paper`, a database at `data\biotradeai.db`, and about 400 tests passing.

See the whole pipeline work on fixtures (scripted LLM, fake broker):

```powershell
python -m biotradeai evaluate --fixture data\fixtures\scenarios\onco.yaml --auto-reject
```

```powershell
python -m biotradeai sell-replay
```

```powershell
python -m biotradeai rag eval
```

`rag eval` downloads the `BAAI/bge-small-en-v1.5` embedding model the first time and
should print recall 1.000 with zero bleed.

## 4. Pick your LLM

Agents use a model spec `<provider>:<model>`, switchable per agent role.

| Option | Cost | Quality | Setup |
|---|---|---|---|
| `ollama:llama3.2:3b` | free, local | weak at structured reasoning; ~30–80 s per ticker | `ollama pull llama3.2:3b` |
| `anthropic:claude-sonnet-5-5` | API usage | much stronger | API key from https://console.anthropic.com/ |

For local Ollama:

```powershell
ollama pull llama3.2:3b
```

```powershell
python -m biotradeai llm-ping --llm ollama:llama3.2:3b
```

You can mix models: e.g. Claude for the Research generator, a local model for the
critic, via `BIOTRADEAI_LLM_RESEARCH` / `BIOTRADEAI_LLM_CRITIC` in `.env`. The Risk Agent
uses no LLM at all.

## 5. Create accounts and keys

You do these steps yourself; the software never creates accounts or handles your
passwords.

**a) Alpaca paper trading (required for Phase 8)**
1. Sign up at https://alpaca.markets/ (a paper account is free and needs no deposit).
2. In the dashboard, switch to **Paper Trading**.
3. Open **API Keys** → **Generate New Keys**. Copy the *Key ID* and *Secret Key* (the
   secret is shown only once).
4. Paper keys only work against `paper-api.alpaca.markets`, which is the only endpoint
   this build talks to.

**b) SEC EDGAR User-Agent (required, no account)**
SEC asks every automated client to identify itself with a name and contact email, e.g.
`BioTradeAI Jane Doe jane@example.com`. That string goes in `SEC_USER_AGENT`.

**c) Approval signing secret (required, generate locally)**

```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```

Copy the 64-character output. It signs the one-time approval tokens for your orders.

**d) Anthropic API key (optional)** from https://console.anthropic.com/ if you use Claude.

## 6. Fill in `.env`

```powershell
copy .env.example .env
```

```powershell
notepad .env
```

Fill in:

```dotenv
BIOTRADEAI_LLM=ollama:llama3.2:3b        # or anthropic:claude-sonnet-5-5
ALPACA_API_KEY=<paper key id>
ALPACA_API_SECRET=<paper secret>
APPROVAL_HMAC_SECRET=<64 hex chars from step 5c>
SEC_USER_AGENT="BioTradeAI <your name> <your email>"
BIOTRADEAI_LIVE=0
# ANTHROPIC_API_KEY=<key>                # only if using Claude
```

`.env` is git-ignored; never commit it. Leave `LANGSMITH_TRACING` unset (it would send
prompts to an external service).

## 7. Verify each live connection

```powershell
python -m biotradeai account
```

Shows paper equity (the account number is never printed).

```powershell
python -m biotradeai runway MRNA
```

Live SEC data: prints cash, quarterly burn and runway months.

```powershell
python -m biotradeai trial NCT04368728
```

Live ClinicalTrials.gov record.

```powershell
pytest -m integration
```

Runs the live, read-only probes (SEC, CT.gov, Alpaca paper account, local Ollama). None place
orders.

## 8. Add your tickers and drugs

**a) Catalysts** — `data\catalysts.csv` is the source of truth for event dates (5–10 rows):

```csv
ticker,drug,event,date,source_url
ABCD,ABC-123,PHASE3_READOUT,2027-03-15,https://investors.example.com/press-release
WXYZ,wexitinib,PDUFA,2027-02-01,https://investors.example.com/pdufa
```

- `event`: `PHASE3_READOUT` or `PDUFA` are evaluated (others are listed but skipped,
  since there are no base rates for them yet).
- `date`: from the company's 8-K or press release, not from CT.gov (CT.gov dates are
  sponsor estimates).

```powershell
python -m biotradeai catalysts
```

**b) Drug names** — `data\synonyms.yaml` maps every name of a drug to one ID. Add every
pipeline code, generic name, brand name and old code:

```yaml
entities:
  ENT-ABCD-ABC123:
    ticker: ABCD
    names: [ABC-123, abcinib, Abcora]
    previous_codes: [XY-77]
```

```powershell
python -m biotradeai ontology resolve abcinib
```

**c) Base rates** — review `data\base_rates.yaml`. The headline rates are cited from
BIO/Informa/QLS 2021; the "partial" splits and every return figure are assumptions.
When you have checked them, set `reviewed: true`.

**d) Placeholders** in `config\guardrails.yaml` to set deliberately:
`tot.residual_scenario_return` (−0.70) and `sell.trailing_stop_pct` (0.10).

## 9. Load documents into the knowledge base

For each drug, save its sources as PDF, HTML or TXT and ingest them with the right tier
and published date:

| Tier | Source | Where |
|---|---|---|
| 1 | ClinicalTrials.gov record, FDA briefing docs | clinicaltrials.gov (save page as HTML), fda.gov |
| 2 | 10-K / 10-Q (Item 1A Risk Factors), 8-K | sec.gov EDGAR filing index |
| 3 | PubMed abstracts, press releases | pubmed.ncbi.nlm.nih.gov, company IR site |

```powershell
python -m biotradeai rag ingest C:\docs\abcd-10k.htm --ticker ABCD --tier 2 --date 2026-03-01 --url https://www.sec.gov/... --drug ABC-123
```

Check what the system would see (point-in-time with `--as-of`):

```powershell
python -m biotradeai rag query ABCD ABC-123 "liver safety signals?"
```

Aim for at least two relevant Tier-1/Tier-2 documents per drug. With fewer, confidence
stays below 0.70 and the ticker is routed to you as INCONCLUSIVE, which is by design.

## 10. Dry run, then your first paper cycle

Dry run: the fake broker, never approves anything.

```powershell
python -m biotradeai paper run --broker fake --auto-reject
```

Real paper cycle: Alpaca paper, asks you to approve each order.

```powershell
python -m biotradeai paper run
```

For each BUY candidate, it shows the order, EV, confidence, runway, sources and every
risk check. Type `approve` to send it to Alpaca paper; anything else rejects it.

## 11. Schedule the daily run

Run one cycle each trading day, e.g. 30 minutes after the open (10:00 New York time).
Because each order needs typed approval, run it in a terminal you watch, or schedule the
dry run and approve manually.

Windows Task Scheduler (dry run each weekday at 10:00 local time; adjust the paths):

```powershell
schtasks /Create /TN "BioTradeAI paper dry run" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 10:00 /TR "C:\path\to\BioTradeAI\.venv\Scripts\python.exe -m biotradeai paper run --auto-reject"
```

Linux cron (same idea): `0 10 * * 1-5 cd /path/BioTradeAI && .venv/bin/python -m biotradeai paper run --auto-reject`.

## 12. Daily routine

1. `python -m biotradeai paper run` and approve or reject each prompt.
2. `python -m biotradeai paper review`: decisions, orders, exits, skips, breaches, and
   an audit-chain check (must say `OK`).
3. Enter any readout results in `data\outcomes.csv` the same day:

   ```csv
   ticker,drug,date,result,source_url
   ABCD,ABC-123,2027-03-15,positive,https://investors.example.com/topline
   ```

   `result` is one of `positive | negative | crl | mixed`. The next cycle uses it to
   scale out (positive) or exit fully (the rest).
4. Weekly: `python -m biotradeai paper gate` shows paper days and breaches toward the
   90-day live gate.

If anything looks wrong: `python -m biotradeai killswitch trip` stops everything and
cancels open orders. Only `killswitch reset` (typed `RESET`) re-enables trading.

## 13. Troubleshooting

| Message | Fix |
|---|---|
| `SEC_USER_AGENT must be set` | add it to `.env` (step 6) |
| `ALPACA_API_KEY / ALPACA_API_SECRET are not set` | add paper keys to `.env` |
| `APPROVAL_HMAC_SECRET must be at least 32 characters` | regenerate (step 5c) |
| `unknown drug name ... add it to data/synonyms.yaml` | step 8b |
| ticker always `INCONCLUSIVE` with low confidence | ingest more Tier-1/Tier-2 documents (step 9) |
| `no usable quote ... market closed?` | run during market hours |
| `kill-switch is tripped` | check `killswitch status` and `paper review`, then `killswitch reset` |
| PowerShell won't activate the venv | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `.venv` not found | `cd` into `BioTradeAI` first, or activate by absolute path |
| Ollama connection refused | start the Ollama app; check `OLLAMA_HOST` |
