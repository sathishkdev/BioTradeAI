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
8. [Build your watchlist (scout + your approval)](#8-build-your-watchlist-scout--your-approval)
9. [Let the agent collect documents and events](#9-let-the-agent-collect-documents-and-events)
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

**e) NCBI contact email and API key (optional, free)** from https://www.ncbi.nlm.nih.gov/account/.
NCBI asks automated PubMed clients for a contact email. A key raises our PubMed rate
from 2 to 5 requests/s.

**f) openFDA API key (optional, free)** from https://open.fda.gov/apis/authentication/.
Without a key we cap openFDA at 500 requests/day, under their 1,000/day per-IP limit.

None of these is needed to start. Every source works without its optional key, just
more slowly. How each source is protected from overload: see the README section
*Data sources and network guardrails*.

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
# NCBI_EMAIL=<your email>                # optional (step 5e)
# NCBI_API_KEY=<key>                     # optional (step 5e)
# OPENFDA_API_KEY=<key>                  # optional (step 5f)
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

## 8. Build your watchlist (scout + your approval)

The agent finds candidates itself. You only approve or reject company+drug pairs.

**a) Run the scout** (weekly is enough):

```powershell
python -m biotradeai watchlist scout
```

It reads the SEC daily filing index for 8-Ks from listed biotech companies, and
ClinicalTrials.gov for industry Phase 3 trials finishing in 3–12 months. It matches
trial sponsors to listed companies, and drops:
* profitable companies;
* anything with under 6 months of runway, or no runway data;
* drugs already FDA-approved (an openFDA check, which also drops label-expansion
  trials);
* pairs you have already decided.

It then ranks what is left and proposes up to 10 pairs.

Add `--with-quotes` to include spread and market cap from Alpaca. That also drops
companies above $5B (change it with `--max-market-cap-billions`). Spread is shown on
the card only; it is checked again on a live quote at trade time. On the first run, the SEC index scan builds a
company cache, which takes a few minutes at our polite 5 requests/second.

**b) Review each proposal's card and decide:**

```powershell
python -m biotradeai watchlist list
```

```powershell
python -m biotradeai watchlist show ABCD:ABC-123
```

The card shows:
* the company, how it was matched, cash runway, and market data;
* the drug's names, trials, endpoints and enrollment;
* each catalyst with the **exact sentence and SEC link** it came from;
* red flags (going concern, shelf offerings, no runway data), concentration, and
  whether the event type is tradable.

```powershell
python -m biotradeai watchlist approve ABCD:ABC-123 --note "clean Phase 2, 18 mo cash"
```

```powershell
python -m biotradeai watchlist reject WXYZ:WX-9 --reason "big pharma, readout won't move it"
```

Approving adds the drug's names to `data\synonyms.yaml` (refused if a name already
belongs to another drug). From then on the agent follows the drug through every phase
until it reaches the market. You are asked again only when:
* something material happens: acquisition or merger, licensing, a clinical hold, a
  terminated trial, delisting;
* the yearly check-in comes round;
* the first full quarter after launch has been reported.

A rejected pair is not proposed again unless something material changes, such as a
new catalyst type.

**c) Manual entries still work.** For a catalyst the scout didn't find, add a row to
`data\catalysts.csv` (`ticker,drug,event,date,source_url`; events `PHASE3_READOUT` or
`PDUFA`) and the drug's names to `data\synonyms.yaml`. Paper runs use both sources.

**d) Base rates** — review `data\base_rates.yaml`. The headline rates are cited from
BIO/Informa/QLS 2021; the "partial" splits and every return figure are assumptions.
When you have checked them, set `reviewed: true`.

**e) Placeholders** in `config\guardrails.yaml` to set deliberately:
`tot.residual_scenario_return` (−0.70) and `sell.trailing_stop_pct` (0.10).

## 9. Let the agent collect documents and events

```powershell
python -m biotradeai watchlist follow
```

**The first run after you approve a drug** loads:
* the latest 10-K and 10-Q risk factors;
* 180 days of 8-Ks, with their press-release exhibits;
* the drug's ClinicalTrials.gov records, including posted results;
* recent PubMed abstracts.

Every document keeps its real publication date.

**After that, each daily run reads only new filings** for followed companies (from the
SEC daily index) and:
* updates catalyst dates (`watchlist events`), keeping every earlier date;
* records outcomes, FDA approval and launch;
* pauses the pair on material changes;
* refreshes each drug's PubMed abstracts about once a week.

```powershell
python -m biotradeai watchlist events
```

```powershell
python -m biotradeai rag query ABCD ABC-123 "liver safety signals?"
```

Catalyst dates come only from company filings, with the source sentence kept;
ClinicalTrials.gov estimates are never used. Two filings on the same day that disagree
mark the event as *conflicting*, and it isn't traded until a newer filing settles it.
FDA briefing PDFs and company websites aren't fetched automatically. If you have one,
add it with `rag ingest <file> --ticker ... --tier 1 --date ... --url ... --drug ...`.

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

Suggested schedule (local times; adjust the paths):

| Job | When | Why |
|---|---|---|
| `watchlist scout --spread-minutes 120` | Sunday night | weekly proposals, spread over 2 hours |
| `watchlist follow --spread-minutes 60` | weekdays ~6:00 | new filings from the previous day |
| `paper run` | weekdays ~10:00 | needs you at the keyboard to approve orders |

```powershell
schtasks /Create /TN "BioTradeAI scout" /SC WEEKLY /D SUN /ST 22:00 /TR "C:\path\to\BioTradeAI\.venv\Scripts\python.exe -m biotradeai watchlist scout --spread-minutes 120"
```

```powershell
schtasks /Create /TN "BioTradeAI follow" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 06:00 /TR "C:\path\to\BioTradeAI\.venv\Scripts\python.exe -m biotradeai watchlist follow --spread-minutes 60"
```

Because each order needs typed approval, run `paper run` in a terminal you watch, or
schedule the dry run (`paper run --auto-reject`) and approve manually.

Linux cron (same idea):
`0 22 * * 0 cd /path/BioTradeAI && .venv/bin/python -m biotradeai watchlist scout --spread-minutes 120`.

## 12. Daily routine

1. `python -m biotradeai paper run` and approve or reject each prompt. It evaluates
   catalysts for **approved** watchlist drugs plus any rows in `catalysts.csv`.
2. `python -m biotradeai paper review`: decisions, orders, exits, skips, breaches, and
   an audit-chain check (must say `OK`).
3. `python -m biotradeai watchlist list --status paused`: anything waiting for you.
   Re-approve it with `watchlist approve <id>` or end it with
   `watchlist retire <id> --reason ...`.
4. Readout results are read from 8-Ks automatically. If the company announced one
   only on its website, add it to `data\outcomes.csv`
   (`ticker,drug,date,result,source_url`; result `positive | negative | crl | mixed`).
5. Weekly: `watchlist show` on new proposals, and `paper gate` for progress toward the
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
| `... daily request budget ... is spent` | a source hit its daily cap; it resumes tomorrow (UTC). Raise it in `data/netpolicy.py` only if the provider allows more |
| `... switched off until tomorrow` | 3 failed requests in a row (provider down or rate-limiting); it resumes tomorrow (UTC) |
| `... is not an allowed data source` | the code tried an unlisted host; that's the guardrail working. Add a host only if it is an official API |
