# Phase 3 — Robinhood Trading MCP spike (read-only)

Date: 2026-11-20. Scope: confirm how Robinhood's official Trading MCP behaves before any
adapter work (plan section 8, Phase 3). **No orders were placed**, and nothing about
the owner's accounts is recorded here, by design: no account numbers, nicknames, types,
balances, settings or history. Only tool *behaviour* is documented.

Access path: the Robinhood connector attached to the Claude host session. Tools used:
`get_accounts`, `get_trade_approval_setting`, `get_trade_approvals`, `get_equity_orders`,
`get_equity_quotes`, `get_equity_price_book`, `get_equity_tradability`, and
`review_equity_order` (simulation only).

## Answers to the plan's questions

### 1. Can a custom MCP client authenticate and keep a session?
**Not answered by this spike.** Here the connector was authenticated by the host (Claude),
not by BioTradeAI's own MCP client. A standalone client
(`mcp` SDK, streamable HTTP to `https://agent.robinhood.com/mcp/trading`) still has to
be tested in Phase 9 before the adapter is wired. Until then, `RobinhoodBroker` keeps
raising `SchemaNotVerifiedError` for every call that isn't mapped.

### 2. With approvals ON, does `place_equity_order` create a pending approval?
**Not tested: placing orders is out of scope for this phase.** What was confirmed:
* `get_trade_approval_setting(account_number)` returns `human_must_approve_trades`
  (bool) plus an `updated_at` timestamp. Only the customer can change it, in the app.
  No tool can.
* `get_trade_approvals(account_number)` lists pending approvals. The tool's guidance
  says approvals expire on their own and must be re-read before every statement.
* `review_equity_order` returns `customer_approval_required` (bool), which reflects the
  setting.
* **Live gate (plan section 7) is confirmed implementable:** read
  `human_must_approve_trades`, and refuse to start live mode unless it is `true`.

### 3. Are limit orders accepted pre-market / after-hours?
**Inconclusive. Keep `orders.allow_extended_hours: false`.**
* `get_equity_tradability` reports `all_day_tradability`. For every small-cap biotech
  sampled it was `untradable`. The tool's guidance ties extended/overnight sessions to
  that flag.
* `review_equity_order(..., market_hours="extended_hours")` returned **no session
  error**, so the review does not validate session eligibility. Only a real order would
  settle it.

### 4. What do `review_equity_order` warnings look like for thin biotech names?
* Alerts come in `order_checks` as `{"alertType": ..., "<alertType>Details": {...}}`,
  e.g. `EQUITY_NOT_ENOUGH_BP` with a `depositAmount`. An empty object `{}` means no
  broker alerts, **not** that the order is safe.
* **No spread or thin-book warning was returned** for very wide after-hours books, so our
  own spread rule (`max_spread_pct`) remains essential.
* **A sell of shares not held returned no alert.** The review does not enforce "no
  shorting", so `rules.check_sell_quantity` must stay in `execution.py`'s pre-trade checks.
* The response includes `market_data_disclosure`, a compliance quote line that must be
  shown **verbatim** whenever review prices are displayed. Its venue/price can differ
  from `quote_data` (seen: a different ask in the two). The approval screen and audit
  log must carry it unchanged.
* `quote_data` is a consolidated quote: last trade, last non-regular-hours trade,
  bid/ask with venue times, previous close.

### 5. Cash vs limited-margin settlement for our trade frequency
`get_accounts` exposes each account's trading type (`cash` / `limited_margin` /
`margin`) and `unsettled_funds`. BioTradeAI holds at most 3 positions for weeks, so T+1
settlement in a cash account is not a constraint. Limited margin mainly avoids
good-faith-violation risk on quick re-entry. No change to the design.

## Other findings that shape the adapter (Phase 9)
1. **Account selection:** `get_accounts` lists *all* the customer's accounts, with
   `agentic_allowed` marking the single one this agent may trade. The adapter must
   select it by that flag, never by nickname or order, and must treat the others as
   read-only.
2. **Identifiers everywhere:** `account_number`, `rhs_account_number`,
   `rhc_account_number`, `nickname`, `unsettled_funds`, buying power and deposit
   amounts all appear in responses. `redact.py` now strips all of them (tested), and
   the adapter must still drop them before anything reaches an LLM or the audit log.
3. **The tools' guidance conflicts with our safety model.** `review_equity_order`'s
   guidance tells an agent to call `place_equity_order` immediately when approval is
   required and no alerts are present. BioTradeAI must **not** follow tool guidance for
   control flow: `execution.py` stays the only order path, behind our own HMAC
   approval token. The Robinhood in-app approval is an *additional* gate in live mode.
4. **Tool names confirmed** against the connector (no names were invented):
   `get_accounts`, `get_portfolio`, `get_equity_quotes`, `get_equity_price_book`,
   `get_equity_tradability`, `review_equity_order`, `place_equity_order`,
   `cancel_equity_order`, `get_equity_orders`, `get_equity_positions`,
   `get_trade_approvals`, plus `get_trade_approval_setting` and
   `decline_trade_approval`. Exact field schemas beyond those quoted above are still
   to be captured in Phase 9 from a standalone client's `list_tools`.
5. **Market data quality:** consolidated quotes confirmed that some candidates trade
   below $1, with very thin after-hours books. Consider a minimum share-price filter in
   the scout (not yet implemented).

## Status
Phase 3 questions 2 (partly), 3, 4 and 5 answered; question 1 and the
approval-flow half of question 2 move to Phase 9, which needs a standalone client and
the owner's explicit go-ahead. Paper mode is unchanged. Robinhood stays shadow-only, and
the adapter still refuses unmapped calls.
