"""Operator app: agent host (schedules + runs), approve-later queue, service layer, UI.

Layers (outer may import inner, never the reverse):
    desktop / api  ->  service modules (jobs, trades, settings, watchlist_ops, safety, views)
                   ->  db (SQLite tables in the main database)  ->  existing BioTradeAI code

The UI never talks to a broker or places an order itself: trade approvals go through
``operator/trades.py``, which issues the same HMAC token as the CLI and hands the order
to ``brokers/execution.py`` (still the only order path).
"""
