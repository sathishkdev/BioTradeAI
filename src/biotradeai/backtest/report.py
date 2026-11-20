"""Write a backtest report (Markdown + JSON) to ``reports/`` (git-ignored)."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from biotradeai.backtest.metrics import (
    BRIER_TARGET,
    GROUNDEDNESS_TARGET,
    LATENCY_TARGET_S,
    Metrics,
)
from biotradeai.paths import REPO_ROOT


def reports_dir() -> Path:
    return REPO_ROOT / "reports"


def _f(x: float | None, fmt: str = ".4f") -> str:
    return "n/a" if x is None else format(x, fmt)


def _mark(ok: bool | None) -> str:
    return "n/a" if ok is None else ("PASS" if ok else "FAIL")


def render_markdown(m: Metrics, meta: dict[str, Any]) -> str:
    c = m.checks()
    lines = [
        f"# Backtest report — {meta['started_at']}",
        "",
        f"- cases: {m.n_cases} ({meta['cases_file']})",
        f"- generator / critic: `{meta['generator']}` / `{meta['critic']}`",
        f"- embedder: `{meta['embedder']}`",
        f"- base rates reviewed by a human: {meta['base_rates_reviewed']}",
        f"- decisions: {', '.join(f'{k} {v}' for k, v in sorted(m.decisions.items()))}",
        "",
        "> " + meta["caveat"],
        "",
        "| Metric | Value | Target | Result |",
        "|---|---|---|---|",
        f"| Brier score (n={m.brier_n}) | {_f(m.brier)} | <= {BRIER_TARGET} | "
        f"{_mark(c['brier'])} |",
        f"| Brier, base rates only | {_f(m.brier_baseline)} | reference | - |",
        f"| Groundedness | {_f(m.groundedness, '.2%')} | >= {GROUNDEDNESS_TARGET:.0%} | "
        f"{_mark(c['groundedness'])} |",
        f"| Guardrail breaches | {len(m.breaches)} | 0 | {_mark(c['guardrail_compliance'])} |",
        f"| Escalation accuracy | {_f(m.escalation_accuracy, '.0%')} | 100% | "
        f"{_mark(c['escalation_accuracy'])} |",
        f"| Latency mean / max | {m.latency_mean_s:.2f} s / {m.latency_max_s:.2f} s | "
        f"<= {LATENCY_TARGET_S:.0f} s | {_mark(c['latency'])} |",
        f"| Point-in-time leaks | {meta['leaks']} | 0 | {_mark(meta['leaks'] == 0)} |",
        "",
        "## Per case",
        "",
        "| Case | Ticker | Decision | Outcome | P(success) | Conf. | Runway mo | EV | s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for p in m.per_case:
        lines.append(
            f"| {p['id']} | {p['ticker']} | {p['decision']} | {p['outcome']} | "
            f"{_f(p['p_success'], '.3f')} | {p['confidence']:.3f} | {p['runway_months']:.1f} | "
            f"{_f(p['risk_ev'], '+.3f')} | {p['latency_s']:.2f} |")
    for title, items in (("Guardrail breaches", m.breaches),
                         ("Ungrounded items", m.ungrounded),
                         ("Escalations without an expected conflict", m.false_escalations)):
        lines += ["", f"## {title}", ""] + ([f"- {x}" for x in items] or ["- none"])
    lines += ["", "## Why cases did not trade", ""]
    for p in m.per_case:
        why = p["inconclusive_reason"] if p["decision"] == "INCONCLUSIVE" else \
            "; ".join(p["veto_reasons"])
        if why:
            lines.append(f"- {p['id']} ({p['decision']}): {why}")
    return "\n".join(lines) + "\n"


def write_report(m: Metrics, meta: dict[str, Any], out_dir: Path | None = None) -> Path:
    d = out_dir or reports_dir()
    d.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    md = d / f"backtest-{stamp}.md"
    md.write_text(render_markdown(m, meta), encoding="utf-8", newline="\n")
    (d / f"backtest-{stamp}.json").write_text(
        json.dumps({"meta": meta, "metrics": asdict(m), "checks": m.checks()}, indent=2,
                   default=str), encoding="utf-8", newline="\n")
    return md
