"""Settings the operator can change: guardrail values, base-rate review, secrets.

Guardrails (``config/guardrails.yaml``)
    Each scalar value can be edited. The new file is validated with the same strict
    model the agents load (hard ceilings such as the 5% position cap live in code and
    cannot be raised), then only that value is replaced in place, so comments and
    layout survive. Every change is audit-logged with the old and new value.
    ``mode`` is not editable here: switching to shadow/live has its own gates.

Secrets (``.env``)
    Never shown. The UI sees only "set / not set" and can replace a value
    (write-only). Values are never logged.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from biotradeai.agents.base_rates import base_rates_path, load_base_rates
from biotradeai.config import ConfigError, parse_config
from biotradeai.paths import REPO_ROOT, config_path
from biotradeai.store.sqlite_store import SQLiteStore

READ_ONLY = {"mode": "Changed only through the shadow/live gates, not from here.",
             "orders.allowed_types": "Limit orders only, by design.",
             "rag.tier_weights": "Edit in the file if you must; affects retrieval tests."}

GROUPS = {
    "": "Risk limits", "runway": "Cash runway", "rag": "Retrieval (RAG)",
    "orders": "Orders and approvals", "tot": "Research (Tree of Thoughts)",
    "agent": "Research agent (tool loop)",
    "sell": "Exit rules", "live_gate": "Live gate",
}

HELP = {
    "min_ev": "Minimum expected value to stage a buy.",
    "min_runway_months": "Risk Agent vetoes below this cash runway.",
    "fatal_runway_months": "Research prunes immediately below this runway.",
    "max_position_pct": "Max share of equity per ticker (hard ceiling 5%).",
    "max_concurrent_positions": "Max open positions at once.",
    "min_confidence": "Below this evidence confidence the run is INCONCLUSIVE.",
    "max_spread_pct": "Max bid/ask spread at trade time.",
    "ttl_hours": "Data older than this is stale.",
    "daily_drawdown_limit": "Kill-switch trips at this daily loss.",
    "max_api_errors": "Kill-switch trips after this many broker errors in a row.",
    "orders.approval_ttl_minutes": "How long an approval token stays valid.",
    "orders.pending_approval_hours": "How long a staged order waits for you.",
    "orders.max_price_drift_pct": "Approval is blocked if the price moved more than this.",
    "tot.residual_scenario_return": "Return assumed for pruned branches (placeholder).",
    "tot.max_prob_adjustment": "Largest evidence-based nudge to a base rate.",
    "agent.enabled": "Let the research agent search and fetch evidence before the tree.",
    "agent.max_tool_calls": "Most tool calls in one search round.",
    "agent.max_turns": "Most model calls in one search round.",
    "agent.max_revisions": "Rounds where Critic rejections send the agent back to search.",
    "agent.max_evidence": "Most evidence chunks one decision may rest on.",
    "agent.max_tokens_per_turn": "Reply length limit for each agent turn.",
    "sell.profit_target_pct": "Run-up that triggers taking profit.",
    "sell.trailing_stop_pct": "Trailing stop after a positive readout (placeholder).",
    "sell.completion_delay_months": "Exit if the trial slips more than this.",
    "live_gate.min_paper_days": "Clean paper days required before live.",
}


class SettingsError(ValueError):
    pass


@dataclass(frozen=True)
class Field:
    path: str
    group: str
    value: Any
    kind: str  # bool | int | float | str | other
    editable: bool
    note: str
    comment: str


def _kind(v: Any) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    return "other"


def _comments(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    stack: list[tuple[int, str]] = []
    for line in text.splitlines():
        m = re.match(r"^(\s*)([A-Za-z_][\w]*):(.*)$", line)
        if not m:
            continue
        indent, key, rest = len(m.group(1)), m.group(2), m.group(3)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path = ".".join([k for _, k in stack] + [key])
        if "#" in rest:
            out[path] = rest.split("#", 1)[1].strip()
        if rest.split("#", 1)[0].strip() == "":
            stack.append((indent, key))
    return out


def guardrail_fields(path: Path | None = None) -> list[Field]:
    p = path or config_path()
    text = p.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    notes = _comments(text)
    out: list[Field] = []

    def walk(d: dict[str, Any], prefix: str, group: str) -> None:
        for k, v in d.items():
            path_ = f"{prefix}{k}"
            if isinstance(v, dict) and path_ not in READ_ONLY:
                walk(v, f"{path_}.", str(k))
                continue
            kind = _kind(v)
            ro = path_ in READ_ONLY or kind in ("other", "str")
            out.append(Field(path_, GROUPS.get(group, group), v, kind, not ro,
                             READ_ONLY.get(path_, HELP.get(path_, "")), notes.get(path_, "")))

    walk(raw, "", "")
    return out


def _coerce(kind: str, raw: str) -> Any:
    s = raw.strip()
    if kind == "bool":
        if s.lower() in ("true", "on", "1", "yes"):
            return True
        if s.lower() in ("false", "off", "0", "no", ""):
            return False
        raise SettingsError("true or false")
    try:
        return int(s) if kind == "int" else float(s)
    except ValueError:
        raise SettingsError("a number is required") from None


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return repr(v) if isinstance(v, float) else str(v)


def set_scalar_text(text: str, path: str, value: Any) -> str:
    """Replace one scalar in YAML text, keeping indentation and the trailing comment."""
    keys = path.split(".")
    lines = text.splitlines(keepends=True)
    stack: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = re.match(r"^(\s*)([A-Za-z_][\w]*):(\s*)([^#\n]*?)(\s*)(#.*)?(\r?\n)?$", line)
        if not m:
            continue
        indent, key = len(m.group(1)), m.group(2)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        here = [k for _, k in stack] + [key]
        val = m.group(4)
        if here == keys and val.strip():
            gap = m.group(3) or " "
            comment = m.group(6) or ""
            pad = m.group(5) or ""
            newval = _fmt(value)
            if comment:  # keep the comment column where possible
                width = len(val) + len(pad)
                pad = " " * max(1, width - len(newval))
            lines[i] = (f"{m.group(1)}{key}:{gap}{newval}{pad}{comment}"
                        f"{m.group(7) or ''}")
            return "".join(lines)
        if not val.strip():
            stack.append((indent, key))
    raise SettingsError(f"setting {path!r} not found")


def _atomic_write(p: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, p)


def update_guardrail(path_: str, raw_value: str, *, by: str, store: SQLiteStore,
                     file: Path | None = None) -> Field:
    p = file or config_path()
    fields = {f.path: f for f in guardrail_fields(p)}
    f = fields.get(path_)
    if f is None:
        raise SettingsError(f"unknown setting {path_!r}")
    if not f.editable:
        raise SettingsError(f.note or "this setting is read-only here")
    value = _coerce(f.kind, raw_value)
    text = p.read_text(encoding="utf-8")
    new_text = set_scalar_text(text, path_, value)
    try:
        parse_config(yaml.safe_load(new_text))
    except ConfigError as exc:
        msg = str(exc).split("\n", 2)
        raise SettingsError(msg[-1].strip()[:300]) from None
    _atomic_write(p, new_text)
    store.append_audit("config.changed", {"setting": path_, "before": f.value, "after": value,
                                          "by": by})
    return {x.path: x for x in guardrail_fields(p)}[path_]


# ---------------------------------------------------------------------- base rates


def base_rates_summary() -> dict[str, Any]:
    r = load_base_rates()
    events = []
    for name, ev in r.events.items():
        c = ev.clinical
        events.append({"name": name, "description": ev.description,
                       "success": c["success"].p, "partial": c["partial"].p,
                       "failure": c["failure"].p,
                       "returns": {k: o.return_ for k, o in c.items()}})
    return {"reviewed": r.reviewed, "events": events,
            "sources": [s.citation for s in r.sources]}


def set_base_rates_reviewed(reviewed: bool, *, by: str, store: SQLiteStore,
                            file: Path | None = None) -> None:
    p = file or base_rates_path()
    text = p.read_text(encoding="utf-8")
    new_text = re.sub(r"(?m)^reviewed:\s*(true|false)\b",
                      f"reviewed: {'true' if reviewed else 'false'}", text, count=1)
    if new_text == text and f"reviewed: {'true' if reviewed else 'false'}" not in text:
        raise SettingsError("could not find 'reviewed:' in base_rates.yaml")
    _atomic_write(p, new_text)
    load_base_rates(p)  # still valid
    store.append_audit("base_rates.reviewed", {"reviewed": reviewed, "by": by})


# ---------------------------------------------------------------------- secrets

SECRETS: tuple[tuple[str, str, bool], ...] = (
    # (env var, what it is, required)
    ("ALPACA_API_KEY", "Alpaca paper key", True),
    ("ALPACA_API_SECRET", "Alpaca paper secret", True),
    ("APPROVAL_HMAC_SECRET", "Approval signing secret", True),
    ("SEC_USER_AGENT", "SEC EDGAR contact (User-Agent)", True),
    ("ANTHROPIC_API_KEY", "Anthropic (Claude) API key", False),
    ("NCBI_API_KEY", "NCBI / PubMed API key", False),
    ("NCBI_EMAIL", "NCBI contact email", False),
    ("OPENFDA_API_KEY", "openFDA API key", False),
)
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{1,40}$")


def env_path() -> Path:
    return REPO_ROOT / ".env"


def secrets_status() -> list[dict[str, Any]]:
    out = []
    for var, label, required in SECRETS:
        val = os.environ.get(var, "")
        ok = bool(val)
        warn = ""
        if var == "APPROVAL_HMAC_SECRET" and ok and len(val) < 32:
            ok, warn = False, "too short (needs 32+ characters)"
        out.append({"var": var, "label": label, "required": required, "set": ok,
                    "warning": warn})
    return out


def replace_secret(var: str, value: str, *, by: str, store: SQLiteStore,
                   file: Path | None = None) -> None:
    """Write-only: sets VAR=value in .env (and this process). The value is never logged."""
    if var not in {v for v, _, _ in SECRETS} or not _ENV_NAME.match(var):
        raise SettingsError("unknown secret")
    value = value.strip()
    if not value or "\n" in value or "\r" in value:
        raise SettingsError("enter a single-line value")
    if var == "APPROVAL_HMAC_SECRET" and len(value) < 32:
        raise SettingsError("the approval secret needs at least 32 characters")
    p = file or env_path()
    text = p.read_text(encoding="utf-8") if p.exists() else ""
    line = f"{var}={value}"
    pattern = re.compile(rf"(?m)^\s*#?\s*{re.escape(var)}\s*=.*$")
    new_text = (pattern.sub(lambda _m: line, text, count=1) if pattern.search(text)
                else text + ("" if text.endswith("\n") or not text else "\n") + line + "\n")
    p.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(p, new_text)
    os.environ[var] = value
    store.append_audit("secret.replaced", {"var": var, "by": by})  # never the value
