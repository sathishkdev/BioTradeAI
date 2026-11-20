"""Structural guarantees enforced as tests (MUST #2: one order path)."""

from __future__ import annotations

import ast
from pathlib import Path

from biotradeai.paths import REPO_ROOT

SRC = REPO_ROOT / "src" / "biotradeai"
ALLOWED_CALLERS = {SRC / "brokers" / "execution.py"}
# Broker implementations define place_order; they may not call another broker's.
BROKER_IMPLS = {SRC / "brokers" / n for n in
                ("base.py", "fake_broker.py", "alpaca_broker.py", "robinhood_broker.py")}


def _calls(path: Path, attr: str) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == attr]


def test_only_execution_calls_place_order() -> None:
    offenders = {
        str(p.relative_to(REPO_ROOT)): lines
        for p in SRC.rglob("*.py")
        if p not in ALLOWED_CALLERS and (lines := _calls(p, "place_order"))
    }
    assert offenders == {}, f"place_order called outside execution.py: {offenders}"


def test_agents_and_graph_do_not_import_brokers() -> None:
    for pkg in ("agents", "graph"):
        for p in (SRC / pkg).rglob("*.py") if (SRC / pkg).exists() else []:
            tree = ast.parse(p.read_text(encoding="utf-8"))
            for n in ast.walk(tree):
                mod = n.module if isinstance(n, ast.ImportFrom) else None
                names = [a.name for a in n.names] if isinstance(n, ast.Import) else []
                for m in [mod or "", *names]:
                    assert "biotradeai.brokers" not in m, f"{p} imports {m}"


def test_execution_file_exists() -> None:
    assert set(SRC.rglob("*.py")) >= ALLOWED_CALLERS
    assert set(SRC.rglob("*.py")) >= BROKER_IMPLS


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module)
            out.update(f"{n.module}.{a.name}" for a in n.names)
        elif isinstance(n, ast.Import):
            out.update(a.name for a in n.names)
    return out


def test_workers_cannot_call_each_other() -> None:
    """Research and Risk agents only talk through the Supervisor's state."""
    research = _imports(SRC / "agents" / "research.py")
    risk = _imports(SRC / "agents" / "risk.py")
    assert not any("agents.risk" in m for m in research)
    assert not any("agents.research" in m or "agents.llm" in m for m in risk)
    for mods in (research, risk):
        assert not any(m.startswith("biotradeai.graph") for m in mods)


def test_risk_agent_has_no_llm() -> None:
    src = (SRC / "agents" / "risk.py").read_text(encoding="utf-8")
    assert "LLMClient" not in src and "complete(" not in src
