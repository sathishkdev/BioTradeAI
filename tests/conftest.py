from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from biotradeai.paths import REPO_ROOT
from biotradeai.store.sqlite_store import SQLiteStore


@pytest.fixture
def raw_config() -> dict[str, Any]:
    """A fresh, mutable copy of the real guardrails.yaml."""
    with (REPO_ROOT / "config" / "guardrails.yaml").open(encoding="utf-8") as fh:
        return copy.deepcopy(yaml.safe_load(fh))


@pytest.fixture
def store(tmp_path: Path) -> SQLiteStore:
    return SQLiteStore(tmp_path / "test.db")


@pytest.fixture(autouse=True)
def _clean_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("BIOTRADEAI_LLM", "BIOTRADEAI_LLM_RESEARCH", "BIOTRADEAI_LLM_RISK",
                "BIOTRADEAI_CONFIG", "BIOTRADEAI_DB"):
        monkeypatch.delenv(var, raising=False)
