"""Filesystem locations, resolved with pathlib so they work on Windows and Linux."""

from __future__ import annotations

import os
from pathlib import Path

# src/biotradeai/paths.py -> parents[2] is the repository root.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def _env_path(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    if not raw:
        return default
    p = Path(raw)
    return p if p.is_absolute() else REPO_ROOT / p


def config_path() -> Path:
    return _env_path("BIOTRADEAI_CONFIG", REPO_ROOT / "config" / "guardrails.yaml")


def db_path() -> Path:
    return _env_path("BIOTRADEAI_DB", REPO_ROOT / "data" / "biotradeai.db")


def data_dir() -> Path:
    return REPO_ROOT / "data"
