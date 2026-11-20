from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from biotradeai.config import ConfigError, Mode, load_config, parse_config


def test_repo_config_loads() -> None:
    cfg = load_config()
    assert cfg.mode is Mode.PAPER
    assert cfg.orders.allowed_types == ["limit"]
    assert cfg.fatal_runway_months < cfg.min_runway_months


def test_env_override_path(tmp_path: Path, raw_config: dict[str, Any],
                           monkeypatch: pytest.MonkeyPatch) -> None:
    raw_config["min_ev"] = 0.33
    p = tmp_path / "g.yaml"
    p.write_text(yaml.safe_dump(raw_config), encoding="utf-8")
    monkeypatch.setenv("BIOTRADEAI_CONFIG", str(p))
    assert load_config().min_ev == 0.33


def test_missing_file() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(Path("does/not/exist.yaml"))


def test_non_mapping_rejected() -> None:
    with pytest.raises(ConfigError):
        parse_config(["not", "a", "mapping"])


@pytest.mark.parametrize("key", ["min_ev", "mode", "orders", "tot", "max_api_errors"])
def test_missing_top_level_key_rejected(raw_config: dict[str, Any], key: str) -> None:
    del raw_config[key]
    with pytest.raises(ConfigError):
        parse_config(raw_config)


def test_missing_nested_key_rejected(raw_config: dict[str, Any]) -> None:
    del raw_config["tot"]["residual_scenario_return"]
    with pytest.raises(ConfigError):
        parse_config(raw_config)


def test_unknown_key_rejected(raw_config: dict[str, Any]) -> None:
    raw_config["min_evv"] = 0.2  # typo must not be silently ignored
    with pytest.raises(ConfigError):
        parse_config(raw_config)


@pytest.mark.parametrize("bad", [["market"], ["limit", "market"], ["stop"], []])
def test_only_limit_orders_allowed(raw_config: dict[str, Any], bad: list[str]) -> None:
    raw_config["orders"]["allowed_types"] = bad
    with pytest.raises(ConfigError):
        parse_config(raw_config)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("mode", "yolo"),
        ("max_position_pct", 0.0),
        ("max_position_pct", 0.0500001),  # just above the 5% hard ceiling
        ("max_position_pct", -0.01),
        ("min_confidence", 1.01),
        ("daily_drawdown_limit", 1.0),
        ("max_api_errors", 0),
        ("max_concurrent_positions", 0),
        ("min_ev", -0.1),
        ("ttl_hours", 0),
    ],
)
def test_bad_values_rejected(raw_config: dict[str, Any], key: str, value: object) -> None:
    raw_config[key] = value
    with pytest.raises(ConfigError):
        parse_config(raw_config)


def test_position_ceiling_boundary(raw_config: dict[str, Any]) -> None:
    raw_config["max_position_pct"] = 0.05  # exactly at the ceiling is fine
    assert parse_config(raw_config).max_position_pct == 0.05
    raw_config["max_position_pct"] = 0.0499
    raw_config["live_gate"]["pilot_position_pct"] = 0.01
    assert parse_config(raw_config).max_position_pct == 0.0499


@pytest.mark.parametrize(("fatal", "ok"), [(5.99, True), (6.0, False), (6.01, False)])
def test_fatal_runway_must_be_below_veto(raw_config: dict[str, Any], fatal: float,
                                         ok: bool) -> None:
    raw_config["min_runway_months"] = 6
    raw_config["fatal_runway_months"] = fatal
    if ok:
        parse_config(raw_config)
    else:
        with pytest.raises(ConfigError, match="fatal_runway_months"):
            parse_config(raw_config)


@pytest.mark.parametrize(
    ("value", "ok"), [(-1.0, True), (0.0, True), (0.01, False), (-1.01, False)]
)
def test_residual_return_bounds(raw_config: dict[str, Any], value: float, ok: bool) -> None:
    raw_config["tot"]["residual_scenario_return"] = value
    if ok:
        parse_config(raw_config)
    else:
        with pytest.raises(ConfigError):
            parse_config(raw_config)


def test_pilot_cannot_exceed_main_caps(raw_config: dict[str, Any]) -> None:
    raw_config["max_position_pct"] = 0.02
    raw_config["live_gate"]["pilot_position_pct"] = 0.03
    with pytest.raises(ConfigError, match="pilot_position_pct"):
        parse_config(raw_config)
    raw_config["live_gate"]["pilot_position_pct"] = 0.01
    raw_config["live_gate"]["pilot_max_positions"] = 4
    with pytest.raises(ConfigError, match="pilot_max_positions"):
        parse_config(raw_config)


def test_config_is_immutable() -> None:
    cfg = load_config()
    with pytest.raises(Exception):  # noqa: B017  pydantic frozen model
        cfg.min_ev = 0.0  # type: ignore[misc]
