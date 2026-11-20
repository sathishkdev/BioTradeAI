from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from biotradeai.schemas import (
    ApprovalToken,
    CatalystState,
    EvidenceBundle,
    EvidenceStatus,
    OrderSide,
    OrderType,
    Scenario,
    ScenarioSet,
    SourceRecord,
    SourceTier,
    StagedOrder,
    ToTNode,
)

TOL = 0.001
H = hashlib.sha256(b"doc").hexdigest()


def _scen(label: str, p: float, r: float = 0.0) -> Scenario:
    return Scenario(depth=3, label=label, probability=p, expected_return=r)


def _src(ticker: str = "ONCO", entity: str = "ENT-001", **kw: Any) -> SourceRecord:
    base: dict[str, Any] = {
        "tier": SourceTier.SEC,
        "published_at": datetime(2026, 1, 1, tzinfo=UTC),
        "sha256": H,
        "url": "https://www.sec.gov/x",
        "ticker": ticker,
        "entity_id": entity,
    }
    base.update(kw)
    return SourceRecord(**base)


# ------------------------------------------------------------------ probability sum


def test_onco_style_set_must_include_failure_branch() -> None:
    # The PDF's $ONCO example summed surviving branches to 0.90. That must fail.
    with pytest.raises(ValidationError, match="sum to"):
        ScenarioSet(scenarios=[_scen("approval", 0.6, 0.8), _scen("partial", 0.3, 0.1)],
                    tolerance=TOL)
    ok = ScenarioSet(
        scenarios=[_scen("approval", 0.6, 0.8), _scen("partial", 0.3, 0.1),
                   _scen("crl", 0.1, -0.7)],
        tolerance=TOL,
    )
    assert len(ok.scenarios) == 3


@pytest.mark.parametrize(
    ("third", "ok"),
    [
        (0.1, True),               # exact
        (0.1 + TOL * 0.9, True),   # just inside +tolerance
        (0.1 - TOL * 0.9, True),   # just inside -tolerance
        (0.1 + TOL * 1.1, False),  # just outside +tolerance
        (0.1 - TOL * 1.1, False),  # just outside -tolerance
    ],
)
def test_probability_sum_tolerance_boundaries(third: float, ok: bool) -> None:
    scen = [_scen("a", 0.6), _scen("b", 0.3), _scen("c", third)]
    if ok:
        ScenarioSet(scenarios=scen, tolerance=TOL)
    else:
        with pytest.raises(ValidationError):
            ScenarioSet(scenarios=scen, tolerance=TOL)


def test_duplicate_labels_rejected() -> None:
    with pytest.raises(ValidationError, match="unique"):
        ScenarioSet(scenarios=[_scen("a", 0.5), _scen("a", 0.5)], tolerance=TOL)


@pytest.mark.parametrize("p", [-0.01, 1.01])
def test_probability_range(p: float) -> None:
    with pytest.raises(ValidationError):
        _scen("x", p)


def test_return_cannot_lose_more_than_everything() -> None:
    with pytest.raises(ValidationError):
        _scen("x", 0.5, -1.01)


# ------------------------------------------------------------------ sources / evidence


def test_source_requires_tz_aware_and_valid_hash() -> None:
    with pytest.raises(ValidationError):
        _src(published_at=datetime(2026, 1, 1))  # naive datetime
    with pytest.raises(ValidationError):
        _src(sha256="abc")
    with pytest.raises(ValidationError):
        _src(ticker="onco")  # lowercase


def test_evidence_bundle_blocks_semantic_bleed() -> None:
    with pytest.raises(ValidationError, match="semantic bleed"):
        EvidenceBundle(ticker="ONCO", entity_id="ENT-001", sources=[_src(ticker="OTHR")],
                       confidence=0.9, status=EvidenceStatus.OK)
    with pytest.raises(ValidationError, match="semantic bleed"):
        EvidenceBundle(ticker="ONCO", entity_id="ENT-001", sources=[_src(entity="ENT-999")],
                       confidence=0.9, status=EvidenceStatus.OK)


def test_inconclusive_needs_reason() -> None:
    with pytest.raises(ValidationError):
        EvidenceBundle(ticker="ONCO", entity_id="ENT-001", sources=[], confidence=0.4,
                       status=EvidenceStatus.INCONCLUSIVE)


def test_pruned_node_needs_reason() -> None:
    with pytest.raises(ValidationError):
        ToTNode(node_id="n1", parent_id=None, depth=1, label="fail", base_rate=0.3,
                adjustment=0.0, probability=0.3, step_return=-0.5, justification="",
                pruned=True)


# ------------------------------------------------------------------ orders / tokens


def test_limit_order_requires_price() -> None:
    with pytest.raises(ValidationError):
        StagedOrder(ticker="ONCO", side=OrderSide.BUY, qty=10, order_type=OrderType.LIMIT,
                    limit_price=None)


def test_order_qty_and_price_positive() -> None:
    with pytest.raises(ValidationError):
        StagedOrder(ticker="ONCO", side=OrderSide.BUY, qty=0, order_type=OrderType.LIMIT,
                    limit_price=Decimal("1.00"))
    with pytest.raises(ValidationError):
        StagedOrder(ticker="ONCO", side=OrderSide.BUY, qty=1, order_type=OrderType.LIMIT,
                    limit_price=Decimal("0"))


def test_signing_payload_is_canonical() -> None:
    kw: dict[str, Any] = {
        "ticker": "ONCO", "side": OrderSide.BUY, "qty": 10,
        "expiry": datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=5),
        "nonce": "n" * 32, "signature": H,
    }
    a = ApprovalToken(limit_price=Decimal("4.5"), **kw)
    b = ApprovalToken(limit_price=Decimal("4.5000"), **kw)
    assert a.signing_payload() == b.signing_payload()
    c = ApprovalToken(limit_price=Decimal("4.51"), **kw)
    assert a.signing_payload() != c.signing_payload()


def test_catalyst_state_roundtrip_json() -> None:
    s = CatalystState(run_id="r1", ticker="ONCO", entity_id="ENT-001",
                      catalyst_event="PDUFA", catalyst_date=date(2026, 3, 1),
                      as_of=datetime(2026, 1, 1, tzinfo=UTC))
    assert CatalystState.model_validate_json(s.model_dump_json()) == s


def test_contracts_reject_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Scenario(depth=1, label="a", probability=0.5, expected_return=0.1,
                 sneaky_ev=99)  # type: ignore[call-arg]
