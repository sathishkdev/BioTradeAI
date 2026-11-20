from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from biotradeai.brokers.approval import ApprovalAuthority, ApprovalError, load_secret
from tests.unit.conftest import SECRET, Clock, make_order


def test_valid_token_verifies(authority: ApprovalAuthority) -> None:
    o = make_order()
    authority.verify(authority.issue(o), o)


def test_forged_signature_rejected(authority: ApprovalAuthority) -> None:
    o = make_order()
    t = authority.issue(o)
    forged = t.model_copy(update={"signature": "a" * 64})
    with pytest.raises(ApprovalError, match="signature"):
        authority.verify(forged, o)


def test_token_from_other_secret_rejected(authority: ApprovalAuthority, clock: Clock) -> None:
    other = ApprovalAuthority(b"y" * 32, ttl=timedelta(minutes=15), clock=clock)
    o = make_order()
    with pytest.raises(ApprovalError, match="signature"):
        authority.verify(other.issue(o), o)


@pytest.mark.parametrize("field,value", [
    ("qty", 101), ("limit_price", Decimal("10.01")), ("ticker", "OTHR"),
])
def test_tampered_fields_break_signature(authority: ApprovalAuthority, field: str,
                                         value: object) -> None:
    o = make_order()
    t = authority.issue(o).model_copy(update={field: value})
    with pytest.raises(ApprovalError, match="signature"):
        authority.verify(t, o)


def test_token_for_different_order_rejected(authority: ApprovalAuthority) -> None:
    t = authority.issue(make_order(qty=100))
    with pytest.raises(ApprovalError, match="does not match order: qty"):
        authority.verify(t, make_order(qty=1000))


def test_expiry_boundary(authority: ApprovalAuthority, clock: Clock) -> None:
    o = make_order()
    t = authority.issue(o)
    clock.advance(minutes=14, seconds=59)
    authority.verify(t, o)
    clock.advance(seconds=1)  # exactly at expiry -> expired
    with pytest.raises(ApprovalError, match="expired"):
        authority.verify(t, o)


def test_nonces_are_unique(authority: ApprovalAuthority) -> None:
    o = make_order()
    assert len({authority.issue(o).nonce for _ in range(50)}) == 50


def test_secret_must_be_long(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APPROVAL_HMAC_SECRET", "short")
    with pytest.raises(ApprovalError):
        load_secret()
    assert load_secret(SECRET.decode()) == SECRET
