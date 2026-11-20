"""HMAC-SHA256 human-approval tokens (paper/shadow modes).

A token binds exactly one order: {ticker, side, qty, limit_price, expiry, nonce}.
``execution.py`` verifies the signature, that the token matches the order being placed,
that it has not expired, and (via the SQLite nonce table) that it is used only once.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from biotradeai.schemas import ApprovalToken, StagedOrder

MIN_SECRET_BYTES = 32
_PLACEHOLDER_SIG = "0" * 64


class ApprovalError(RuntimeError):
    pass


def load_secret(value: str | None = None) -> bytes:
    raw = value if value is not None else os.environ.get("APPROVAL_HMAC_SECRET", "")
    secret = raw.encode("utf-8")
    if len(secret) < MIN_SECRET_BYTES:
        raise ApprovalError(
            f"APPROVAL_HMAC_SECRET must be at least {MIN_SECRET_BYTES} characters. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    return secret


def _sign(secret: bytes, token: ApprovalToken) -> str:
    return hmac.new(secret, token.signing_payload(), hashlib.sha256).hexdigest()


class ApprovalAuthority:
    def __init__(self, secret: bytes, *, ttl: timedelta,
                 clock: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None:
        if len(secret) < MIN_SECRET_BYTES:
            raise ApprovalError("approval secret too short")
        self._secret = secret
        self._ttl = ttl
        self._clock = clock

    def issue(self, order: StagedOrder) -> ApprovalToken:
        """Called only from the human approval prompt, after the human types 'approve'."""
        if order.limit_price is None:
            raise ApprovalError("only limit orders can be approved")
        unsigned = ApprovalToken(
            ticker=order.ticker, side=order.side, qty=order.qty,
            limit_price=order.limit_price, expiry=self._clock() + self._ttl,
            nonce=secrets.token_hex(16), signature=_PLACEHOLDER_SIG,
        )
        return unsigned.model_copy(update={"signature": _sign(self._secret, unsigned)})

    def verify(self, token: ApprovalToken, order: StagedOrder) -> None:
        """Raise ApprovalError unless the token is authentic, unexpired and matches order."""
        expected = _sign(self._secret, token)
        if not hmac.compare_digest(expected, token.signature):
            raise ApprovalError("approval token signature is invalid (forged or altered)")
        if self._clock() >= token.expiry:
            raise ApprovalError(f"approval token expired at {token.expiry.isoformat()}")
        mismatches = [
            name for name, a, b in (
                ("ticker", token.ticker, order.ticker),
                ("side", token.side, order.side),
                ("qty", token.qty, order.qty),
                ("limit_price", token.limit_price, order.limit_price),
            ) if a != b
        ]
        if mismatches:
            raise ApprovalError(f"approval token does not match order: {', '.join(mismatches)}")
