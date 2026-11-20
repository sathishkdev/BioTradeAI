"""Scrub credentials and account identifiers before anything is logged or sent to an LLM.

Two layers:
1. Key-based: any mapping key that looks sensitive has its value replaced.
2. Value-based: the literal values of known secret environment variables are
   replaced wherever they appear inside strings.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from typing import Any

REDACTED = "[REDACTED]"

_SENSITIVE_KEY = re.compile(
    r"(api[_-]?key|secret|password|passwd|token|authorization|auth|cookie|"
    r"account[_-]?(number|no|id)|acct)",
    re.IGNORECASE,
)

SECRET_ENV_VARS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "ALPACA_API_KEY",
    "ALPACA_API_SECRET",
    "APPROVAL_HMAC_SECRET",
)

# Values shorter than this are not scrubbed by substring (would cause false hits).
_MIN_SECRET_LEN = 6


def _known_secrets(extra: Iterable[str] = ()) -> list[str]:
    vals = [os.environ.get(v, "") for v in SECRET_ENV_VARS]
    vals.extend(extra)
    return sorted({v for v in vals if len(v) >= _MIN_SECRET_LEN}, key=len, reverse=True)


def redact_text(text: str, extra_secrets: Iterable[str] = ()) -> str:
    for secret in _known_secrets(extra_secrets):
        text = text.replace(secret, REDACTED)
    return text


def redact(obj: Any, extra_secrets: Iterable[str] = ()) -> Any:
    """Return a redacted deep copy of ``obj`` (dicts, lists, tuples, strings)."""
    secrets = list(extra_secrets)
    if isinstance(obj, dict):
        out: dict[Any, Any] = {}
        for k, v in obj.items():
            if isinstance(k, str) and _SENSITIVE_KEY.search(k):
                out[k] = REDACTED
            else:
                out[k] = redact(v, secrets)
        return out
    if isinstance(obj, list | tuple):
        return [redact(v, secrets) for v in obj]
    if isinstance(obj, str):
        return redact_text(obj, secrets)
    return obj
