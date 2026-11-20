"""Vendor-neutral LLM access.

Agents depend on the ``LLMClient`` protocol, never on a vendor SDK. Pick a model
with a spec string ``<provider>:<model>``:

    anthropic:claude-sonnet-5-5
    ollama:llama3.2:3b          (model names may themselves contain ':')
    fake                        (deterministic, for tests)

Resolution order for an agent role (e.g. "research"):
    1. explicit ``spec`` argument (CLI ``--llm``)
    2. env ``BIOTRADEAI_LLM_<ROLE>``  (e.g. BIOTRADEAI_LLM_RESEARCH)
    3. env ``BIOTRADEAI_LLM``
    4. DEFAULT_SPEC

Every prompt is passed through ``redact`` before leaving the process.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import httpx

from biotradeai.redact import redact_text

DEFAULT_SPEC = "anthropic:claude-sonnet-5-5"
PROVIDERS = ("anthropic", "ollama", "fake")

Role = Literal["system", "user", "assistant"]


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class LLMClient(Protocol):
    provider: str
    model: str

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse: ...


def _clean(
    messages: Sequence[LLMMessage], system: str | None
) -> tuple[list[LLMMessage], str | None]:
    cleaned = [LLMMessage(m.role, redact_text(m.content)) for m in messages]
    return cleaned, (redact_text(system) if system else None)


# --------------------------------------------------------------------------- fake


class FakeLLMClient:
    """Deterministic client for tests. Returns queued replies or a function's output."""

    provider = "fake"

    def __init__(
        self,
        replies: Sequence[str] | Callable[[Sequence[LLMMessage]], str] = (),
        model: str = "fake",
    ) -> None:
        self.model = model
        self._fn = replies if callable(replies) else None
        self._queue = [] if callable(replies) else list(replies)
        self.calls: list[list[LLMMessage]] = []

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        msgs, _ = _clean(messages, system)
        self.calls.append(msgs)
        if self._fn is not None:
            text = self._fn(msgs)
        elif self._queue:
            text = self._queue.pop(0)
        else:
            raise LLMError("FakeLLMClient has no more queued replies")
        return LLMResponse(text=text, provider=self.provider, model=self.model)


# --------------------------------------------------------------------------- ollama


class OllamaClient:
    """Local models via Ollama's HTTP API (``POST /api/chat``)."""

    provider = "ollama"

    def __init__(
        self,
        model: str,
        host: str | None = None,
        timeout_s: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self._http = httpx.Client(base_url=self.host, timeout=timeout_s, transport=transport)

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        msgs, sys_txt = _clean(messages, system)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": ([{"role": "system", "content": sys_txt}] if sys_txt else [])
            + [{"role": m.role, "content": m.content} for m in msgs],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_schema is not None:
            payload["format"] = json_schema  # Ollama structured outputs
        try:
            resp = self._http.post("/api/chat", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama request failed: {exc}") from exc
        data = resp.json()
        usage = {
            k: int(data[k]) for k in ("prompt_eval_count", "eval_count") if k in data
        }
        return LLMResponse(
            text=data["message"]["content"], provider=self.provider, model=self.model, usage=usage
        )


# --------------------------------------------------------------------------- anthropic


class AnthropicClient:
    """Claude via the official ``anthropic`` SDK (optional dependency)."""

    provider = "anthropic"

    def __init__(self, model: str, client: Any | None = None) -> None:
        self.model = model
        if client is None:
            try:
                import anthropic  # type: ignore[import-not-found, unused-ignore]
            except ImportError as exc:
                raise LLMError(
                    "anthropic package not installed: pip install -e .[anthropic]"
                ) from exc
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise LLMError("ANTHROPIC_API_KEY is not set")
            client = anthropic.Anthropic()
        self._client = client

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        msgs, sys_txt = _clean(messages, system)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": m.role, "content": m.content} for m in msgs if m.role != "system"
            ],
        }
        if sys_txt:
            kwargs["system"] = sys_txt
        try:
            resp = self._client.messages.create(**kwargs)
        except Exception as exc:
            raise LLMError(f"anthropic request failed: {exc}") from exc
        text = "".join(getattr(b, "text", "") for b in resp.content)
        usage = {}
        if getattr(resp, "usage", None) is not None:
            usage = {
                "input_tokens": int(resp.usage.input_tokens),
                "output_tokens": int(resp.usage.output_tokens),
            }
        return LLMResponse(text=text, provider=self.provider, model=self.model, usage=usage)


# --------------------------------------------------------------------------- factory


def parse_spec(spec: str) -> tuple[str, str]:
    spec = spec.strip()
    provider, _, model = spec.partition(":")
    provider = provider.lower()
    if provider not in PROVIDERS:
        raise LLMError(f"unknown LLM provider {provider!r}; expected one of {PROVIDERS}")
    if provider != "fake" and not model:
        raise LLMError(f"LLM spec {spec!r} is missing a model, e.g. 'ollama:llama3.2:3b'")
    return provider, model or "fake"


def resolve_spec(role: str | None = None, spec: str | None = None) -> str:
    if spec:
        return spec
    if role:
        per_role = os.environ.get(f"BIOTRADEAI_LLM_{role.upper()}")
        if per_role:
            return per_role
    return os.environ.get("BIOTRADEAI_LLM") or DEFAULT_SPEC


def make_llm_client(role: str | None = None, spec: str | None = None) -> LLMClient:
    provider, model = parse_spec(resolve_spec(role, spec))
    if provider == "ollama":
        return OllamaClient(model)
    if provider == "anthropic":
        return AnthropicClient(model)
    return FakeLLMClient(model=model)
