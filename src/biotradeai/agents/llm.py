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

Two ways to call a model:

* ``complete(messages)`` -> text. One question, one answer (Generator, Critic).
* ``converse(turns, tools=...)`` -> text and/or tool calls. The model may ask the
  caller to run a tool; the caller runs it and sends the result back as a ``tool`` turn.
  This is the loop behind the research agent (``agents/research_agent.py``).
  Each provider's native tool-use format is translated to and from one neutral shape
  (``ToolSpec``, ``ToolCall``, ``ChatTurn``), so agents never see vendor formats.
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


# --------------------------------------------------------------------------- tool use

TurnRole = Literal["user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model may call. ``parameters`` is a JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass(frozen=True)
class ChatTurn:
    """One turn of a tool-using conversation, in a provider-neutral shape.

    user:      ``content`` is the message.
    assistant: ``content`` is any text; ``tool_calls`` are the tools it asked for.
    tool:      ``content`` is a tool's result for ``tool_call_id`` (named ``name``).
    """

    role: TurnRole
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content,
                "tool_calls": [c.as_dict() for c in self.tool_calls],
                "tool_call_id": self.tool_call_id, "name": self.name}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChatTurn:
        return cls(role=d["role"], content=str(d.get("content") or ""),
                   tool_calls=tuple(ToolCall(str(c["id"]), str(c["name"]),
                                             dict(c.get("arguments") or {}))
                                    for c in d.get("tool_calls") or []),
                   tool_call_id=d.get("tool_call_id"), name=d.get("name"))


@dataclass(frozen=True)
class ToolResponse:
    text: str
    tool_calls: tuple[ToolCall, ...]
    provider: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class ToolLLMClient(Protocol):
    provider: str
    model: str

    def converse(
        self,
        turns: Sequence[ChatTurn],
        *,
        tools: Sequence[ToolSpec],
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ToolResponse: ...


def _clean_turns(turns: Sequence[ChatTurn]) -> list[ChatTurn]:
    return [ChatTurn(t.role, redact_text(t.content),
                     tuple(ToolCall(c.id, c.name, _redact_args(c.arguments))
                           for c in t.tool_calls), t.tool_call_id, t.name) for t in turns]


def _redact_args(args: dict[str, Any]) -> dict[str, Any]:
    return {k: redact_text(v) if isinstance(v, str) else v for k, v in args.items()}


def _as_args(raw: Any) -> dict[str, Any]:
    """Tool arguments arrive as a dict or a JSON string, depending on the provider."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        import json

        try:
            v = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return v if isinstance(v, dict) else {}
    return {}


# --------------------------------------------------------------------------- fake


FakeToolReply = ToolResponse | dict[str, Any]


class FakeLLMClient:
    """Deterministic client for tests. Returns queued replies or a function's output.

    ``tool_replies`` scripts ``converse``: each item is a ``ToolResponse`` or a dict
    ``{"text": "...", "tool_calls": [{"name": "...", "arguments": {...}}]}``. With no
    scripted tool replies, ``converse`` answers with plain text and no tool calls,
    which a tool loop treats as "done".
    """

    provider = "fake"

    def __init__(
        self,
        replies: Sequence[str] | Callable[[Sequence[LLMMessage]], str] = (),
        model: str = "fake",
        *,
        tool_replies: Sequence[FakeToolReply]
        | Callable[[Sequence[ChatTurn]], FakeToolReply] = (),
    ) -> None:
        self.model = model
        self._fn = replies if callable(replies) else None
        self._queue = [] if callable(replies) else list(replies)
        self.calls: list[list[LLMMessage]] = []
        self._tool_fn = tool_replies if callable(tool_replies) else None
        self._tool_queue: list[FakeToolReply] = (
            [] if callable(tool_replies) else list(tool_replies))
        self.tool_calls_seen: list[list[ChatTurn]] = []

    def converse(
        self,
        turns: Sequence[ChatTurn],
        *,
        tools: Sequence[ToolSpec],
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ToolResponse:
        cleaned = _clean_turns(turns)
        self.tool_calls_seen.append(cleaned)
        if self._tool_fn is not None:
            r: FakeToolReply = self._tool_fn(cleaned)
        elif self._tool_queue:
            r = self._tool_queue.pop(0)
        else:
            r = {"text": "no further searches"}
        if isinstance(r, ToolResponse):
            return r
        calls = tuple(ToolCall(str(c.get("id") or f"call_{len(self.tool_calls_seen)}_{i}"),
                               str(c["name"]), dict(c.get("arguments") or {}))
                      for i, c in enumerate(r.get("tool_calls") or []))
        return ToolResponse(str(r.get("text") or ""), calls, self.provider, self.model)

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

    def converse(
        self,
        turns: Sequence[ChatTurn],
        *,
        tools: Sequence[ToolSpec],
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ToolResponse:
        """Ollama's native tool calling (``tools`` on ``/api/chat``)."""
        msgs: list[dict[str, Any]] = (
            [{"role": "system", "content": redact_text(system)}] if system else [])
        for t in _clean_turns(turns):
            if t.role == "assistant":
                m: dict[str, Any] = {"role": "assistant", "content": t.content}
                if t.tool_calls:
                    m["tool_calls"] = [{"function": {"name": c.name, "arguments": c.arguments}}
                                       for c in t.tool_calls]
                msgs.append(m)
            elif t.role == "tool":
                msgs.append({"role": "tool", "content": t.content, "tool_name": t.name or ""})
            else:
                msgs.append({"role": "user", "content": t.content})
        payload: dict[str, Any] = {
            "model": self.model, "messages": msgs, "stream": False,
            "tools": [{"type": "function", "function": {
                "name": s.name, "description": s.description, "parameters": s.parameters}}
                for s in tools],
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        try:
            resp = self._http.post("/api/chat", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama request failed: {exc}") from exc
        data = resp.json()
        msg = data.get("message") or {}
        calls = tuple(
            ToolCall(f"call_{i}", str((c.get("function") or {}).get("name", "")),
                     _as_args((c.get("function") or {}).get("arguments")))
            for i, c in enumerate(msg.get("tool_calls") or []))
        usage = {k: int(data[k]) for k in ("prompt_eval_count", "eval_count") if k in data}
        return ToolResponse(str(msg.get("content") or ""), calls, self.provider, self.model,
                            usage)


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

    def converse(
        self,
        turns: Sequence[ChatTurn],
        *,
        tools: Sequence[ToolSpec],
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ToolResponse:
        """Claude's native tool use: ``tool_use`` blocks out, ``tool_result`` blocks in."""
        msgs: list[dict[str, Any]] = []
        for t in _clean_turns(turns):
            if t.role == "assistant":
                blocks: list[dict[str, Any]] = (
                    [{"type": "text", "text": t.content}] if t.content else [])
                blocks += [{"type": "tool_use", "id": c.id, "name": c.name,
                            "input": c.arguments} for c in t.tool_calls]
                msgs.append({"role": "assistant", "content": blocks or t.content})
            elif t.role == "tool":
                block = {"type": "tool_result", "tool_use_id": t.tool_call_id or "",
                         "content": t.content}
                # Consecutive tool results go back together in one user message.
                if msgs and msgs[-1]["role"] == "user" and isinstance(msgs[-1]["content"], list):
                    msgs[-1]["content"].append(block)
                else:
                    msgs.append({"role": "user", "content": [block]})
            else:
                msgs.append({"role": "user", "content": t.content})
        kwargs: dict[str, Any] = {
            "model": self.model, "max_tokens": max_tokens, "temperature": temperature,
            "messages": msgs,
            "tools": [{"name": s.name, "description": s.description,
                       "input_schema": s.parameters} for s in tools],
        }
        if system:
            kwargs["system"] = redact_text(system)
        try:
            resp = self._client.messages.create(**kwargs)
        except Exception as exc:
            raise LLMError(f"anthropic request failed: {exc}") from exc
        text = "".join(getattr(b, "text", "") or "" for b in resp.content
                       if getattr(b, "type", "") == "text")
        calls = tuple(ToolCall(str(b.id), str(b.name), _as_args(b.input))
                      for b in resp.content if getattr(b, "type", "") == "tool_use")
        usage = {}
        if getattr(resp, "usage", None) is not None:
            usage = {"input_tokens": int(resp.usage.input_tokens),
                     "output_tokens": int(resp.usage.output_tokens)}
        return ToolResponse(text, calls, self.provider, self.model, usage)


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


def make_tool_llm_client(role: str | None = None, spec: str | None = None) -> ToolLLMClient:
    """Same resolution as ``make_llm_client``; every built-in client supports tools."""
    client = make_llm_client(role, spec)
    if not isinstance(client, ToolLLMClient):  # pragma: no cover - all built-ins do
        raise LLMError(f"{client.provider} client does not support tool calling")
    return client
