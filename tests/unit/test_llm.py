from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from biotradeai.agents.llm import (
    DEFAULT_SPEC,
    AnthropicClient,
    FakeLLMClient,
    LLMClient,
    LLMError,
    LLMMessage,
    OllamaClient,
    make_llm_client,
    parse_spec,
    resolve_spec,
)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("ollama:llama3.2:3b", ("ollama", "llama3.2:3b")),
        ("anthropic:claude-sonnet-5-5", ("anthropic", "claude-sonnet-5-5")),
        ("  OLLAMA:qwen2.5:7b ", ("ollama", "qwen2.5:7b")),
        ("fake", ("fake", "fake")),
    ],
)
def test_parse_spec(spec: str, expected: tuple[str, str]) -> None:
    assert parse_spec(spec) == expected


@pytest.mark.parametrize("bad", ["openai:gpt", "ollama", "ollama:", ""])
def test_parse_spec_rejects(bad: str) -> None:
    with pytest.raises(LLMError):
        parse_spec(bad)


def test_resolution_order(monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_spec("research") == DEFAULT_SPEC
    monkeypatch.setenv("BIOTRADEAI_LLM", "ollama:llama3.2:3b")
    assert resolve_spec("research") == "ollama:llama3.2:3b"
    monkeypatch.setenv("BIOTRADEAI_LLM_RESEARCH", "anthropic:claude-sonnet-5-5")
    assert resolve_spec("research") == "anthropic:claude-sonnet-5-5"
    assert resolve_spec("risk") == "ollama:llama3.2:3b"
    assert resolve_spec("research", "fake") == "fake"


def test_factory_builds_selected_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIOTRADEAI_LLM", "ollama:llama3.2:3b")
    c = make_llm_client("research")
    assert isinstance(c, OllamaClient) and c.model == "llama3.2:3b"
    assert isinstance(make_llm_client(spec="fake"), FakeLLMClient)


def test_fake_client_queue_and_protocol() -> None:
    c = FakeLLMClient(["one", "two"])
    assert isinstance(c, LLMClient)
    assert c.complete([LLMMessage("user", "hi")]).text == "one"
    assert c.complete([LLMMessage("user", "hi")]).text == "two"
    with pytest.raises(LLMError):
        c.complete([LLMMessage("user", "hi")])


def test_prompts_are_redacted_before_leaving(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "PKSECRETKEY999")
    c = FakeLLMClient(lambda msgs: msgs[0].content)
    out = c.complete([LLMMessage("user", "my key is PKSECRETKEY999")]).text
    assert "PKSECRETKEY999" not in out


def test_ollama_client_wire_format() -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        assert req.url.path == "/api/chat"
        return httpx.Response(200, json={
            "message": {"role": "assistant", "content": "pong"},
            "prompt_eval_count": 7, "eval_count": 2,
        })

    c = OllamaClient("llama3.2:3b", host="http://ollama.test",
                     transport=httpx.MockTransport(handler))
    r = c.complete([LLMMessage("user", "ping")], system="be terse",
                   json_schema={"type": "object"})
    assert r.text == "pong" and r.usage == {"prompt_eval_count": 7, "eval_count": 2}
    assert seen["model"] == "llama3.2:3b" and seen["stream"] is False
    assert seen["messages"][0] == {"role": "system", "content": "be terse"}
    assert seen["format"] == {"type": "object"}


def test_ollama_http_error_wrapped() -> None:
    c = OllamaClient("m", host="http://ollama.test",
                     transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    with pytest.raises(LLMError):
        c.complete([LLMMessage("user", "x")])


def test_anthropic_client_with_injected_sdk() -> None:
    captured: dict[str, Any] = {}

    def create(**kw: Any) -> Any:
        captured.update(kw)
        return SimpleNamespace(content=[SimpleNamespace(text="pong")],
                               usage=SimpleNamespace(input_tokens=5, output_tokens=1))

    fake_sdk = SimpleNamespace(messages=SimpleNamespace(create=create))
    c = AnthropicClient("claude-sonnet-5-5", client=fake_sdk)
    r = c.complete([LLMMessage("user", "ping")], system="terse", max_tokens=5)
    assert r.text == "pong" and r.usage == {"input_tokens": 5, "output_tokens": 1}
    assert captured["system"] == "terse" and captured["model"] == "claude-sonnet-5-5"


@pytest.mark.integration
def test_live_ollama_roundtrip() -> None:
    c = OllamaClient("llama3.2:3b")
    r = c.complete([LLMMessage("user", "Reply with exactly: pong")], max_tokens=10)
    assert r.text.strip()
