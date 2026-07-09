"""Inference provider tests using httpx MockTransport (no live server)."""

from __future__ import annotations

import json as jsonlib
from collections.abc import Callable

import httpx
import pytest

from smartrag.config import load_config
from smartrag.inference import (
    ChatMessage,
    HuggingFaceProvider,
    NvidiaProvider,
    OllamaProvider,
    ProviderUnreachableError,
    build_inference_provider,
)
from smartrag.inference._openai_compat import OpenAICompatibleProvider

Handler = Callable[[httpx.Request], httpx.Response]


def _ollama(handler: Handler) -> OllamaProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OllamaProvider(
        host="http://ollama.test", default_model="glm-ocr:latest", client=client, backoff_s=0.0
    )


def _openai(handler: Handler) -> OpenAICompatibleProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAICompatibleProvider(
        base_url="http://nim.test/v1", model="m", client=client, backoff_s=0.0
    )


def test_ollama_generate() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/generate"
        return httpx.Response(200, json={"response": "hello"})

    assert _ollama(handler).generate("hi") == "hello"


def test_ollama_chat() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/chat"
        return httpx.Response(200, json={"message": {"content": "hi there"}})

    assert _ollama(handler).chat([ChatMessage(role="user", content="hi")]) == "hi there"


def test_ollama_vision_base64_encodes_images() -> None:
    captured: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["images"] = jsonlib.loads(req.content)["images"]
        return httpx.Response(200, json={"response": "text in image"})

    out = _ollama(handler).vision("read", [b"\x89PNGrawbytes"])
    assert out == "text in image"
    images = captured["images"]
    assert isinstance(images, list) and isinstance(images[0], str)  # base64 string


def test_ollama_unreachable_fails_fast() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=req)

    with pytest.raises(ProviderUnreachableError) as excinfo:
        _ollama(handler).health_check()
    assert "ollama.test" in str(excinfo.value)
    assert "ollama pull" in str(excinfo.value)


def test_ollama_retries_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            raise httpx.ConnectError("transient", request=req)
        return httpx.Response(200, json={"response": "ok"})

    assert _ollama(handler).generate("x") == "ok"
    assert calls["n"] == 2


def test_openai_compat_generate() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1/chat/completions"
        return httpx.Response(200, json={"choices": [{"message": {"content": "answer"}}]})

    assert _openai(handler).generate("q") == "answer"


def test_openai_compat_unreachable() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=req)

    with pytest.raises(ProviderUnreachableError):
        _openai(handler).health_check()


def test_factory_default_is_ollama() -> None:
    assert isinstance(build_inference_provider(load_config()), OllamaProvider)


def test_factory_selects_nvidia(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMARTRAG_INFERENCE__PROVIDER", "nvidia")
    monkeypatch.setenv("SMARTRAG_INFERENCE__NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")
    assert isinstance(build_inference_provider(load_config()), NvidiaProvider)


def test_factory_selects_huggingface(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMARTRAG_INFERENCE__PROVIDER", "huggingface")
    monkeypatch.setenv("SMARTRAG_INFERENCE__HUGGINGFACE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    assert isinstance(build_inference_provider(load_config()), HuggingFaceProvider)
