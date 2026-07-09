"""Ollama inference provider (native API).

Shared transport for local GLM-OCR and VLM calls. Uses Ollama's ``/api/generate`` and
``/api/chat`` endpoints; vision requests pass base64-encoded images.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from typing import Any

import httpx

from smartrag.inference.base import (
    ChatMessage,
    InferenceProvider,
    request_json,
)

DEFAULT_TIMEOUT_S = 120.0


class OllamaProvider(InferenceProvider):
    """Talks to a local (or remote) Ollama server over its native HTTP API."""

    def __init__(
        self,
        *,
        host: str,
        default_model: str,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = 3,
        backoff_s: float = 0.5,
        client: httpx.Client | None = None,
    ) -> None:
        self._host = host.rstrip("/")
        self._default_model = default_model
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self._client = client or httpx.Client(timeout=timeout_s)

    def _unreachable(self, model: str) -> str:
        return (
            f"Cannot reach Ollama at {self._host}. Is `ollama serve` running and reachable, "
            f"and has the model '{model}' been pulled (`ollama pull {model}`)?"
        )

    def _post(self, path: str, payload: dict[str, Any], model: str) -> dict[str, Any]:
        return request_json(
            self._client,
            "POST",
            f"{self._host}{path}",
            json=payload,
            max_retries=self._max_retries,
            backoff_s=self._backoff_s,
            unreachable_hint=self._unreachable(model),
        )

    def generate(self, prompt: str, *, model: str | None = None, temperature: float = 0.0) -> str:
        model = model or self._default_model
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        return str(self._post("/api/generate", payload, model).get("response", ""))

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        model = model or self._default_model
        payload = {
            "model": model,
            "stream": False,
            "options": {"temperature": temperature},
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        message = self._post("/api/chat", payload, model).get("message") or {}
        return str(message.get("content", ""))

    def vision(
        self,
        prompt: str,
        images: Sequence[bytes],
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        model = model or self._default_model
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [base64.b64encode(img).decode("ascii") for img in images],
            "stream": False,
            "options": {"temperature": temperature},
        }
        return str(self._post("/api/generate", payload, model).get("response", ""))

    def health_check(self) -> None:
        request_json(
            self._client,
            "GET",
            f"{self._host}/api/tags",
            max_retries=self._max_retries,
            backoff_s=self._backoff_s,
            unreachable_hint=self._unreachable(self._default_model),
        )

    def close(self) -> None:
        self._client.close()
