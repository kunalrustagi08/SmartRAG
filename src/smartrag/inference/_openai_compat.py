"""Shared OpenAI-compatible provider (used by NVIDIA NIM and the HF router).

Both NVIDIA inference endpoints and the Hugging Face router expose an OpenAI-compatible
``/chat/completions`` API, so a single httpx-based implementation backs both.
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


class OpenAICompatibleProvider(InferenceProvider):
    """Provider for any OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = 3,
        backoff_s: float = 0.5,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(timeout=timeout_s, headers=headers)

    def _unreachable(self) -> str:
        return f"Cannot reach inference endpoint at {self._base_url}. Check the URL and network."

    def _chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str | None,
        temperature: float,
    ) -> str:
        data = request_json(
            self._client,
            "POST",
            f"{self._base_url}/chat/completions",
            json={
                "model": model or self._model,
                "messages": messages,
                "temperature": temperature,
                "stream": False,
            },
            max_retries=self._max_retries,
            backoff_s=self._backoff_s,
            unreachable_hint=self._unreachable(),
        )
        choices = data.get("choices") or []
        if not choices:
            return ""
        return str(choices[0].get("message", {}).get("content", "") or "")

    def generate(self, prompt: str, *, model: str | None = None, temperature: float = 0.0) -> str:
        return self._chat_completion([{"role": "user", "content": prompt}], model, temperature)

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        payload = [{"role": m.role, "content": m.content} for m in messages]
        return self._chat_completion(payload, model, temperature)

    def vision(
        self,
        prompt: str,
        images: Sequence[bytes],
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            uri = "data:image/png;base64," + base64.b64encode(img).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": uri}})
        return self._chat_completion([{"role": "user", "content": content}], model, temperature)

    def health_check(self) -> None:
        request_json(
            self._client,
            "GET",
            f"{self._base_url}/models",
            max_retries=self._max_retries,
            backoff_s=self._backoff_s,
            unreachable_hint=self._unreachable(),
        )

    def close(self) -> None:
        self._client.close()
