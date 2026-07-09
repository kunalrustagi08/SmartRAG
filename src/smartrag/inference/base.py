"""Inference provider interface, shared HTTP transport, and error types.

All providers are reached over ``httpx`` (not heavy provider SDKs) so the interface stays
thin and swappable. :func:`request_json` centralizes retries and fail-fast behavior:
transport errors are retried with backoff; an unreachable endpoint raises
:class:`ProviderUnreachableError` with an actionable message.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal

import httpx
from pydantic import BaseModel
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    """A single chat message."""

    role: Role
    content: str


class InferenceError(RuntimeError):
    """Base class for inference failures."""


class ProviderUnreachableError(InferenceError):
    """Raised when the inference endpoint cannot be reached (fail-fast)."""


class InferenceProvider(ABC):
    """Swappable text/vision inference backend."""

    @abstractmethod
    def generate(self, prompt: str, *, model: str | None = None, temperature: float = 0.0) -> str:
        """Single-prompt text completion."""

    @abstractmethod
    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Multi-turn chat completion."""

    @abstractmethod
    def vision(
        self,
        prompt: str,
        images: Sequence[bytes],
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Multimodal completion over a prompt plus one or more images."""

    @abstractmethod
    def health_check(self) -> None:
        """Fail fast (raise :class:`ProviderUnreachableError`) if the endpoint is unreachable."""

    def close(self) -> None:
        """Release underlying resources. Default no-op."""
        return None


def request_json(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    json: dict[str, Any] | None = None,
    max_retries: int,
    backoff_s: float,
    unreachable_hint: str,
) -> dict[str, Any]:
    """Perform an HTTP request with retries and return the decoded JSON body.

    Retries transport errors with exponential backoff. Converts a persistent connection
    failure into :class:`ProviderUnreachableError` and an HTTP error status into
    :class:`InferenceError`, both with actionable context.
    """
    try:
        for attempt in Retrying(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=backoff_s, max=8.0),
            retry=retry_if_exception_type(httpx.TransportError),
            reraise=True,
        ):
            with attempt:
                response = client.request(method, url, json=json)
                response.raise_for_status()
                data: dict[str, Any] = response.json()
                return data
    except httpx.ConnectError as exc:
        raise ProviderUnreachableError(unreachable_hint) from exc
    except httpx.HTTPStatusError as exc:
        body = exc.response.text[:200]
        raise InferenceError(f"{url} returned {exc.response.status_code}: {body}") from exc
    raise InferenceError("retry loop exhausted without a response")  # pragma: no cover
