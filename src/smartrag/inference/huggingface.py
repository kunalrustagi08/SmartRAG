"""Hugging Face inference provider (OpenAI-compatible router)."""

from __future__ import annotations

import os

import httpx

from smartrag.inference._openai_compat import OpenAICompatibleProvider
from smartrag.inference.base import InferenceError

DEFAULT_HF_BASE_URL = "https://router.huggingface.co/v1"


class HuggingFaceProvider(OpenAICompatibleProvider):
    """Hugging Face router over the OpenAI-compatible API.

    The token is read from ``HF_TOKEN`` (or ``HUGGINGFACE_TOKEN``); never from config.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        backoff_s: float = 0.5,
        client: httpx.Client | None = None,
    ) -> None:
        if not model:
            raise InferenceError(
                "Hugging Face provider requires a model (set inference.huggingface_model)."
            )
        token = api_key or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        super().__init__(
            base_url=base_url or DEFAULT_HF_BASE_URL,
            model=model,
            api_key=token,
            timeout_s=timeout_s,
            max_retries=max_retries,
            backoff_s=backoff_s,
            client=client,
        )
