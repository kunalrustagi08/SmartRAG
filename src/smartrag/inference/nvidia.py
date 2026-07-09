"""NVIDIA inference provider (OpenAI-compatible NIM endpoints)."""

from __future__ import annotations

import os

import httpx

from smartrag.inference._openai_compat import OpenAICompatibleProvider
from smartrag.inference.base import InferenceError

DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


class NvidiaProvider(OpenAICompatibleProvider):
    """NVIDIA NIM / integrate.api.nvidia.com over the OpenAI-compatible API.

    The API key is read from the ``NVIDIA_API_KEY`` environment variable (never from config).
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
                "NVIDIA provider requires a model (set inference.nvidia_model in config)."
            )
        super().__init__(
            base_url=base_url or DEFAULT_NVIDIA_BASE_URL,
            model=model,
            api_key=api_key or os.environ.get("NVIDIA_API_KEY"),
            timeout_s=timeout_s,
            max_retries=max_retries,
            backoff_s=backoff_s,
            client=client,
        )
