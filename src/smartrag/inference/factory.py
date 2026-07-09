"""Build the configured inference provider (swappable via ``inference.provider``)."""

from __future__ import annotations

from typing_extensions import assert_never

from smartrag.config import Settings
from smartrag.inference.base import InferenceProvider
from smartrag.inference.huggingface import HuggingFaceProvider
from smartrag.inference.nvidia import NvidiaProvider
from smartrag.inference.ollama import OllamaProvider


def build_inference_provider(cfg: Settings) -> InferenceProvider:
    """Return the inference provider selected by ``cfg.inference.provider``."""
    provider = cfg.inference.provider
    if provider == "ollama":
        return OllamaProvider(
            host=cfg.ollama.host,
            default_model=cfg.ollama.vlm_model,
            timeout_s=cfg.ollama.timeout_s,
            max_retries=cfg.inference.max_retries,
        )
    if provider == "nvidia":
        return NvidiaProvider(
            model=cfg.inference.nvidia_model or "",
            base_url=cfg.inference.nvidia_base_url,
            timeout_s=cfg.ollama.timeout_s,
            max_retries=cfg.inference.max_retries,
        )
    if provider == "huggingface":
        return HuggingFaceProvider(
            model=cfg.inference.huggingface_model or "",
            timeout_s=cfg.ollama.timeout_s,
            max_retries=cfg.inference.max_retries,
        )
    assert_never(provider)
