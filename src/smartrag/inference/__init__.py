"""Swappable inference providers."""

from __future__ import annotations

from smartrag.inference.base import (
    ChatMessage,
    InferenceError,
    InferenceProvider,
    ProviderUnreachableError,
)
from smartrag.inference.factory import build_inference_provider
from smartrag.inference.huggingface import HuggingFaceProvider
from smartrag.inference.nvidia import NvidiaProvider
from smartrag.inference.ollama import OllamaProvider

__all__ = [
    "ChatMessage",
    "HuggingFaceProvider",
    "InferenceError",
    "InferenceProvider",
    "NvidiaProvider",
    "OllamaProvider",
    "ProviderUnreachableError",
    "build_inference_provider",
]
