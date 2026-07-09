"""Build the GLM-OCR backend from config.

GLM-OCR always runs through Ollama regardless of the text inference provider, so this builds
a dedicated :class:`OllamaProvider` pinned to the configured OCR model.
"""

from __future__ import annotations

from smartrag.config import Settings
from smartrag.inference.ollama import OllamaProvider
from smartrag.ocr.base import OcrBackend
from smartrag.ocr.glm_ocr_ollama import GlmOcrOllama


def build_ocr_backend(cfg: Settings) -> OcrBackend:
    """Return the configured OCR backend (GLM-OCR via Ollama)."""
    provider = OllamaProvider(
        host=cfg.ollama.host,
        default_model=cfg.ollama.ocr_model,
        timeout_s=cfg.ollama.timeout_s,
        max_retries=cfg.inference.max_retries,
    )
    return GlmOcrOllama(provider=provider, model=cfg.ollama.ocr_model)
