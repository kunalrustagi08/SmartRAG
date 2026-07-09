"""OCR / VLM backends."""

from __future__ import annotations

from smartrag.ocr.base import OcrBackend, OcrResult
from smartrag.ocr.factory import build_ocr_backend
from smartrag.ocr.glm_ocr_ollama import GlmOcrOllama

__all__ = [
    "GlmOcrOllama",
    "OcrBackend",
    "OcrResult",
    "build_ocr_backend",
]
