"""OCR / VLM backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class OcrResult(BaseModel):
    """Text extracted from an image plus a confidence signal in [0, 1]."""

    text: str
    confidence: float


class OcrBackend(ABC):
    """Swappable OCR/VLM backend operating on a single page image."""

    @abstractmethod
    def ocr_image(self, image: bytes, *, prompt: str | None = None) -> OcrResult:
        """Transcribe the text in ``image``."""

    @abstractmethod
    def health_check(self) -> None:
        """Fail fast if the backend is unreachable."""
