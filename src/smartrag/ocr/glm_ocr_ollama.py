"""GLM-OCR backend served through Ollama.

Ollama does not expose token-level confidence, so ``OcrResult.confidence`` is a heuristic
placeholder: 0.0 for empty output, reduced when the model emits a refusal/failure phrase,
otherwise a high default. A better signal can replace this without changing the interface.
"""

from __future__ import annotations

from smartrag.inference.ollama import OllamaProvider
from smartrag.ocr.base import OcrBackend, OcrResult

DEFAULT_OCR_PROMPT = (
    "Transcribe all text in this image exactly, preserving the natural reading order. "
    "Output only the transcribed text, with no commentary."
)

_FAILURE_MARKERS = (
    "i cannot",
    "i can't",
    "i'm unable",
    "i am unable",
    "sorry, i can",
    "cannot process",
    "no text",
)


def heuristic_confidence(text: str) -> float:
    """Rough confidence for VLM OCR output (see module docstring)."""
    stripped = text.strip().lower()
    if not stripped:
        return 0.0
    if any(marker in stripped for marker in _FAILURE_MARKERS):
        return 0.3
    return 0.9


class GlmOcrOllama(OcrBackend):
    """OCR pages via the GLM-OCR model over an :class:`OllamaProvider`."""

    def __init__(self, *, provider: OllamaProvider, model: str) -> None:
        self._provider = provider
        self._model = model

    def ocr_image(self, image: bytes, *, prompt: str | None = None) -> OcrResult:
        text = self._provider.vision(prompt or DEFAULT_OCR_PROMPT, [image], model=self._model)
        return OcrResult(text=text, confidence=heuristic_confidence(text))

    def health_check(self) -> None:
        self._provider.health_check()
