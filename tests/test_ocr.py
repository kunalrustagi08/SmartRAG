"""GLM-OCR backend tests with a stubbed provider."""

from __future__ import annotations

from collections.abc import Sequence

from smartrag.config import load_config
from smartrag.ocr import GlmOcrOllama, build_ocr_backend
from smartrag.ocr.glm_ocr_ollama import heuristic_confidence


class _StubProvider:
    """Duck-typed stand-in for OllamaProvider used by the OCR backend."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.health_checked = False

    def vision(
        self,
        prompt: str,
        images: Sequence[bytes],
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        return self._text

    def health_check(self) -> None:
        self.health_checked = True


def test_ocr_returns_text_and_confidence() -> None:
    backend = GlmOcrOllama(provider=_StubProvider("Hello world"), model="glm-ocr:latest")  # type: ignore[arg-type]
    result = backend.ocr_image(b"imgbytes")
    assert result.text == "Hello world"
    assert 0.0 < result.confidence <= 1.0


def test_ocr_empty_output_zero_confidence() -> None:
    backend = GlmOcrOllama(provider=_StubProvider("   "), model="m")  # type: ignore[arg-type]
    assert backend.ocr_image(b"x").confidence == 0.0


def test_heuristic_confidence_failure_marker() -> None:
    assert heuristic_confidence("I cannot read this image") == 0.3
    assert heuristic_confidence("Section 1. Introduction") == 0.9
    assert heuristic_confidence("") == 0.0


def test_ocr_health_check_delegates() -> None:
    stub = _StubProvider("x")
    backend = GlmOcrOllama(provider=stub, model="m")  # type: ignore[arg-type]
    backend.health_check()
    assert stub.health_checked is True


def test_ocr_factory_builds_glm_backend() -> None:
    assert isinstance(build_ocr_backend(load_config()), GlmOcrOllama)
