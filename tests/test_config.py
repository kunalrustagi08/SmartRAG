"""Config loading + override-precedence tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from smartrag.config import load_config


def test_defaults_from_yaml() -> None:
    cfg = load_config()
    assert cfg.ollama.host == "http://localhost:11434"
    assert cfg.ollama.ocr_model == "glm-ocr:latest"
    assert cfg.inventory.holdout_size == 10
    assert cfg.chunk.tokenizer == "BAAI/bge-large-en-v1.5"
    assert cfg.chunk.target_tokens == 512
    assert cfg.table.format == "markdown_html"
    assert cfg.figures.caption_strategy == "conditional"
    assert cfg.reading_order.strategy == "docling_vlm_fallback"


def test_env_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMARTRAG_OLLAMA__HOST", "http://gpu-box:11434")
    monkeypatch.setenv("SMARTRAG_CHUNK__TARGET_TOKENS", "256")
    cfg = load_config()
    # Overridden by env.
    assert cfg.ollama.host == "http://gpu-box:11434"
    assert cfg.chunk.target_tokens == 256
    # Untouched values still come from YAML.
    assert cfg.inventory.holdout_size == 10
    assert cfg.ollama.ocr_model == "glm-ocr:latest"


def test_custom_config_file(tmp_path: Path) -> None:
    custom = tmp_path / "custom.yaml"
    custom.write_text("inventory:\n  holdout_size: 3\nchunk:\n  target_tokens: 128\n")
    cfg = load_config(custom)
    # Values from the custom file.
    assert cfg.inventory.holdout_size == 3
    assert cfg.chunk.target_tokens == 128
    # Sections absent from the custom file fall back to model defaults.
    assert cfg.ollama.host == "http://localhost:11434"


def test_unknown_yaml_key_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("inventory:\n  holdout_size: 5\n  bogus_field: 1\n")
    with pytest.raises(ValueError):
        load_config(bad)
