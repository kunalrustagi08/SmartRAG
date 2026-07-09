"""Configuration — the single source of truth for tunable behavior.

Values load from ``configs/default.yaml`` and can be overridden by environment variables of
the form ``SMARTRAG_<SECTION>__<FIELD>`` (e.g. ``SMARTRAG_OLLAMA__HOST``). Precedence is:

    env vars  >  YAML file  >  model defaults

Nothing else in the codebase should hardcode thresholds, hosts, sizes, or endpoints — read
them from :func:`load_config`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")

_STRICT = ConfigDict(extra="forbid")


class OllamaConfig(BaseModel):
    """Ollama transport for GLM-OCR and the VLM."""

    model_config = _STRICT

    host: str = "http://localhost:11434"
    ocr_model: str = "glm-ocr:latest"
    vlm_model: str = "glm-ocr:latest"
    timeout_s: float = 120.0


class InferenceConfig(BaseModel):
    """Swappable inference provider selection and endpoints."""

    model_config = _STRICT

    provider: Literal["ollama", "nvidia", "huggingface"] = "ollama"
    nvidia_base_url: str | None = None
    nvidia_model: str | None = None
    huggingface_model: str | None = None
    max_retries: int = 3


class InventoryConfig(BaseModel):
    """Corpus scan and held-out sampling."""

    model_config = _STRICT

    holdout_size: int = 10
    seed: int = 42


class TriageConfig(BaseModel):
    """Page-level routing thresholds and escalation guardrail."""

    model_config = _STRICT

    extractable_text_ratio_min: float = 0.2
    image_density_max: float = 0.5
    escalation_rate_target: float = 0.15


class ParseConfig(BaseModel):
    """Parser-specific knobs."""

    model_config = _STRICT

    docling_do_ocr: bool = False


class TableConfig(BaseModel):
    """Table serialization format."""

    model_config = _STRICT

    format: Literal["markdown_html", "markdown", "html"] = "markdown_html"


class FigureConfig(BaseModel):
    """Figure handling strategy."""

    model_config = _STRICT

    caption_strategy: Literal["conditional", "all", "reference"] = "conditional"


class ReadingOrderConfig(BaseModel):
    """Reading-order reconstruction strategy."""

    model_config = _STRICT

    strategy: Literal["docling", "docling_vlm_fallback", "geometric"] = "docling_vlm_fallback"


class ChunkConfig(BaseModel):
    """Structure-aware chunking sizing."""

    model_config = _STRICT

    tokenizer: str = "BAAI/bge-large-en-v1.5"
    target_tokens: int = 512
    overlap_tokens: int = 64


class PathsConfig(BaseModel):
    """Filesystem locations for corpus and outputs."""

    model_config = _STRICT

    corpus_dir: Path = Path("data/corpus")
    output_dir: Path = Path("output")
    figures_dir: Path = Path("output/figures")


class Settings(BaseSettings):
    """Top-level resolved configuration."""

    model_config = SettingsConfigDict(
        env_prefix="SMARTRAG_",
        env_nested_delimiter="__",
        yaml_file=str(DEFAULT_CONFIG_PATH),
        extra="forbid",
    )

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    inventory: InventoryConfig = Field(default_factory=InventoryConfig)
    triage: TriageConfig = Field(default_factory=TriageConfig)
    parse: ParseConfig = Field(default_factory=ParseConfig)
    table: TableConfig = Field(default_factory=TableConfig)
    figures: FigureConfig = Field(default_factory=FigureConfig)
    reading_order: ReadingOrderConfig = Field(default_factory=ReadingOrderConfig)
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Order = priority (highest first): env > YAML > model defaults.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


def load_config(config_file: str | Path | None = None) -> Settings:
    """Load settings from YAML (default: ``configs/default.yaml``) with env overrides.

    Pass ``config_file`` to read a different YAML file; env vars still take precedence.
    """
    if config_file is None:
        return Settings()

    class _ConfiguredSettings(Settings):
        model_config = SettingsConfigDict(
            env_prefix="SMARTRAG_",
            env_nested_delimiter="__",
            yaml_file=str(config_file),
            extra="forbid",
        )

    return _ConfiguredSettings()
