"""Inventory & triage records.

:class:`FileRecord` / :class:`PageRecord` capture the corpus scan and page-level routing
decisions. ``FileRecord.file_hash`` is the key used for resumability in the pipeline state.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DocType(str, Enum):
    """Source document format."""

    PDF = "pdf"
    DOCX = "docx"


class PageRoute(str, Enum):
    """Page-level routing decision from triage."""

    BORN_DIGITAL = "born_digital"
    SCANNED = "scanned"
    MIXED = "mixed"


class PageRecord(BaseModel):
    """Per-page triage signals and the resulting route."""

    model_config = ConfigDict(extra="forbid")

    page_number: int
    extractable_text_ratio: float
    image_density: float
    route: PageRoute | None = None
    deciding_signal: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FileRecord(BaseModel):
    """Per-file inventory metadata plus its page records."""

    model_config = ConfigDict(extra="forbid")

    path: str
    file_hash: str
    doc_type: DocType
    page_count: int
    text_layer_present: bool
    image_density: float | None = None
    layout_type: str | None = None
    is_holdout: bool = False
    pages: list[PageRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
