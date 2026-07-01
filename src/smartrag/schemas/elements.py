"""Normalized element schema — the convergence point for every parse path.

Every parser/OCR/VLM path emits a stream of :class:`NormalizedElement`. This module is the
contract the rest of the pipeline builds on, so the models forbid unknown fields to catch
drift early.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ElementType(str, Enum):
    """Kind of structural element extracted from a page."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    CODE = "code"
    FORMULA = "formula"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"


class SourcePath(str, Enum):
    """Which processing path produced an element."""

    BORN_DIGITAL = "born_digital"
    OCR = "ocr"
    VLM = "vlm"


class QualityFlag(str, Enum):
    """Quality/exception signals attached to elements and chunks."""

    LOW_CONFIDENCE = "low_confidence"
    MALFORMED_TABLE = "malformed_table"
    UNCERTAIN_READING_ORDER = "uncertain_reading_order"
    OCR_FALLBACK = "ocr_fallback"
    EMPTY = "empty"


class BBox(BaseModel):
    """Bounding box on a single page, in the parser's coordinate space."""

    model_config = ConfigDict(extra="forbid")

    page: int
    x0: float
    y0: float
    x1: float
    y1: float


class TablePayload(BaseModel):
    """Serialized table: Markdown for readability plus HTML for span-preserving structure."""

    model_config = ConfigDict(extra="forbid")

    markdown: str
    html: str
    n_rows: int | None = None
    n_cols: int | None = None


class FigurePayload(BaseModel):
    """Figure stored by reference and/or captioned by the VLM."""

    model_config = ConfigDict(extra="forbid")

    image_ref: str | None = None
    caption: str | None = None
    caption_source: Literal["vlm", "original", "none"] = "none"


class NormalizedElement(BaseModel):
    """One structural element in a document's ordered stream."""

    model_config = ConfigDict(extra="forbid")

    id: str
    doc_id: str
    source_file: str
    page_numbers: list[int]
    element_type: ElementType
    reading_order: int
    section_path: list[str] = Field(default_factory=list)
    level: int | None = None
    text: str | None = None
    table: TablePayload | None = None
    figure: FigurePayload | None = None
    bbox: BBox | None = None
    source_path: SourcePath
    confidence: float | None = None
    quality_flags: list[QualityFlag] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
