"""Pydantic schemas: the contract between pipeline stages."""

from __future__ import annotations

from smartrag.schemas.chunks import Chunk
from smartrag.schemas.elements import (
    BBox,
    ElementType,
    FigurePayload,
    NormalizedElement,
    QualityFlag,
    SourcePath,
    TablePayload,
)
from smartrag.schemas.inventory import DocType, FileRecord, PageRecord, PageRoute

__all__ = [
    "BBox",
    "Chunk",
    "DocType",
    "ElementType",
    "FigurePayload",
    "FileRecord",
    "NormalizedElement",
    "PageRecord",
    "PageRoute",
    "QualityFlag",
    "SourcePath",
    "TablePayload",
]
