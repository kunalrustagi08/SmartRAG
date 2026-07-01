"""Final chunk schema — the Phase 1 deliverable.

A :class:`Chunk` is produced by structure-aware chunking and carries full provenance back to
the :class:`~smartrag.schemas.elements.NormalizedElement` instances it was built from.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from smartrag.schemas.elements import ElementType, QualityFlag


class Chunk(BaseModel):
    """A retrieval-ready chunk with breadcrumbs, citations, and provenance."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    doc_id: str
    source_file: str
    page_numbers: list[int]
    element_types: list[ElementType]
    section_path: list[str] = Field(default_factory=list)
    breadcrumb: str
    text: str
    token_count: int
    element_ids: list[str] = Field(default_factory=list)
    quality_flags: list[QualityFlag] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
