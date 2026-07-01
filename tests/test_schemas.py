"""Round-trip serialization tests for the stage-contract schemas."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from smartrag.schemas import (
    BBox,
    Chunk,
    DocType,
    ElementType,
    FigurePayload,
    FileRecord,
    NormalizedElement,
    PageRecord,
    PageRoute,
    QualityFlag,
    SourcePath,
    TablePayload,
)

TABLE_ELEMENT = NormalizedElement(
    id="e-table",
    doc_id="d1",
    source_file="report.pdf",
    page_numbers=[3, 4],
    element_type=ElementType.TABLE,
    reading_order=7,
    section_path=["Results", "Benchmarks"],
    table=TablePayload(markdown="| a | b |", html="<table></table>", n_rows=2, n_cols=2),
    bbox=BBox(page=3, x0=10.0, y0=20.0, x1=200.0, y1=120.0),
    source_path=SourcePath.BORN_DIGITAL,
    confidence=0.98,
    quality_flags=[QualityFlag.MALFORMED_TABLE],
    metadata={"rotation": 0},
)

FIGURE_ELEMENT = NormalizedElement(
    id="e-figure",
    doc_id="d1",
    source_file="report.pdf",
    page_numbers=[5],
    element_type=ElementType.FIGURE,
    reading_order=9,
    figure=FigurePayload(
        image_ref="figures/d1_p5_fig1.png",
        caption="Throughput vs. batch size.",
        caption_source="vlm",
    ),
    source_path=SourcePath.VLM,
)

PARAGRAPH_ELEMENT = NormalizedElement(
    id="e-para",
    doc_id="d1",
    source_file="report.pdf",
    page_numbers=[1],
    element_type=ElementType.PARAGRAPH,
    reading_order=0,
    text="Hello world.",
    source_path=SourcePath.OCR,
    quality_flags=[QualityFlag.OCR_FALLBACK, QualityFlag.LOW_CONFIDENCE],
)

CHUNK = Chunk(
    chunk_id="d1-c0",
    doc_id="d1",
    source_file="report.pdf",
    page_numbers=[1, 2],
    element_types=[ElementType.HEADING, ElementType.PARAGRAPH],
    section_path=["Introduction"],
    breadcrumb="Report > Introduction",
    text="Introduction\n\nHello world.",
    token_count=42,
    element_ids=["e-head", "e-para"],
    quality_flags=[],
    metadata={"overlap": True},
)

PAGE_RECORD = PageRecord(
    page_number=2,
    extractable_text_ratio=0.12,
    image_density=0.87,
    route=PageRoute.SCANNED,
    deciding_signal="extractable_text_ratio<0.2",
)

FILE_RECORD = FileRecord(
    path="corpus/report.pdf",
    file_hash="sha256:abc123",
    doc_type=DocType.PDF,
    page_count=10,
    text_layer_present=True,
    image_density=0.4,
    layout_type="multi_column",
    is_holdout=True,
    pages=[PAGE_RECORD],
    metadata={"scanned_pages": 3},
)

MODELS = [
    TABLE_ELEMENT,
    FIGURE_ELEMENT,
    PARAGRAPH_ELEMENT,
    CHUNK,
    PAGE_RECORD,
    FILE_RECORD,
]


@pytest.mark.parametrize("model", MODELS, ids=lambda m: type(m).__name__)
def test_json_roundtrip(model: BaseModel) -> None:
    cls = type(model)
    assert cls.model_validate_json(model.model_dump_json()) == model


@pytest.mark.parametrize("model", MODELS, ids=lambda m: type(m).__name__)
def test_dict_roundtrip(model: BaseModel) -> None:
    cls = type(model)
    assert cls.model_validate(model.model_dump()) == model
