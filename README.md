# SmartRAG

Document processing pipeline that turns large PDFs and Word docs (potentially thousands of
pages) into clean, **chunk-ready structured output with metadata**.

> **Phase 1 scope:** document processing only — no embedding, no vector DB, no retrieval.
> The deliverable is structured, chunk-ready output with full provenance metadata.

## Pipeline stages

1. **Inventory & sampling** — scan the corpus, record per-file metadata, reserve a held-out
   hard-case set for validation.
2. **Triage / routing** — classify each *page* (born-digital / scanned / mixed) and route it
   to the cheap path or the OCR path.
3. **Parse** — born-digital pages via PyMuPDF; scanned/complex pages via Docling + GLM-OCR.
4. **Exception handling** — low-confidence / malformed pages escalate to the GLM-OCR/VLM path.
5. **Normalization** — unify all paths into one element schema (text, tables, figures).
6. **Structure-aware chunking** — chunk along structure into the final `Chunk` schema.

Parser, OCR/VLM backend, and inference provider each sit behind a swappable interface.

## Getting started

```bash
uv sync            # install runtime + dev dependencies
uv run smartrag --help
```

## Configuration defaults

| Setting | Default |
| --- | --- |
| Ollama host | `http://localhost:11434` |
| GLM-OCR model tag | `glm-ocr:latest` |
| Held-out set size | 10 documents |
| Chunk tokenizer | `BAAI/bge-large-en-v1.5` (~512-token soft cap + overlap) |
| Table format | Markdown + HTML |

All thresholds, hosts, sizes, and endpoints live in config (`configs/default.yaml`), not in code.
