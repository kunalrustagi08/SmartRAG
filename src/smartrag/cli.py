"""SmartRAG command-line interface.

Thin `typer` app exposing one command per pipeline stage. Commands are stubs for now;
each is implemented by its corresponding issue/stage.
"""

from __future__ import annotations

from pathlib import Path

import typer

from smartrag.config import load_config
from smartrag.inference.base import InferenceError
from smartrag.ocr import build_ocr_backend

app = typer.Typer(
    name="smartrag",
    help="Document processing pipeline for RAG: PDFs/Word docs -> chunk-ready structured output.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def config() -> None:
    """Print the resolved configuration (YAML + env overrides) as JSON."""
    typer.echo(load_config().model_dump_json(indent=2))


@app.command("ocr-check")
def ocr_check(image: Path) -> None:
    """Smoke-test the GLM-OCR backend on a single image file."""
    backend = build_ocr_backend(load_config())
    try:
        backend.health_check()
        result = backend.ocr_image(image.read_bytes())
    except InferenceError as exc:
        typer.secho(f"OCR failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"confidence={result.confidence:.2f}")
    typer.echo(result.text)


@app.command()
def inventory() -> None:
    """Scan the corpus and record per-file metadata."""
    typer.echo("inventory: not yet implemented")


@app.command()
def triage() -> None:
    """Classify pages and route them to the cheap or OCR path."""
    typer.echo("triage: not yet implemented")


@app.command()
def parse() -> None:
    """Parse pages into normalized elements."""
    typer.echo("parse: not yet implemented")


@app.command()
def run() -> None:
    """Run the full pipeline end to end."""
    typer.echo("run: not yet implemented")


@app.command()
def validate() -> None:
    """Run the pipeline on the held-out hard cases and dump inspectable output."""
    typer.echo("validate: not yet implemented")


def main() -> None:
    """Console-script entry point (see ``[project.scripts]`` in pyproject.toml)."""
    app()


if __name__ == "__main__":
    main()
