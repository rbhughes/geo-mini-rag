"""Route a file to a domain handler, or to the generic extractor."""

from __future__ import annotations

from pathlib import Path

from geo_mini_rag import ep
from geo_mini_rag.rag.extract import HEAD_BYTES, Extracted, extract
from geo_mini_rag.rag.trace import OFF, Tracer


def parse(path: Path, cfg: dict, trace: Tracer = OFF) -> Extracted:
    with path.open("rb") as f:
        head = f.read(HEAD_BYTES)
    handler = ep.find(path, head)
    if handler is not None:
        trace("handler", f"{handler.name} claims {path.name}")
        return handler.parse(path, cfg, trace)
    return extract(
        path,
        max_pdf_pages=cfg["extract"]["max_pdf_pages"],
        max_text_bytes=cfg["extract"]["max_text_bytes"],
        trace=trace,
    )
