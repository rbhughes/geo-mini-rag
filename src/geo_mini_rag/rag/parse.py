"""Partitioning: route a file to a parser and get text back.

The shape follows Unstructured's `partition`: detect the type, dispatch to a
parser for that type, and choose a strategy within it. For PDFs the strategy is
the one that matters — read the text layer, and where there isn't one, OCR the
document and read it again. That replaces the separate probe-and-quarantine
stages: "needs OCR" is a routing decision, not a state a file sits in.

Domain handlers registered in `geo_mini_rag.ep` are consulted first, so a
format whose meaning is not its raw text (LAS, shapefile) never reaches the
generic path.
"""

from __future__ import annotations

from pathlib import Path

from geo_mini_rag import ep, settings
from geo_mini_rag.rag.extract import HEAD_BYTES, Extracted, extract
from geo_mini_rag.rag.trace import OFF, Tracer


def _chars_per_page(ex: Extracted) -> float:
    pages = [len((text or "").strip()) for _, text in ex.segments] or [0]
    return sum(pages) / len(pages)


def needs_ocr(ex: Extracted, cfg: dict) -> bool:
    """A PDF whose pages carry no usable text layer."""
    if ex.kind != "pdf":
        return False
    threshold = cfg["extract"].get("ocr_below_chars_per_page", 50)
    return _chars_per_page(ex) < threshold


def parse(path: Path, cfg: dict, trace: Tracer = OFF, *, allow_ocr: bool = True) -> Extracted:
    with path.open("rb") as f:
        head = f.read(HEAD_BYTES)

    handler = ep.find(path, head)
    if handler is not None:
        trace("handler", f"{handler.name} claims {path.name}")
        return handler.parse(path, cfg, trace)

    ex = extract(
        path,
        max_pdf_pages=cfg["extract"]["max_pdf_pages"],
        max_text_bytes=cfg["extract"]["max_text_bytes"],
        trace=trace,
    )
    if not (allow_ocr and needs_ocr(ex, cfg)):
        return ex

    from geo_mini_rag.rag import ocr

    trace("partition", f"{_chars_per_page(ex):.1f} chars/page: no text layer, switching to OCR")
    try:
        out = ocr.ocr_to(path, language=cfg["extract"].get("ocr_language", "eng"), trace=trace)
    except (ocr.OcrUnavailable, RuntimeError) as exc:
        trace("partition", f"OCR unavailable or failed, keeping what the text layer gave: {exc}")
        ex.notes.append(f"ocr_failed: {exc}")
        return ex

    after = extract(
        out,
        max_pdf_pages=cfg["extract"]["max_pdf_pages"],
        max_text_bytes=cfg["extract"]["max_text_bytes"],
        trace=trace,
    )
    after.metadata["ocr_path"] = str(out.relative_to(settings.ROOT))
    after.metadata["ocr_language"] = cfg["extract"].get("ocr_language", "eng")
    after.notes.append("ocr")
    trace("partition", f"OCR yielded {_chars_per_page(after):.1f} chars/page")
    return after
