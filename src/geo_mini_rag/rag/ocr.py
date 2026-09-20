"""Text recovery: OCR the PDFs that reached the index with no text layer.

Appraisal will eventually decide what deserves OCR. Until then the held set is
simply the documents ingest recorded as "no extractable text" — on an E&P drive
that is leases, unit agreements, mud logs and completion reports, which is where
most of the real content lives.

Each file is OCR'd into data/ocr/<its path under the drive>, re-extracted from
that copy, and its chunks are stored under the *original* document's id, so a
citation still points at the file on the drive. The row keeps `ocr_path` so it
is obvious where its text came from.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from geo_mini_rag import settings
from geo_mini_rag.errors import UserError
from geo_mini_rag.rag.chunk import chunk_segments
from geo_mini_rag.rag.extract import Skip, extract
from geo_mini_rag.rag.index import (
    DB_PATH,
    _doc_row,
    _sql,
    connect,
    embed_and_store,
    migrate,
)
from geo_mini_rag.rag.trace import OFF, Tracer

HOLD_REASON = "no extractable text"


@dataclass
class OcrEvent:
    path: str
    status: str  # indexed | empty | failed | skipped
    detail: str = ""
    cost: float = 0.0
    seconds: float = 0.0


def require_ocrmypdf() -> str:
    exe = shutil.which("ocrmypdf")
    if exe is None:
        raise UserError(
            "ocrmypdf is not installed: `brew install ocrmypdf` on macOS, "
            "`apt install ocrmypdf` on Debian/Ubuntu."
        )
    return exe


def ocr_pdf(exe: str, src: Path, out: Path, *, language: str, timeout_s: float, trace: Tracer) -> None:
    """Run ocrmypdf, writing a searchable copy to `out`."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        exe,
        "--skip-text",       # leave pages that already carry text alone
        "--rotate-pages",    # scans arrive sideways
        "--deskew",
        "--language", language,
        "--quiet",
        str(src),
        str(out),
    ]
    trace("ocr", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired as exc:
        out.unlink(missing_ok=True)
        raise RuntimeError(f"ocrmypdf timed out after {timeout_s:.0f}s") from exc
    if proc.returncode != 0:
        out.unlink(missing_ok=True)
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise RuntimeError(f"ocrmypdf exit {proc.returncode}: {detail[-1] if detail else 'no output'}")


def held(con, reason: str = HOLD_REASON) -> list[tuple[str, str]]:
    """(doc_id, path) for documents that were ingested but yielded no text."""
    return con.execute(
        "SELECT doc_id, path FROM documents WHERE status = 'skipped' AND reason = ? ORDER BY path",
        [reason],
    ).fetchall()


def run(
    *,
    db: Path = DB_PATH,
    limit: int | None = None,
    force: bool = False,
    language: str = "eng",
    timeout_s: float = 900.0,
    on_event: Callable[[OcrEvent], None] = lambda e: None,
    trace: Tracer = OFF,
) -> None:
    exe = require_ocrmypdf()
    cfg = settings.load_rag_config()
    model = cfg["embed"]["model"]
    batch_size = cfg["embed"]["batch_size"]
    chunk_chars, chunk_overlap = cfg["chunk"]["chars"], cfg["chunk"]["overlap"]

    with connect(db) as con:
        migrate(con, trace)
        targets = held(con)
        trace("ocr", f"{len(targets)} held documents in {db}")
        for n, (doc_id, rel) in enumerate(targets):
            if limit is not None and n >= limit:
                trace("ocr", f"--limit {limit} reached; stopping")
                break
            src = settings.ROOT / rel
            if not src.exists():
                on_event(OcrEvent(rel, "failed", "source file is gone"))
                continue
            out = settings.OCR_DIR / rel
            trace("file", f"#{n + 1} {rel}")

            started = time.monotonic()
            if out.exists() and not force and out.stat().st_mtime >= src.stat().st_mtime:
                trace("ocr", f"reusing {out}")
            else:
                try:
                    ocr_pdf(exe, src, out, language=language, timeout_s=timeout_s, trace=trace)
                except RuntimeError as exc:
                    on_event(OcrEvent(rel, "failed", str(exc), seconds=time.monotonic() - started))
                    continue
            elapsed = time.monotonic() - started

            try:
                ex = extract(
                    out,
                    max_pdf_pages=cfg["extract"]["max_pdf_pages"],
                    max_text_bytes=cfg["extract"]["max_text_bytes"],
                    trace=trace,
                )
            except Skip as s:
                on_event(OcrEvent(rel, "failed", f"re-extract: {s}", seconds=elapsed))
                continue
            chunks = chunk_segments(ex.segments, chars=chunk_chars, overlap=chunk_overlap)
            n_chars = sum(len(t) for _, t in ex.segments)
            st = src.stat()
            if not chunks:
                # OCR ran and still found nothing: a photograph, a blank page, a plat.
                _sql(con, trace, "BEGIN")
                _sql(con, trace, "DELETE FROM documents WHERE doc_id = ?", [doc_id])
                _doc_row(con, trace, doc_id, rel, st, "skipped", ex,
                         "no text after OCR", ocr_path=str(out.relative_to(settings.ROOT)))
                _sql(con, trace, "COMMIT")
                on_event(OcrEvent(rel, "empty", "OCR produced no text", seconds=elapsed))
                continue

            _sql(con, trace, "BEGIN")
            _sql(con, trace, "DELETE FROM chunks WHERE doc_id = ?", [doc_id])
            _sql(con, trace, "DELETE FROM documents WHERE doc_id = ?", [doc_id])
            try:
                tokens, cost = embed_and_store(con, doc_id, chunks, model, batch_size, trace)
            except Exception as exc:  # noqa: BLE001 - keep the held row, retry next run
                _sql(con, trace, "ROLLBACK")
                on_event(OcrEvent(rel, "failed", f"embedding: {type(exc).__name__}: {exc}", seconds=elapsed))
                continue
            _doc_row(con, trace, doc_id, rel, st, "indexed", ex, None, n_chars=n_chars,
                     n_chunks=len(chunks), embed_tokens=tokens, embed_cost=cost,
                     ocr_path=str(out.relative_to(settings.ROOT)))
            _sql(con, trace, "COMMIT")
            on_event(OcrEvent(
                rel, "indexed",
                f"{ex.pages} pages, {n_chars:,} chars, {len(chunks)} chunks, ${cost:.5f}",
                cost, elapsed,
            ))
