"""OCR as a partitioning strategy.

A scanned PDF has no text layer, so text extraction returns nothing and the
document is invisible to retrieval — in E&P material that is most leases, unit
agreements and completion reports. ocrmypdf writes a searchable copy into
data/ocr/, mirroring the source layout, and partitioning reads that copy
instead. The original is never modified.

Copies are reused: OCR costs seconds per page and nothing in dollars, so the
expensive part is worth keeping between runs.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from geo_mini_rag import settings
from geo_mini_rag.rag.trace import OFF, Tracer


class OcrUnavailable(RuntimeError):
    """ocrmypdf is not installed."""


def executable() -> str:
    exe = shutil.which("ocrmypdf")
    if exe is None:
        raise OcrUnavailable(
            "ocrmypdf is not installed: `brew install ocrmypdf` on macOS, "
            "`apt install ocrmypdf` on Debian/Ubuntu."
        )
    return exe


def output_path(src: Path) -> Path:
    """Where the searchable copy of this file lives."""
    try:
        return settings.OCR_DIR / src.relative_to(settings.ROOT)
    except ValueError:
        return settings.OCR_DIR / src.name


def ocr_to(
    src: Path,
    *,
    language: str = "eng",
    timeout_s: float = 900.0,
    force: bool = False,
    trace: Tracer = OFF,
) -> Path:
    """OCR `src`, returning the path to the searchable copy."""
    exe = executable()
    out = output_path(src)
    if out.exists() and not force and out.stat().st_mtime >= src.stat().st_mtime:
        trace("ocr", f"reusing {out}")
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        exe,
        "--skip-text",       # pages that already carry text are left alone
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
    return out
