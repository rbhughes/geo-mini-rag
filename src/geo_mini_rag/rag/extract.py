"""Reading the ordinary formats: PDF, DOCX, HTML and plain text.

The type comes from the file's leading bytes rather than its extension, using a
few signature checks that are certain. Anything else is handed to libmagic, and
if libmagic cannot name something we can read, the file is skipped with that
name as the reason, so `stats` can say what a collection is full of.

E&P formats are not here: `geo_mini_rag.ep` claims those before this runs.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import ClassVar

from geo_mini_rag.errors import UserError
from geo_mini_rag.rag.trace import OFF, Tracer, hexdump

HTML_EXTS = {".html", ".htm", ".xhtml", ".shtml", ".asp", ".aspx", ".php", ".jsp"}
HEAD_BYTES = 8192
# Microsoft Compound File Binary header, shared by legacy .doc, .xls, .ppt, .msg and others.
OLE2_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
TEXTUAL_MIN_PRINTABLE = 0.95


@dataclass
class Extracted:
    kind: str                                   # pdf | docx | html | text
    segments: list[tuple[int | None, str]]      # (page number or None, text)
    pages: int | None = None
    truncated: bool = False
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)  # document-level fields a handler lifted out
    atomic: bool = False  # segments are already chunk-sized; do not merge or split them


class Skip(Exception):
    """The file is not something this extractor handles; str(self) is the reason."""


def sniff(path: Path, head: bytes) -> str:
    return sniff_explained(path, head)[0]


def sniff_explained(path: Path, head: bytes) -> tuple[str, str]:
    """Return (kind, the rule that decided it)."""
    if head.startswith(b"%PDF"):
        return "pdf", "starts with %PDF"
    if head.startswith(OLE2_SIGNATURE):
        return "binary", "OLE2 compound file (legacy Office .doc/.xls/.ppt or similar)"
    if head.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(path) as z:
                names = z.namelist()
        except zipfile.BadZipFile:
            return "binary", "zip signature, but not a readable zip"
        if "word/document.xml" in names:
            return "docx", "zip containing word/document.xml"
        return "binary", f"zip archive without word/document.xml ({len(names)} members, e.g. {names[:3]})"
    if not head:
        return "binary", "empty"
    if b"\x00" in head:
        return "binary", f"NUL byte at offset {head.index(b'\x00')} of first {len(head)} bytes"
    ratio = _printable_ratio(head)
    if ratio <= TEXTUAL_MIN_PRINTABLE:
        return "binary", f"printable ratio {ratio:.3f} <= {TEXTUAL_MIN_PRINTABLE}"
    lowered = head[:1024].lower()
    if path.suffix.lower() in HTML_EXTS:
        return "html", f"printable ratio {ratio:.3f}; extension {path.suffix} is HTML"
    if b"<html" in lowered or b"<!doctype html" in lowered:
        return "html", f"printable ratio {ratio:.3f}; <html or <!doctype html in first 1024 bytes"
    return "text", f"printable ratio {ratio:.3f} > {TEXTUAL_MIN_PRINTABLE}; no HTML markers"


def _printable_ratio(head: bytes) -> float:
    sample = head.decode("latin-1")
    return sum(ch.isprintable() or ch in "\r\n\t\f" for ch in sample) / len(sample)


def magic_label(path: Path) -> tuple[str, str]:
    """(mime type, libmagic's description). libmagic reads the whole file, so it
    sees zip directories and OLE2 streams that our 8KB head does not."""
    try:
        import magic
    except ImportError as exc:  # the C library is missing, not the Python package
        raise UserError(
            "python-magic needs the libmagic C library: `brew install libmagic` on macOS, "
            "`apt install libmagic1` on Debian/Ubuntu."
        ) from exc
    try:
        return magic.from_file(str(path), mime=True), magic.from_file(str(path))
    except magic.MagicException as exc:
        return "application/octet-stream", f"libmagic failed: {exc}"


def extract(path: Path, *, max_pdf_pages: int, max_text_bytes: int, trace: Tracer = OFF) -> Extracted:
    with path.open("rb") as f:
        head = f.read(HEAD_BYTES)
    trace("read", f"first {len(head):,} bytes: {hexdump(head)}")
    if not head:
        raise Skip("empty file")
    kind, why = sniff_explained(path, head)
    trace("sniff", f"kind={kind}  because {why}")
    if kind == "pdf":
        return _pdf(path, max_pdf_pages, trace)
    if kind == "docx":
        return _docx(path, trace)
    if kind in ("html", "text"):
        size = path.stat().st_size
        raw = path.read_bytes()[:max_text_bytes]
        text, encoding = _decode(raw)
        trace("decode", f"read {len(raw):,} of {size:,} bytes (cap {max_text_bytes:,}); decoded as {encoding} "
                        f"-> {len(text):,} chars")
        if kind == "html":
            stripped = _html_text(text)
            trace("html", f"stripped tags, script, style, head: {len(text):,} -> {len(stripped):,} chars")
            text = stripped
        trace("text", trace.text(text))
        return Extracted(kind, [(None, text)], truncated=size > max_text_bytes)
    mime, description = magic_label(path)
    trace("magic", f"libmagic: {mime}  {description}")
    if mime in ("application/octet-stream", "text/plain", "application/x-empty"):
        # Too generic to be worth reporting on its own; keep libmagic's wording too.
        raise Skip(f"unsupported format: {mime} ({description[:60]})")
    raise Skip(f"unsupported format: {mime}")


def _pdf(path: Path, max_pages: int, trace: Tracer) -> Extracted:
    import pypdfium2 as pdfium

    try:
        pdf = pdfium.PdfDocument(path)
    except pdfium.PdfiumError as exc:
        raise Skip(f"unreadable pdf: {exc}") from exc
    try:
        n = len(pdf)
        read = min(n, max_pages)
        trace("pdf", f"{n} pages; reading {read} (max_pdf_pages={max_pages})")
        segs = []
        for i in range(read):
            page = pdf[i]
            textpage = page.get_textpage()
            text = textpage.get_text_range()
            segs.append((i + 1, text))
            trace("pdf", f"page {i + 1}: {len(text):,} chars  {trace.text(text)}")
            textpage.close()
            page.close()
    finally:
        pdf.close()
    return Extracted("pdf", segs, pages=n, truncated=n > max_pages)


def _docx(path: Path, trace: Tracer) -> Extracted:
    import docx

    try:
        d = docx.Document(str(path))
    except Exception as exc:
        raise Skip(f"unreadable docx: {exc}") from exc
    parts = [p.text for p in d.paragraphs]
    n_paragraphs = len(parts)
    for table in d.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    text = "\n".join(parts)
    trace("docx", f"{n_paragraphs} paragraphs, {len(d.tables)} tables ({len(parts) - n_paragraphs} rows) "
                  f"-> {len(text):,} chars")
    trace("text", trace.text(text))
    return Extracted("docx", [(None, text)])


def _decode(raw: bytes) -> tuple[str, str]:
    for enc in ("utf-8", "cp1252"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1"), "latin-1"


class _TextOnly(HTMLParser):
    SKIP: ClassVar[frozenset[str]] = frozenset({"script", "style", "noscript", "head"})
    BLOCK: ClassVar[frozenset[str]] = frozenset(
        {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section", "article"}
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.depth += 1
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.depth:
            self.depth -= 1
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.depth:
            self.out.append(data)


def _html_text(html: str) -> str:
    parser = _TextOnly()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001, S110 - malformed markup; keep whatever was parsed
        pass
    return re.sub(r"\n\s*\n+", "\n\n", "".join(parser.out))
