"""Chunking over document elements.

Following Unstructured's model: partitioning yields elements — paragraphs,
table rows, pages — and chunking combines sequential elements up to
`max_characters` rather than cutting raw text at fixed offsets. Overlap is
applied only where a single element is too large to fit and has to be split,
which is the one case where a boundary falls mid-sentence.

Keeping page breaks means a chunk never mixes text from two pages, so the page
a citation names is the page the text came from.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

PARAGRAPH_BREAK = re.compile(r"\n\s*\n+")
WHITESPACE = re.compile(r"[ \t\f\r]+")


@dataclass
class Element:
    text: str
    page: int | None = None
    atomic: bool = False   # a handler's record: keep it whole if it fits


@dataclass
class Chunk:
    ord: int
    text: str
    page: int | None    # page the chunk starts on
    start: int = 0      # offsets within the element stream, for tracing
    end: int = 0


def elements_from(segments: Sequence[tuple[int | None, str]], *, atomic: bool = False) -> list[Element]:
    """Split extracted segments into elements: paragraphs, or whole records."""
    out: list[Element] = []
    for page, raw in segments:
        text = WHITESPACE.sub(" ", raw or "")
        if atomic:
            body = text.strip()
            if body:
                out.append(Element(body, page, atomic=True))
            continue
        for part in PARAGRAPH_BREAK.split(text):
            body = part.strip()
            if body:
                out.append(Element(body, page))
    return out


def chunk_elements(
    elements: Sequence[Element],
    *,
    max_characters: int,
    overlap: int,
    respect_page_breaks: bool = True,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_page: int | None = None
    position = 0
    start = 0

    def flush() -> None:
        nonlocal buffer, buffer_page, start
        if buffer:
            chunks.append(Chunk(len(chunks), "\n\n".join(buffer), buffer_page, start, position))
            buffer = []
            buffer_page = None
        start = position

    for element in elements:
        if respect_page_breaks and buffer and element.page != buffer_page:
            flush()
        if len(element.text) > max_characters:
            flush()
            for piece in _split(element.text, max_characters, overlap):
                chunks.append(Chunk(len(chunks), piece, element.page, position, position + len(piece)))
                position += len(piece)
            start = position
            continue
        projected = sum(len(b) for b in buffer) + 2 * len(buffer) + len(element.text)
        if buffer and projected > max_characters:
            flush()
        if not buffer:
            buffer_page = element.page
        buffer.append(element.text)
        position += len(element.text)
    flush()
    return chunks


def _split(text: str, max_characters: int, overlap: int) -> list[str]:
    """One oversized element, cut on whitespace, with overlap between pieces."""
    pieces: list[str] = []
    i = 0
    while i < len(text):
        end = min(i + max_characters, len(text))
        if end < len(text):
            brk = text.rfind(" ", i + max_characters // 2, end)
            if brk != -1:
                end = brk
        body = text[i:end].strip()
        if body:
            pieces.append(body)
        if end >= len(text):
            break
        i = max(end - overlap, i + 1)
    return pieces


def chunks_from(
    segments: Sequence[tuple[int | None, str]],
    *,
    max_characters: int,
    overlap: int,
    atomic: bool = False,
    respect_page_breaks: bool = True,
) -> list[Chunk]:
    """Segments as extraction produced them, to chunks ready for embedding."""
    return chunk_elements(
        elements_from(segments, atomic=atomic),
        max_characters=max_characters,
        overlap=overlap,
        respect_page_breaks=respect_page_breaks,
    )
