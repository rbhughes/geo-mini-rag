"""Fixed-size character chunks with overlap, breaking on whitespace where possible."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class Chunk:
    ord: int
    text: str
    page: int | None  # page where the chunk starts, when the source has pages
    start: int = 0  # offsets into the joined, whitespace-normalized text
    end: int = 0


def join_segments(
    segments: Sequence[tuple[int | None, str]],
) -> tuple[str, list[tuple[int, int | None]]]:
    """Normalize whitespace, drop empty segments, and join with blank lines.

    Returns the joined text and (offset, page) where each kept segment starts.
    """
    starts: list[tuple[int, int | None]] = []
    parts: list[str] = []
    pos = 0
    for page, text in segments:
        text = re.sub(r"[ \t\f\r]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
        if not text:
            continue
        starts.append((pos, page))
        parts.append(text)
        pos += len(text) + 2
    return "\n\n".join(parts), starts


def chunk_segments(
    segments: Sequence[tuple[int | None, str]], *, chars: int, overlap: int
) -> list[Chunk]:
    full, starts = join_segments(segments)
    chunks: list[Chunk] = []
    i = 0
    while i < len(full):
        end = min(i + chars, len(full))
        if end < len(full):
            brk = full.rfind(" ", i + chars // 2, end)
            if brk != -1:
                end = brk
        body = full[i:end].strip()
        if body:
            chunks.append(Chunk(len(chunks), body, _page_at(starts, i), i, end))
        if end >= len(full):
            break
        i = max(end - overlap, i + 1)
    # print("................")
    # print(chunks)
    # print("................")
    return chunks


def chunks_from(segments: Sequence[tuple[int | None, str]], *, chars: int, overlap: int,
                atomic: bool = False) -> list[Chunk]:
    """Chunk a document. `atomic` keeps each segment whole (a LAS header card,
    a SEGY textual header), splitting one only if it is far past the chunk size."""
    if not atomic:
        return chunk_segments(segments, chars=chars, overlap=overlap)
    chunks: list[Chunk] = []
    for page, text in segments:
        text = text.strip()
        if not text:
            continue
        if len(text) <= chars * 2:
            chunks.append(Chunk(len(chunks), text, page, 0, len(text)))
            continue
        for part in chunk_segments([(page, text)], chars=chars, overlap=overlap):
            chunks.append(Chunk(len(chunks), part.text, page, part.start, part.end))
    return chunks


def _page_at(starts: list[tuple[int, int | None]], offset: int) -> int | None:
    page = None
    for start, p in starts:
        if start > offset:
            break
        page = p
    return page
