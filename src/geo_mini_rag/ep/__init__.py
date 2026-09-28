"""Domain parsers for E&P formats, and the registry that routes files to them.

A generic RAG pipeline reads a file and chunks whatever text it finds. That is
wrong for most E&P data: a LAS file is 95% floating point, a SEGY file's only
prose is its 3200-byte textual header, and a shapefile's meaning lives in its
attribute table. Each of those wants a parser that knows the format and emits
a short, well-formed summary plus structured fields.

A handler declares:

    name        short label, stored as the document's `kind`
    matches()   cheap test on the path and first bytes; no full reads
    parse()     returns an Extracted with
                  .segments   text to embed, usually one or two short records
                  .metadata   document-level fields: well, api, field, date...
                  .atomic     True when each segment is already chunk-sized

Handlers run in registration order, first match wins, and anything unclaimed
falls through to the generic extractor (PDF, DOCX, HTML, plain text).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol, runtime_checkable

from geo_mini_rag.rag.extract import Extracted
from geo_mini_rag.rag.trace import Tracer


@runtime_checkable
class Handler(Protocol):
    name: str

    def matches(self, path: Path, head: bytes) -> bool: ...

    def parse(self, path: Path, cfg: dict, trace: Tracer) -> Extracted: ...


HANDLERS: list[Handler] = []


def register(handler: Handler) -> Handler:
    HANDLERS.append(handler)
    return handler


def find(path: Path, head: bytes) -> Handler | None:
    """The handler that claims this file, or None.

    Setting GEO_NO_EP_HANDLERS=1 turns every handler off, so the same corpus
    can be indexed as a generic pipeline would index it. That is the baseline
    arm of the experiment this project exists to run: without it a LAS file is
    423 MB of floating point read as prose, and a SEG-Y or shapefile is binary
    nobody can read at all.
    """
    if os.environ.get("GEO_NO_EP_HANDLERS") == "1":
        return None
    for handler in HANDLERS:
        if handler.matches(path, head):
            return handler
    return None


def _load_builtin() -> None:
    from geo_mini_rag.ep.las import LasHandler
    from geo_mini_rag.ep.segp1 import SegP1Handler
    from geo_mini_rag.ep.segy import SegyHandler
    from geo_mini_rag.ep.shapefile import ShapefileHandler
    from geo_mini_rag.ep.xls import XlsHandler

    register(LasHandler())
    register(SegyHandler())
    register(SegP1Handler())
    register(ShapefileHandler())
    register(XlsHandler())


_load_builtin()
