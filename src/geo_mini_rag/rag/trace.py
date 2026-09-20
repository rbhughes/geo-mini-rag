"""Step-by-step tracing for ingest. Off unless a caller passes an emit function."""

from __future__ import annotations

from collections.abc import Callable

Emit = Callable[[str, str], None]  # (stage, message)


class Tracer:
    def __init__(self, emit: Emit | None = None, preview_chars: int = 160) -> None:
        self._emit = emit
        self.preview_chars = preview_chars   # 0 means print text in full

    @property
    def on(self) -> bool:
        return self._emit is not None

    def __call__(self, stage: str, message: str) -> None:
        if self._emit is not None:
            self._emit(stage, message)

    def text(self, s: str) -> str:
        """A text sample for trace output: head and tail with the middle elided, repr-escaped."""
        n = self.preview_chars
        if n <= 0 or len(s) <= n:
            return repr(s)
        half = n // 2
        return f"{s[:half]!r} … [{len(s) - 2 * half:,} chars] … {s[-half:]!r}"


OFF = Tracer()


def hexdump(b: bytes, width: int = 16) -> str:
    head = b[:width]
    hexed = " ".join(f"{x:02x}" for x in head)
    ascii_ = "".join(chr(x) if 32 <= x < 127 else "." for x in head)
    return f"{hexed}  |{ascii_}|"
