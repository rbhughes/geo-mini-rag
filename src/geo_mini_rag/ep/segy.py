"""SEG-Y: read the headers, never the traces.

A SEG-Y file opens with two headers before any seismic data:

    bytes 0-3199      textual header, 40 cards of 80 characters, historically
                      EBCDIC. Free text: client, area, line, shot and processing
                      dates, the processing sequence. This is the part with
                      words in it, and the reason a seismic volume belongs in a
                      document index at all.
    bytes 3200-3599   binary header, fixed big-endian fields: sample interval,
                      samples per trace, format code, measurement system.

The traces after that are numbers — often gigabytes of them — and are never
read: a handler opens 3,600 bytes regardless of file size.

Standard library only; `cp037` is the EBCDIC code page SEG-Y writers use, with
`cp500` seen in some international files.

    from geo_mini_rag.ep.segy import read_headers
    header = read_headers("survey.sgy")
    header.text                       # the 40 cards
    header.binary["sample_interval_us"]

"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from geo_mini_rag.rag.extract import Extracted, Skip
from geo_mini_rag.rag.trace import Tracer

TEXTUAL_BYTES = 3200
BINARY_BYTES = 400
CARD_LENGTH = 80
CARDS = 40
ENCODINGS = ("cp037", "cp500", "ascii")
PRINTABLE_FLOOR = 0.75   # measured: real headers 1.00, prose as EBCDIC 0.24, random bytes 0.36

ByteOrder = Literal["big", "little"]

# (name, zero-based offset in the 400-byte block, struct code). The standard
# numbers these bytes 3201-3600 of the file, so "3217-3218 sample interval" is
# offset 16 here. Verified against files whose textual header states the sample
# rate: a header reading "SAMPLE RATE 4 MS" decodes to sample_interval_us 4000.
BINARY_FIELDS: tuple[tuple[str, int, str], ...] = (
    ("job_id", 0, "i"),
    ("line_number", 4, "i"),
    ("reel_number", 8, "i"),
    ("traces_per_ensemble", 12, "h"),
    ("aux_traces_per_ensemble", 14, "h"),
    ("sample_interval_us", 16, "h"),
    ("sample_interval_us_original", 18, "h"),
    ("samples_per_trace", 20, "h"),
    ("samples_per_trace_original", 22, "h"),
    ("format_code", 24, "h"),
    ("ensemble_fold", 26, "h"),
    ("trace_sorting_code", 28, "h"),
    ("vertical_sum_code", 30, "h"),
    ("sweep_frequency_start_hz", 32, "h"),
    ("sweep_frequency_end_hz", 34, "h"),
    ("sweep_length_ms", 36, "h"),
    ("sweep_type_code", 38, "h"),
    ("correlated_traces_code", 48, "h"),
    ("binary_gain_recovered_code", 50, "h"),
    ("amplitude_recovery_code", 52, "h"),
    ("measurement_system_code", 54, "h"),
    ("impulse_polarity_code", 56, "h"),
    ("vibratory_polarity_code", 58, "h"),
    ("segy_revision", 300, "h"),
    ("fixed_length_trace_flag", 302, "h"),
    ("extended_textual_headers", 304, "h"),
)

FORMAT_CODES = {
    1: "4-byte IBM floating point",
    2: "4-byte two's complement integer",
    3: "2-byte two's complement integer",
    4: "4-byte fixed point with gain (obsolete)",
    5: "4-byte IEEE floating point",
    6: "8-byte IEEE floating point",
    7: "3-byte two's complement integer",
    8: "1-byte two's complement integer",
    9: "8-byte two's complement integer",
    10: "4-byte unsigned integer",
    11: "2-byte unsigned integer",
    12: "8-byte unsigned integer",
    15: "3-byte unsigned integer",
    16: "1-byte unsigned integer",
}
MEASUREMENT_SYSTEMS = {1: "meters", 2: "feet"}

# What each numeric field is allowed to hold. Most of the binary block is
# optional, and a writer that never set a field leaves whatever was in memory
# there, so 45 of 950 facts read off this corpus were things like an ensemble
# fold of -13,922 or a reel number of -1,868,250,301. A count cannot be
# negative and a sample interval cannot be zero; a value outside its range is
# not a measurement and is dropped rather than stored.
NUMERIC_LIMITS: dict[str, tuple[int, int]] = {
    "line_number": (0, 2**31 - 1),
    "reel_number": (0, 2**31 - 1),
    "traces_per_ensemble": (0, 32767),
    "aux_traces_per_ensemble": (0, 32767),
    "sample_interval_us": (1, 32767),
    "sample_interval_us_original": (1, 32767),
    "samples_per_trace": (1, 32767),
    "samples_per_trace_original": (1, 32767),
    "ensemble_fold": (0, 32767),
    "sweep_frequency_start_hz": (0, 32767),
    "sweep_frequency_end_hz": (0, 32767),
    "sweep_length_ms": (0, 32767),
}
SORTING_CODES = {
    -1: "other", 0: "unknown", 1: "as recorded", 2: "CDP ensemble",
    3: "single fold continuous profile", 4: "horizontally stacked",
    5: "common source point", 6: "common receiver point", 7: "common offset point",
    8: "common mid-point", 9: "common conversion point",
}

# Labels conventionally written on the C-cards. Free text is free text, so only
# these are lifted into facts; anything else stays in the chunk to be searched
# semantically rather than becoming a half-parsed field.
TEXT_LABELS = {
    "CLIENT": "client", "COMPANY": "client", "AREA": "area", "PROSPECT": "area",
    "FIELD": "field", "LINE": "line", "SURVEY": "survey", "CONTRACTOR": "contractor",
    "SHOT BY": "shot_by", "PROCESSED BY": "processed_by", "DATUM": "datum",
    "PROJECTION": "projection", "MAP PROJECTION": "projection", "ZONE": "zone",
    "MEAS UNITS": "units", "UNITS": "units",
}
LABEL = re.compile(
    r"(?P<label>[A-Z][A-Z &/.]{2,18}?)\s*:\s*(?P<value>[^:]*?)"
    r"(?=\s{2,}[A-Z][A-Z &/.]{2,18}\s*:|$)"
)
CARD_NUMBER = re.compile(r"^C\s*\d{0,2}\s?")
# Half the headers in this corpus write the line name without a colon, on a
# card of their own: "LINE 700", "LINE CPB-3", "LINE 71-117-277  LOUISIANA".
# Only the start of a card counts, and only across a single space: "LINE
# NUMBER:  17   LONG   LINE RECORD" is a byte-offset map, and the template
# card "CLIENT                COMPANY" is an unfilled form, not a client named
# COMPANY. A value carrying a colon is a label of its own, so it is not one.
# The value also stops at the next column's name, because a template card fills
# one column and leaves the next heading standing right beside it:
#   C 1 CLIENT ENCANA OIL & GAS (USA) INC. COMPANY          CREW NO
# without this the client is read as "... INC. COMPANY".
_COLUMN_NAMES = "|".join(
    re.escape(name) for name in sorted(TEXT_LABELS, key=len, reverse=True)
)
BARE_LABEL = re.compile(
    r"^(?P<label>[A-Z][A-Z &/.]{2,18}?) (?P<value>\S[^\n]*?)"
    rf"(?=\s{{2,}}|\s+(?:{_COLUMN_NAMES})\b|$)"
)


class NotSegy(ValueError):
    """The file does not hold SEG-Y headers."""


@dataclass
class SegyHeader:
    path: Path
    encoding: str
    cards: list[str] = field(default_factory=list)
    binary: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        """The textual header as it reads on the page, blank cards dropped."""
        return "\n".join(card.rstrip() for card in self.cards if card.strip())

    def described(self) -> dict[str, object]:
        """Coded fields spelled out, derived quantities as numbers.

        Numbers stay numbers: a fact stored as "17988" cannot answer
        `--where trace_length_ms>10000`.
        """
        out: dict[str, object] = {}
        if (code := self.binary.get("format_code")) in FORMAT_CODES:
            out["sample_format"] = FORMAT_CODES[code]
        if (code := self.binary.get("measurement_system_code")) in MEASUREMENT_SYSTEMS:
            out["measurement_system"] = MEASUREMENT_SYSTEMS[code]
        if (code := self.binary.get("trace_sorting_code")) in SORTING_CODES:
            out["trace_sorting"] = SORTING_CODES[code]
        interval = self.binary.get("sample_interval_us")
        samples = self.binary.get("samples_per_trace")
        if interval:
            out["sample_interval_ms"] = interval / 1000
        if interval and samples:
            out["trace_length_ms"] = interval * samples / 1000
        return out

    def numbers(self) -> dict[str, int]:
        """Binary fields that hold a usable measurement.

        The coded fields are left out on purpose: `trace_sorting_code: 4` is
        not something anyone searches for, and `described()` already offers it
        as "horizontally stacked". What remains is counts and intervals, each
        inside the range its field allows.
        """
        return {
            name: value
            for name, value in self.binary.items()
            if name in NUMERIC_LIMITS
            and NUMERIC_LIMITS[name][0] <= value <= NUMERIC_LIMITS[name][1]
        }

    def labels(self) -> dict[str, str]:
        """Best-effort fields off the C-cards, limited to conventional labels."""
        found: dict[str, str] = {}
        for card in self.cards:
            body = CARD_NUMBER.sub("", card)
            for match in LABEL.finditer(body):
                key = TEXT_LABELS.get(" ".join(match["label"].split()))
                value = " ".join(match["value"].split()).strip(" .-")
                if key and value and key not in found:
                    found[key] = value
            if bare := BARE_LABEL.match(body):
                key = TEXT_LABELS.get(" ".join(bare["label"].split()))
                value = " ".join(bare["value"].split()).strip(" .-")
                if key and value and ":" not in value and key not in found:
                    found[key] = value
        return found


def _printable_ratio(text: str) -> float:
    """How much of a decoding is printable ASCII, which is what a header is.

    Counting `str.isalnum` instead would score arbitrary bytes decoded as EBCDIC
    highly, because accented letters are alphabetic to Python.
    """
    return sum(1 for ch in text if 32 <= ord(ch) < 127) / len(text) if text else 0.0


def decode_textual(raw: bytes, encoding: str | None = None) -> tuple[str, str]:
    """(text, encoding). The header does not say whether it is EBCDIC or ASCII,
    so every candidate is decoded and the one that reads as text wins."""
    if encoding:
        if encoding not in ENCODINGS:
            raise NotSegy(f"unsupported encoding {encoding!r}; use one of {', '.join(ENCODINGS)}")
        return raw.decode(encoding, errors="replace"), encoding

    scored = [
        (_printable_ratio(text := raw.decode(candidate, errors="replace")), -n, candidate, text)
        for n, candidate in enumerate(ENCODINGS)
    ]
    ratio, _, winner, text = max(scored)
    if ratio < PRINTABLE_FLOOR:
        raise NotSegy(
            f"textual header does not decode as EBCDIC or ASCII "
            f"({ratio:.0%} printable, {PRINTABLE_FLOOR:.0%} needed)"
        )
    return text, winner


def split_cards(text: str) -> list[str]:
    """40 fixed-width cards; the header carries no line breaks of its own."""
    return [
        text[n * CARD_LENGTH : (n + 1) * CARD_LENGTH].replace("\x00", " ").rstrip()
        for n in range(CARDS)
    ]


def decode_binary(raw: bytes, byteorder: ByteOrder = "big") -> dict[str, int]:
    """Named fields from the binary header. Zeros mean "not recorded" in nearly
    every field, so they are left out rather than stored as facts."""
    prefix = ">" if byteorder == "big" else "<"
    values: dict[str, int] = {}
    for name, offset, code in BINARY_FIELDS:
        fmt = prefix + code
        if offset + struct.calcsize(fmt) > len(raw):
            continue
        (value,) = struct.unpack_from(fmt, raw, offset)
        if value:
            values[name] = value
    return values


def implausible(binary: dict[str, int]) -> str | None:
    """Why this binary header cannot be SEG-Y, or None if it might be.

    Any text file longer than 3,200 bytes decodes as a "textual header", so the
    binary block is what identifies the format: its fields have ranges, and
    ASCII read as big-endian integers lands far outside them.
    """
    code = binary.get("format_code")
    if code not in FORMAT_CODES:
        return f"sample format code {code} is not one of {sorted(FORMAT_CODES)}"
    if not 0 < binary.get("sample_interval_us", 0) <= 32767:
        return f"sample interval {binary.get('sample_interval_us', 0)} us is out of range"
    if not 0 < binary.get("samples_per_trace", 0) <= 32767:
        return f"samples per trace {binary.get('samples_per_trace', 0)} is out of range"
    return None


def read_headers(
    path: str | Path,
    *,
    strict: bool = True,
    byteorder: ByteOrder = "big",
    encoding: str | None = None,
) -> SegyHeader:
    """Read both headers. Reads 3,600 bytes; trace data is never touched.

    `strict=False` keeps a file whose binary block is damaged but whose textual
    header still reads — that happens, and the text is the part worth indexing.
    """
    path = Path(path)
    with path.open("rb") as f:
        head = f.read(TEXTUAL_BYTES + BINARY_BYTES)
    if len(head) < TEXTUAL_BYTES + BINARY_BYTES:
        raise NotSegy(
            f"{path.name} is {len(head):,} bytes, too short for the "
            f"{TEXTUAL_BYTES + BINARY_BYTES:,}-byte SEG-Y header"
        )

    text, used = decode_textual(head[:TEXTUAL_BYTES], encoding)
    binary = decode_binary(head[TEXTUAL_BYTES:], byteorder)
    warnings: list[str] = []
    if complaint := implausible(binary):
        if strict:
            raise NotSegy(f"{path.name} is not SEG-Y: {complaint}")
        warnings.append(f"binary header is implausible: {complaint}")

    header = SegyHeader(path=path, encoding=used, cards=split_cards(text), binary=binary)
    if not header.text.strip():
        warnings.append("textual header is blank")
    header.warnings = warnings
    return header


def summary(header: SegyHeader) -> str:
    """Both headers as one readable block, for the command line."""
    lines = [f"{header.path.name} — textual header ({header.encoding})", "", header.text]
    if header.binary:
        lines += ["", "Binary header"] + [f"  {k}: {v}" for k, v in header.numbers().items()]
    if described := header.described():
        lines += ["", "Meaning"] + [f"  {k}: {v}" for k, v in described.items()]
    if labels := header.labels():
        lines += ["", "Labels"] + [f"  {k}: {v}" for k, v in labels.items()]
    if header.warnings:
        lines += ["", "Warnings"] + [f"  {w}" for w in header.warnings]
    return "\n".join(lines)


class SegyHandler:
    """Index the words in a seismic volume, not its samples."""

    name = "segy"
    extensions = (".sgy", ".segy")

    def matches(self, path: Path, head: bytes) -> bool:
        return path.suffix.lower() in self.extensions

    def parse(self, path: Path, cfg: dict, trace: Tracer) -> Extracted:
        try:
            header = read_headers(path, strict=False)
        except (NotSegy, OSError) as exc:
            raise Skip(f"unreadable segy: {exc}") from exc
        if not header.text.strip():
            raise Skip("segy textual header is blank")

        meta: dict[str, object] = {"text_encoding": header.encoding}
        meta.update(header.labels())
        meta.update(header.numbers())
        meta.update(header.described())

        card_text = f"Seismic survey header (SEG-Y) for {path.name}.\n{header.text}"
        acquisition = _acquisition_text(path, header)
        trace("segy", f"{header.encoding}, {len(header.binary)} binary fields, "
                      f"{len(header.labels())} labels, {len(header.warnings)} warnings")
        return Extracted(
            kind="segy",
            segments=[(None, card_text), (None, acquisition)],
            metadata=meta,
            atomic=True,
            notes=list(header.warnings),
        )


def _acquisition_text(path: Path, header: SegyHeader) -> str:
    """The binary header as a sentence, so it can be retrieved semantically."""
    described = header.described()
    lines = [f"Recording parameters in {path.name}:"]
    for key, label in (
        ("sample_interval_ms", "Sample interval (ms)"),
        ("trace_length_ms", "Trace length (ms)"),
        ("sample_format", "Sample format"),
        ("measurement_system", "Measurement system"),
        ("trace_sorting", "Trace sorting"),
    ):
        if key in described:
            lines.append(f"{label}: {described[key]}")
    for key, label in (
        ("samples_per_trace", "Samples per trace"),
        ("ensemble_fold", "Ensemble fold"),
        ("traces_per_ensemble", "Traces per ensemble"),
        ("sweep_frequency_start_hz", "Sweep start (Hz)"),
        ("sweep_frequency_end_hz", "Sweep end (Hz)"),
        ("sweep_length_ms", "Sweep length (ms)"),
        ("line_number", "Line number"),
        ("reel_number", "Reel number"),
    ):
        if key in header.binary:
            lines.append(f"{label}: {header.binary[key]}")
    return "\n".join(lines)
