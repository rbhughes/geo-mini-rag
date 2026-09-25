import struct

import pytest

from geo_mini_rag import settings
from geo_mini_rag.ep.segy import (
    BINARY_BYTES,
    TEXTUAL_BYTES,
    NotSegy,
    SegyHandler,
    read_headers,
)
from geo_mini_rag.rag.extract import Skip
from geo_mini_rag.rag.store import metadata_rows
from geo_mini_rag.rag.trace import OFF

CARDS = [
    "C01CLIENT: LITHOPROBE   AREA: ABITIBI - GRENVILLE '93  LINE:55",
    "C02SHOT BY: ENERTEC GEOPHYSICAL               DATE: AUG 1993",
    "C03PROCESSED BY: CGG GEOPHYSICS CANADA LTD.",
    "C09SAMPLE RATE..................4 MS",
]


def segy_bytes(*, encoding: str = "cp037", interval: int = 4000, samples: int = 4497,
               fmt: int = 1, system: int = 1, byteorder: str = ">") -> bytes:
    text = "".join(card.ljust(80) for card in CARDS).ljust(TEXTUAL_BYTES)[:TEXTUAL_BYTES]
    binary = bytearray(BINARY_BYTES)
    struct.pack_into(byteorder + "h", binary, 16, interval)
    struct.pack_into(byteorder + "h", binary, 20, samples)
    struct.pack_into(byteorder + "h", binary, 24, fmt)
    struct.pack_into(byteorder + "h", binary, 54, system)
    return text.encode(encoding) + bytes(binary)


@pytest.fixture
def survey(tmp_path):
    path = tmp_path / "line55.sgy"
    path.write_bytes(segy_bytes())
    return path


def test_offsets_agree_with_the_textual_header(survey):
    """The card says SAMPLE RATE 4 MS; the binary header must agree."""
    header = read_headers(survey)
    assert header.binary["sample_interval_us"] == 4000
    assert header.binary["samples_per_trace"] == 4497
    assert header.binary["format_code"] == 1
    described = header.described()
    assert described["sample_interval_ms"] == 4.0
    assert described["sample_format"] == "4-byte IBM floating point"


def test_derived_values_are_numbers_not_strings(survey):
    """Facts must be filterable: --where trace_length_ms>10000."""
    described = read_headers(survey).described()
    assert isinstance(described["trace_length_ms"], float)
    assert described["trace_length_ms"] == 17988.0


@pytest.mark.parametrize("encoding", ["cp037", "cp500", "ascii"])
def test_both_ebcdic_pages_and_ascii_are_read(tmp_path, encoding):
    path = tmp_path / f"{encoding}.sgy"
    path.write_bytes(segy_bytes(encoding=encoding))
    header = read_headers(path)
    assert "LITHOPROBE" in header.text


def test_byte_swapped_files_can_be_read(tmp_path):
    path = tmp_path / "swapped.sgy"
    path.write_bytes(segy_bytes(byteorder="<"))
    with pytest.raises(NotSegy):
        read_headers(path)
    header = read_headers(path, byteorder="little")
    assert header.binary["sample_interval_us"] == 4000


def test_labels_come_off_the_cards(survey):
    labels = read_headers(survey).labels()
    assert labels["client"] == "LITHOPROBE"
    assert labels["area"] == "ABITIBI - GRENVILLE '93"
    assert labels["line"] == "55"
    assert labels["shot_by"] == "ENERTEC GEOPHYSICAL"
    assert labels["processed_by"] == "CGG GEOPHYSICS CANADA LTD"


def test_text_files_are_not_mistaken_for_segy(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("The unit operator shall drill the initial test well. " * 100)
    with pytest.raises(NotSegy, match="not SEG-Y|too short"):
        read_headers(path)


def test_short_and_unreadable_files_are_rejected(tmp_path):
    short = tmp_path / "stub.sgy"
    short.write_bytes(b"\xc3\xf0\xf1" + b"\x40" * 500)
    with pytest.raises(NotSegy, match="too short"):
        read_headers(short)

    blob = tmp_path / "blob.sgy"
    blob.write_bytes(bytes(range(256)) * 20)
    with pytest.raises(NotSegy, match="does not decode"):
        read_headers(blob)


def test_loose_mode_keeps_a_damaged_binary_header(tmp_path):
    path = tmp_path / "damaged.sgy"
    path.write_bytes(segy_bytes(fmt=99, interval=0, samples=0))
    with pytest.raises(NotSegy):
        read_headers(path)
    header = read_headers(path, strict=False)
    assert "LITHOPROBE" in header.text
    assert header.warnings, "the damage is reported rather than hidden"


def test_handler_emits_text_and_typed_facts(survey):
    ex = SegyHandler().parse(survey, settings.load_rag_config(), OFF)
    assert ex.kind == "segy"
    assert ex.atomic and len(ex.segments) == 2
    assert "LITHOPROBE" in ex.segments[0][1]
    assert "Sample interval (ms): 4.0" in ex.segments[1][1]

    numeric = {k: n for _, k, _, n in metadata_rows("d", ex.metadata) if n is not None}
    assert numeric["sample_interval_us"] == 4000
    assert numeric["trace_length_ms"] == 17988.0
    assert ex.metadata["client"] == "LITHOPROBE"


def test_handler_claims_only_segy_extensions(tmp_path):
    handler = SegyHandler()
    assert handler.matches(tmp_path / "a.sgy", b"")
    assert handler.matches(tmp_path / "a.SEGY", b"")
    assert not handler.matches(tmp_path / "a.seg", b""), "that is SEG-P1's business"
    assert not handler.matches(tmp_path / "a.las", b"")


def test_handler_skips_a_file_it_cannot_read(tmp_path):
    path = tmp_path / "empty.sgy"
    path.write_bytes(b"")
    with pytest.raises(Skip, match="unreadable segy"):
        SegyHandler().parse(path, settings.load_rag_config(), OFF)
