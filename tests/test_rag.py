import pytest

from geo_mini_rag.rag.answer import build_messages
from geo_mini_rag.rag.chunk import chunk_segments
from geo_mini_rag.rag.extract import Skip, extract, sniff
from geo_mini_rag.rag.index import Hit


def test_chunks_overlap_and_track_pages():
    segs = [(1, "alpha " * 300), (2, "beta " * 300)]
    chunks = chunk_segments(segs, chars=500, overlap=100)
    assert all(len(c.text) <= 500 for c in chunks)
    assert chunks[0].page == 1
    assert chunks[-1].page == 2
    assert [c.ord for c in chunks] == list(range(len(chunks)))
    # consecutive chunks share text
    assert chunks[0].text[-50:].split()[-1] in chunks[1].text


def test_chunking_empty_text_yields_nothing():
    assert chunk_segments([(1, "   \n\n "), (2, "")], chars=500, overlap=100) == []


def test_sniff_ignores_extension(tmp_path):
    fake = tmp_path / "report.pdf"
    fake.write_bytes(b"<html><body><p>hello</p></body></html>")
    assert sniff(fake, fake.read_bytes()) == "html"
    binary = tmp_path / "notes.txt"
    binary.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 100)
    assert sniff(binary, binary.read_bytes()) == "binary"


def test_extract_html_drops_scripts(tmp_path):
    page = tmp_path / "page.htm"
    page.write_text("<html><head><title>t</title></head><body><script>var x=1;</script><p>Well 42</p></body></html>")
    ex = extract(page, max_pdf_pages=10, max_text_bytes=10_000)
    assert ex.kind == "html"
    assert "Well 42" in ex.segments[0][1]
    assert "var x" not in ex.segments[0][1]


def test_legacy_office_is_named_by_libmagic(tmp_path):
    doc = tmp_path / "old.doc"
    doc.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 600)
    with pytest.raises(Skip, match="ole-storage|msword"):
        extract(doc, max_pdf_pages=10, max_text_bytes=10_000)


def test_extract_skips_empty(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_bytes(b"")
    with pytest.raises(Skip, match="empty"):
        extract(empty, max_pdf_pages=10, max_text_bytes=10_000)


def test_prompt_numbers_sources():
    hits = [Hit(1, 0.9, "a.pdf", 3, "first"), Hit(2, 0.8, "b.txt", None, "second")]
    msgs = build_messages("q?", hits)
    assert "[1] (a.pdf, page 3)" in msgs[1]["content"]
    assert "[2] (b.txt)" in msgs[1]["content"]


def test_missing_index_is_a_user_error(tmp_path):
    from geo_mini_rag.errors import UserError
    from geo_mini_rag.rag.index import connect

    with pytest.raises(UserError, match="no index at"):
        connect(tmp_path / "absent.duckdb", read_only=True)
