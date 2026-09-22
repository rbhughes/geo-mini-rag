import pytest

from geo_mini_rag.rag.answer import build_messages
from geo_mini_rag.rag.chunk import chunks_from
from geo_mini_rag.rag.extract import Skip, extract, sniff
from geo_mini_rag.rag.index import Hit


def test_oversized_elements_split_with_overlap_and_keep_pages():
    segs = [(1, "alpha " * 300), (2, "beta " * 300)]
    chunks = chunks_from(segs, max_characters=500, overlap=100)
    assert all(len(c.text) <= 500 for c in chunks)
    assert chunks[0].page == 1
    assert chunks[-1].page == 2
    assert [c.ord for c in chunks] == list(range(len(chunks)))
    assert chunks[0].text[-50:].split()[-1] in chunks[1].text, "split pieces overlap"


def test_paragraphs_are_combined_not_cut():
    paragraphs = "\n\n".join(f"Paragraph {n} about the well and the completion report." for n in range(6))
    chunks = chunks_from([(None, paragraphs)], max_characters=200, overlap=50)
    assert len(chunks) > 1
    for chunk in chunks:
        for line in chunk.text.split("\n\n"):
            assert line.startswith("Paragraph"), "elements are kept whole"
            assert line.endswith("report."), "elements are kept whole"


def test_page_breaks_are_not_crossed():
    chunks = chunks_from([(1, "short one"), (2, "short two")], max_characters=1000, overlap=0)
    assert len(chunks) == 2, "two pages, two chunks, even though both would fit in one"
    assert [c.page for c in chunks] == [1, 2]


def test_chunking_empty_text_yields_nothing():
    assert chunks_from([(1, "   \n\n "), (2, "")], max_characters=500, overlap=100) == []


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


def test_needs_ocr_only_for_pdfs_without_a_text_layer():
    from geo_mini_rag.rag.extract import Extracted
    from geo_mini_rag.rag.parse import needs_ocr

    cfg = {"extract": {"ocr_below_chars_per_page": 50}}
    scan = Extracted("pdf", [(1, ""), (2, "  ")])
    typed = Extracted("pdf", [(1, "The lease is made between the parties named below. " * 5)])
    text_file = Extracted("text", [(None, "")])
    assert needs_ocr(scan, cfg) is True
    assert needs_ocr(typed, cfg) is False
    assert needs_ocr(text_file, cfg) is False, "only PDFs have a text layer to be missing"


def test_ocr_output_mirrors_the_source_layout():
    from geo_mini_rag import settings
    from geo_mini_rag.rag import ocr

    out = ocr.output_path(settings.ROOT / "data/raw/leases/WY_00537.pdf")
    assert out == settings.OCR_DIR / "data/raw/leases/WY_00537.pdf"
