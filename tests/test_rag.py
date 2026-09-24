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


def test_metadata_rows_expand_lists_and_type_numbers():
    from geo_mini_rag.rag.index import metadata_rows

    rows = metadata_rows("doc1", {
        "well": "NPR #3 #13SX11-11",
        "curve": ["GRD", "RHOB", "CALD"],
        "depth_max": 570.0,
        "depth_step": "0.5",
        "blank": "",
    })
    keys = [(k, v, n) for _, k, v, n in rows]
    assert ("curve", "GRD", None) in keys and ("curve", "RHOB", None) in keys
    assert sum(1 for k, _, _ in keys if k == "curve") == 3, "one row per curve"
    assert ("depth_max", "570.0", 570.0) in keys, "numbers keep a numeric reading"
    assert ("depth_step", "0.5", 0.5) in keys, "numeric strings too"
    assert ("well", "NPR #3 #13SX11-11", None) in keys
    assert not any(k == "blank" for k, _, _ in keys), "empty values are dropped"


def test_idf_weights_rarity():
    from geo_mini_rag.rag.index import _idf

    assert _idf(1, 1000) == pytest.approx(1.0), "a value only one document carries"
    assert _idf(1000, 1000) == 0.0, "a value every document carries is no evidence"
    assert _idf(10, 1000) > _idf(500, 1000)


def _tiny_index(path, rows, dim=3):
    """An index of hand-written vectors: (doc_id, path, text, embedding)."""
    import duckdb

    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE meta (key VARCHAR PRIMARY KEY, value VARCHAR)")
    con.executemany("INSERT INTO meta VALUES (?, ?)",
                    [("embed_model", "test-model"), ("dim", str(dim))])
    con.execute("CREATE TABLE documents (doc_id VARCHAR, path VARCHAR)")
    con.execute("CREATE TABLE doc_meta (doc_id VARCHAR, key VARCHAR, value VARCHAR, num_value DOUBLE)")
    con.execute(f"CREATE TABLE chunks (doc_id VARCHAR, ord INTEGER, page INTEGER,"
                f" text VARCHAR, embedding FLOAT[{dim}])")
    for i, (doc_id, doc_path, text, vector) in enumerate(rows):
        con.execute("INSERT INTO documents VALUES (?, ?)", [doc_id, doc_path])
        con.execute("INSERT INTO chunks VALUES (?, ?, NULL, ?, ?)", [doc_id, i, text, vector])
    con.close()


def test_one_document_cannot_take_every_place_in_the_answer(tmp_path, monkeypatch):
    """A shapefile of 2,111 wells is 452 chunks that read alike; without a limit
    it held ranks 1 to 5 for any question about wells."""
    from geo_mini_rag import openrouter
    from geo_mini_rag.rag import index

    db = tmp_path / "tiny.duckdb"
    _tiny_index(db, [
        *[("big", "layer.shp", f"well {n}", [1.0, 0.0, 0.02 * (5 - n)]) for n in range(5)],
        ("las1", "one.las", "log one", [0.9, 0.1, 0.0]),
        ("las2", "two.las", "log two", [0.85, 0.1, 0.0]),
    ])
    monkeypatch.setattr(openrouter, "embed",
                        lambda *a, **k: openrouter.EmbedResult("test-model", "test-model", [[1.0, 0.0, 0.0]]))

    cfg = {"retrieve": {"metadata_boost": 0.1, "per_document": 0}}
    hits, _ = index.search("q", 4, db, cfg=cfg)
    assert {h.path for h in hits} == {"layer.shp"}, "unlimited, the big file takes them all"

    cfg["retrieve"]["per_document"] = 2
    hits, _ = index.search("q", 4, db, cfg=cfg)
    assert [h.path for h in hits] == ["layer.shp", "layer.shp", "one.las", "two.las"]
    assert [h.rank for h in hits] == [1, 2, 3, 4]
    assert hits[0].score >= hits[-1].score, "still ordered by score"


def test_an_identifier_in_a_question_is_looked_up_not_ranked(tmp_path, monkeypatch):
    """`well 4902511080` scores 0.324 against the log that carries it and 0.729
    against a page of unrelated digits. Ranking cannot find it; a lookup can."""
    import duckdb

    from geo_mini_rag import openrouter
    from geo_mini_rag.rag import index

    db = tmp_path / "ids.duckdb"
    _tiny_index(db, [
        ("noise", "digits.txt", "81374982496e85828 4041 9903", [1.0, 0.0, 0.0]),
        ("well", "log.las", "Well log header. API number: 490251108000", [0.3, 0.9, 0.0]),
    ])
    con = duckdb.connect(str(db))
    con.execute("INSERT INTO doc_meta VALUES ('well', 'api', '4902511080', NULL)")
    con.close()
    monkeypatch.setattr(openrouter, "embed",
                        lambda *a, **k: openrouter.EmbedResult("m", "m", [[1.0, 0.0, 0.0]]))

    cfg = {"retrieve": {"metadata_boost": 0.1, "identifier_lookup": False}}
    hits, _ = index.search("well 4902511080", 2, db, cfg=cfg)
    assert hits[0].path == "digits.txt", "ranked, the digits win"

    cfg["retrieve"]["identifier_lookup"] = True
    hits, _ = index.search("well 4902511080", 2, db, cfg=cfg)
    assert [h.path for h in hits] == ["log.las"], "looked up, only the well that carries it"
    assert "api=4902511080" in hits[0].matched


def test_an_identifier_the_index_does_not_hold_changes_nothing(tmp_path, monkeypatch):
    """Otherwise a question about a well we lack would return nothing at all."""
    from geo_mini_rag import openrouter
    from geo_mini_rag.rag import index

    db = tmp_path / "absent.duckdb"
    _tiny_index(db, [("a", "one.txt", "something", [1.0, 0.0, 0.0])])
    monkeypatch.setattr(openrouter, "embed",
                        lambda *a, **k: openrouter.EmbedResult("m", "m", [[1.0, 0.0, 0.0]]))

    hits, _ = index.search("what about well 4902599999", 2, db,
                           cfg={"retrieve": {"identifier_lookup": True}})
    assert [h.path for h in hits] == ["one.txt"]
