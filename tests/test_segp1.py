import pytest

from geo_mini_rag import settings
from geo_mini_rag.ep.segp1 import NotSegP1, SegP1Handler, read_survey
from geo_mini_rag.rag.extract import Skip
from geo_mini_rag.rag.store import metadata_rows
from geo_mini_rag.rag.trace import OFF

SURVEY = """H                              SEISMIC SURVEY DATA
 -------------------------------------------------------------------------------
 CLIENT      : ARCTIC OIL
 PROSPECT    :
 UNITS       : DECIMETERS
 SURVEYOR    :
 FILE NUMBER :MDH
 -------------------------------------------------------------------------------
 <.....LINE.....><..SP..>.<..LAT..><..LONG..><..EA..><..NO..><ELV><...><....>
 101                  101                     405080864496387 480000000000000
 101                  105                     405077364495191 480600000000000
 101                  109                     405073964493994 480800000000000
 102                  113                     405070564492797 480800000000000
"""


@pytest.fixture
def survey_file(tmp_path):
    path = tmp_path / "MHD-101.SEG"
    path.write_text(SURVEY)
    return path


def test_header_labels_are_read(survey_file):
    survey = read_survey(survey_file)
    assert survey.title == "SEISMIC SURVEY DATA"
    assert survey.labels["client"] == "ARCTIC OIL"
    assert survey.labels["units"] == "DECIMETERS"
    assert survey.labels["file_number"] == "MDH"
    assert "prospect" not in survey.labels, "empty labels are not facts"


def test_columns_come_from_the_files_own_legend(survey_file):
    names = [name for name, _, _ in read_survey(survey_file).columns]
    assert names[:4] == ["line", "shotpoint", "latitude", "longitude"]
    assert "elevation" in names


def test_points_are_counted_and_lines_collected(survey_file):
    survey = read_survey(survey_file)
    assert survey.point_count == 4
    assert survey.lines == ["101", "102"]
    assert survey.shotpoint_range == (101, 113)


def test_a_file_with_no_header_block_is_rejected(tmp_path):
    path = tmp_path / "plain.seg"
    path.write_text("just some text\nwith no header block\n")
    with pytest.raises(NotSegP1, match="no SEG-P1 header"):
        read_survey(path)


def test_binary_files_are_rejected(tmp_path):
    path = tmp_path / "binary.seg"
    path.write_bytes(bytes(range(256)) * 4)
    with pytest.raises(NotSegP1, match="not ASCII"):
        read_survey(path)


def test_handler_emits_text_and_typed_facts(survey_file):
    ex = SegP1Handler().parse(survey_file, settings.load_rag_config(), OFF)
    assert ex.kind == "segp1"
    assert ex.atomic and len(ex.segments) == 2
    assert "Client: ARCTIC OIL" in ex.segments[0][1]
    assert "Surveyed points" in ex.segments[1][1]

    rows = metadata_rows("d", ex.metadata)
    numeric = {k: n for _, k, _, n in rows if n is not None}
    assert numeric["point_count"] == 4
    assert numeric["shotpoint_min"] == 101
    assert numeric["shotpoint_max"] == 113
    lines = sorted(v for _, k, v, _ in rows if k == "seismic_line")
    assert lines == ["101", "102"], "one row per line, not a joined string"


def test_handler_claims_ascii_seg_files_only(survey_file, tmp_path):
    handler = SegP1Handler()
    assert handler.matches(survey_file, survey_file.read_bytes()[:512])
    ebcdic = tmp_path / "seismic.seg"
    ebcdic.write_bytes(b"\xc3\xf0\xf1\x40" * 100)
    assert not handler.matches(ebcdic, ebcdic.read_bytes()), "EBCDIC .seg is not SEG-P1"
    assert not handler.matches(tmp_path / "a.sgy", b"H  SURVEY")


def test_handler_skips_what_it_cannot_read(tmp_path):
    path = tmp_path / "nothing.seg"
    path.write_text("no header here\n")
    with pytest.raises(Skip, match="unreadable segp1"):
        SegP1Handler().parse(path, settings.load_rag_config(), OFF)
