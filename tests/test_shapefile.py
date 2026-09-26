import struct

import pytest

from geo_mini_rag import settings
from geo_mini_rag.ep.shapefile import (
    DEFAULTS,
    Field,
    NotShapefile,
    ShapefileHandler,
    detail_fields,
    feature_chunks,
    read_layer,
    read_prj,
)
from geo_mini_rag.rag.extract import Skip
from geo_mini_rag.rag.store import metadata_rows
from geo_mini_rag.rag.trace import OFF

WKT = (
    'PROJCS["NAD_1927_StatePlane_Wyoming_East_Central_FIPS_4902",'
    'GEOGCS["GCS_North_American_1927",DATUM["D_North_American_1927",'
    'SPHEROID["Clarke_1866",6378206.4,294.9786982]],PRIMEM["Greenwich",0.0],'
    'UNIT["Degree",0.0174532925199433]],AUTHORITY["EPSG","32056"]]'
)
XML = """<metadata><idinfo>
  <citation><citeinfo><title>Teapot Wells</title></citeinfo></citation>
  <descript><abstract>Wells drilled for oil and gas in the Teapot Dome area.</abstract>
            <purpose>To show well locations.</purpose></descript>
  <keywords><theme><themekey>well</themekey><themekey>oil</themekey></theme></keywords>
</idinfo><dataqual><lineage><procstep><procdesc>Dataset copied.</procdesc></procstep></lineage></dataqual></metadata>"""


def write_bundle(directory, name, fields, rows, *, prj=True, xml=True, shape_type=1):
    """A minimal but real bundle: .shp header, .dbf table, .prj, .shp.xml."""
    shp = bytearray(100)
    struct.pack_into(">i", shp, 0, 9994)
    struct.pack_into("<i", shp, 32, shape_type)
    struct.pack_into("<4d", shp, 36, 396923.0, 4783670.0, 407717.0, 4804120.0)
    (directory / f"{name}.shp").write_bytes(bytes(shp))

    record_length = 1 + sum(length for _, _, length in fields)
    header_length = 32 + 32 * len(fields) + 1
    dbf = bytearray()
    dbf += struct.pack("<B3B", 0x03, 124, 1, 1)
    dbf += struct.pack("<IHH", len(rows), header_length, record_length)
    dbf += bytes(20)
    for fname, ftype, length in fields:
        dbf += fname.encode("latin-1")[:11].ljust(11, b"\x00")
        dbf += ftype.encode() + bytes(4) + bytes([length]) + bytes(15)
    dbf += b"\x0d"
    for row in rows:
        dbf += b" "
        for (fname, _, length), value in zip(fields, row, strict=True):
            dbf += str(value).encode("latin-1")[:length].ljust(length)
    (directory / f"{name}.dbf").write_bytes(bytes(dbf))

    if prj:
        (directory / f"{name}.prj").write_text(WKT)
    if xml:
        (directory / f"{name}.shp.xml").write_text(XML)
    return directory / f"{name}.shp"


def _features(chunk):
    """The feature lines of a chunk, without its two heading lines."""
    return chunk.split("\n")[2:]


@pytest.fixture
def wells(tmp_path):
    fields = [("WELL_NAME", "C", 12), ("COMPANY", "C", 16), ("FIELD_NAME", "C", 12),
              ("TOTAL_DEPTH", "N", 8), ("NOTE", "C", 10)]
    rows = [
        (f"No. {n}", "TEAPOT OIL" if n % 2 else "AMERADA HESS", "TEAPOT DOME", 3000 + n * 10, "")
        for n in range(60)
    ]
    return write_bundle(tmp_path, "Teapot_Wells", fields, rows)


def test_reads_geometry_and_extent_from_the_shp_header(wells):
    layer = read_layer(wells)
    assert layer.geometry == "point"
    assert layer.bbox == (396923.0, 4783670.0, 407717.0, 4804120.0)
    assert layer.feature_count == 60


def test_projection_gives_crs_name_datum_units_and_epsg(tmp_path):
    prj = tmp_path / "layer.prj"
    prj.write_text(WKT)
    crs = read_prj(prj)
    assert crs["crs_name"] == "NAD 1927 StatePlane Wyoming East Central FIPS 4902"
    assert crs["crs_datum"] == "North American 1927"
    assert crs["crs_units"] == "Degree"
    assert crs["crs_epsg"] == "32056"


def test_metadata_is_read_but_stubs_are_ignored(wells):
    layer = read_layer(wells)
    assert layer.metadata["title"] == "Teapot Wells"
    assert "Teapot Dome area" in layer.metadata["abstract"]
    assert layer.metadata["keywords"] == "well, oil"
    assert "lineage" not in layer.metadata, "'Dataset copied.' is boilerplate, not lineage"


def test_fields_are_classified_by_repetition_not_by_name(wells):
    roles = {f.name: f.role(DEFAULTS) for f in read_layer(wells).fields}
    assert roles["COMPANY"] == "categorical", "2 operators over 60 wells is a category"
    assert roles["WELL_NAME"] == "identifier", "60 distinct names over 60 wells"
    assert roles["TOTAL_DEPTH"] == "number"
    assert roles["NOTE"] == "empty", "a field nobody filled in is not a fact"


def test_facts_are_filterable_and_typed(wells):
    ex = ShapefileHandler().parse(wells, settings.load_rag_config(), OFF)
    rows = metadata_rows("d", ex.metadata)
    values = {(k, v) for _, k, v, _ in rows}
    numeric = {k: n for _, k, _, n in rows if n is not None}
    assert ("company", "TEAPOT OIL") in values and ("company", "AMERADA HESS") in values
    assert ("crs_epsg", "32056") in values
    assert numeric["feature_count"] == 60
    assert numeric["bbox_min_x"] == 396923.0
    assert numeric["total_depth_max"] == 3590.0
    assert ("field", "WELL_NAME") in values, "the schema is searchable too"


def test_features_are_grouped_and_nameless_layers_get_none(wells, tmp_path):
    chunks = feature_chunks(read_layer(wells), DEFAULTS)
    assert len(chunks) == 3, "60 features at 25 per chunk"
    assert chunks[0].startswith("Teapot_Wells is a point map layer of 60 features")
    assert "Feature group 1 of 3:" in chunks[0]
    assert "WELL_NAME: No. 0" in chunks[0]
    assert "NOTE" not in chunks[0], "empty fields are not described"
    assert "FIELD_NAME" not in chunks[0], "TEAPOT DOME on every well is a layer fact"

    contours = write_bundle(tmp_path, "Structure", [("Id", "N", 4), ("TVDSS", "N", 6)],
                            [(0, 1020 + n) for n in range(40)], prj=False, xml=False)
    assert feature_chunks(read_layer(contours), DEFAULTS) == [], "no words, no feature chunks"


def test_handler_claims_only_real_shapefiles(tmp_path, wells):
    handler = ShapefileHandler()
    assert handler.matches(wells, wells.read_bytes()[:100])
    impostor = tmp_path / "fake.shp"
    impostor.write_bytes(b"not a shapefile at all" * 10)
    assert not handler.matches(impostor, impostor.read_bytes()[:100])
    assert not handler.matches(tmp_path / "a.dbf", b"")


def test_handler_skips_geometry_with_nothing_to_search(tmp_path):
    """A .shp with no .dbf, .prj or metadata is coordinates and nothing else."""
    path = tmp_path / "bare.shp"
    path.write_bytes(struct.pack(">i", 9994) + bytes(96))
    with pytest.raises(Skip, match="no attributes, projection or metadata"):
        ShapefileHandler().parse(path, settings.load_rag_config(), OFF)


def test_handler_skips_an_unreadable_bundle(tmp_path):
    path = tmp_path / "truncated.shp"
    path.write_bytes(struct.pack(">i", 9994) + bytes(20))
    with pytest.raises(Skip, match="unreadable shapefile"):
        ShapefileHandler().parse(path, settings.load_rag_config(), OFF)


def test_short_files_are_rejected(tmp_path):
    path = tmp_path / "stub.shp"
    path.write_bytes(b"\x00" * 20)
    with pytest.raises(NotShapefile, match="too short"):
        read_layer(path)


def test_empty_field_reports_no_role():
    assert Field("BLANK", "text", 10, ["", "  ", ""]).role(DEFAULTS) == "empty"


def test_no_feature_is_dropped_however_many_chunks_that_takes(tmp_path):
    """A layer is never part-indexed: the handler adds chunks, it does not truncate."""
    fields = [("WELL_NAME", "C", 12), ("COMPANY", "C", 16)]
    rows = [(f"No. {n}", "TEAPOT OIL" if n % 2 else "AMERADA HESS") for n in range(500)]
    layer = read_layer(write_bundle(tmp_path, "Many_Wells", fields, rows))

    chunks = feature_chunks(layer, DEFAULTS, budget=60)   # one feature per chunk
    assert len(chunks) == 500, "a tight budget buys more chunks, not fewer features"
    assert sum(len(_features(chunk)) for chunk in chunks) == 500
    assert "WELL_NAME: No. 499" in chunks[-1], "the last feature is described too"


def test_a_value_on_nearly_every_feature_is_left_to_the_layer(tmp_path):
    """SURF_TYPE is 3 on 5,806 Denver road segments: that describes the layer."""
    fields = [("WELL_NAME", "C", 12), ("SURF_TYPE", "C", 4), ("STATUS", "C", 8)]
    rows = [(f"No. {n}", "3" if n else "4", "ACTIVE" if n % 3 else "PLUGGED") for n in range(60)]
    layer = read_layer(write_bundle(tmp_path, "Roads", fields, rows))

    kept = [c.name for c in detail_fields(layer, DEFAULTS["dominant_max_share"])]
    assert kept == ["WELL_NAME", "STATUS"], "SURF_TYPE varies on one row in sixty"


def test_bare_number_fields_are_not_worth_a_sentence(tmp_path):
    """Retrieval is text. SEGMID 1401 names nothing, and makes two rows look distinct."""
    fields = [("ROUTENAME", "C", 12), ("FROM_DESCR", "C", 10), ("SEGMID", "C", 6)]
    names = ["E 14TH AVE", "W 10TH AVE", "N DEXTER ST"]
    rows = [(names[i % 3], f"MILE {i}", str(1400 + i * 2 + half))
            for i in range(15) for half in (0, 1)]
    layer = read_layer(write_bundle(tmp_path, "Segments", fields, rows))

    assert [c.name for c in detail_fields(layer, DEFAULTS["dominant_max_share"])] == ["ROUTENAME", "FROM_DESCR"]
    chunks = feature_chunks(layer, DEFAULTS)
    assert "SEGMID" not in chunks[0], "a segment id is not a word"
    assert len(_features(chunks[0])) == 15, "thirty segments, fifteen descriptions"
    assert chunks[0].count("(x2 features)") == 15, "the count says what each stands for"


def test_every_feature_chunk_says_what_layer_it_belongs_to(wells):
    """A list of names reads as "about wells" for any question about wells; the
    sentence gives the chunk something to be about."""
    chunk = feature_chunks(read_layer(wells), DEFAULTS)[1]
    first, second = chunk.split("\n")[:2]
    assert first == ("Teapot_Wells is a point map layer of 60 features, titled Teapot Wells."
                     " Wells drilled for oil and gas in the Teapot Dome area.")
    assert second == "Feature group 2 of 3:"


def test_the_heading_is_paid_for_out_of_the_budget(wells):
    """It repeats on every chunk, so it cannot push one over the chunk size."""
    for chunk in feature_chunks(read_layer(wells), DEFAULTS, budget=600):
        assert len(chunk) <= 600


def test_a_column_of_well_numbers_is_found_without_being_named(tmp_path):
    """GeoGraphix layers keep well numbers under names like DataId and WellID.
    The whole column is the evidence: a column of valid API numbers is one,
    a single number in prose is not, which is why prose needs a label."""
    from geo_mini_rag.ep.shapefile import api_values

    fields = [("DataId", "C", 12), ("TypeId", "C", 8), ("NAME", "C", 10)]
    rows = [(f"49025{n + 10000:05d}00", f"2098{n:03d}", f"well {n}") for n in range(6)]
    layer = read_layer(write_bundle(tmp_path, "Posted", fields, rows))

    found = api_values(layer)
    assert len(found) == 6 and found[0].startswith("49025")
    assert not any(v.startswith("2098") for v in found), "TypeId is not a well number"


def test_a_numeric_api_column_is_kept_as_the_file_writes_it(tmp_path):
    """Teapot_Wells stores API_NUMBER as a numeric column, so 2,111 wells
    collapsed to a min and a max. 2500153 is Natrona 025 and well 00153 with
    the state missing, and the reader does not guess the rest."""
    from geo_mini_rag.ep.shapefile import api_values

    fields = [("API_NUMBER", "N", 20), ("WELL", "C", 8)]
    rows = [("2.50638700000e+006", "No. 1"), ("2500153", "No. 2"), ("2506390", "No. 3")]
    layer = read_layer(write_bundle(tmp_path, "Wells", fields, rows))

    assert api_values(layer) == ["2506387", "2500153", "2506390"]
