import yaml

from geo_mini_rag import settings
from geo_mini_rag.appraisal import pass0

POLICY = yaml.safe_load((settings.CONFIG_DIR / "policy.yaml").read_text())


def build(tmp_path, files: dict[str, bytes]):
    for name, content in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
    return {r.path.rsplit("/", 1)[-1]: r for r in pass0.inventory(str(tmp_path), POLICY)}


def test_drops_junk_and_empty_files(tmp_path):
    rows = build(tmp_path, {
        ".DS_Store": b"whatever",
        "~$report.doc": b"lock file",
        "empty.txt": b"",
        "keeper.txt": b"real content here",
    })
    assert rows[".DS_Store"].verdict == "EXCLUDE" and rows[".DS_Store"].reason == "junk_name"
    assert rows["~$report.doc"].reason == "junk_prefix"
    assert rows["empty.txt"].reason == "zero_bytes"
    assert rows["keeper.txt"].verdict == "PENDING"


def test_excludes_non_document_formats(tmp_path):
    rows = build(tmp_path, {
        "survey.dlis": b"\x00" * 4000,
        "model.dwg": b"AC1015" + b"\x00" * 200,
        "notes.txt": b"plain text",
    })
    assert rows["survey.dlis"].verdict == "EXCLUDE"
    assert "not_a_document" in rows["survey.dlis"].reason
    assert rows["model.dwg"].verdict == "EXCLUDE"
    assert rows["notes.txt"].verdict == "PENDING"


def test_shapefile_primary_is_a_special_class_and_sidecars_are_excluded(tmp_path):
    rows = build(tmp_path, {
        "layers/leases.shp": b"\x00\x00\x27\x0a" + b"\x00" * 200,
        "layers/leases.dbf": b"\x03" + b"\x00" * 200,
        "layers/leases.prj": b'PROJCS["NAD_1927_StatePlane"]',
        "layers/leases.shp.xml": b"<metadata><abstract>lease polygons</abstract></metadata>",
    })
    primary = rows["leases.shp"]
    assert primary.verdict == "PENDING"
    assert "special_class" in primary.reason, "the handler decides what a shapefile means"
    for sidecar in ("leases.dbf", "leases.prj", "leases.shp.xml"):
        assert rows[sidecar].verdict == "EXCLUDE", sidecar
        assert rows[sidecar].reason.startswith("bundle_sidecar"), sidecar
        assert rows[sidecar].part_of.endswith("leases.shp")


def test_las_is_flagged_as_its_own_class_not_excluded(tmp_path):
    rows = build(tmp_path, {"log.las": b"~VERSION INFORMATION\nVERS. 2.0:\n"})
    assert rows["log.las"].verdict == "PENDING"
    assert "special_class" in rows["log.las"].reason


def test_seismic_formats_are_special_classes_now_that_handlers_read_them(tmp_path):
    """A SEG-Y volume is not a document, but its textual header is."""
    rows = build(tmp_path, {
        "survey.sgy": b"\xc3\xf0\xf1" + b"\x40" * 4000,
        "positions.seg": b"H  SEISMIC SURVEY DATA\n CLIENT : ARCTIC OIL\n",
    })
    for name in ("survey.sgy", "positions.seg"):
        assert rows[name].verdict == "PENDING", name
        assert "special_class" in rows[name].reason, name


def test_extension_mismatch_is_recorded_but_kept(tmp_path):
    rows = build(tmp_path, {"report.txt": b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n" + b"0" * 500})
    row = rows["report.txt"]
    assert row.ext_mismatch is True
    assert row.mime == "application/pdf"
    assert row.verdict == "PENDING"   # a misnamed document is still a document


def test_manifest_id_changes_with_policy_and_inventory(tmp_path):
    rows = build(tmp_path, {"a.txt": b"one", "b.txt": b"two"})
    rows = list(rows.values())
    base = pass0.manifest_id("policy: v1", rows)
    assert base == pass0.manifest_id("policy: v1", rows), "same inputs must give the same id"
    assert base != pass0.manifest_id("policy: v2", rows), "policy change must change the id"
    assert base != pass0.manifest_id("policy: v1", rows[:1]), "inventory change must change the id"


def test_shapefile_sidecars_point_at_their_primary(tmp_path):
    rows = build(tmp_path, {
        "wells/leases.shp": b"\x00\x00\x27\x0a" + b"\x00" * 200,
        "wells/leases.shx": b"\x00\x00\x27\x0a" + b"\x00" * 50,
        "wells/leases.dbf": b"\x03" + b"\x00" * 200,
        "wells/leases.prj": b'GEOGCS["GCS_North_American_1927"]',
        "wells/leases.shp.xml": b"<metadata/>",
        "wells/unrelated.txt": b"not part of the bundle",
    })
    primary = rows["leases.shp"]
    assert primary.part_of is None, "the primary is not a sidecar of itself"
    for sidecar in ("leases.shx", "leases.dbf", "leases.prj", "leases.shp.xml"):
        assert rows[sidecar].part_of.endswith("leases.shp"), sidecar
    assert rows["unrelated.txt"].part_of is None


def test_sidecar_without_a_primary_stays_unlinked(tmp_path):
    rows = build(tmp_path, {"orphan.dbf": b"\x03" + b"\x00" * 200})
    assert rows["orphan.dbf"].part_of is None
