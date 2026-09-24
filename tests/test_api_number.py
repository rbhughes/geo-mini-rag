import pytest

from geo_mini_rag.ep.api_number import OFFSHORE, SOURCE, WellId, find
from geo_mini_rag.ep.well_ids import enrich

LAS_HEADER = """~Well Information Block
 WELL.                NPR #3 #13SX11-11:  WELL
 API .       490251108000:   API NUMBER
 UWI .       490251108000:   UNIQUE WELL ID
"""


def test_reads_the_api_out_of_a_las_header():
    found = find(LAS_HEADER)
    assert [w.api for w in found] == ["4902511080"], "one well, however many times it is named"
    assert found[0].state == "WY"
    assert found[0].counties == ("Natrona",)
    assert found[0].suffix == "00", "the sidetrack digits are kept as written"


@pytest.mark.parametrize("written", [
    "API 49-025-11080",
    "API: 49 025 11080",
    "API No. 4902511080",
    "api_number=490251108000",
    "UWI . 49025110800001 : UNIQUE WELL ID",
])
def test_the_shapes_a_number_is_written_in(written):
    assert [w.api for w in find(written)] == ["4902511080"]


def test_an_unlabelled_number_is_not_an_api_number():
    """The label is the discriminator: 4,607 digit runs in GovDocs1 pass on structure."""
    assert find("Invoice total 4902511080 paid on 2026-01-05") == []
    assert find("Reference 49-025-11080, see attached") == []


def test_a_digit_between_the_label_and_the_number_breaks_the_claim():
    assert find("API wells 2 through 9 total 4902511080") == []


def test_codes_that_are_not_in_the_table_are_not_wells():
    assert find("API 99-025-11080") == [], "99 is not a state"
    assert find("API 49-999-11080") == [], "999 is not a county of Wyoming"


def test_nothing_is_inferred_from_a_partial_number():
    """Teapot_Wells.dbf stores 2500153: Natrona 025 and well 00153, state missing."""
    assert find("API 2500153") == []


def test_one_well_is_one_fact_however_it_was_written():
    facts, notes = enrich(
        {"api": "490251108000"},
        [(None, "API . 490251108000\nsee also API 49-025-11080 and API 05-123-45678")],
    )
    assert facts["api"] == ["4902511080", "0512345678"]
    assert facts["api_state"] == ["CO", "WY"]
    assert facts["api_county"] == ["Natrona", "Weld"]
    assert notes == []


def test_every_identifier_is_kept_unless_a_cap_is_asked_for():
    text = "\n".join(f"API 49-025-{n:05d}" for n in range(1, 301))
    facts, notes = enrich({}, [(None, text)])
    assert len(facts["api"]) == 300 and notes == []

    facts, notes = enrich({}, [(None, text)], {"max_per_document": 100})
    assert len(facts["api"]) == 100
    assert notes == ["api: kept 100 of 300 identifiers"]


def test_a_code_naming_more_than_one_place_reports_both():
    """Virginia gives 45-003 to both Albemarle county and Charlottesville city."""
    assert find("API 45-003-11080")[0].counties == ("Albemarle", "Charlottesville ( City )")


def test_the_table_is_attributed():
    assert "freezer" in SOURCE and "api_codes.csv" in SOURCE


def test_a_document_with_no_wells_gets_no_facts():
    facts, notes = enrich({}, [(None, "A report about nothing in particular.")])
    assert facts == {} and notes == []


def test_well_id_is_hashable_so_callers_can_deduplicate():
    assert len({WellId("x", "4902511080", "WY", ("Natrona",)),
                WellId("x", "4902511080", "WY", ("Natrona",))}) == 1


def test_a_uwi_header_field_is_filed_the_same_way():
    facts, _ = enrich({"uwi": "490251108000"}, [(None, "API . 490251108000")])
    assert facts["api"] == facts["uwi"] == ["4902511080"], "one well, one shape"


def test_a_canadian_uwi_is_left_exactly_as_found():
    """Not an API number. The shapes are known; nothing here is tested against them."""
    facts, _ = enrich({"uwi": "100/04-11-082-04W6/00"}, [(None, "API 49-025-11080")])
    assert facts["uwi"] == ["100/04-11-082-04W6/00"]


def test_offshore_wells_are_numbered_under_pseudo_states():
    """The CSV holds counties, and the Gulf of Mexico has none."""
    found = find("API 60-817-40161")
    assert found[0].api == "6081740161"
    assert found[0].state == "Northern Gulf of Mexico"
    assert found[0].counties == (), "an area code is not a county, and is not checked"
    assert set(OFFSHORE) == {"55", "56", "60", "61"}


def test_a_well_number_of_all_zeros_is_not_a_well():
    """Well numbers run 00001-99999."""
    assert find("API 49-025-00000") == []


def test_texas_county_codes_run_past_the_fips_range():
    """API county codes are not FIPS: Zavala is 42-507, and the table knows it."""
    assert find("API 42-507-11080")[0].counties == ("Zavala",)
