import pytest

from geo_mini_rag.ep.api_number import WellId, enrich, find, in_question, offshore

LAS_HEADER = """~Well Information Block
 WELL.                NPR #3 #13SX11-11:  WELL
 API .       490251108000:   API NUMBER
 UWI .       490251108000:   UNIQUE WELL ID
"""


def test_reads_the_api_out_of_a_las_header():
    found = find(LAS_HEADER)
    assert [w.api for w in found] == ["490251108000"], (
        "every digit the header wrote, and it wrote twelve")
    assert found[0].state == "WY"
    assert found[0].counties == ("Natrona",)
    assert found[0].suffix == "00", "the sidetrack digits are kept as written"
    assert len(found[0].api) == 12, "no length is imposed on the number"


@pytest.mark.parametrize("written, digits", [
    ("API 49-025-11080", "4902511080"),
    ("API: 49 025 11080", "4902511080"),
    ("API No. 4902511080", "4902511080"),
    ("api_number=490251108000", "490251108000"),
    ("UWI . 49025110800001 : UNIQUE WELL ID", "49025110800001"),
])
def test_the_shapes_a_number_is_written_in(written, digits):
    """Separators go, length stays: a vendor's fourteen digits carry a
    sidetrack and a completion that ten would throw away."""
    assert [w.api for w in find(written)] == [digits]


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


def test_a_document_keeps_every_length_it_wrote():
    """The same well at two lengths stays two facts, because neither length is
    the right one: twelve digits carry a sidetrack that ten would discard, and
    seven is what a map layer holds when the state is left off. Bridging the
    lengths is the searching side's job."""
    facts = enrich(
        {"api": "490251108000"},
        [(None, "API . 490251108000\nsee also API 49-025-11080 and API 05-123-45678")],
    )
    assert facts["api"] == ["490251108000", "4902511080", "0512345678"]
    assert facts["api_state"] == ["CO", "WY"]
    assert facts["api_county"] == ["Natrona", "Weld"]


def test_every_identifier_a_document_names_is_kept():
    """There is no cap. A loader report listing 4,937 wells is a document about
    4,937 wells, and a fact costs a row."""
    text = "\n".join(f"API 49-025-{n:05d}" for n in range(1, 301))
    assert len(enrich({}, [(None, text)])["api"]) == 300


def test_a_code_naming_more_than_one_place_reports_both():
    """Virginia gives 45-003 to both Albemarle county and Charlottesville city."""
    assert find("API 45-003-11080")[0].counties == ("Albemarle", "Charlottesville ( City )")


def test_the_table_covers_every_state_and_the_offshore_areas():
    from geo_mini_rag.ep.api_number import _tables

    states, counties = _tables()
    assert len(states) == 55, "51 states and the District, plus four offshore areas"
    assert states["49"] == "WY" and states["60"] == "Northern Gulf of Mexico"
    assert ("49", "025") in counties


def test_a_document_with_no_wells_gets_no_facts():
    facts = enrich({}, [(None, "A report about nothing in particular.")])
    assert facts == {}


def test_well_id_is_hashable_so_callers_can_deduplicate():
    assert len({WellId("x", "4902511080", "WY", ("Natrona",)),
                WellId("x", "4902511080", "WY", ("Natrona",))}) == 1


def test_a_uwi_header_field_is_filed_the_same_way():
    facts = enrich({"uwi": "490251108000"}, [(None, "API . 490251108000")])
    assert facts["api"] == facts["uwi"] == ["490251108000"], "the digits the header wrote"


def test_a_canadian_uwi_is_left_exactly_as_found():
    """Not an API number. The shapes are known; nothing here is tested against them."""
    facts = enrich({"uwi": "100/04-11-082-04W6/00"}, [(None, "API 49-025-11080")])
    assert facts["uwi"] == ["100/04-11-082-04W6/00"]


def test_offshore_wells_are_numbered_under_pseudo_states():
    """The CSV holds counties, and the Gulf of Mexico has none."""
    found = find("API 60-817-40161")
    assert found[0].api == "6081740161"
    assert found[0].state == "Northern Gulf of Mexico"
    assert found[0].counties == (), "an area code is not a county, and is not checked"
    assert offshore() == {"55", "56", "60", "61"}


def test_a_well_number_of_all_zeros_is_not_a_well():
    """Well numbers run 00001-99999."""
    assert find("API 49-025-00000") == []


def test_texas_county_codes_run_past_the_fips_range():
    """API county codes are not FIPS: Zavala is 42-507, and the table knows it."""
    assert find("API 42-507-11080")[0].counties == ("Zavala",)


def test_counties_added_since_the_table_was_first_written():
    """La Paz was split from Yuma in 1983 and took an even code, as Cibola did."""
    assert find("API 02-012-11080")[0].counties == ("La Paz",)
    assert find("API 02-027-11080")[0].counties == ("Yuma",)
    assert find("API 30-006-11080")[0].counties == ("Cibola",)


def test_counties_the_table_was_missing_or_had_wrong():
    """Both found from evidence: 122 labelled 05-014 numbers in data/raw, and
    Kentucky's own alphabetical odd-code sequence with Nicholas absent."""
    assert find("API 05-014-11080")[0].counties == ("Broomfield",)
    assert find("API 16-181-11080")[0].counties == ("Nicholas",)
    assert find("API 16-179-11080")[0].counties == ("Nelson",)


def test_nothing_can_cap_the_identifiers_a_document_yields():
    """The cap existed only to be set to zero, and an edit meant for
    retrieve.per_document matched max_per_document and set it to 2 instead.
    Six thousand facts went missing quietly. The knob is gone."""
    from geo_mini_rag.ep.api_number import DEFAULTS
    from geo_mini_rag.settings import load_rag_config

    assert not [k for k in DEFAULTS if "max" in k]
    assert "enrich" not in load_rag_config()


@pytest.mark.parametrize("written, expected", [
    ("*2506325", "*2506325"),
    ("2506325*", "2506325*"),
    ("*2506325*", "*2506325*"),
])
def test_a_fragment_is_starred_the_way_a_glob_is(written, expected):
    """A vendor's fourteen digits carry a sidetrack after the well, so a
    fragment taken from a map layer lands in the middle of the longer number
    rather than at its end: 2506325 ends 2506325 and sits inside 490250632500."""
    assert in_question(written) == [("api", expected)]


def test_a_number_with_no_star_is_not_a_fragment():
    assert in_question("4902506325") == [("api", "4902506325")]
    assert in_question("which well is 2506325") == []


def test_a_trusted_length_yields_its_state_and_county():
    """Twelve digits put the county where the numbering says it is."""
    from geo_mini_rag.ep.api_number import codes_of

    assert codes_of(["490250632500"]) == {"api_state": ["WY"], "api_county": ["Natrona"]}


@pytest.mark.parametrize("value", ["2500153", "2506325", "4902511080"])
def test_an_untrusted_length_yields_nothing(value):
    """A fragment has lost its state, and the last ten digits of a fourteen read
    as a different state entirely: 0250632500 looks like Arizona county 506. A
    wrong county is worse than no county, so only 12 and 14 are read."""
    from geo_mini_rag.ep.api_number import codes_of

    assert codes_of([value]) == {}


@pytest.mark.parametrize("question, expected", [
    ("what LAS files are in TX", ["TX"]),
    ("which wells are in Texas", ["TX"]),
    ("wells in texas", ["TX"]),
    ("wells in or near the field", []),     # OR is Oregon, in lower case it is a word
    ("ok, which wells are in me", []),      # OK and ME likewise
])
def test_a_state_is_read_by_capitalised_code_or_by_name(question, expected):
    from geo_mini_rag.ep.api_number import states_in_question

    assert states_in_question(question) == expected
