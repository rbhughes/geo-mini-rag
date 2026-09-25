import pytest

from geo_mini_rag.ep.uwi_ca import find
from geo_mini_rag.ep.well_ids import enrich

DLS = "100/04-11-082-04W6/00"
NTS = "200/a-096-H/094-A-15/00"


@pytest.mark.parametrize("written", [DLS, "100041108204W600", "100 04-11-082-04W6 00"])
def test_the_shapes_a_dls_uwi_is_written_in(written):
    found = find(written)
    assert [w.uwi for w in found] == ["100041108204W600"]
    assert found[0].survey == "dls"
    assert found[0].location == "041108204W6"


@pytest.mark.parametrize("written", [NTS, "200A096H094A1500", "200/A 096 H 094 A 15/00"])
def test_the_shapes_an_nts_uwi_is_written_in(written):
    found = find(written)
    assert [w.uwi for w in found] == ["200A096H094A1500"]
    assert found[0].survey == "nts"


def test_a_complete_uwi_needs_no_label():
    """Its letters sit where a date or a phone number cannot put them."""
    assert find(f"Spudded {DLS} in March.")[0].uwi == "100041108204W600"


@pytest.mark.parametrize("bad, why", [
    ("100/17-11-082-04W6/00", "legal subdivisions stop at 16"),
    ("100/04-40-082-04W6/00", "sections stop at 36"),
    ("100/04-11-999-04W6/00", "townships stop at 126"),
    ("100/04-11-082-99W6/00", "ranges stop at 34"),
    ("100/04-11-082-04W9/00", "there is no ninth meridian"),
    ("100/04-11-082-04E6/00", "only two meridians east of the prime"),
    ("200/a-996-H/094-A-15/00", "units stop at 100"),
    ("200/a-096-H/094-A-99/00", "map sheets stop at 16"),
])
def test_a_part_outside_its_range_is_not_a_well(bad, why):
    assert find(bad) == [], why


def test_a_bare_location_counts_only_where_a_label_vouches_for_it():
    found = find("UWI 04-11-082-04W6")
    assert [(w.uwi, w.location) for w in found] == [("", "041108204W6")]
    assert find("part number 04-11-082-04W6 shipped") == []


def test_nothing_is_padded_into_a_uwi_the_document_did_not_write():
    facts = enrich({}, [(None, "UWI 04-11-082-04W6")])
    assert facts == {"well_location": ["041108204W6"]}, "a location is not a UWI"


def test_a_uwi_is_not_read_twice_as_its_own_location():
    found = find(f"UWI {DLS}")
    assert len(found) == 1 and found[0].uwi


def test_both_countries_can_appear_in_one_document():
    facts = enrich({}, [(None, f"API 49-025-11080 and UWI {DLS}")])
    assert facts["api"] == ["4902511080"]
    assert facts["uwi"] == ["100041108204W600"]
    assert facts["api_county"] == ["Natrona"]


def test_a_canadian_uwi_in_a_header_is_not_renumbered_as_an_api():
    facts = enrich({"uwi": "100/04-11-082-04W6/00"}, [(None, f"UWI {DLS}")])
    assert facts["uwi"] == ["100/04-11-082-04W6/00", "100041108204W600"], (
        "the header's own wording is kept; only API numbers are renumbered"
    )
