"""The spreadsheet handler: columns measured, not column names trusted."""

import datetime as dt

import pytest

from geo_mini_rag.ep.xls import (
    DEFAULTS,
    POSITIONAL,
    XlsHandler,
    _cell,
    api_values,
    columns_of,
    facts_of,
    row_chunks,
)

LIMITS = DEFAULTS


def _sheet(rows):
    return columns_of("Sheet1", rows, LIMITS)


def test_a_first_row_of_short_distinct_labels_is_a_header():
    fields = _sheet([["Well", "Operator"], ["A-1", "ACME"], ["A-2", "ACME"]])
    assert [f.name for f in fields] == ["Well", "Operator"]
    assert fields[0].values == ["A-1", "A-2"]


@pytest.mark.parametrize("first", [
    ["Well", "Well"],            # not distinct
    ["Well", ""],                # not filled
    ["1", "2"],                  # bare digits are data
])
def test_a_first_row_that_is_not_a_header_is_not_treated_as_one(first):
    """Naming columns after a row of data loses the row and invents the name."""
    fields = _sheet([first, ["A-1", "ACME"], ["A-2", "ACME"]])
    assert all(f.name.startswith(POSITIONAL) for f in fields)
    assert fields[0].values[0] == first[0], "the first row is data and is kept"


def test_an_unnamed_column_contributes_no_fact_key():
    """A filter needs a name, and `column 21` is not one the sheet gave."""
    # A first row of repeats is not a header, so no column here has a name.
    fields = _sheet([["ACME", "ACME"], ["ACME", "x"], ["BETA", "y"]])
    assert all(f.name.startswith(POSITIONAL) for f in fields)
    assert "field" not in facts_of([("Sheet1", fields)], LIMITS)


def test_a_repeating_column_becomes_a_fact_and_a_distinct_one_does_not():
    rows = [["Well", "Operator"]] + [[f"A-{i}", "ACME" if i % 2 else "BETA"] for i in range(10)]
    facts = facts_of([("Sheet1", _sheet(rows))], LIMITS)
    assert sorted(facts["operator"]) == ["ACME", "BETA"], "repeats: filterable"
    assert "well" not in facts, "ten distinct values name individual things"
    assert facts["field"] == ["Well", "Operator"]


def test_a_number_column_becomes_a_range():
    rows = [["Depth"]] + [[str(d)] for d in (100, 250, 3124)]
    facts = facts_of([("Sheet1", _sheet(rows))], LIMITS)
    assert facts["depth_min"] == 100.0
    assert facts["depth_max"] == 3124.0


def test_a_column_of_bare_numbers_is_not_repeated_per_row():
    """Retrieval is text; a number naming nothing only makes rows look distinct."""
    rows = [["SEGMID", "Street"]] + [[str(1400 + i), "Mulberry"] for i in range(8)]
    chunks = row_chunks(_sheet(rows), LIMITS)
    assert not any("1401" in c for c in chunks)


def test_identical_rows_are_merged_and_counted():
    rows = [["Street", "City"]] + [["Mulberry", "Fort Collins"]] * 4 + [["Oak", "Fort Collins"]]
    text = "\n".join(row_chunks(_sheet(rows), LIMITS))
    assert "[4 rows]" in text
    assert text.count("Mulberry") == 1


def test_a_sheet_that_names_nothing_gets_no_row_chunks():
    """Counts and codes are already in the summary; repeating them adds noise."""
    rows = [["Count", "Code"]] + [[str(i), "A"] for i in range(10)]
    assert row_chunks(_sheet(rows), LIMITS) == []


def test_a_column_of_well_numbers_needs_no_label():
    """The column is the evidence, exactly as in a .dbf."""
    wells = ["4902511080", "4902510421", "4902510399", "4902510409"]
    fields = _sheet([["mystery"]] + [[w] for w in wells])
    assert api_values(fields) == wells


@pytest.mark.parametrize("value, expected", [
    (2007.0, "2007"),                       # a year, not 2007.0
    (3124.5, "3124.5"),
    (dt.date(1977, 9, 26), "1977-09-26"),
    (True, "true"),
    (None, ""),
])
def test_a_cell_is_written_the_way_a_person_would_search_for_it(value, expected):
    assert _cell(value) == expected


@pytest.mark.parametrize("head, claimed", [
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1rest", True),    # OLE: Excel 5.0 and later
    (b"\x09\x04\x06\x00\x00\x00", True),                # bare BIFF4: Excel 4.0
    (b"PK\x03\x04", False),                             # .xlsx in a .xls coat
    (b"~VERSION INFORMATION", False),
])
def test_the_handler_claims_a_workbook_by_its_first_bytes(tmp_path, head, claimed):
    path = tmp_path / "book.xls"
    path.write_bytes(head)
    assert XlsHandler().matches(path, head) is claimed


def test_the_handler_does_not_claim_another_extension(tmp_path):
    path = tmp_path / "book.doc"
    assert XlsHandler().matches(path, b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1") is False
