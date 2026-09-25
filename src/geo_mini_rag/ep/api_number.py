"""US API well numbers found in any document, validated against a code table.

Not a handler: handlers own a format, and a well identifier is not a format. A
PDF completion report, a scanned permit, a CSV export and a LAS header all name
wells the same way, so this runs over whatever text partitioning produced and
adds the identifiers it finds to the document's facts.

An API number is 2 digits of state, 3 of county, 5 for the well, and optionally
2 for a sidetrack and 2 for a completion event: 49-025-11080, 490251108000.
Every part is a code, so `4902511080` is a number an embedding can do nothing
with, which is exactly why it belongs in the metadata that filtering and the
IDF boost work on.

Two rules keep it honest, and the order matters:

1.  A label must precede the digits. Measured on 992 GovDocs1 files, which have
    nothing to do with wells: 139,629 bare 10/12/14-digit runs, of which 4,607
    carry a plausible state and county code and would be accepted on structure
    alone. Requiring `API` or `UWI` in the preceding characters, with no other
    digits in between, leaves none of them. The label is what discriminates;
    the code table only confirms.

2.  The state and county must exist. `data/api_codes.csv` is the authority for
    that, so nothing here is inferred from a name or a shape, and a labelled
    number whose codes are not in it is not recorded. Offshore rows carry no
    county code, since offshore wells are numbered by area; for those the area
    code is accepted as given. The well number itself is checked only for
    00000, which the numbering does not use.

Canadian UWIs are not handled. Their shapes are well defined -- DLS
100/04-11-082-04W6/00, NTS 200/a-096-H/094-A-15/00 -- but there is no Canadian
file in this corpus to test a detector against, and a detector nobody can test
is a guess. `UWI` appears here only as a label, because LAS headers write the
API number under that mnemonic.

Nothing is repaired. Teapot_Wells.dbf stores its API as `2500153`, which is
Natrona county 025 and well 00153 with the state code absent and the leading
zero eaten by numeric storage; the full number is guessable and this module
does not guess it. Partial identifiers are left to the handler that found them.

    from geo_mini_rag.ep.api_number import find
    [w.api for w in find("API .   490251108000:  API NUMBER")]
    ['4902511080']
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

# This project's table of API state and county codes, maintained here. Its rows
# follow the numbering described at en.wikipedia.org/wiki/API_well_number:
# states 01-51 in alphabetical order, county codes usually odd with even ones
# for counties created later, and four offshore pseudo-states. An offshore row
# has no county code, because offshore wells are numbered by area instead; the
# area code is then accepted unchecked and the label carries the evidence.
CODES = Path(__file__).parent / "data" / "api_codes.csv"

# The label that has to be there. `well no` is deliberately absent: "Well No. 5"
# names a well without claiming to be an API number.
LABEL = re.compile(r"(?i)(?:\b|_)(api|uwi)(?:\b|_)")
# 2-3-5, then an optional sidetrack pair and an optional event pair. The
# lookarounds pin the run to its full length, so only 10, 12 and 14 digits match.
NUMBER = re.compile(
    r"(?<!\d)(\d{2})[-._/ ]?(\d{3})[-._/ ]?(\d{5})(?:[-._/ ]?(\d{2}))?(?:[-._/ ]?(\d{2}))?(?!\d)"
)

DEFAULTS = {
    "label_window": 60,      # characters before the digits in which the label must appear
}


@dataclass(frozen=True)
class WellId:
    """One identifier, as written and as normalised."""

    text: str                   # exactly what the document said
    api: str                    # the first ten digits, separators removed
    state: str                  # two-letter abbreviation from the code table
    counties: tuple[str, ...]   # usually one; a handful of codes name a county and a city
    suffix: str = ""            # sidetrack and event digits, when present


@cache
def offshore() -> frozenset[str]:
    """State codes that name an area instead of a county."""
    with CODES.open(newline="", encoding="utf-8-sig") as f:
        return frozenset(row["state_code"].strip() for row in csv.DictReader(f)
                         if not row["county_code"].strip())


@cache
def _tables() -> tuple[dict[str, str], dict[tuple[str, str], tuple[str, ...]]]:
    """(state code -> abbreviation, (state, county) -> names) from the CSV."""
    states: dict[str, str] = {}
    counties: dict[tuple[str, str], list[str]] = {}
    with CODES.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            state, county = row["state_code"].strip(), row["county_code"].strip()
            states[state] = row["state"].strip()
            if not county:
                continue                # offshore: numbered by area, not by county
            names = counties.setdefault((state, county), [])
            if (name := row["county"].strip()) not in names:
                names.append(name)
    return states, {key: tuple(names) for key, names in counties.items()}


def _labelled(text: str, start: int, window: int) -> bool:
    """Is there an API or UWI label close before this number, with no digits between?"""
    before = text[max(0, start - window) : start]
    labels = list(LABEL.finditer(before))
    if not labels:
        return False
    return not any(character.isdigit() for character in before[labels[-1].end() :])


def find(text: str, limits: dict | None = None, *, require_label: bool = True) -> list[WellId]:
    """Every labelled, code-valid well identifier in the text, in order, once each.

    `require_label=False` is for questions, not documents. The label rule was
    calibrated on 139,629 digit runs across 992 files; a question is a dozen
    words someone typed on purpose, where a number that validates against the
    code table is the thing they are asking about.
    """
    limits = {**DEFAULTS, **(limits or {})}
    states, counties = _tables()
    out: list[WellId] = []
    seen: set[str] = set()

    for match in NUMBER.finditer(text):
        if require_label and not _labelled(text, match.start(), limits["label_window"]):
            continue
        state_code, county_code, well = match.group(1), match.group(2), match.group(3)
        if well == "00000":
            continue                    # well numbers run 00001-99999
        if state_code in offshore():
            state, county_names = states[state_code], ()
        elif state_code in states and (state_code, county_code) in counties:
            state, county_names = states[state_code], counties[(state_code, county_code)]
        else:
            continue
        api = state_code + county_code + well
        if api in seen:
            continue
        seen.add(api)
        out.append(
            WellId(
                text=match.group(0),
                api=api,
                state=state,
                counties=county_names,
                suffix="".join(group for group in match.group(4, 5) if group),
            )
        )
    return out


def ten(value: str) -> str:
    """An API number as its ten identifying digits, whatever shape it was written in.

    Anything with a letter in it is left exactly as found: a Canadian UWI is not
    an API number and must not be filed as one.
    """
    if any(character.isalpha() for character in value):
        return value.strip()
    digits = re.sub(r"\D", "", value)
    return digits[:10] if len(digits) in (10, 12, 14) else value.strip()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)
    for path in args.paths:
        found = find(path.read_text(encoding="latin-1", errors="replace"))
        print(f"{path}: {len(found)} identifiers")
        for well in found[:20]:
            counties = "/".join(well.counties)
            print(f"  {well.api}{well.suffix}  {well.state} {counties:22} as written: {well.text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
