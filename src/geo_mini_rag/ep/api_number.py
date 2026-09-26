"""API well numbers: find them in any document, and in any question.

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

US numbers only. `UWI` appears here as a label, not a format, because LAS
headers write the API number under that mnemonic.

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
    api: str                    # every digit, separators removed: 10, 12 or 14
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
        api = "".join(part for part in match.groups() if part)
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


def digits_of(value: str) -> str:
    """An API number with its separators removed and nothing else changed.

    No length is imposed. A vendor writes fourteen digits, a state agency ten,
    a map layer sometimes seven with the state left off; every one of those is
    what that system holds, and deciding which is "the" number would throw away
    the sidetrack and completion that the longer forms carry. Matching across
    lengths is the searching side's job, not the storing side's.

    Anything with a letter in it is left exactly as found: a Canadian UWI is not
    an API number and must not be filed as one.
    """
    if any(character.isalpha() for character in value):
        return value.strip()
    digits = re.sub(r"\D", "", value)
    return digits if 7 <= len(digits) <= 14 else value.strip()
def find_bare(value: str) -> WellId | None:
    """One value that is an API number on its own, with nothing around it.

    A column of these is evidence in a way a single one in prose is not: the
    label rule exists because 139,629 digit runs turned up in 992 documents,
    and a .dbf column is not prose. The caller decides on the column.
    """
    match = NUMBER.fullmatch(value.strip())
    if not match:
        return None
    found = find(value.strip(), require_label=False)
    return found[0] if found else None


def enrich(metadata: dict, segments: list[tuple[int | None, str]]) -> dict:
    """The `api`, `api_state` and `api_county` facts a document carries.

    Every identifier a document names is kept: a loader report listing 4,937
    wells is a document about 4,937 wells, and a fact costs a row. A handler
    that already lifted one out of a header keeps it -- the sets are merged --
    and one well is one fact in one shape, so a header writing 490251108000 and
    a report writing 49-025-11080 do not count as two wells.
    """
    found = find("\n".join(part for _, part in segments if part))
    if not found:
        return {}

    def merged(key: str, values: list[str], canonical=lambda v: v) -> list[str]:
        existing = metadata.get(key, [])
        existing = list(existing) if isinstance(existing, (list, tuple, set)) else [existing]
        out: list[str] = []
        for value in [*(str(v).strip() for v in existing), *values]:
            value = canonical(value)
            if value and value not in out:
                out.append(value)
        return out

    facts = {
        "api": merged("api", [w.api for w in found], canonical=digits_of),
        "api_state": merged("api_state", sorted({w.state for w in found})),
        "api_county": merged("api_county", sorted({c for w in found for c in w.counties})),
    }
    if metadata.get("uwi"):   # a header that wrote the API number under UWI
        facts["uwi"] = merged("uwi", [], canonical=digits_of)
    return facts


# A well number written as a tail: *2500153 finds 4902500153. Archives record
# the same well at different lengths -- a shapefile keeps county and well and
# drops the state, a log keeps all ten digits -- and nobody should have to know
# which before they can search.
PARTIAL = re.compile(r"(?<![\d*])\*(?P<tail>\d{4,14})(?!\d)")


def in_question(question: str) -> list[tuple[str, str]]:
    """(key, value) pairs a question names, for looking up rather than ranking.

    `well 4902511080` scores 0.324 against the log that carries it and 0.729
    against a page of unrelated digits, so no boost small enough to be safe can
    rescue it. Looked up, it is exact. No label is required here: a question is
    a dozen words typed on purpose, and the code table alone decides.
    """
    found = [("api", "*" + m["tail"]) for m in PARTIAL.finditer(question)]
    return found + [("api", well.api) for well in find(question, require_label=False)]
