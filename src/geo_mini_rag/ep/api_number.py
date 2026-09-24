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

2.  The state and county must exist. `api_codes.csv` is the authority for that
    (see the module's `SOURCE`), so nothing here is inferred from a name or a
    shape. A labelled number whose codes are not in the table is not recorded.

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

CODES = Path(__file__).parent / "data" / "api_codes.csv"
SOURCE = "rbhughes/freezer, cat-rasputin/api_codes.csv"

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
    "max_per_document": 0,   # 0: keep every one. A loader report naming 4,937 wells names them.
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
def _tables() -> tuple[dict[str, str], dict[tuple[str, str], tuple[str, ...]]]:
    """(state code -> abbreviation, (state, county) -> names) from the CSV."""
    states: dict[str, str] = {}
    counties: dict[tuple[str, str], list[str]] = {}
    with CODES.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            state = row["STATE_API_CODE"].strip()
            county = row["COUNTY_API_CODE"].strip()
            states[state] = row["STATE_ABBR"].strip()
            counties.setdefault((state, county), []).append(row["COUNTY_NAME"].strip())
    return states, {key: tuple(names) for key, names in counties.items()}


def _labelled(text: str, start: int, window: int) -> bool:
    """Is there an API or UWI label close before this number, with no digits between?"""
    before = text[max(0, start - window) : start]
    labels = list(LABEL.finditer(before))
    if not labels:
        return False
    return not any(character.isdigit() for character in before[labels[-1].end() :])


def find(text: str, limits: dict | None = None) -> list[WellId]:
    """Every labelled, code-valid well identifier in the text, in order, once each."""
    limits = {**DEFAULTS, **(limits or {})}
    states, counties = _tables()
    out: list[WellId] = []
    seen: set[str] = set()

    for match in NUMBER.finditer(text):
        if not _labelled(text, match.start(), limits["label_window"]):
            continue
        state_code, county_code, well = match.group(1), match.group(2), match.group(3)
        if state_code not in states or (state_code, county_code) not in counties:
            continue
        api = state_code + county_code + well
        if api in seen:
            continue
        seen.add(api)
        out.append(
            WellId(
                text=match.group(0),
                api=api,
                state=states[state_code],
                counties=counties[(state_code, county_code)],
                suffix="".join(group for group in match.group(4, 5) if group),
            )
        )
    return out


def _ten(value: str) -> str:
    """An API number as its ten identifying digits, whatever shape it was written in.

    Anything with a letter in it is left exactly as found: a Canadian UWI is not
    an API number and must not be filed as one.
    """
    if any(character.isalpha() for character in value):
        return value.strip()
    digits = re.sub(r"\D", "", value)
    return digits[:10] if len(digits) in (10, 12, 14) else value.strip()


def enrich(metadata: dict, segments: list[tuple[int | None, str]], limits: dict | None = None
           ) -> tuple[dict, list[str]]:
    """Add `api`, `api_state` and `api_county` facts. Returns (facts, notes).

    A handler that already lifted an API out of a header keeps it: the two sets
    are merged, so a LAS whose header declares one number and whose body lists
    twenty ends up with twenty-one. One well is one fact, in one shape -- a LAS
    header writing 490251108000 and a report writing 49-025-11080 are the same
    well, and both are stored as the ten digits that identify it.
    """
    limits = {**DEFAULTS, **(limits or {})}
    found = find("\n".join(text for _, text in segments if text), limits)
    notes: list[str] = []
    cap = limits["max_per_document"]
    if cap and len(found) > cap:
        notes.append(f"api: kept {cap:,} of {len(found):,} identifiers")
        found = found[:cap]
    if not found:
        return {}, notes

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
        "api": merged("api", [w.api for w in found], canonical=_ten),
        # A header that wrote the same number under UWI meant the same well.
        **({"uwi": merged("uwi", [], canonical=_ten)} if metadata.get("uwi") else {}),
        "api_state": merged("api_state", sorted({w.state for w in found})),
        "api_county": merged("api_county", sorted({c for w in found for c in w.counties})),
    }
    return facts, notes


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
