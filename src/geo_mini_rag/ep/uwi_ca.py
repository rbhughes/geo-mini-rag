"""Canadian Unique Well Identifiers: the DLS and NTS survey systems.

A Canadian UWI is 16 characters that spell out where the well is, under one of
two survey systems, and it is written a dozen ways:

    DLS   100/04-11-082-04W6/00   100041108204W600   100 04-11-082-04W6 00
          1     survey system, 1 for the Dominion Land Survey
          00    location exception, 00 for the first well at that location
          04    legal subdivision, 01-16
          11    section, 01-36
          082   township, 001-126
          04    range, 01-34
          W6    meridian, W1-W6 west of the prime meridians, E1-E2 in Manitoba
          00    event sequence

    NTS   200/a-096-H/094-A-15/00   200A096H094A1500
          2     survey system, 2 for the National Topographic System
          00    location exception
          a     quarter unit, a-d
          096   unit, 001-100
          H     block, A-L
          094   map sheet number, 001-120
          A     map area, A-P
          15    map sheet, 01-16
          00    event sequence

Unlike an API number these carry letters in fixed places -- the meridian, the
block, the map area -- so the shape is evidence in itself and no label is
required for a complete UWI. Measured against 992 GovDocs1 documents, which
contain no Canadian wells: zero matches. That is a precision test, and the only
one available here.

**Recall is untested.** There is no Canadian file in this corpus. The ranges
above come from the survey systems' own definitions, but which forms actually
occur in a given vendor's export is a thing to check against real data, not to
assume. Treat this module as unproven until it has seen some.

A bare legal description with no survey system or event sequence -- `04-11-082-
04W6` -- is a location, not a UWI. It is reported as one, under `location`, and
only when a label vouches for it: nothing is padded out into a UWI that the
document did not write.

    from geo_mini_rag.ep.uwi_ca import find
    [w.uwi for w in find("UWI 100/04-11-082-04W6/00")]
    ['100041108204W600']
"""

from __future__ import annotations

import re
from dataclasses import dataclass

LABEL = re.compile(r"(?i)(?:\b|_)(uwi|api|well\s*id|licen[cs]e)(?:\b|_)")

# Separators people put between the parts: slashes, dashes, spaces, or nothing.
_S = r"[-/ ]?"
DLS = re.compile(
    rf"(?<![0-9A-Za-z])(1){_S}(\d{{2}}){_S}(\d{{2}}){_S}(\d{{2}}){_S}(\d{{3}}){_S}"
    rf"(\d{{2}}){_S}([WwEe]){_S}(\d){_S}(\d{{2}})(?![0-9A-Za-z])"
)
NTS = re.compile(
    rf"(?<![0-9A-Za-z])(2){_S}(\d{{2}}){_S}([A-Da-d]){_S}(\d{{3}}){_S}([A-La-l]){_S}"
    rf"(\d{{3}}){_S}([A-Pa-p]){_S}(\d{{2}}){_S}(\d{{2}})(?![0-9A-Za-z])"
)
# The same descriptions without the survey system and event sequence around them.
DLS_LOCATION = re.compile(
    rf"(?<![0-9A-Za-z])(\d{{2}}){_S}(\d{{2}}){_S}(\d{{3}}){_S}(\d{{2}}){_S}"
    rf"([WwEe]){_S}(\d)(?![0-9A-Za-z])"
)
NTS_LOCATION = re.compile(
    rf"(?<![0-9A-Za-z])([A-Da-d]){_S}(\d{{3}}){_S}([A-La-l]){_S}(\d{{3}}){_S}"
    rf"([A-Pa-p]){_S}(\d{{2}})(?![0-9A-Za-z])"
)

DEFAULTS = {"label_window": 60}


@dataclass(frozen=True)
class CanadianWell:
    """One identifier, as written and as normalised to its 16 characters."""

    text: str                # exactly what the document said
    uwi: str                 # the 16-character form, or "" for a bare location
    location: str            # the survey description alone: 0411082004W6, a096H094A15
    survey: str              # dls | nts


def _in_range(value: str, low: int, high: int) -> bool:
    return low <= int(value) <= high


def _dls(match: re.Match, complete: bool) -> CanadianWell | None:
    if complete:
        system, exception, lsd, section, township, rng, hemi, meridian, event = match.groups()
    else:
        system = exception = event = ""
        lsd, section, township, rng, hemi, meridian = match.groups()
    hemi = hemi.upper()
    if not (_in_range(lsd, 1, 16) and _in_range(section, 1, 36)
            and _in_range(township, 1, 126) and _in_range(rng, 1, 34)):
        return None
    # Six meridians west of the prime, two east of it in Manitoba.
    if not (1 <= int(meridian) <= (6 if hemi == "W" else 2)):
        return None
    location = f"{lsd}{section}{township}{rng}{hemi}{meridian}"
    return CanadianWell(
        text=match.group(0),
        uwi=f"{system}{exception}{location}{event}" if complete else "",
        location=location,
        survey="dls",
    )


def _nts(match: re.Match, complete: bool) -> CanadianWell | None:
    if complete:
        system, exception, quarter, unit, block, sheet, area, number, event = match.groups()
    else:
        system = exception = event = ""
        quarter, unit, block, sheet, area, number = match.groups()
    if not (_in_range(unit, 1, 100) and _in_range(sheet, 1, 120) and _in_range(number, 1, 16)):
        return None
    location = f"{quarter.upper()}{unit}{block.upper()}{sheet}{area.upper()}{number}"
    return CanadianWell(
        text=match.group(0),
        uwi=f"{system}{exception}{location}{event}" if complete else "",
        location=location,
        survey="nts",
    )


def _labelled(text: str, start: int, window: int) -> bool:
    """A label close before, with no digits between it and the description."""
    before = text[max(0, start - window) : start]
    labels = list(LABEL.finditer(before))
    if not labels:
        return False
    return not any(character.isdigit() for character in before[labels[-1].end() :])


def find(text: str, limits: dict | None = None, *, require_label: bool = True
         ) -> list[CanadianWell]:
    """Every Canadian well identifier in the text, in order, once each.

    A complete UWI stands on its own: its letters sit in places a phone number
    or a date cannot put them. A bare legal description does not, so it counts
    only where a label vouches for it.
    """
    limits = {**DEFAULTS, **(limits or {})}
    out: list[CanadianWell] = []
    seen: set[str] = set()
    taken: list[tuple[int, int]] = []

    for pattern, build, complete in (
        (DLS, _dls, True), (NTS, _nts, True),
        (DLS_LOCATION, _dls, False), (NTS_LOCATION, _nts, False),
    ):
        for match in pattern.finditer(text):
            if any(start < match.end() and match.start() < end for start, end in taken):
                continue        # already read as part of a complete UWI
            if not complete and require_label and not _labelled(
                    text, match.start(), limits["label_window"]):
                continue
            well = build(match, complete)
            if well is None:
                continue
            key = well.uwi or well.location
            if key in seen:
                continue
            seen.add(key)
            taken.append((match.start(), match.end()))
            out.append(well)
    return out
