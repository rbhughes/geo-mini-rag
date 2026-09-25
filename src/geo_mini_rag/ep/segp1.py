"""SEG-P1: shot point positions, as text.

SEG-P1 is the SEG's exchange format for seismic positioning — where the shot
points and receivers actually are. Unlike SEG-Y it is plain ASCII and holds no
trace samples at all: a header block naming the client, prospect, datum and
units, then one fixed-width record per surveyed point.

Files in this family are commonly named `.seg`, which invites confusion with
SEG-Y. They are not SEG-Y: they have no 3,200-byte EBCDIC header and no binary
block, so `segy.py` rejects them and this handler claims them instead.

Column layout is read from the file's own legend line where one is present:

    <.....LINE.....><..SP..>.<..LAT..><..LONG..><..EA..><..NO..><ELV><...><....>

rather than assumed, because column positions vary between vintages of the
format and between the vendors who write it.

    from geo_mini_rag.ep.segp1 import read_survey
    survey = read_survey("MHD-101.SEG")
    survey.labels["client"], survey.point_count

"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from geo_mini_rag.rag.extract import Extracted, Skip
from geo_mini_rag.rag.trace import Tracer

MAX_BYTES = 5_000_000
LABEL = re.compile(r"^\s*(?P<label>[A-Z][A-Z ()/.]{2,20}?)\s*:\s*(?P<value>.*?)\s*$")
# Three header conventions turned up in real files. A bare "H" then the title; G3236.SEGP1 writes numbered
# H-records, H1500 Geodetic Datum: NAD_1927; esw_1.segp writes # comments,
# "# Type: scattered data (SEGP1-3 Format)". Both are "name: value" once the
# record marker is off the front, so neither needs a table of codes.
H_RECORD = re.compile(r"^H\d{0,4}\s+(?P<rest>.*?)\s*$")
HASH_RECORD = re.compile(r"^#\s*(?P<rest>.*?)\s*$")
NAMED = re.compile(r"^(?P<label>[A-Za-z][\w ()/.]{2,32}?)\s*:\s*(?P<value>.+?)\s*$")
# esw_1.segp declares its own columns: "# Field: LINEID     2 17 non-numeric"
FIELD_DECL = re.compile(r"^Field:\s*(?P<name>\w+)\s+(?P<start>\d+)\s+(?P<end>\d+)")
LEGEND = re.compile(r"<[^>]*>")
RULE = re.compile(r"^[\s|-]{20,}$")

# Legend names as written, mapped to what they mean. Anything unrecognised keeps
# the file's own name, lowercased.
COLUMNS = {
    "LINE": "line", "SP": "shotpoint", "LAT": "latitude", "LONG": "longitude",
    "EA": "easting", "NO": "northing", "ELV": "elevation",
}
LABELS = {
    "CLIENT": "client", "PROSPECT": "prospect", "LINE": "line",
    "CONTRACTOR": "contractor", "ORIGIN": "origin", "DATUM": "datum",
    "UNITS": "units", "SURVEYOR": "surveyor", "SURVEY DATE": "survey_date",
    "PRODUCED BY": "produced_by", "FILE NUMBER": "file_number",
}


class NotSegP1(ValueError):
    """The file does not look like SEG-P1 positioning data."""


@dataclass
class Survey:
    path: Path
    title: str = ""
    labels: dict[str, str] = field(default_factory=dict)
    columns: list[tuple[str, int, int]] = field(default_factory=list)  # (name, start, end)
    lines: list[str] = field(default_factory=list)                     # seismic line names
    shotpoints: list[int] = field(default_factory=list)
    point_count: int = 0

    @property
    def shotpoint_range(self) -> tuple[int, int] | None:
        return (min(self.shotpoints), max(self.shotpoints)) if self.shotpoints else None


def _columns_from_legend(legend: str) -> list[tuple[str, int, int]]:
    """Field spans taken from the file's own legend, not from an assumed layout."""
    columns = []
    for match in LEGEND.finditer(legend):
        name = match.group().strip("<>").strip(". ").upper()
        columns.append((COLUMNS.get(name, name.lower() or "field"), match.start(), match.end()))
    return columns


def read_survey(path: str | Path, *, max_bytes: int = MAX_BYTES) -> Survey:
    """Read the header block and summarise the point records."""
    path = Path(path)
    raw = path.read_bytes()[:max_bytes]
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        try:
            text = raw.decode("cp1252")
        except UnicodeDecodeError as exc:
            raise NotSegP1(f"{path.name} is not ASCII text") from exc

    survey = Survey(path=path)
    body: list[str] = []
    for line in text.splitlines():
        if RULE.match(line) or not line.strip():
            continue
        record = H_RECORD.match(line) or HASH_RECORD.match(line)
        if record:
            rest = record["rest"]
            if declared := FIELD_DECL.match(rest):
                # the file states its own column spans; 1-based, inclusive
                survey.columns.append((declared["name"].lower(),
                                       int(declared["start"]) - 1, int(declared["end"])))
                continue
            if named := NAMED.match(rest):
                key = " ".join(named["label"].split()).lower().replace(" ", "_")
                survey.labels.setdefault(key, named["value"])
            elif not survey.title and len(rest) > 3 and not rest.endswith(":"):
                survey.title = rest
            continue
        if legend := _columns_from_legend(line):
            survey.columns = legend
            continue
        if (found := LABEL.match(line)) and not survey.columns:
            key = LABELS.get(" ".join(found["label"].split()))
            if key and found["value"]:
                survey.labels[key] = found["value"]
            continue
        if survey.columns:
            body.append(line)

    if not survey.title and not survey.labels:
        raise NotSegP1(f"{path.name} has no SEG-P1 header block")

    spans = {name: (start, end) for name, start, end in survey.columns}
    seen_lines: dict[str, None] = {}
    for record in body:
        survey.point_count += 1
        if "line" in spans:
            start, end = spans["line"]
            if name := record[start:end].strip():
                seen_lines[name] = None
        if "shotpoint" in spans:
            start, end = spans["shotpoint"]
            point = record[start:end].strip()
            if point.lstrip("-").isdigit():
                survey.shotpoints.append(int(point))
    survey.lines = list(seen_lines)
    return survey


def survey_text(survey: Survey) -> str:
    """The header block as prose, for retrieval."""
    lines = [f"Seismic survey positions (SEG-P1) in {survey.path.name}."]
    if survey.title:
        lines.append(survey.title)
    for key, value in survey.labels.items():
        lines.append(f"{key.replace('_', ' ').title()}: {value}")
    return "\n".join(lines)


def points_text(survey: Survey) -> str:
    """What was surveyed, as a sentence rather than a table of coordinates."""
    lines = [f"Surveyed points in {survey.path.name}: {survey.point_count}"]
    if survey.lines:
        lines.append(f"Seismic lines: {', '.join(survey.lines)}")
    if span := survey.shotpoint_range:
        lines.append(f"Shot points: {span[0]} to {span[1]}")
    if survey.columns:
        lines.append(f"Columns recorded: {', '.join(name for name, _, _ in survey.columns)}")
    if units := survey.labels.get("units"):
        lines.append(f"Coordinate units: {units}")
    return "\n".join(lines)


class SegP1Handler:
    """Positioning data: a header worth reading, coordinates worth counting."""

    name = "segp1"
    # .segp1 and .segp arrived in data/raw/seis2 and fell through to the text
    # reader, though both files name their own format in their first line.
    extensions = (".seg", ".p1", ".sp1", ".segp1", ".segp")

    def matches(self, path: Path, head: bytes) -> bool:
        if path.suffix.lower() not in self.extensions:
            return False
        # Cheap content check: SEG-P1 is ASCII and opens with a header block, so
        # a .seg file holding EBCDIC or binary belongs to another reader.
        sample = head[:512]
        return bool(sample) and all(ch in (9, 10, 13) or 32 <= ch < 127 for ch in sample)

    def parse(self, path: Path, cfg: dict, trace: Tracer) -> Extracted:
        try:
            survey = read_survey(path, max_bytes=cfg["extract"]["max_text_bytes"])
        except (NotSegP1, OSError) as exc:
            raise Skip(f"unreadable segp1: {exc}") from exc

        meta: dict[str, object] = dict(survey.labels)
        meta["point_count"] = survey.point_count
        if survey.lines:
            meta["seismic_line"] = survey.lines
        if span := survey.shotpoint_range:
            meta["shotpoint_min"], meta["shotpoint_max"] = span
        if survey.columns:
            meta["column"] = [name for name, _, _ in survey.columns]

        trace("segp1", f"{len(survey.labels)} labels, {survey.point_count} points, "
                       f"{len(survey.lines)} lines, {len(survey.columns)} columns")
        return Extracted(
            kind="segp1",
            segments=[(None, survey_text(survey)), (None, points_text(survey))],
            metadata=meta,
            atomic=True,
        )
