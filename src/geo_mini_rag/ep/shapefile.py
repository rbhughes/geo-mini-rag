"""ESRI shapefiles: the words in a map layer.

A shapefile is a bundle — geometry in `.shp`, attributes in `.dbf`, the
coordinate system in `.prj`, and often metadata in `.shp.xml`. The geometry is
coordinates and is not indexed; everything a person would search for lives in
the other three.

What each part is worth:

    .shp        bounding box and geometry type, from its 100-byte header. The
                box is taken from here rather than the XML because it is always
                present and cannot be stale.
    .prj        coordinate system name, datum, units, and an EPSG code when the
                WKT carries one.
    .shp.xml    title, abstract, purpose, keywords, lineage — when it is more
                than a stub.
    .dbf        the attribute table, treated by measurement rather than by
                field name: a layer of 2,111 wells carrying operator, formation
                and lease is worth indexing feature by feature, while a layer of
                nameless geometry with SHAPE_LENG and OBJECTID is worth one
                summary and nothing more.

Standard library only: the `.shp` header is a struct, the `.dbf` is a
fixed-format table, and `.shp.xml` is XML.

    from geo_mini_rag.ep.shapefile import read_layer
    layer = read_layer("leases.shp")
    layer.crs_name, layer.feature_count, layer.fields

    python -m geo_mini_rag.ep.shapefile leases.shp
"""

from __future__ import annotations

import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from geo_mini_rag.rag.extract import Extracted, Skip
from geo_mini_rag.rag.trace import Tracer

SHP_MAGIC = 9994
SHP_HEADER_BYTES = 100
DBF_HEADER_BYTES = 32

# Defaults; `handlers.shapefile` in config/rag.yaml overrides them.
DEFAULTS = {
    "features_per_chunk": 25,
    "max_feature_chunks": 200,      # a 54,000-feature layer is summarised, not transcribed
    "max_features_read": 20_000,
    "categorical_max_ratio": 0.2,   # each value recurs at least five times
    "prose_min_length": 40,         # average characters, above which it is text not a label
    "max_fact_values": 200,
}

GEOMETRY_TYPES = {
    0: "null", 1: "point", 3: "polyline", 5: "polygon", 8: "multipoint",
    11: "pointZ", 13: "polylineZ", 15: "polygonZ", 18: "multipointZ",
    21: "pointM", 23: "polylineM", 25: "polygonM", 28: "multipointM",
    31: "multipatch",
}
DBF_TYPES = {"C": "text", "N": "number", "F": "number", "D": "date", "L": "boolean", "M": "memo"}

WKT_NAME = re.compile(r'^\s*(?:PROJCS|GEOGCS|GEOGCRS|PROJCRS)\s*\[\s*"([^"]+)"')
WKT_DATUM = re.compile(r'DATUM\s*\[\s*"([^"]+)"')
WKT_UNIT = re.compile(r'UNIT\s*\[\s*"([^"]+)"')
WKT_EPSG = re.compile(r'AUTHORITY\s*\[\s*"EPSG"\s*,\s*"(\d+)"\s*\]\s*\]?\s*$', re.IGNORECASE)
# FGDC and ESRI metadata put the same things in different places.
XML_PATHS = {
    "title": ("idinfo/citation/citeinfo/title", "dataIdInfo/idCitation/resTitle"),
    "abstract": ("idinfo/descript/abstract", "dataIdInfo/idAbs"),
    "purpose": ("idinfo/descript/purpose", "dataIdInfo/idPurp"),
    "lineage": ("dataqual/lineage/procstep/procdesc",),
}
XML_STUBS = {"dataset copied.", "required: a brief narrative summary of the data set.", ""}


class NotShapefile(ValueError):
    """The file is not a readable shapefile bundle."""


@dataclass
class Field:
    name: str
    kind: str                       # text | number | date | boolean | memo
    length: int
    values: list[str] = field(default_factory=list)

    @property
    def filled(self) -> list[str]:
        """Values that say something. A field of blanks is not a field."""
        return [v.strip() for v in self.values if v.strip()]

    @property
    def distinct(self) -> int:
        return len(set(self.filled))

    @property
    def average_length(self) -> float:
        filled = self.filled
        return sum(len(v) for v in filled) / len(filled) if filled else 0.0

    def role(self, limits: dict) -> str:
        """empty | categorical | identifier | prose | number — measured, not guessed.

        Repetition is what makes a category: COMPANY holds 70 operators across
        2,111 wells, so every value recurs about thirty times and the field is
        worth filtering on. WELL_NUMBE holds 1,564 values in the same rows and
        names individual things, so it belongs in the text instead.
        """
        filled = self.filled
        if not filled:
            return "empty"
        if self.kind == "number":
            return "number"
        if self.average_length >= limits["prose_min_length"]:
            return "prose"
        if self.distinct / len(filled) <= limits["categorical_max_ratio"]:
            return "categorical"
        return "identifier"

    def numbers(self) -> list[float]:
        out = []
        for value in self.filled:
            try:
                out.append(float(value))
            except ValueError:
                continue
        return out


@dataclass
class Layer:
    path: Path
    geometry: str = "unknown"
    feature_count: int = 0
    bbox: tuple[float, float, float, float] | None = None
    crs_name: str = ""
    crs_datum: str = ""
    crs_units: str = ""
    crs_epsg: str = ""
    metadata: dict[str, str] = field(default_factory=dict)   # from .shp.xml
    fields: list[Field] = field(default_factory=list)
    truncated: bool = False

    @property
    def name(self) -> str:
        return self.path.stem


def read_shp_header(path: Path) -> tuple[str, tuple[float, float, float, float] | None]:
    """Geometry type and bounding box from the 100-byte .shp header."""
    with path.open("rb") as f:
        head = f.read(SHP_HEADER_BYTES)
    if len(head) < SHP_HEADER_BYTES:
        raise NotShapefile(f"{path.name} is {len(head)} bytes, too short for a .shp header")
    (magic,) = struct.unpack_from(">i", head, 0)
    if magic != SHP_MAGIC:
        raise NotShapefile(f"{path.name} does not start with the shapefile code {SHP_MAGIC}")
    (shape_type,) = struct.unpack_from("<i", head, 32)
    box = struct.unpack_from("<4d", head, 36)
    return GEOMETRY_TYPES.get(shape_type, f"type {shape_type}"), box


def read_dbf(path: Path, max_records: int) -> tuple[int, list[Field], bool]:
    """(record count, fields with their values, truncated) from a .dbf table."""
    with path.open("rb") as f:
        head = f.read(DBF_HEADER_BYTES)
        if len(head) < DBF_HEADER_BYTES:
            raise NotShapefile(f"{path.name} is too short for a .dbf header")
        count, header_length, record_length = struct.unpack_from("<IHH", head, 4)
        fields: list[Field] = []
        while True:
            descriptor = f.read(32)
            if not descriptor or descriptor[0] == 0x0D:
                break
            name = descriptor[:11].split(b"\x00")[0].decode("latin-1").strip()
            kind = DBF_TYPES.get(chr(descriptor[11]), "text")
            fields.append(Field(name=name, kind=kind, length=descriptor[16]))
        if not fields:
            raise NotShapefile(f"{path.name} has no attribute fields")

        f.seek(header_length)
        wanted = min(count, max_records)
        for _ in range(wanted):
            record = f.read(record_length)
            if len(record) < record_length:
                break
            offset = 1                      # first byte is the deletion flag
            for column in fields:
                column.values.append(record[offset : offset + column.length].decode("latin-1").strip())
                offset += column.length
    return count, fields, count > wanted


def read_prj(path: Path) -> dict[str, str]:
    """Coordinate system name, datum, units and EPSG code, where the WKT says so."""
    try:
        wkt = path.read_text(encoding="latin-1").strip()
    except OSError:
        return {}
    out: dict[str, str] = {}
    if found := WKT_NAME.search(wkt):
        out["crs_name"] = found.group(1).replace("_", " ")
    if found := WKT_DATUM.search(wkt):
        out["crs_datum"] = found.group(1).lstrip("D_").replace("_", " ")
    if found := WKT_UNIT.search(wkt):
        out["crs_units"] = found.group(1)
    if found := WKT_EPSG.search(wkt):
        out["crs_epsg"] = found.group(1)
    return out


def read_shp_xml(path: Path) -> dict[str, str]:
    """Title, abstract, purpose and lineage, skipping ESRI's boilerplate stubs."""
    try:
        root = ET.fromstring(path.read_text(encoding="latin-1"))
    except (OSError, ET.ParseError):
        return {}
    out: dict[str, str] = {}
    for key, paths in XML_PATHS.items():
        for xpath in paths:
            node = root.find(xpath)
            text = " ".join((node.text or "").split()) if node is not None else ""
            if text and text.lower() not in XML_STUBS:
                out[key] = text
                break
    keywords = [
        " ".join((node.text or "").split())
        for node in root.iter()
        if node.tag.endswith(("themekey", "placekey", "keyword")) and node.text
    ]
    if keywords:
        out["keywords"] = ", ".join(dict.fromkeys(keywords))
    return out


def read_layer(path: str | Path, limits: dict | None = None) -> Layer:
    """Read a shapefile bundle: geometry header, attributes, projection, metadata."""
    path = Path(path)
    limits = {**DEFAULTS, **(limits or {})}
    geometry, bbox = read_shp_header(path)
    layer = Layer(path=path, geometry=geometry, bbox=bbox)

    dbf = path.with_suffix(".dbf")
    if dbf.exists():
        layer.feature_count, layer.fields, layer.truncated = read_dbf(
            dbf, limits["max_features_read"]
        )
    prj = path.with_suffix(".prj")
    if prj.exists():
        for key, value in read_prj(prj).items():
            setattr(layer, key, value)
    xml = path.with_name(path.name + ".xml")
    if xml.exists():
        layer.metadata = read_shp_xml(xml)
    return layer


def layer_facts(layer: Layer, limits: dict) -> dict[str, object]:
    """Facts for filtering and ranking: one row per value, numbers as numbers."""
    facts: dict[str, object] = {
        "geometry_type": layer.geometry,
        "feature_count": layer.feature_count,
    }
    if layer.bbox and any(layer.bbox):
        facts["bbox_min_x"], facts["bbox_min_y"], facts["bbox_max_x"], facts["bbox_max_y"] = layer.bbox
    for key in ("crs_name", "crs_datum", "crs_units", "crs_epsg"):
        if value := getattr(layer, key):
            facts[key] = value
    for key, value in layer.metadata.items():
        if key in ("title", "keywords"):
            facts[key] = value

    named: list[str] = []
    for column in layer.fields:
        role = column.role(limits)
        if role == "empty":
            continue
        named.append(column.name)
        if role == "categorical":
            values = sorted(set(column.filled))[: limits["max_fact_values"]]
            facts[column.name.lower()] = values
        elif role == "number":
            numbers = column.numbers()
            if numbers and (min(numbers) or max(numbers)):
                facts[f"{column.name.lower()}_min"] = min(numbers)
                facts[f"{column.name.lower()}_max"] = max(numbers)
    if named:
        facts["field"] = named
    return facts


def layer_text(layer: Layer, limits: dict) -> str:
    """What this layer is, for someone reading or a model retrieving."""
    summary = (
        f"Map layer (shapefile) {layer.name}: {layer.geometry} geometry, "
        f"{layer.feature_count:,} features."
    )
    lines = [summary]
    if title := layer.metadata.get("title"):
        lines.append(f"Title: {title}")
    for key in ("abstract", "purpose", "lineage", "keywords"):
        if value := layer.metadata.get(key):
            lines.append(f"{key.title()}: {value}")
    if layer.crs_name:
        crs = layer.crs_name + (f" (EPSG:{layer.crs_epsg})" if layer.crs_epsg else "")
        lines.append(f"Coordinate system: {crs}")
    if layer.crs_datum:
        lines.append(f"Datum: {layer.crs_datum}")
    if layer.crs_units:
        lines.append(f"Units: {layer.crs_units}")
    if layer.bbox and any(layer.bbox):
        west, south, east, north = layer.bbox
        lines.append(f"Extent: {west:g} to {east:g} east, {south:g} to {north:g} north")

    described = []
    for column in layer.fields:
        role = column.role(limits)
        if role == "empty":
            continue
        if role == "categorical":
            values = sorted(set(column.filled))[:8]
            described.append(f"{column.name} ({column.distinct} values): {', '.join(values)}")
        elif role == "number":
            numbers = column.numbers()
            if numbers:
                described.append(f"{column.name}: {min(numbers):g} to {max(numbers):g}")
        else:
            described.append(f"{column.name}: {column.distinct:,} distinct {role} values")
    if described:
        lines.append("Attributes:")
        lines.extend(f"  {line}" for line in described)
    return "\n".join(lines)


def feature_chunks(layer: Layer, limits: dict, budget: int | None = None) -> list[str]:
    """Features in groups, for layers whose attributes name things.

    A layer of nameless geometry gets none of these: its summary says everything
    there is to say.

    `features_per_chunk` is an upper bound, not a target: a group also stops at
    `budget` characters so that chunking never has to cut a feature in half. A
    well row runs about 200 characters, so a 1,200-character budget holds five
    or six of them; raise chunk.max_characters to fit more.
    """
    roles = {column.name: column.role(limits) for column in layer.fields}
    # A field holding one value for every feature says nothing about any of
    # them; it is already in the layer summary.
    carried = [
        c for c in layer.fields
        if roles[c.name] in ("identifier", "prose", "categorical") and c.distinct > 1
    ]
    if not any(roles[c.name] in ("identifier", "prose") for c in carried):
        return []

    rows = min(len(carried[0].values), limits["max_features_read"])
    per_chunk = limits["features_per_chunk"]
    chunks: list[str] = []
    described: list[str] = []
    first = 0
    size = 0

    def flush(last: int) -> None:
        nonlocal described, size, first
        if described:
            header = f"Features {first + 1}-{last} of {layer.name}:"
            chunks.append(header + "\n" + "\n".join(described))
        described, size, first = [], 0, last

    for index in range(rows):
        if len(chunks) >= limits["max_feature_chunks"]:
            break
        parts = [
            f"{c.name}: {c.values[index]}"
            for c in carried
            if index < len(c.values) and c.values[index].strip()
        ]
        if not parts:
            continue
        line = "; ".join(parts)
        if described and (len(described) >= per_chunk or (budget and size + len(line) > budget)):
            flush(index)
        described.append(line)
        size += len(line) + 1
    if len(chunks) < limits["max_feature_chunks"]:
        flush(rows)
    return chunks


class ShapefileHandler:
    """A map layer's attributes and projection, not its coordinates."""

    name = "shapefile"
    extensions = (".shp",)

    def matches(self, path: Path, head: bytes) -> bool:
        if path.suffix.lower() not in self.extensions:
            return False
        return len(head) >= 4 and struct.unpack_from(">i", head, 0)[0] == SHP_MAGIC

    def parse(self, path: Path, cfg: dict, trace: Tracer) -> Extracted:
        limits = {**DEFAULTS, **(cfg.get("handlers", {}).get("shapefile") or {})}
        try:
            layer = read_layer(path, limits)
        except (NotShapefile, OSError, struct.error) as exc:
            raise Skip(f"unreadable shapefile: {exc}") from exc

        if not layer.fields and not layer.crs_name and not layer.metadata:
            # Geometry with no attributes, projection or metadata: coordinates
            # and nothing a person could search for.
            raise Skip("shapefile has no attributes, projection or metadata")

        segments: list[tuple[int | None, str]] = [(None, layer_text(layer, limits))]
        features = feature_chunks(layer, limits, cfg["chunk"]["max_characters"])
        segments.extend((None, chunk) for chunk in features)

        notes = []
        if layer.truncated:
            notes.append(
                f"attributes read for {limits['max_features_read']:,} of "
                f"{layer.feature_count:,} features"
            )
        trace("shapefile", f"{layer.geometry}, {layer.feature_count:,} features, "
                           f"{len(layer.fields)} fields, {len(features)} feature chunks")
        return Extracted(
            kind="shapefile",
            segments=segments,
            metadata=layer_facts(layer, limits),
            atomic=True,
            notes=notes,
        )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Summarise an ESRI shapefile bundle.")
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--features", action="store_true", help="Print the feature chunks too.")
    args = parser.parse_args(argv)

    failed = False
    for n, path in enumerate(args.files):
        if n:
            print("\n" + "=" * 72 + "\n")
        try:
            layer = read_layer(path)
        except (NotShapefile, OSError, struct.error) as exc:
            failed = True
            print(f"{path}: {exc}")
            continue
        print(layer_text(layer, DEFAULTS))
        if args.features:
            for chunk in feature_chunks(layer, DEFAULTS)[:3]:
                print("\n" + chunk)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
