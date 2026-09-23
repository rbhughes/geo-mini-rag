"""LAS well logs: embed the header, never the curve data.

A 120 KB LAS file is one screen of header followed by thousands of rows of
floating point. Chunking the whole thing produces ~100 chunks of numbers that
match nothing and crowd out real documents. The header, by contrast, is the
most queryable text in an E&P collection: well name, API number, field, county,
operator, service company, log date, depth range and the curve list.

This handler emits one chunk for the well header and one for the curves, and
lifts the identifying fields into document metadata so they can be filtered on.
"""

from __future__ import annotations

import re
from pathlib import Path

from geo_mini_rag.rag.extract import Extracted, Skip
from geo_mini_rag.rag.trace import Tracer

# LAS mnemonic -> the metadata key we store it under
WELL_FIELDS = {
    "WELL": "well",
    "API": "api",
    "UWI": "uwi",
    "FLD": "field",
    "LOC": "location",
    "CNTY": "county",
    "STAT": "state",
    "CTRY": "country",
    "COMP": "operator",
    "SRVC": "service_company",
    "DATE": "log_date",
}
LABELS = {
    "well": "Well", "api": "API number", "uwi": "UWI", "field": "Field",
    "location": "Location", "county": "County", "state": "State", "country": "Country",
    "operator": "Operator", "service_company": "Service company", "log_date": "Log date",
}


class LasHandler:
    name = "las"

    def matches(self, path: Path, head: bytes) -> bool:
        if path.suffix.lower() != ".las":
            return False
        # LAS files open with a version section, sometimes after comment lines
        return b"~V" in head[:4096].upper()

    def parse(self, path: Path, cfg: dict, trace: Tracer) -> Extracted:
        import lasio

        try:
            las = lasio.read(str(path), ignore_data=True)
        except Exception as exc:  # lasio raises many shapes on malformed files
            raise Skip(f"unreadable las: {type(exc).__name__}: {exc}") from exc

        meta: dict[str, object] = {}
        for item in las.well:
            key = WELL_FIELDS.get(item.mnemonic.upper())
            value = str(item.value).strip()
            if key and value and value not in ("-999.25", "None"):
                meta[key] = value

        # One row per fact: each curve is its own value, depths are numbers, so
        # they can be filtered (--where curve=RHOB, --where depth_max>5000) and
        # weighted by how rare they are, rather than sitting inside one string.
        meta["curve"] = [c.mnemonic for c in las.curves if c.mnemonic]
        meta["curve_description"] = sorted(
            {c.descr.strip() for c in las.curves if (c.descr or "").strip()}
        )
        meta["curve_count"] = len(las.curves)
        meta.update(_depth_facts(las))
        year = _log_year(meta.get("log_date"))
        if year:
            meta["log_year"] = year

        depth = _depth_text(meta)
        header = _header_text(path, meta, depth)
        curves = _curve_text(path, meta, las)
        trace("las", f"header {len(header)} chars, curves {len(curves)} chars, "
                     f"{sum(len(v) if isinstance(v, list) else 1 for v in meta.values())} facts "
                     f"over {len(meta)} keys")
        return Extracted(
            kind="las",
            segments=[(None, header), (None, curves)],
            metadata=meta,
            atomic=True,
        )


def _depth_facts(las) -> dict[str, object]:
    """STRT/STOP/STEP as numbers, so they can be compared rather than matched."""
    def well_item(mnemonic):
        try:
            return las.well[mnemonic]
        except (KeyError, AttributeError):
            return None

    facts: dict[str, object] = {}
    for mnemonic, key in (("STRT", "depth_min"), ("STOP", "depth_max"), ("STEP", "depth_step")):
        item = well_item(mnemonic)
        if item is None or item.value in (None, ""):
            continue
        try:
            facts[key] = float(item.value)
        except (TypeError, ValueError):
            continue
    start = well_item("STRT")
    if start is not None and (start.unit or "").strip():
        facts["depth_units"] = start.unit.strip()
    return facts


def _log_year(log_date: object) -> int | None:
    """The year out of a LAS date, when one is unambiguous: 27-FEB-1977 -> 1977."""
    if not log_date:
        return None
    found = re.search(r"(18|19|20)\d{2}", str(log_date))
    return int(found.group()) if found else None


def _depth_text(meta: dict) -> str:
    """The logged interval, for the header card a reader sees."""
    if "depth_min" not in meta or "depth_max" not in meta:
        return ""
    units = f" {meta['depth_units']}" if meta.get("depth_units") else ""
    text = f"{meta['depth_min']:g} to {meta['depth_max']:g}{units}"
    return f"{text}, step {meta['depth_step']:g}" if "depth_step" in meta else text


def _header_text(path: Path, meta: dict, depth: str) -> str:
    lines = [f"Well log header (LAS) for {path.name}."]
    for key, label in LABELS.items():
        if meta.get(key):
            lines.append(f"{label}: {meta[key]}")
    if depth:
        lines.append(f"Logged interval: {depth}")
    lines.append(f"Curves recorded: {meta.get('curve_count', 0)}")
    return "\n".join(lines)


def _curve_text(path: Path, meta: dict, las) -> str:
    who = meta.get("well") or path.name
    lines = [f"Log curves in {path.name}" + (f" for well {who}" if meta.get("well") else "") + ":"]
    for c in las.curves:
        unit = f" [{c.unit}]" if c.unit else ""
        descr = f" — {c.descr}" if c.descr else ""
        lines.append(f"{c.mnemonic}{unit}{descr}")
    return "\n".join(lines)
