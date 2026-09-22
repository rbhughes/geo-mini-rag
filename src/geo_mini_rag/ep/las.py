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

        meta = {}
        for item in las.well:
            key = WELL_FIELDS.get(item.mnemonic.upper())
            value = str(item.value).strip()
            if key and value and value not in ("-999.25", "None"):
                meta[key] = value

        depth = _depth_range(las)
        if depth:
            meta["depth_range"] = depth
        meta["curve_count"] = str(len(las.curves))
        meta["curves"] = ", ".join(c.mnemonic for c in las.curves)

        header = _header_text(path, meta, depth)
        curves = _curve_text(path, meta, las)
        trace("las", f"header {len(header)} chars, curves {len(curves)} chars, "
                     f"metadata: {', '.join(f'{k}={v}' for k, v in meta.items() if k != 'curves')}")
        return Extracted(
            kind="las",
            segments=[(None, header), (None, curves)],
            metadata=meta,
            atomic=True,
        )


def _depth_range(las) -> str:
    def well_value(mnemonic):
        try:
            return las.well[mnemonic].value
        except (KeyError, AttributeError):
            return None

    start, stop, step = well_value("STRT"), well_value("STOP"), well_value("STEP")
    if start is None or stop is None:
        return ""
    try:
        unit = las.well["STRT"].unit or ""
    except (KeyError, AttributeError):
        unit = ""
    text = f"{start} to {stop} {unit}".strip()
    return f"{text}, step {step}" if step is not None else text


def _header_text(path: Path, meta: dict, depth: str) -> str:
    lines = [f"Well log header (LAS) for {path.name}."]
    for key, label in LABELS.items():
        if meta.get(key):
            lines.append(f"{label}: {meta[key]}")
    if depth:
        lines.append(f"Logged interval: {depth}")
    lines.append(f"Curves recorded: {meta.get('curve_count', '0')}")
    return "\n".join(lines)


def _curve_text(path: Path, meta: dict, las) -> str:
    who = meta.get("well") or path.name
    lines = [f"Log curves in {path.name}" + (f" for well {who}" if meta.get("well") else "") + ":"]
    for c in las.curves:
        unit = f" [{c.unit}]" if c.unit else ""
        descr = f" — {c.descr}" if c.descr else ""
        lines.append(f"{c.mnemonic}{unit}{descr}")
    return "\n".join(lines)
