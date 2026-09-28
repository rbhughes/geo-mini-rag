"""Excel workbooks: the sheets that hold records, read as columns.

A spreadsheet is a table that happens to be in a file format, and the mistake
is to read it as prose. A row poured into an embedder is a line of numbers with
a few words in it; a thousand such rows are a thousand near-identical chunks
that crowd out everything else. So a sheet is treated exactly like a `.dbf`
attribute table, through the same measurement in `ep.columns`: a column whose
values repeat is a category and becomes a filterable fact, one whose values are
distinct names individual things and goes into the text, one that is nearly all
the same value describes the sheet rather than the row, and a column of bare
digits is dropped.

Reading is `python-calamine`, which is a Rust parser with no Python
dependencies and no system libraries -- the point being that legacy Excel does
not have to mean a LibreOffice install and a subprocess per file. It handles
every BIFF version that ships inside an OLE container. Three files in data/raw
predate that: Excel 4.0 wrote raw BIFF with no container, and `xlrd` still
reads those, so it is the fallback rather than the main reader.

Nothing here writes, and no formula is evaluated: a cached value is what the
sheet says, and a formula is not text anyone searches for.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from geo_mini_rag.ep import api_number
from geo_mini_rag.ep.columns import BARE_NUMBER, Field
from geo_mini_rag.ep.columns import DEFAULTS as COLUMN_DEFAULTS
from geo_mini_rag.rag.extract import Extracted, Skip
from geo_mini_rag.rag.trace import Tracer

DEFAULTS = {
    **COLUMN_DEFAULTS,
    "rows_per_chunk": 25,
    "max_rows_per_sheet": 0,        # 0 means every row; a sheet is not truncated
    "header_max_length": 64,        # a first-row cell longer than this is data, not a header
}

# A column name the sheet did not give. Prefixing them marks the difference,
# because a row written as "column 21: Total RNA" is worse than one written as
# "Total RNA": the key is not a key, and it is two thirds of the characters.
POSITIONAL = "column "

# The first bytes of the two things a .xls can be: an OLE compound file, which
# is every Excel from 5.0 on, and a bare BIFF stream, which is Excel 4.0 and
# earlier writing its records straight to disk.
OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
BIFF_MAGIC = (b"\x09\x00", b"\x09\x02", b"\x09\x04", b"\x09\x06", b"\x09\x08")


class NotWorkbook(ValueError):
    """The file is not a readable workbook."""


def _cell(value) -> str:
    """One cell as the text a person would search for.

    A date is written the way the sheet shows it rather than as a timestamp, and
    a whole number keeps no decimal tail: 2007 is a year, 2007.0 is nothing.
    """
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()[:10]
    if isinstance(value, dt.time):
        return value.isoformat()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _rows_calamine(path: Path) -> list[tuple[str, list[list[str]]]]:
    import python_calamine

    book = python_calamine.CalamineWorkbook.from_path(str(path))
    out = []
    for name in book.sheet_names:
        rows = [[_cell(c) for c in row] for row in book.get_sheet_by_name(name).to_python()]
        out.append((name, rows))
    return out


def _rows_xlrd(path: Path) -> list[tuple[str, list[list[str]]]]:
    """Excel 4.0 and earlier: raw BIFF, which calamine does not open."""
    import xlrd

    book = xlrd.open_workbook(str(path))
    out = []
    for index in range(book.nsheets):
        sheet = book.sheet_by_index(index)
        rows = [[_cell(sheet.cell_value(r, c)) for c in range(sheet.ncols)]
                for r in range(sheet.nrows)]
        out.append((sheet.name, rows))
    return out


def read_sheets(path: Path) -> list[tuple[str, list[list[str]]]]:
    """Every sheet as rows of strings, by whichever reader can open the file."""
    try:
        return _rows_calamine(path)
    except ImportError:
        raise
    except Exception as first:   # noqa: BLE001 - a 1992 file malforms in ways no base class covers
        try:
            return _rows_xlrd(path)
        except ImportError:
            raise
        except Exception as second:   # noqa: BLE001 - likewise
            raise NotWorkbook(f"{first}; xlrd: {second}") from first


def columns_of(name: str, rows: list[list[str]], limits: dict) -> list[Field]:
    """A sheet's columns, named by its first row when that row is a header.

    A header row is one whose cells are short, present and distinct -- which is
    what a header is. When the first row fails that test the sheet has no
    header, the data starts at row one, and the columns are named by position
    so that nothing is invented.
    """
    if not rows:
        return []
    width = max(len(r) for r in rows)
    first = [(rows[0][i] if i < len(rows[0]) else "") for i in range(width)]
    labelled = (
        len(rows) > 1
        and all(cell for cell in first)
        and len(set(first)) == width
        and all(len(cell) <= limits["header_max_length"] for cell in first)
        and not all(BARE_NUMBER.match(cell) for cell in first)
    )
    names = first if labelled else [f"{POSITIONAL}{i + 1}" for i in range(width)]
    body = rows[1:] if labelled else rows
    if limits["max_rows_per_sheet"]:
        body = body[: limits["max_rows_per_sheet"]]

    fields = []
    for i, column_name in enumerate(names):
        values = [(r[i] if i < len(r) else "") for r in body]
        kind = "number" if values and all(
            not v or BARE_NUMBER.match(v) for v in values
        ) else "text"
        fields.append(Field(name=str(column_name), kind=kind, values=values))
    return fields


def api_values(fields: list[Field]) -> list[str]:
    """Well numbers a sheet holds, found the way a .dbf column is.

    A column is evidence where a sentence is not: if most of a column validates
    against the code table it is a column of well numbers whatever the header
    calls it, and no label is required. See `ep.shapefile.api_values`, which
    does the same thing for an attribute table.
    """
    from geo_mini_rag.ep.shapefile import (
        API_COLUMN_NAME,
        API_COLUMN_SHARE,
        MIN_API_ROWS,
    )

    out: list[str] = []
    for column in fields:
        filled = [v for v in column.filled if v]
        if len(filled) < MIN_API_ROWS:
            continue
        if API_COLUMN_NAME.match(column.name):
            out += filled
            continue
        valid = [v for v in filled if api_number.find_bare(v)]
        if len(valid) / len(filled) >= API_COLUMN_SHARE:
            out += valid
    seen: dict[str, None] = {}
    for value in out:
        seen.setdefault(api_number.digits_of(value), None)
    return list(seen)


def facts_of(sheets: list[tuple[str, list[Field]]], limits: dict) -> dict:
    """Filterable facts: categories as values, numbers as ranges."""
    facts: dict = {}
    named: list[str] = []
    wells: list[str] = []
    for _, fields in sheets:
        wells += api_values(fields)
        for column in fields:
            role = column.role(limits)
            if role == "empty":
                continue
            if column.name.startswith(POSITIONAL):
                continue        # an unnamed column cannot be filtered on by name
            named.append(column.name)
            if role == "categorical" and column.dominant_share < limits["dominant_max_share"]:
                facts.setdefault(column.name.lower(), [])
                for value in sorted(set(column.filled)):
                    if value not in facts[column.name.lower()]:
                        facts[column.name.lower()].append(value)
            elif role == "number":
                numbers = column.numbers()
                if numbers and (min(numbers) or max(numbers)):
                    facts[f"{column.name.lower()}_min"] = min(numbers)
                    facts[f"{column.name.lower()}_max"] = max(numbers)
    if named:
        facts["field"] = list(dict.fromkeys(named))
    if wells:
        facts["api"] = wells
        facts.update(api_number.codes_of(wells))
    return facts


def sheet_text(path: Path, sheets: list[tuple[str, list[Field]]], limits: dict) -> str:
    """What this workbook is, for someone reading or a model retrieving."""
    plural = "s" if len(sheets) != 1 else ""
    lines = [f"Spreadsheet {path.name}: {len(sheets)} sheet{plural}."]
    for name, fields in sheets:
        rows = len(fields[0].values) if fields else 0
        lines.append(f"Sheet {name}: {rows:,} rows, {len(fields)} columns.")
        described = []
        for column in fields:
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
        lines.extend(f"  {line}" for line in described)
    return "\n".join(lines)


def row_chunks(fields: list[Field], limits: dict, budget: int | None = None) -> list[str]:
    """Rows in groups, for sheets whose columns name things.

    A sheet of nothing but counts and codes gets none of these: its summary says
    everything there is to say. Nothing is truncated, and identical rows are
    merged and counted, since a second copy of the same sentence adds nothing to
    a text index.
    """
    carried = [
        column
        for column in fields
        if column.role(limits) in ("identifier", "prose", "categorical")
        and column.distinct > 1
        and column.dominant_share < limits["dominant_max_share"]
        and not all(BARE_NUMBER.match(value) for value in column.filled)
    ]
    if not any(c.role(limits) in ("identifier", "prose") for c in carried):
        return []

    seen: dict[str, int] = {}
    for index in range(len(carried[0].values)):
        parts = [
            f"{c.name}: {c.values[index]}" if not c.name.startswith(POSITIONAL)
            else c.values[index]
            for c in carried
            if index < len(c.values) and c.values[index].strip()
        ]
        if parts:
            line = "; ".join(parts)
            seen[line] = seen.get(line, 0) + 1

    chunks, group, size = [], [], 0
    for line, count in seen.items():
        rendered = line if count == 1 else f"{line} [{count} rows]"
        if group and (len(group) >= limits["rows_per_chunk"]
                      or (budget and size + len(rendered) > budget)):
            chunks.append("\n".join(group))
            group, size = [], 0
        group.append(rendered)
        size += len(rendered) + 1
    if group:
        chunks.append("\n".join(group))
    return chunks


class XlsHandler:
    """A workbook's sheets read as columns, not as prose."""

    name = "xls"
    extensions = (".xls",)

    def matches(self, path: Path, head: bytes) -> bool:
        if path.suffix.lower() not in self.extensions:
            return False
        return head.startswith(OLE_MAGIC) or head.startswith(BIFF_MAGIC)

    def parse(self, path: Path, cfg: dict, trace: Tracer) -> Extracted:
        limits = {**DEFAULTS, **(cfg.get("handlers", {}).get("xls") or {})}
        try:
            raw = read_sheets(path)
        except NotWorkbook as exc:
            raise Skip(f"unreadable workbook: {exc}") from exc
        except ImportError as exc:
            raise Skip(f"no reader installed: {exc}") from exc

        sheets = [(name, columns_of(name, rows, limits)) for name, rows in raw]
        sheets = [(name, fields) for name, fields in sheets if any(f.filled for f in fields)]
        if not sheets:
            raise Skip("workbook has no filled cells")

        segments: list[tuple[int | None, str]] = [(None, sheet_text(path, sheets, limits))]
        budget = cfg["chunk"]["max_characters"]
        detail = 0
        for _, fields in sheets:
            for chunk in row_chunks(fields, limits, budget):
                segments.append((None, chunk))
                detail += 1

        rows = sum(len(f.values) for _, fields in sheets for f in fields[:1])
        trace("xls", f"{len(sheets)} sheets, {rows:,} rows, {detail} row chunks")
        return Extracted(
            kind="xls",
            segments=segments,
            metadata=facts_of(sheets, limits),
            atomic=True,
        )
