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
    "header_min_share": 0.6,        # this much of the first row must be labelled for it to be one
    "numeric_min_share": 0.9,       # this much of a column numeric, and a stray label is an outlier
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


def _rows_calamine(path: Path) -> list[tuple[str, list[list]]]:
    import python_calamine

    book = python_calamine.CalamineWorkbook.from_path(str(path))
    return [(name, book.get_sheet_by_name(name).to_python()) for name in book.sheet_names]


def _rows_xlrd(path: Path) -> list[tuple[str, list[list[str]]]]:
    """Excel 4.0 and earlier: raw BIFF, which calamine does not open."""
    import xlrd

    book = xlrd.open_workbook(str(path))
    out = []
    for index in range(book.nsheets):
        sheet = book.sheet_by_index(index)
        rows = []
        for r in range(sheet.nrows):
            row = []
            for c in range(sheet.ncols):
                cell = sheet.cell(r, c)
                if cell.ctype == xlrd.XL_CELL_DATE:
                    # A spreadsheet date carries no zone; inventing one would be worse.
                    row.append(dt.datetime(*xlrd.xldate_as_tuple(   # noqa: DTZ001
                        cell.value, book.datemode)))
                else:
                    row.append(cell.value)
            rows.append(row)
        out.append((sheet.name, rows))
    return out


def read_sheets(path: Path) -> list[tuple[str, list[list]]]:
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


def _kind(raw: list, numeric_min_share: float) -> str:
    """text | number | date, from what the cells are rather than how they print.

    A date cannot be told from a number by looking at the string: 1977-09-26 is
    digits and separators, and so is 1,977.26. The workbook knows which is
    which, so it is asked rather than guessed.

    A share rather than all of them, because one stray label does not make a
    column of readings textual. A sheet with a two-row header leaves its second
    label row stranded in the data, and requiring purity there let 4,610
    measurements through as prose.
    """
    present = [v for v in raw if v is not None and v != ""]
    if not present:
        return "text"
    dates = sum(isinstance(v, (dt.datetime, dt.date, dt.time)) for v in present)
    numbers = sum(isinstance(v, (int, float)) and not isinstance(v, bool) for v in present)
    if dates / len(present) >= numeric_min_share:
        return "date"
    if numbers / len(present) >= numeric_min_share:
        return "number"
    return "text"


def columns_of(name: str, rows: list[list], limits: dict) -> list[Field]:
    """A sheet's columns, named by its first row when that row is a header.

    A header row is one whose cells are short, present and distinct -- which is
    what a header is. When the first row fails that test the sheet has no
    header, the data starts at row one, and the columns are named by position
    so that nothing is invented.
    """
    if not rows:
        return []
    width = max(len(r) for r in rows)
    first = [_cell(rows[0][i]) if i < len(rows[0]) else "" for i in range(width)]
    # Most cells, not all: a real header often leaves a spacer column blank,
    # and demanding a full row rejected it and left the labels stranded in the
    # data below. The filled ones still have to be distinct, short, and not all
    # digits, which is what makes a header a header.
    named = [cell for cell in first if cell]
    labelled = (
        len(rows) > 1
        and len(named) >= limits["header_min_share"] * width
        and len(set(named)) == len(named)
        and all(len(cell) <= limits["header_max_length"] for cell in named)
        and not all(BARE_NUMBER.match(cell) for cell in named)
    )
    names = [cell or f"{POSITIONAL}{i + 1}" for i, cell in enumerate(first)] if labelled \
        else [f"{POSITIONAL}{i + 1}" for i in range(width)]
    body = rows[1:] if labelled else rows
    if limits["max_rows_per_sheet"]:
        body = body[: limits["max_rows_per_sheet"]]

    fields = []
    for i, column_name in enumerate(names):
        raw = [(r[i] if i < len(r) else None) for r in body]
        fields.append(Field(name=str(column_name),
                            kind=_kind(raw, limits["numeric_min_share"]),
                            values=[_cell(v) for v in raw]))
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


def carried_columns(fields: list[Field], limits: dict) -> list[Field]:
    """The columns whose values are worth indexing: the textual ones.

    Stricter than the rule for a .dbf, and deliberately. An attribute table is
    mostly names; a spreadsheet is mostly arithmetic, and nobody searches for a
    number that does not identify something. Two lab workbooks in data/raw made
    the case: 4,611 and 2,076 rows of microarray readings produced 8,100 of the
    10,300 spreadsheet chunks, none of them answerable by any question.

    So a numeric or date column contributes no values at all -- not to the text,
    not as a range. Its header still counts, because "Latitude" or "Spud Date"
    says what the sheet is about even when no reading in it is searchable.

    The exception is the identifier, which is what an API number is: those are
    read out of any column by `api_values`, numeric or not, and filed as facts,
    where identifiers belong in this pipeline anyway. A phone number or a
    postcode is the same shape of exception and is not implemented, because
    neither appears in this corpus and guessing at one would be inventing a
    format.
    """
    return [
        column
        for column in fields
        if column.kind == "text"
        and column.role(limits) in ("identifier", "prose", "categorical")
        and column.distinct > 1
        and column.dominant_share < limits["dominant_max_share"]
        and not all(BARE_NUMBER.match(value) for value in column.filled)
    ]


def facts_of(sheets: list[tuple[str, list[Field]]], limits: dict) -> dict:
    """Filterable facts: textual categories, well numbers, and the headers."""
    facts: dict = {}
    named: list[str] = []
    wells: list[str] = []
    for _, fields in sheets:
        wells += api_values(fields)
        for column in fields:
            if not column.filled or column.name.startswith(POSITIONAL):
                continue        # an unnamed column cannot be filtered on by name
            named.append(column.name)   # the header counts even when its values do not
            if column.kind != "text":
                continue
            if (column.role(limits) == "categorical"
                    and column.dominant_share < limits["dominant_max_share"]):
                facts.setdefault(column.name.lower(), [])
                for value in sorted(set(column.filled)):
                    if value not in facts[column.name.lower()]:
                        facts[column.name.lower()].append(value)
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
        described, plain = [], []
        for column in fields:
            if not column.filled:
                continue
            if column.kind != "text":
                # Named, not described: the values are unsearchable by design.
                if not column.name.startswith(POSITIONAL):
                    plain.append(f"{column.name} ({column.kind})")
                continue
            role = column.role(limits)
            if role == "categorical":
                values = sorted(set(column.filled))[:8]
                described.append(f"{column.name} ({column.distinct} values): {', '.join(values)}")
            else:
                described.append(f"{column.name}: {column.distinct:,} distinct {role} values")
        lines.extend(f"  {line}" for line in described)
        if plain:
            lines.append(f"  Numeric and date columns: {', '.join(plain)}")
    return "\n".join(lines)


def row_chunks(fields: list[Field], limits: dict, budget: int | None = None) -> list[str]:
    """Rows in groups, for sheets whose columns name things.

    A sheet of nothing but counts and codes gets none of these: its summary says
    everything there is to say. Nothing is truncated, and identical rows are
    merged and counted, since a second copy of the same sentence adds nothing to
    a text index.
    """
    carried = carried_columns(fields, limits)
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
