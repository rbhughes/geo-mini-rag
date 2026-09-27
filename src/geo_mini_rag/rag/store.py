"""The DuckDB file: connecting to it, its tables, and what it can report.

One file holds everything: a row per document with why it was or was not
indexed, a row per chunk with its embedding, and a row per metadata fact.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import duckdb

from geo_mini_rag import settings
from geo_mini_rag.errors import UserError
from geo_mini_rag.rag.trace import OFF, Tracer

DB_PATH = settings.INDEX_DIR / "rag.duckdb"


class IndexBusy(UserError):
    """Another process holds the DuckDB write lock, normally a running ingest."""


def connect(db: Path = DB_PATH, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    if read_only and not db.exists():
        raise UserError(f"no index at {db}; run `geo-mini-rag ingest` first")
    db.parent.mkdir(parents=True, exist_ok=True)
    try:
        return duckdb.connect(str(db), read_only=read_only)
    except duckdb.IOException as exc:
        if "lock" not in str(exc).lower():
            raise
        pid = re.search(r"PID (\d+)", str(exc))
        holder = f"process {pid.group(1)}" if pid else "another process"
        raise IndexBusy(
            f"{db} is in use by {holder}, probably `geo-mini-rag ingest`. "
            "DuckDB allows one process at a time while it writes; try again when that finishes."
        ) from exc


def _sql(
    con: duckdb.DuckDBPyConnection,
    trace: Tracer,
    sql: str,
    params: list[Any] | None = None,
):
    """Execute and, when tracing, show the statement with a readable parameter summary."""
    if trace.on:
        shown = " ".join(sql.split())
        if params:
            shown += f"   params={[_param(p, trace) for p in params]}"
        trace("sql", shown)
    return con.execute(sql, params)


def _param(p: Any, trace: Tracer) -> str:
    if isinstance(p, str):
        return trace.text(p) if len(p) > 60 else repr(p)
    return repr(p)


def _init(
    con: duckdb.DuckDBPyConnection, model: str, rebuild: bool, trace: Tracer
) -> None:
    if rebuild:
        # doc_meta was added after this list and left out of it, so facts for
        # files since deleted from disk survived a rebuild as orphans.
        for table in ("chunks", "documents", "doc_meta", "meta"):
            _sql(con, trace, f"DROP TABLE IF EXISTS {table}")
    _sql(
        con,
        trace,
        "CREATE TABLE IF NOT EXISTS meta (key VARCHAR PRIMARY KEY, value VARCHAR)",
    )
    row = _sql(
        con, trace, "SELECT value FROM meta WHERE key = 'embed_model'"
    ).fetchone()
    if row and row[0] != model:
        raise UserError(
            f"index was built with {row[0]}, not {model}. Run ingest --rebuild."
        )
    _sql(con, trace, "INSERT OR REPLACE INTO meta VALUES ('embed_model', ?)", [model])
    _sql(
        con,
        trace,
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id VARCHAR PRIMARY KEY, path VARCHAR, size BIGINT, mtime DOUBLE,
            status VARCHAR, kind VARCHAR, reason VARCHAR,
            pages INTEGER, n_chars BIGINT, n_chunks INTEGER, truncated BOOLEAN,
            embed_tokens BIGINT, embed_cost DOUBLE, ocr_path VARCHAR, sha256 VARCHAR
        )""",
    )
    _sql(con, trace, """
        CREATE TABLE IF NOT EXISTS doc_meta (
            doc_id VARCHAR, key VARCHAR, value VARCHAR, num_value DOUBLE
        )""")


def _ensure_chunks_table(
    con: duckdb.DuckDBPyConnection, dim: int, trace: Tracer
) -> None:
    """The vector width is only known once the first embedding comes back."""
    row = _sql(con, trace, "SELECT value FROM meta WHERE key = 'dim'").fetchone()
    if row and int(row[0]) != dim:
        raise UserError(
            f"index vectors are {row[0]} wide but the model returned {dim}. Run ingest --rebuild."
        )
    if not row:
        _sql(con, trace, "INSERT INTO meta VALUES ('dim', ?)", [str(dim)])
    _sql(
        con,
        trace,
        f"""
        CREATE TABLE IF NOT EXISTS chunks (
            doc_id VARCHAR, ord INTEGER, page INTEGER, text VARCHAR, embedding FLOAT[{dim}]
        )""",
    )


def _replace_rows(con, trace: Tracer, doc_id: str) -> None:
    if _has_table(con, "chunks"):
        _sql(con, trace, "DELETE FROM chunks WHERE doc_id = ?", [doc_id])
    _sql(con, trace, "DELETE FROM doc_meta WHERE doc_id = ?", [doc_id])
    _sql(con, trace, "DELETE FROM documents WHERE doc_id = ?", [doc_id])


def _as_number(value) -> float | None:
    """The numeric reading of a value, when it has one: 570.0 is a depth, 'TEAPOT' is not."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def metadata_rows(doc_id: str, metadata: dict) -> list[tuple[str, str, str, float | None]]:
    """One row per fact. A list value becomes one row per item, so a log with ten
    curves gets ten `curve` rows rather than one string holding all of them."""
    rows: list[tuple[str, str, str, float | None]] = []
    for key, value in metadata.items():
        items = value if isinstance(value, (list, tuple, set)) else [value]
        for item in items:
            if item is None or str(item).strip() == "":
                continue
            rows.append((doc_id, key, str(item).strip(), _as_number(item)))
    return rows


def _store_metadata(con, doc_id: str, metadata: dict, trace: Tracer) -> None:
    """Document-level facts a domain handler lifted out, for filtering and ranking."""
    rows = metadata_rows(doc_id, metadata)
    if not rows:
        return
    numeric = sum(1 for r in rows if r[3] is not None)
    trace("meta", f"{len(rows)} facts over {len(metadata)} keys ({numeric} numeric)")
    con.executemany("INSERT INTO doc_meta VALUES (?, ?, ?, ?)", rows)


def _record_serving(con: duckdb.DuckDBPyConnection, served: dict[str, str], trace: Tracer) -> None:
    """Remember who actually answered: the id we asked for is not the whole story."""
    for key, value in served.items():
        row = _sql(con, trace, "SELECT value FROM meta WHERE key = ?", [key]).fetchone()
        if row and row[0] == value:
            continue
        _sql(con, trace, "INSERT OR REPLACE INTO meta VALUES (?, ?)", [key, value])


def build_vector_index(con: duckdb.DuckDBPyConnection, trace: Tracer = OFF) -> None:
    """Build the HNSW index the nearest-neighbour search reads.

    Without it every query is a full scan of every vector, which is 0.42s at
    68,000 chunks and linear from there. The planner only reaches the index
    through `array_cosine_distance`, so the ranking is written as a distance
    to minimise rather than a similarity to maximise; the two carry the same
    order, and only one of them can be indexed.
    """
    if not _has_table(con, "chunks"):
        return
    _sql(con, trace, "INSTALL vss")
    _sql(con, trace, "LOAD vss")
    _sql(con, trace, "SET hnsw_enable_experimental_persistence=true")
    _sql(con, trace, "DROP INDEX IF EXISTS chunks_hnsw")
    con.execute("CREATE INDEX chunks_hnsw ON chunks USING HNSW (embedding) "
                "WITH (metric = 'cosine')")
    trace("sql", "rebuilt the HNSW index over chunks.embedding")


# How many candidates the index keeps in play while it walks the graph. The
# default cost 12% of the true nearest neighbours here, which showed up as 14
# points of recall@5, for no speed at all: 128 overlaps the exact answer
# completely and is no slower than 64. An approximate index is only worth
# having if you check how approximate it is being.
HNSW_EF_SEARCH = 128


def load_vector_index(con: duckdb.DuckDBPyConnection) -> None:
    """Make the extension available to a reader, so the planner can use it."""
    try:
        con.execute("LOAD vss")
        con.execute(f"SET hnsw_ef_search={HNSW_EF_SEARCH}")
    except duckdb.Error:
        pass


def build_text_index(con: duckdb.DuckDBPyConnection, trace: Tracer = OFF) -> None:
    """Rebuild the BM25 index over the chunk text.

    Dense retrieval cannot place an identifier or a quantity, and this project
    added five detectors to work around that. BM25 does the general case: it
    matches the words as written. Neither one wins alone, so the two are fused.

    The index is a snapshot, not a live view, so it is rebuilt whenever chunks
    change -- 4 seconds for 68,000 of them.
    """
    if not _has_table(con, "chunks"):
        return
    _sql(con, trace, "INSTALL fts")
    _sql(con, trace, "LOAD fts")
    con.execute("PRAGMA create_fts_index('chunks', 'rowid', 'text', overwrite=1)")
    trace("sql", "rebuilt the BM25 index over chunks.text")


def has_text_index(con: duckdb.DuckDBPyConnection) -> bool:
    """Whether a BM25 index is present to search."""
    try:
        con.execute("LOAD fts")
        con.execute("SELECT 1 FROM fts_main_chunks.docs LIMIT 1")
        return True
    except duckdb.Error:
        return False


def _has_table(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    return bool(
        con.execute(
            "SELECT 1 FROM duckdb_tables() WHERE table_name = ?", [name]
        ).fetchone()
    )


def _doc_row(
    con,
    trace,
    doc_id,
    rel,
    st,
    status,
    ex,
    reason,
    *,
    n_chars=0,
    n_chunks=0,
    embed_tokens=0,
    embed_cost=0.0,
    ocr_path=None,
    sha256=None,
):
    _sql(
        con,
        trace,
        "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            doc_id,
            rel,
            st.st_size,
            st.st_mtime,
            status,
            ex.kind if ex else None,
            reason,
            ex.pages if ex else None,
            n_chars,
            n_chunks,
            ex.truncated if ex else False,
            embed_tokens,
            embed_cost,
            ocr_path,
            sha256,
        ],
    )


def metadata_keys(con) -> list[tuple[str, int, int]]:
    """(key, documents carrying it, distinct values) — what can be filtered on."""
    if not _has_table(con, "doc_meta"):
        return []
    return con.execute(
        "SELECT key, count(*), count(DISTINCT value) FROM doc_meta GROUP BY 1 ORDER BY 2 DESC"
    ).fetchall()


def stats(db: Path = DB_PATH) -> dict:
    with connect(db, read_only=True) as con:
        by_status = con.execute(
            "SELECT status, coalesce(kind, '-'), count(*), sum(n_chunks), sum(embed_tokens), sum(embed_cost)"
            " FROM documents GROUP BY ALL ORDER BY ALL"
        ).fetchall()
        reasons = con.execute(
            "SELECT reason, count(*) FROM documents WHERE status <> 'indexed' GROUP BY ALL ORDER BY 2 DESC LIMIT 15"
        ).fetchall()
        meta = dict(con.execute("SELECT key, value FROM meta").fetchall())
    return {
        "meta": meta,
        "embed_model": meta.get("embed_model", "unknown"),
        "by_status": by_status,
        "reasons": reasons,
    }

