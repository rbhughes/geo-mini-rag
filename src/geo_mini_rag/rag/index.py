"""DuckDB store for documents, chunks, and OpenRouter embeddings."""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from geo_mini_rag import openrouter, settings
from geo_mini_rag.errors import UserError
from geo_mini_rag.rag.chunk import chunks_from
from geo_mini_rag.rag.extract import Extracted, Skip
from geo_mini_rag.rag.parse import parse
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
        for table in ("chunks", "documents", "meta"):
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
            embed_tokens BIGINT, embed_cost DOUBLE, ocr_path VARCHAR
        )""",
    )
    migrate(con, trace)


def migrate(con: duckdb.DuckDBPyConnection, trace: Tracer = OFF) -> None:
    """Bring an index built by an older version up to the current schema."""
    _sql(con, trace, "ALTER TABLE documents ADD COLUMN IF NOT EXISTS ocr_path VARCHAR")
    _sql(con, trace, """
        CREATE TABLE IF NOT EXISTS doc_meta (
            doc_id VARCHAR, key VARCHAR, value VARCHAR, num_value DOUBLE
        )""")
    _sql(con, trace, "ALTER TABLE doc_meta ADD COLUMN IF NOT EXISTS num_value DOUBLE")


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


def _embed_batches(
    model: str, texts: list[str], batch_size: int, workers: int, trace: Tracer
) -> tuple[list[list[float]], int, float, dict[str, str]]:
    """Embed many texts, several batches in flight at once.

    Batches are filled across documents rather than per document: most files
    produce a handful of chunks, and sending those one request at a time left
    the pipeline waiting on round trips instead of using the 100-input limit.
    """
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    results: list[openrouter.EmbedResult | None] = [None] * len(batches)
    trace("embed", f"{len(texts):,} texts in {len(batches)} batches, {workers} in flight")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(openrouter.embed, model, batch): n for n, batch in enumerate(batches)}
        for future in as_completed(futures):
            n = futures[future]
            res = future.result()   # a failed batch aborts the group; the caller records it
            results[n] = res
            trace(
                "embed",
                f"batch {n + 1}/{len(batches)}: {len(res.vectors)} vectors, "
                f"prompt_tokens={res.usage.get('prompt_tokens')} cost=${res.usage.get('cost')} "
                f"{res.latency_s:.2f}s attempts={res.attempts}",
            )

    vectors: list[list[float]] = []
    tokens, cost = 0, 0.0
    served: dict[str, str] = {}
    for res in results:
        assert res is not None
        vectors.extend(res.vectors)
        tokens += int(res.usage.get("prompt_tokens") or 0)
        cost += float(res.usage.get("cost") or 0)
        served = {
            "embed_model_served": res.served_model or "unreported",
            "embed_provider": res.provider or "unreported",
        }
    return vectors, tokens, cost, served


@dataclass
class _Parsed:
    """A document waiting for its chunks to be embedded with everyone else's."""

    doc_id: str
    rel: str
    st: os.stat_result
    ex: Extracted
    chunks: list
    n_chars: int

    @property
    def chars(self) -> int:
        return sum(len(c.text) for c in self.chunks)


@dataclass
class IngestEvent:
    path: str
    status: str  # indexed | skipped | unchanged | error
    detail: str = ""
    cost: float = 0.0


def _walk(root: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if not name.startswith("."):
                yield Path(dirpath) / name


def ingest(
    root: str,
    *,
    db: Path = DB_PATH,
    rebuild: bool = False,
    limit: int | None = None,
    embed_model: str | None = None,
    paths: list[Path] | None = None,
    on_event: Callable[[IngestEvent], None] = lambda e: None,
    trace: Tracer = OFF,
) -> None:
    if paths is None and "://" in root:
        raise UserError(
            "ingest reads local paths for now; fsspec URLs come with appraisal"
        )
    cfg = settings.load_rag_config()
    model = embed_model or cfg["embed"]["model"]
    batch_size = cfg["embed"]["batch_size"]
    workers = cfg["embed"].get("concurrency", 6)
    chunking = cfg["chunk"]
    root_path = Path(root)
    trace(
        "config",
        f"db={db} root={root_path} embed_model={model} batch_size={batch_size} "
        f"concurrency={workers} "
        f"max_characters={chunking['max_characters']} overlap={chunking['overlap']} "
        f"max_pdf_pages={cfg['extract']['max_pdf_pages']} max_text_bytes={cfg['extract']['max_text_bytes']:,}",
    )

    source = iter(paths) if paths is not None else _walk(root_path)
    if paths is not None:
        trace("walk", f"{len(paths)} files from the manifest; not walking {root_path}")

    pending: list[_Parsed] = []
    pending_chunks = 0
    flush_at = batch_size * workers

    with connect(db) as con:
        _init(con, model, rebuild, trace)

        def flush() -> None:
            nonlocal pending_chunks
            _flush_group(con, pending, model=model, batch_size=batch_size,
                         workers=workers, on_event=on_event, trace=trace)
            pending_chunks = 0

        for n, path in enumerate(source):
            if not path.exists():
                on_event(IngestEvent(str(path), "error", "listed in the manifest but missing"))
                continue
            if limit is not None and n >= limit:
                trace("walk", f"--limit {limit} reached; stopping")
                break
            rel = str(
                path.relative_to(settings.ROOT)
                if path.is_relative_to(settings.ROOT)
                else path
            )
            doc_id = hashlib.sha1(rel.encode()).hexdigest()[:16]
            st = path.stat()
            trace("file", f"#{n + 1} {rel}")
            trace(
                "walk",
                f"size={st.st_size:,} bytes mtime={st.st_mtime} doc_id={doc_id} (sha1 of path, first 16 hex)",
            )
            prev = _sql(
                con,
                trace,
                "SELECT size, mtime FROM documents WHERE doc_id = ?",
                [doc_id],
            ).fetchone()
            if prev and prev[0] == st.st_size and prev[1] == st.st_mtime:
                trace("walk", "same size and mtime as the indexed copy; skipping")
                on_event(IngestEvent(rel, "unchanged"))
                continue
            trace(
                "walk",
                "not indexed yet"
                if not prev
                else f"changed since indexing (was size={prev[0]} mtime={prev[1]})",
            )
            _sql(con, trace, "BEGIN")
            try:
                ex = parse(path, cfg, trace)
            except Skip as s:
                trace("skip", str(s))
                _replace_rows(con, trace, doc_id)
                _doc_row(con, trace, doc_id, rel, st, "skipped", None, str(s))
                _sql(con, trace, "COMMIT")
                on_event(IngestEvent(rel, "skipped", str(s)))
                continue
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
                trace("error", f"extract raised {type(exc).__name__}: {exc}")
                _replace_rows(con, trace, doc_id)
                _doc_row(
                    con,
                    trace,
                    doc_id,
                    rel,
                    st,
                    "error",
                    None,
                    f"{type(exc).__name__}: {exc}",
                )
                _sql(con, trace, "COMMIT")
                on_event(IngestEvent(rel, "error", f"{type(exc).__name__}: {exc}"))
                continue

            n_chars = sum(len(t) for _, t in ex.segments)
            chunks = chunks_from(
                ex.segments,
                max_characters=chunking["max_characters"],
                overlap=chunking["overlap"],
                atomic=ex.atomic,
                respect_page_breaks=chunking.get("respect_page_breaks", True),
            )
            if trace.on:
                for c in chunks:
                    page = f" page={c.page}" if c.page is not None else ""
                    trace("chunk", f"chunk {c.ord}{page} {len(c.text):,} chars  {trace.text(c.text)}")
            if not chunks:
                trace("skip", "extraction produced no text to chunk")
                _replace_rows(con, trace, doc_id)
                _doc_row(
                    con, trace, doc_id, rel, st, "skipped", ex, "no extractable text"
                )
                _sql(con, trace, "COMMIT")
                on_event(IngestEvent(rel, "skipped", f"{ex.kind}: no extractable text"))
                continue
            # Nothing is written yet: this document waits for a full batch.
            _sql(con, trace, "COMMIT")
            pending.append(_Parsed(doc_id, rel, st, ex, chunks, n_chars))
            pending_chunks += len(chunks)
            if pending_chunks >= flush_at:
                flush()

        flush()


def _flush_group(
    con,
    pending: list[_Parsed],
    *,
    model: str,
    batch_size: int,
    workers: int,
    on_event: Callable[[IngestEvent], None],
    trace: Tracer,
) -> None:
    """Embed a group of documents together, then write them in one transaction."""
    if not pending:
        return
    texts = [c.text for doc in pending for c in doc.chunks]
    try:
        vectors, tokens, cost, served = _embed_batches(model, texts, batch_size, workers, trace)
    except Exception as exc:  # noqa: BLE001 - the group is not written; a rerun retries it
        trace("error", f"embedding raised {type(exc).__name__}: {exc}")
        for doc in pending:
            on_event(IngestEvent(doc.rel, "error", f"embedding failed: {type(exc).__name__}: {exc}"))
        pending.clear()
        return

    group_chars = sum(doc.chars for doc in pending) or 1
    _sql(con, trace, "BEGIN")
    _ensure_chunks_table(con, len(vectors[0]), trace)
    _record_serving(con, served, trace)
    offset = 0
    arrow_rows: list[tuple[str, int, int | None, str, list[float]]] = []
    for doc in pending:
        take = len(doc.chunks)
        _replace_rows(con, trace, doc.doc_id)
        arrow_rows.extend(
            (doc.doc_id, c.ord, c.page, c.text, v)
            for c, v in zip(doc.chunks, vectors[offset : offset + take], strict=True)
        )
        offset += take
        # Batches span documents, so usage is apportioned by share of characters.
        share = doc.chars / group_chars
        doc_cost = cost * share
        _doc_row(
            con, trace, doc.doc_id, doc.rel, doc.st, "indexed", doc.ex, None,
            n_chars=doc.n_chars, n_chunks=take,
            embed_tokens=int(tokens * share), embed_cost=doc_cost,
            ocr_path=doc.ex.metadata.get("ocr_path"),
        )
        _store_metadata(con, doc.doc_id, doc.ex.metadata, trace)
        on_event(IngestEvent(doc.rel, "indexed", f"{doc.ex.kind}, {take} chunks, ${doc_cost:.5f}", doc_cost))
    _insert_chunks(con, arrow_rows, len(vectors[0]), trace)
    _sql(con, trace, "COMMIT")
    trace("sql", f"wrote {len(pending)} documents, {len(vectors):,} chunks in one transaction")
    pending.clear()


def _insert_chunks(con, rows: list[tuple], dim: int, trace: Tracer) -> None:
    """Bulk-load chunks through Arrow.

    Row-by-row `executemany` moves each of the 1536 floats across the Python
    boundary on its own: measured at 7 rows/s, against 2,150 rows/s for the
    documented bulk path of registering an Arrow table and selecting from it.
    """
    if not rows:
        return
    import pyarrow as pa

    flat = np.fromiter(
        (value for row in rows for value in row[4]), dtype=np.float32, count=len(rows) * dim
    )
    table = pa.table({
        "doc_id": pa.array([r[0] for r in rows], pa.string()),
        "ord": pa.array([r[1] for r in rows], pa.int32()),
        "page": pa.array([r[2] for r in rows], pa.int32()),
        "text": pa.array([r[3] for r in rows], pa.string()),
        "embedding": pa.FixedSizeListArray.from_arrays(pa.array(flat), dim),
    })
    trace("sql", f"INSERT INTO chunks SELECT * FROM <arrow table>   {len(rows):,} rows x {dim} dims")
    con.register("chunk_batch", table)
    try:
        con.execute("INSERT INTO chunks SELECT * FROM chunk_batch")
    finally:
        con.unregister("chunk_batch")


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
):
    _sql(
        con,
        trace,
        "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
        ],
    )


@dataclass
class Hit:
    rank: int
    score: float
    path: str
    page: int | None
    text: str
    matched: str = ""   # metadata values from the question that this document carries


def metadata_keys(con) -> list[tuple[str, int, int]]:
    """(key, documents carrying it, distinct values) — what can be filtered on."""
    if not _has_table(con, "doc_meta"):
        return []
    return con.execute(
        "SELECT key, count(*), count(DISTINCT value) FROM doc_meta GROUP BY 1 ORDER BY 2 DESC"
    ).fetchall()


def mentioned_metadata(con, question: str, cfg: dict) -> list[tuple[str, str, str, int]]:
    """(doc_id, key, value) where a value stored in the index appears in the question.

    Identifiers are what embeddings are worst at: 1,400 near-identical LAS
    headers rank alike for "the API number of NPR #3 #13SX11-11". Matching the
    question against values already in the index is exact, needs no model, and
    is limited to the vocabulary the handlers actually extracted.

    Both sides are reduced to words first, because neither punctuation nor word
    boundaries survive the trip from a header to a question. A LAS calls a well
    FLUOR 41 "X" #1-2 and the person asking writes FLUOR 41 X #1-2; matching the
    raw strings missed it. Going the other way, a bare substring test matched
    the curve named DEPT inside the word "depth", which lifted 1,400 logs for
    any question that mentioned depth.
    """
    if not _has_table(con, "doc_meta"):
        return []
    retrieve = cfg.get("retrieve", {})
    skip = set(retrieve.get("metadata_skip_keys", []))
    min_len = retrieve.get("metadata_min_value_length", 4)
    words = "trim(regexp_replace(lower({}), '[^a-z0-9]+', ' ', 'g'))"
    rows = con.execute(
        f"""
        SELECT m.doc_id, m.key, m.value, c.docs
        FROM doc_meta m
        JOIN (
            SELECT key, value, count(DISTINCT doc_id) AS docs FROM doc_meta GROUP BY 1, 2
        ) c USING (key, value)
        WHERE length({words.format("m.value")}) >= ?
          AND ' ' || {words.format("?")} || ' '
              LIKE '%' || ' ' || {words.format("m.value")} || ' ' || '%'
        """,
        [min_len, question],
    ).fetchall()
    return [r for r in rows if r[1] not in skip]


COMPARISONS = (">=", "<=", "!=", ">", "<", "=")


def _as_clauses(where) -> list[tuple[str, str, str]]:
    """Accept {key: value} or [(key, op, value)] and normalise to clauses."""
    if not where:
        return []
    if isinstance(where, dict):
        return [(k, "=", v) for k, v in where.items()]
    return list(where)


def _idf(docs_with_value: int, total_docs: int) -> float:
    """1.0 for a value only one document carries, 0.0 for one they all carry."""
    if total_docs <= 1 or docs_with_value >= total_docs:
        return 0.0
    return math.log(total_docs / max(docs_with_value, 1)) / math.log(total_docs)


def _identifiers(con: duckdb.DuckDBPyConnection, question: str) -> list[tuple[str, str]]:
    """Well identifiers the question names that this index actually holds."""
    from geo_mini_rag.ep.well_ids import in_question

    held = []
    for key, value in in_question(question):
        row = con.execute(
            "SELECT 1 FROM doc_meta WHERE key = ? AND value = ? LIMIT 1", [key, value]
        ).fetchone()
        if row:
            held.append((key, value))
    return held


def search(
    question: str,
    k: int,
    db: Path = DB_PATH,
    *,
    where: dict[str, str] | list[tuple[str, str, str]] | None = None,
    cfg: dict | None = None,
) -> tuple[list[Hit], openrouter.EmbedResult]:
    """Embed the question and rank chunks, with metadata in both stages.

    A well identifier in the question is looked up rather than ranked; `where`
    restricts the candidates; any other metadata value the question names lifts
    its documents up the ranking; and no document may take more than
    `per_document` of the k places, so one big file cannot fill the answer.
    """
    cfg = cfg if cfg is not None else settings.load_rag_config()
    boost = cfg.get("retrieve", {}).get("metadata_boost", 0.15)
    per_document = cfg.get("retrieve", {}).get("per_document", 0)
    lookup = cfg.get("retrieve", {}).get("identifier_lookup", True)

    with connect(db, read_only=True) as con:
        meta = dict(con.execute("SELECT key, value FROM meta").fetchall())
        if "dim" not in meta:
            raise UserError(f"{db} has no chunks yet; run `geo-mini-rag ingest`")
        model, dim = meta["embed_model"], meta["dim"]

        total_docs = con.execute("SELECT count(DISTINCT doc_id) FROM doc_meta").fetchone()[0] or 1
        by_doc: dict[str, list[str]] = {}
        looked_up: list[str] = []
        weights: dict[str, float] = {}
        for doc_id, key, value, docs in mentioned_metadata(con, question, cfg):
            by_doc.setdefault(doc_id, []).append(f"{key}={value}")
            # Rarity is the evidence, the same idea as inverse document frequency
            # in lexical search: a well name held by one document says far more
            # than state=WYOMING, which 1,375 documents carry. Scaled to [0, 1]
            # so a unique value takes the full boost.
            weights[doc_id] = max(weights.get(doc_id, 0.0), boost * _idf(docs, total_docs))

        filters, params = [], []
        if lookup:
            for key, value in _identifiers(con, question):
                # An identifier names one well. Ranking cannot find it -- the
                # digits embed close to any other digits -- so it is a lookup,
                # narrowing the candidates the way --where does. Only values the
                # index actually holds get this far, so it never empties a result.
                filters.append(
                    "d.doc_id IN (SELECT doc_id FROM doc_meta WHERE key = ? AND value = ?)"
                )
                params += [key, value]
                looked_up.append(f"{key}={value}")
        for key, op, value in _as_clauses(where):
            if op == "=":
                filters.append(
                    "d.doc_id IN (SELECT doc_id FROM doc_meta WHERE key = ? AND lower(value) = lower(?))"
                )
                params += [key, value]
                continue
            number = _as_number(value)
            if number is None:
                raise UserError(f"--where {key}{op}{value}: {value!r} is not a number")
            filters.append(
                f"d.doc_id IN (SELECT doc_id FROM doc_meta WHERE key = ? AND num_value {op} ?)"
            )
            params += [key, number]

        qres = openrouter.embed(model, [question])
        con.execute("CREATE OR REPLACE TEMP TABLE metadata_boost (doc_id VARCHAR, weight DOUBLE)")
        if weights:
            con.executemany("INSERT INTO metadata_boost VALUES (?, ?)", list(weights.items()))
        scored = f"""
            SELECT array_cosine_similarity(c.embedding, ?::FLOAT[{int(dim)}])
                   + coalesce(b.weight, 0) AS score,
                   d.path, c.page, c.text, d.doc_id
            FROM chunks c
            JOIN documents d USING (doc_id)
            LEFT JOIN metadata_boost b ON b.doc_id = d.doc_id
            {"WHERE " + " AND ".join(filters) if filters else ""}"""
        if per_document:
            # A layer of 2,111 wells is 452 chunks that read alike, and without
            # this it takes every place in the answer. Rank within each document
            # first, then across documents, so the k places go to k different
            # sources wherever there are that many.
            sql = f"""
                SELECT score, path, page, text, doc_id FROM (
                    SELECT *, row_number() OVER (PARTITION BY doc_id ORDER BY score DESC) AS seat
                    FROM ({scored})
                ) WHERE seat <= ? ORDER BY score DESC LIMIT ?"""
            rows = con.execute(sql, [qres.vectors[0], *params, per_document, k]).fetchall()
        else:
            rows = con.execute(f"{scored} ORDER BY score DESC LIMIT ?",
                               [qres.vectors[0], *params, k]).fetchall()

    hits = [
        Hit(i + 1, s, p, pg, t, ", ".join(dict.fromkeys([*looked_up, *by_doc.get(doc_id, [])])))
        for i, (s, p, pg, t, doc_id) in enumerate(rows)
    ]
    return hits, qres


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
