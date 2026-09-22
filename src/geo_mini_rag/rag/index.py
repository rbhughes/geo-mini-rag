"""DuckDB store for documents, chunks, and OpenRouter embeddings."""

from __future__ import annotations

import hashlib
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
            doc_id VARCHAR, key VARCHAR, value VARCHAR
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


def _store_metadata(con, doc_id: str, metadata: dict[str, str], trace: Tracer) -> None:
    """Document-level fields a domain handler lifted out, for filtering later."""
    if not metadata:
        return
    trace("meta", f"{len(metadata)} fields: {', '.join(sorted(metadata))}")
    con.executemany(
        "INSERT INTO doc_meta VALUES (?, ?, ?)",
        [(doc_id, k, v) for k, v in metadata.items()],
    )


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


def search(
    question: str, k: int, db: Path = DB_PATH
) -> tuple[list[Hit], openrouter.EmbedResult]:
    """Embed the question with the index's own model (one small paid call) and rank chunks."""
    with connect(db, read_only=True) as con:
        meta = dict(con.execute("SELECT key, value FROM meta").fetchall())
        if "dim" not in meta:
            raise UserError(f"{db} has no chunks yet; run `geo-mini-rag ingest`")
        model, dim = meta["embed_model"], meta["dim"]
        qres = openrouter.embed(model, [question])
        qvec = qres.vectors[0]
        rows = con.execute(
            f"""
            SELECT array_cosine_similarity(c.embedding, ?::FLOAT[{int(dim)}]) AS score, d.path, c.page, c.text
            FROM chunks c JOIN documents d USING (doc_id)
            ORDER BY score DESC LIMIT ?""",
            [qvec, k],
        ).fetchall()
    return [Hit(i + 1, s, p, pg, t) for i, (s, p, pg, t) in enumerate(rows)], qres


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
