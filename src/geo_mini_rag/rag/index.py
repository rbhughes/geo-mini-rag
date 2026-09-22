"""DuckDB store for documents, chunks, and OpenRouter embeddings."""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from geo_mini_rag import openrouter, settings
from geo_mini_rag.errors import UserError
from geo_mini_rag.rag.chunk import chunks_from
from geo_mini_rag.rag.extract import Skip
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


def _embed_all(
    model: str, texts: list[str], batch_size: int, trace: Tracer
) -> tuple[list[list[float]], int, float, dict[str, str]]:
    vectors: list[list[float]] = []
    tokens, cost = 0, 0.0
    served: dict[str, str] = {}
    n_batches = math.ceil(len(texts) / batch_size)
    for b, i in enumerate(range(0, len(texts), batch_size), 1):
        batch = texts[i : i + batch_size]
        trace(
            "embed",
            f"batch {b}/{n_batches}: POST /embeddings model={model} inputs={len(batch)} "
            f"chars={sum(map(len, batch)):,}",
        )
        res = openrouter.embed(model, batch)
        u = res.usage
        trace(
            "embed",
            f"batch {b}/{n_batches}: provider={res.provider} prompt_tokens={u.get('prompt_tokens')} "
            f"cost=${u.get('cost')} latency={res.latency_s:.2f}s attempts={res.attempts}",
        )
        if trace.on:
            v = res.vectors[0]
            norm = math.sqrt(sum(x * x for x in v))
            trace(
                "embed",
                f"batch {b}/{n_batches}: {len(res.vectors)} vectors x {len(v)} dims; "
                f"vector 0 starts {[round(x, 4) for x in v[:4]]}, L2 norm {norm:.4f}",
            )
        served = {
            "embed_model_served": res.served_model or "unreported",
            "embed_provider": res.provider or "unreported",
        }
        vectors.extend(res.vectors)
        tokens += int(u.get("prompt_tokens") or 0)
        cost += float(u.get("cost") or 0)
    return vectors, tokens, cost, served


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
    chunking = cfg["chunk"]
    root_path = Path(root)
    trace(
        "config",
        f"db={db} root={root_path} embed_model={model} batch_size={batch_size} "
        f"max_characters={chunking['max_characters']} overlap={chunking['overlap']} "
        f"max_pdf_pages={cfg['extract']['max_pdf_pages']} max_text_bytes={cfg['extract']['max_text_bytes']:,}",
    )

    source = iter(paths) if paths is not None else _walk(root_path)
    if paths is not None:
        trace("walk", f"{len(paths)} files from the manifest; not walking {root_path}")

    with connect(db) as con:
        _init(con, model, rebuild, trace)
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
            if _has_table(con, "chunks"):
                _sql(con, trace, "DELETE FROM chunks WHERE doc_id = ?", [doc_id])
            _sql(con, trace, "DELETE FROM doc_meta WHERE doc_id = ?", [doc_id])
            _sql(con, trace, "DELETE FROM documents WHERE doc_id = ?", [doc_id])
            try:
                ex = parse(path, cfg, trace)
            except Skip as s:
                trace("skip", str(s))
                _doc_row(con, trace, doc_id, rel, st, "skipped", None, str(s))
                _sql(con, trace, "COMMIT")
                on_event(IngestEvent(rel, "skipped", str(s)))
                continue
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
                trace("error", f"extract raised {type(exc).__name__}: {exc}")
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
                _doc_row(
                    con, trace, doc_id, rel, st, "skipped", ex, "no extractable text"
                )
                _sql(con, trace, "COMMIT")
                on_event(IngestEvent(rel, "skipped", f"{ex.kind}: no extractable text"))
                continue
            try:
                tokens, cost = embed_and_store(
                    con, doc_id, chunks, model, batch_size, trace
                )
            except Exception as exc:  # noqa: BLE001 - record and move on; rerun retries it
                trace("error", f"embedding raised {type(exc).__name__}: {exc}")
                _sql(con, trace, "ROLLBACK")
                on_event(
                    IngestEvent(
                        rel, "error", f"embedding failed: {type(exc).__name__}: {exc}"
                    )
                )
                continue
            _doc_row(
                con,
                trace,
                doc_id,
                rel,
                st,
                "indexed",
                ex,
                None,
                n_chars=n_chars,
                n_chunks=len(chunks),
                embed_tokens=tokens,
                embed_cost=cost,
                ocr_path=ex.metadata.get("ocr_path"),
            )
            _store_metadata(con, doc_id, ex.metadata, trace)
            _sql(con, trace, "COMMIT")
            on_event(
                IngestEvent(
                    rel,
                    "indexed",
                    f"{ex.kind}, {len(chunks)} chunks, ${cost:.5f}",
                    cost,
                )
            )


def embed_and_store(con, doc_id: str, chunks, model: str, batch_size: int, trace: Tracer) -> tuple[int, float]:
    """Embed a document's chunks and write them. Shared by ingest and the OCR pass."""
    vectors, tokens, cost, served = _embed_all(model, [c.text for c in chunks], batch_size, trace)
    _ensure_chunks_table(con, len(vectors[0]), trace)
    _record_serving(con, served, trace)
    trace(
        "sql",
        f"INSERT INTO chunks VALUES (?, ?, ?, ?, ?)   x{len(chunks)} rows "
        f"(doc_id, ord, page, text, {len(vectors[0])}-float embedding)",
    )
    con.executemany(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
        [(doc_id, c.ord, c.page, c.text, v) for c, v in zip(chunks, vectors, strict=True)],
    )
    return tokens, cost


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
