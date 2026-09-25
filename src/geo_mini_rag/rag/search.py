"""Rank chunks against a question.

Cosine similarity over the embeddings, with three corrections that matter for
this kind of collection: a well identifier in the question is looked up rather
than ranked, metadata the question names lifts the documents that carry it, and
no single document may take more than its share of the answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import duckdb

from geo_mini_rag import openrouter, settings
from geo_mini_rag.errors import UserError
from geo_mini_rag.rag.store import DB_PATH, _as_number, _has_table, connect


@dataclass
class Hit:
    rank: int
    score: float
    path: str
    page: int | None
    text: str
    matched: str = ""   # metadata values from the question that this document carries


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
    return rows


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
    from geo_mini_rag.ep.api_number import in_question

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
        for key, value in _identifiers(con, question):
            # An identifier names one well. Ranking cannot find it -- the digits
            # embed close to any other digits -- so it is a lookup, narrowing the
            # candidates the way --where does. Only values the index actually
            # holds get this far, so it never empties a result.
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

