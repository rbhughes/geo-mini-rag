"""Rank chunks against a question.

Cosine similarity over the embeddings, with three corrections that matter for
this kind of collection: a well identifier in the question is looked up rather
than ranked, metadata the question names lifts the documents that carry it, and
no single document may take more than its share of the answer.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import duckdb

from geo_mini_rag import openrouter, settings
from geo_mini_rag.errors import UserError
from geo_mini_rag.rag.store import (
    DB_PATH,
    _as_number,
    _has_table,
    connect,
    has_text_index,
)


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


# Keys whose value names a group of documents rather than describing one. A
# question that spells out "NAD 1927 UTM Zone 13N" is asking for those layers,
# not for layers a little like them, and the boost cannot do it: 21 layers out
# of 2,520 earn about 0.06, which will not lift them past 68,000 chunks. An
# exact match on one of these narrows the candidates instead.
GROUP_KEYS = frozenset({"crs_name", "crs_datum"})

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


# The same well is written at whatever length the system that recorded it uses:
# a vendor's fourteen digits carry a sidetrack and a completion, a state agency
# writes ten, a map layer sometimes seven with the state left off. Two numbers
# name the same well when one runs on from the other -- sharing a start, where
# both begin at the state, or sharing an end, where one has dropped it. Below
# this many digits that is coincidence rather than evidence.
# Reciprocal rank fusion. Two retrievers disagree about scale -- a cosine
# similarity of 0.66 and a BM25 score of 6.6 are not comparable -- so their
# ranks are combined rather than their scores. 60 is the constant the method
# was published with; it flattens the difference between rank 1 and rank 2 so
# that agreeing on a document matters more than either ranking it first.
RRF_K = 60
CANDIDATES = 60        # how deep each retriever is read before fusing

BRIDGE_DIGITS = 7
WELL_KEYS = frozenset({"api", "uwi"})
# one runs on from the other: shares a start, shares an end, or is the same
_SAME_WELL = (
    "(value LIKE ? || '%' OR ? LIKE value || '%'"
    " OR value LIKE '%' || ? OR ? LIKE '%' || value)"
)


def _value_clause(key: str, value: object) -> tuple[str, list]:
    """How a stored value is matched against one a question or --where names.

    Three cases, and every caller wants the same three: a starred fragment
    matches the way a glob does, a well number matches at any length, and
    anything else matches as text or as a number, since the index stores
    18000.0 and a question says 18000.
    """
    value = str(value)
    if value.startswith("*") or value.endswith("*"):
        pattern = ("%" if value.startswith("*") else "") + value.strip("*") \
            + ("%" if value.endswith("*") else "")
        return "value LIKE ?", [pattern]
    if key in WELL_KEYS and value.isdigit() and len(value) >= BRIDGE_DIGITS:
        return _SAME_WELL, [value] * 4
    return "(lower(value) = lower(?) OR num_value = try_cast(? AS DOUBLE))", [value, value]


def _carrying(key: str, value: object) -> tuple[str, list]:
    """SQL for "the documents holding this fact"."""
    clause, args = _value_clause(key, value)
    return (f"d.doc_id IN (SELECT doc_id FROM doc_meta WHERE key = ? AND {clause})",
            [key, *args])


def _identifiers(con: duckdb.DuckDBPyConnection, question: str) -> list[tuple[str, str]]:
    """Identifiers the question names that this index actually holds.

    A well by its API number, a seismic line by its name. Both are short codes
    an embedding cannot place, and both are exact once looked up.
    """
    from geo_mini_rag.ep import api_number, segy

    held = []
    for key, value in api_number.in_question(question) + segy.in_question(question):
        clause, args = _value_clause(key, value)
        row = con.execute(
            f"SELECT 1 FROM doc_meta WHERE key = ? AND {clause} LIMIT 1", [key, *args]
        ).fetchone()
        if row:
            held.append((key, value))
    return held


def _fuse(dense: list, lexical: list, lexical_weight: float,
          per_document: int, k: int) -> list:
    """Combine two rankings by reciprocal rank, then cap how much one file takes.

    Ranks rather than scores, because a cosine similarity of 0.66 and a BM25
    score of 6.6 do not share a scale. A chunk both retrievers place well beats
    one that either places first.
    """
    fused: dict[int, list] = {}
    scores: dict[int, float] = {}
    for weight, ranking in ((1.0, dense), (lexical_weight, lexical)):
        for rank, row in enumerate(ranking, 1):
            rowid = row[1]
            fused.setdefault(rowid, row)
            scores[rowid] = scores.get(rowid, 0.0) + weight / (RRF_K + rank)

    order = sorted(scores, key=lambda rowid: -scores[rowid])
    out, per_doc = [], Counter()
    for rowid in order:
        _, _, path, page, text, doc_id = fused[rowid]
        if per_document and per_doc[doc_id] >= per_document:
            continue
        per_doc[doc_id] += 1
        out.append((scores[rowid], path, page, text, doc_id))
        if len(out) >= k:
            break
    return out


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
    lexical_weight = cfg.get("retrieve", {}).get("lexical_weight", 1.0)

    with connect(db, read_only=True) as con:
        meta = dict(con.execute("SELECT key, value FROM meta").fetchall())
        if "dim" not in meta:
            raise UserError(f"{db} has no chunks yet; run `geo-mini-rag ingest`")
        model, dim = meta["embed_model"], meta["dim"]

        total_docs = con.execute("SELECT count(DISTINCT doc_id) FROM doc_meta").fetchone()[0] or 1
        by_doc: dict[str, list[str]] = {}
        looked_up: list[str] = []
        weights: dict[str, float] = {}
        groups: list[tuple[str, str]] = []
        for doc_id, key, value, docs in mentioned_metadata(con, question, cfg):
            by_doc.setdefault(doc_id, []).append(f"{key}={value}")
            if key in GROUP_KEYS and (key, value) not in groups:
                groups.append((key, value))
                looked_up.append(f"{key}={value}")
                continue
            # Rarity is the evidence, the same idea as inverse document frequency
            # in lexical search: a well name held by one document says far more
            # than state=WYOMING, which 1,375 documents carry. Scaled to [0, 1]
            # so a unique value takes the full boost.
            weights[doc_id] = max(weights.get(doc_id, 0.0), boost * _idf(docs, total_docs))

        filters, params = [], []
        if groups:
            # Any of the named groups, not all of them: a question naming two
            # projections is asking for either.
            filters.append(
                "d.doc_id IN (SELECT doc_id FROM doc_meta WHERE "
                + " OR ".join(["(key = ? AND value = ?)"] * len(groups))
                + ")"
            )
            params += [part for pair in groups for part in pair]
        for key, value in _identifiers(con, question):
            # An identifier names one well. Ranking cannot find it -- the digits
            # embed close to any other digits -- so it is a lookup, narrowing the
            # candidates the way --where does. Only values the index actually
            # holds get this far, so it never empties a result.
            clause, args = _carrying(key, value)
            filters.append(clause)
            params += args
            looked_up.append(f"{key}={value}")
        for key, op, value in _as_clauses(where):
            if op == "=":
                clause, args = _carrying(key, value)
                filters.append(clause)
                params += args
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
                   c.rowid, d.path, c.page, c.text, d.doc_id
            FROM chunks c
            JOIN documents d USING (doc_id)
            LEFT JOIN metadata_boost b ON b.doc_id = d.doc_id
            {"WHERE " + " AND ".join(filters) if filters else ""}"""
        pool = max(CANDIDATES, k)
        dense = con.execute(f"{scored} ORDER BY score DESC LIMIT ?",
                            [qres.vectors[0], *params, pool]).fetchall()

        lexical = []
        if lexical_weight and has_text_index(con):
            # The same filters, so a lookup narrows both retrievers alike.
            matched = f"""
                SELECT fts_main_chunks.match_bm25(c.rowid, ?) AS score,
                       c.rowid, d.path, c.page, c.text, d.doc_id
                FROM chunks c
                JOIN documents d USING (doc_id)
                {"WHERE " + " AND ".join(filters) if filters else ""}"""
            lexical = con.execute(
                f"SELECT * FROM ({matched}) WHERE score IS NOT NULL ORDER BY score DESC LIMIT ?",
                [question, *params, pool]).fetchall()

        rows = _fuse(dense, lexical, lexical_weight, per_document, k)

    hits = [
        Hit(i + 1, s, p, pg, t, ", ".join(dict.fromkeys([*looked_up, *by_doc.get(doc_id, [])])))
        for i, (s, p, pg, t, doc_id) in enumerate(rows)
    ]
    return hits, qres

