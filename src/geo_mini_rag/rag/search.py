"""Rank chunks against a question.

Cosine similarity over the embeddings, with three corrections that matter for
this kind of collection: a well identifier in the question is looked up rather
than ranked, metadata the question names lifts the documents that carry it, and
no single document may take more than its share of the answer.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import duckdb

from geo_mini_rag import openrouter, settings
from geo_mini_rag.ep import api_number
from geo_mini_rag.errors import UserError
from geo_mini_rag.rag.store import (
    DB_PATH,
    _as_number,
    _has_table,
    connect,
    has_text_index,
    load_vector_index,
)


@dataclass
class Hit:
    rank: int
    score: float
    path: str
    page: int | None
    text: str
    matched: str = ""   # metadata values from the question that this document carries


def mentioned_metadata(con, question: str, cfg: dict) -> list[tuple[str, str, str, int, int]]:
    """(doc_id, key, value, documents holding it, values this document holds for that key).

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
    # A two-letter state code is below the length floor, and has to be, since
    # half the codes are also ordinary words. It comes in through the one door
    # that reads them safely: capitalised in the question, or spelled out as a
    # state name. See api_number.states_in_question.
    codes = api_number.states_in_question(question)
    by_code = ""
    if codes:
        by_code = (" OR (m.key = 'api_state' AND m.value IN ("
                   + ", ".join("?" * len(codes)) + "))")
    rows = con.execute(
        f"""
        SELECT m.doc_id, m.key, m.value, c.docs, h.held
        FROM doc_meta m
        JOIN (
            SELECT key, value, count(DISTINCT doc_id) AS docs FROM doc_meta GROUP BY 1, 2
        ) c USING (key, value)
        JOIN (
            SELECT doc_id, key, count(*) AS held FROM doc_meta GROUP BY 1, 2
        ) h USING (doc_id, key)
        WHERE (
            length({words.format("m.value")}) >= ?
            AND ' ' || {words.format("?")} || ' '
                LIKE '%' || ' ' || {words.format("m.value")} || ' ' || '%'
        ){by_code}
        """,
        [min_len, question, *codes],
    ).fetchall()
    return rows


# Keys whose value names a group of documents rather than describing one. A
# question that spells out "NAD 1927 UTM Zone 13N" is asking for those layers,
# not for layers a little like them, and the boost cannot do it: 21 layers out
# of 2,520 earn about 0.06, which will not lift them past 68,000 chunks. An
# exact match on one of these narrows the candidates instead.
#
# A year is one too: "which wells were logged in 1977" names 169 documents out
# of 1,632, which the boost spreads evenly across and cannot tell apart. It
# needs _subsumed to be safe, because a projection name carries a year that is
# not a year -- NAD 1983 HARN StatePlane Colorado North.
#
# A state and a county are the same kind of thing. "What LAS files are in TX"
# names 11 documents out of 2,520 and the boost left every one of them off the
# first page, returning Wyoming logs; a filter answers it. Values within one
# key are alternatives and are OR'd; separate keys are separate conditions and
# are AND'd, so a question naming a state and a county asks for both.
GROUP_KEYS = frozenset({"crs_name", "crs_datum", "api_state", "api_county", "log_year"})

# A key is near-universal in a kind when almost every document of that kind
# carries it. Read from the index instead of listed, so a new handler that
# always records something gets the same treatment without an edit here.
UNIVERSAL_SHARE = 0.9


def _universal_kinds(con, key: str) -> list[str]:
    """Kinds where this key is near-universal, so not carrying it means not matching."""
    return [kind for kind, in con.execute(
        """
        SELECT d.kind
        FROM documents d
        LEFT JOIN (SELECT DISTINCT doc_id FROM doc_meta WHERE key = ?) m USING (doc_id)
        GROUP BY d.kind
        HAVING count(m.doc_id) >= ? * count(*)
        """, [key, UNIVERSAL_SHARE]).fetchall()]


COMPARISONS = (">=", "<=", "!=", ">", "<", "=")


def _as_clauses(where) -> list[tuple[str, str, str]]:
    """Accept {key: value} or [(key, op, value)] and normalise to clauses."""
    if not where:
        return []
    if isinstance(where, dict):
        return [(k, "=", v) for k, v in where.items()]
    return list(where)


def _as_words(text: str) -> list[str]:
    """The same reduction the match itself uses: lower case, alphanumeric runs."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).split()


def _covers(longer: list[str], shorter: list[str]) -> bool:
    """Whether one word sequence contains the other, whole and unbroken."""
    if not shorter or len(shorter) >= len(longer):
        return False
    return any(longer[i:i + len(shorter)] == shorter
               for i in range(len(longer) - len(shorter) + 1))


# A year is only a year when the question says something was done in it. The
# same rule as the one on API numbers, and for the same reason: four digits are
# four digits, and "the 2012 update readme" is not asking about a log date. On
# 1,632 documents the difference is 169 candidates or all of them.
WHEN_CUES = re.compile(
    r"(?i)\b(logged|logging|recorded|recording|run|ran|surveyed|survey|drilled|"
    r"spudded|completed|shot|acquired|vintage|dated|since|before|after|during)\b")
WHEN_KEYS = frozenset({"log_year"})


def _uncued(mentions, question: str) -> set[tuple[str, str]]:
    """Year matches the question gives no reason to read as a date."""
    if WHEN_CUES.search(question):
        return set()
    return {(key, value) for _, key, value, _, _ in mentions if key in WHEN_KEYS}


def _subsumed(mentions) -> set[tuple[str, str]]:
    """Matches a longer match already accounts for, which are not evidence.

    "Which shapefiles are in NAD 1983 HARN StatePlane Colorado North" names one
    thing, and the index answers with two: the projection, and a log_year of
    1983 that is simply four characters inside the projection's name. Both
    become filters, the filters are combined, and the answer disappears. The
    year in a coordinate system is not a year.

    So a matched value whose words sit whole and unbroken inside another
    matched value is dropped. The longer match explains more of the question
    and already covers the shorter one; keeping both counts the same span
    twice and, where the span was never about the shorter thing, wrongly.
    """
    seqs = {(key, value): _as_words(value) for _, key, value, _, _ in mentions}
    return {a for a, words in seqs.items()
            if any(_covers(other, words) for b, other in seqs.items() if b != a)}


def _idf(docs_with_value: int, total_docs: int) -> float:
    """1.0 for a value only one document carries, 0.0 for one they all carry."""
    if total_docs <= 1 or docs_with_value >= total_docs:
        return 0.0
    return math.log(total_docs / max(docs_with_value, 1)) / math.log(total_docs)


def _specificity(values_held: int, best_held: int) -> float:
    """How singular this match is, next to the most singular the question found.

    Rarity across the corpus is only half the evidence, and on its own it cannot
    tell identity from mention. Asked for well NPR 3 #51-41SX10UP4, the log that
    records it holds one `well` and is that well; a spreadsheet of 967 Teapot
    wells holds 71 `lease number` values, one of which is also unique in the
    corpus, so both took the full boost and the spreadsheet won on similarity.
    This is the document side of the same idea -- the term-frequency half that
    inverse document frequency is usually paired with.

    Relative, not absolute, because holding many values is what makes a document
    right for a question like "which shapefile has Mulberry Street": there the
    answer is a list, and an absolute penalty took that set from 45% to 5% on
    the top hit. So the comparison is against the best match the question
    actually found. When something claims the value as its identity, mentions
    rank below it; when every candidate is a list, none is penalised.

    Read from the index, so no list of keys counts as identifiers, and a
    2,111-row map layer, a 6,164-value text file and a 71-lease spreadsheet are
    treated alike. Identifier questions never reach it: api and uwi are looked
    up, and a projection, state or county filters.
    """
    return (1.0 + math.log(max(best_held, 1))) / (1.0 + math.log(max(values_held, 1)))


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
    vector_index = cfg.get("retrieve", {}).get("vector_index", False)

    with connect(db, read_only=True) as con:
        if vector_index:
            load_vector_index(con)
        meta = dict(con.execute("SELECT key, value FROM meta").fetchall())
        if "dim" not in meta:
            raise UserError(f"{db} has no chunks yet; run `geo-mini-rag ingest`")
        model, dim = meta["embed_model"], meta["dim"]

        total_docs = con.execute("SELECT count(DISTINCT doc_id) FROM doc_meta").fetchone()[0] or 1
        by_doc: dict[str, list[str]] = {}
        looked_up: list[str] = []
        weights: dict[str, float] = {}
        groups: list[tuple[str, str]] = []
        mentions = mentioned_metadata(con, question, cfg)
        covered = _subsumed(mentions) | _uncued(mentions, question)
        mentions = [m for m in mentions if (m[1], m[2]) not in covered]
        best_held = min((held for *_, held in mentions), default=1)
        for doc_id, key, value, docs, held in mentions:
            by_doc.setdefault(doc_id, []).append(f"{key}={value}")
            if key in GROUP_KEYS and (key, value) not in groups:
                groups.append((key, value))
                looked_up.append(f"{key}={value}")
                continue
            # Rarity is half the evidence, the same idea as inverse document
            # frequency in lexical search: a well name held by one document says
            # far more than state=WYOMING, which 1,375 documents carry. The other
            # half is whether the value is what this document is or merely one of
            # many it lists; see _specificity. Both are scaled to [0, 1], so the
            # full boost needs a value that is rare everywhere and singular here.
            weights[doc_id] = max(weights.get(doc_id, 0.0),
                                  boost * _idf(docs, total_docs) * _specificity(held, best_held))

        filters, params = [], []
        for key in dict.fromkeys(key for key, _ in groups):
            # Values under one key are alternatives -- a question naming two
            # projections is asking for either -- so they are OR'd, and each
            # key becomes its own condition.
            values = [value for group_key, value in groups if group_key == key]
            # Whether silence is a contradiction depends on the key, and the
            # index says which. Every shapefile has a .prj, so a layer that
            # does not answer to the projection named is in a different one and
            # goes. Only a document with a parsed well number carries
            # api_state, so a road layer saying nothing about a state is not
            # claiming to be outside it -- excluding those cost the shapefile
            # set seven points. The rule is read off the data rather than
            # listed here: among the kinds where a key is near-universal it
            # must match, and elsewhere silence passes.
            universal = _universal_kinds(con, key)
            clause = ("d.doc_id IN (SELECT doc_id FROM doc_meta WHERE key = ? AND value IN ("
                      + ", ".join("?" * len(values)) + "))")
            args = [key, *values]
            if universal:
                clause += (" OR (d.kind NOT IN (" + ", ".join("?" * len(universal)) + ")"
                           " AND d.doc_id NOT IN (SELECT doc_id FROM doc_meta WHERE key = ?))")
                args += [*universal, key]
            filters.append(f"({clause})")
            params += args
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
        # An HNSW index over the embeddings is built at ingest and switched
        # off here, because it was measured and it costs more than it buys.
        # The planner only reaches it through a bare
        # `ORDER BY array_cosine_distance(...) LIMIT n` over the table, so the
        # nearest chunks must be cut before the filter and the boost are
        # applied -- and both of those have to be exact. A filter must narrow
        # before the nearest are chosen, or a search for one well returns
        # whichever of its chunks happened to make a global top sixty. A boost
        # must reach anywhere: 63 of the 103 corpus questions carry one.
        # Merging the boosted documents back in recovered most of it, but not
        # all: recall@5 came back 78% -> 73% and MRR 0.668 -> 0.637, to save
        # 0.3s of a query whose embedding round trip is most of a second.
        # At 68,000 vectors an exact scan is the better trade. Turn
        # `vector_index` on to reproduce it.
        pool = max(CANDIDATES, k)
        exact = f"""
            SELECT array_cosine_distance(c.embedding, ?::FLOAT[{int(dim)}])
                   - coalesce(b.weight, 0) AS score,
                   c.rowid, d.path, c.page, c.text, d.doc_id
            FROM chunks c
            JOIN documents d USING (doc_id)
            LEFT JOIN metadata_boost b ON b.doc_id = d.doc_id
            WHERE {" AND ".join(filters) or "TRUE"}
            ORDER BY score ASC LIMIT ?"""
        if vector_index and not filters:
            dense = con.execute(f"""
                WITH nearest AS (
                    SELECT rowid FROM chunks
                    ORDER BY array_cosine_distance(embedding, ?::FLOAT[{int(dim)}]) LIMIT ?
                ),
                boosted AS (
                    SELECT rowid FROM chunks WHERE doc_id IN (SELECT doc_id FROM metadata_boost)
                ),
                candidates AS (SELECT rowid FROM nearest UNION SELECT rowid FROM boosted)
                SELECT array_cosine_distance(c.embedding, ?::FLOAT[{int(dim)}])
                       - coalesce(b.weight, 0) AS score,
                       c.rowid, d.path, c.page, c.text, d.doc_id
                FROM candidates n
                JOIN chunks c ON c.rowid = n.rowid
                JOIN documents d USING (doc_id)
                LEFT JOIN metadata_boost b ON b.doc_id = d.doc_id
                ORDER BY score ASC LIMIT ?""",
                [qres.vectors[0], pool, qres.vectors[0], pool]).fetchall()
        else:
            dense = con.execute(exact, [qres.vectors[0], *params, pool]).fetchall()

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

