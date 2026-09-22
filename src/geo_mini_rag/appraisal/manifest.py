"""Manifest storage shared by the passes.

Every pass writes the same row shape to two places: the `appraisal` table in
DuckDB (for querying) and data/manifests/pass<n>_<manifest_id>.jsonl (which the
next pass reads, and which diffs between policy runs).
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from geo_mini_rag import settings
from geo_mini_rag.errors import UserError
from geo_mini_rag.rag.index import _sql
from geo_mini_rag.rag.trace import OFF, Tracer

# Column order for the appraisal table. Passes fill what they know; the rest
# stays null until a later pass has something to say.
COLUMNS = (
    "path", "part_of", "size", "mtime", "ext", "mime", "description",
    "ext_mismatch", "verdict", "reason", "sha256", "dup_of", "family_id", "family_size", "text_class", "signals",
)


def path_for(pass_n: int, manifest_id: str) -> Path:
    return settings.MANIFEST_DIR / f"pass{pass_n}_{manifest_id}.jsonl"


def ensure_tables(con: duckdb.DuckDBPyConnection, trace: Tracer = OFF) -> None:
    _sql(con, trace, """
        CREATE TABLE IF NOT EXISTS appraisal (
            manifest_id VARCHAR, pass INTEGER, path VARCHAR, part_of VARCHAR,
            size BIGINT, mtime DOUBLE,
            ext VARCHAR, mime VARCHAR, description VARCHAR, ext_mismatch BOOLEAN,
            verdict VARCHAR, reason VARCHAR, sha256 VARCHAR, dup_of VARCHAR,
            family_id VARCHAR, family_size INTEGER, text_class VARCHAR, signals VARCHAR
        )""")
    # indexes written before a pass added its columns
    for column, kind in (("sha256", "VARCHAR"), ("dup_of", "VARCHAR"), ("part_of", "VARCHAR"),
                         ("family_id", "VARCHAR"), ("family_size", "INTEGER"),
                         ("text_class", "VARCHAR"), ("signals", "VARCHAR")):
        _sql(con, trace, f"ALTER TABLE appraisal ADD COLUMN IF NOT EXISTS {column} {kind}")
    _sql(con, trace, """
        CREATE TABLE IF NOT EXISTS appraisal_runs (
            manifest_id VARCHAR, pass INTEGER, root VARCHAR, files BIGINT,
            excluded BIGINT, pending BIGINT, seconds DOUBLE, ran_at TIMESTAMP
        )""")


def latest(con: duckdb.DuckDBPyConnection, root: str | None = None) -> tuple[str, int]:
    """(manifest_id, highest pass) of the most recent appraisal run in this database."""
    ensure_tables(con, OFF)
    where, params = "", []
    if root:
        where, params = "WHERE root = ?", [root]
    row = con.execute(
        f"SELECT manifest_id, max(pass) FROM appraisal_runs {where} "
        "GROUP BY manifest_id ORDER BY max(ran_at) DESC LIMIT 1",
        params,
    ).fetchone()
    if not row:
        raise UserError(
            "no appraisal run in this index"
            + (f" for root {root}" if root else "")
            + "; run `geo-mini-rag appraise` first"
        )
    return row[0], int(row[1])


def admitted(pass_n: int, manifest_id: str) -> list[str]:
    """Paths the manifest has not excluded: what ingest should look at."""
    return [r["path"] for r in read(pass_n, manifest_id) if r.get("verdict") != "EXCLUDE"]


def read(pass_n: int, manifest_id: str) -> list[dict]:
    """Rows written by an earlier pass."""
    path = path_for(pass_n, manifest_id)
    if not path.exists():
        raise UserError(f"no manifest at {path}; run the earlier pass first")
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write(
    con: duckdb.DuckDBPyConnection,
    manifest_id: str,
    pass_n: int,
    root: str,
    rows: list[dict],
    seconds: float,
    trace: Tracer = OFF,
) -> Path:
    ensure_tables(con, trace)
    _sql(con, trace, "BEGIN")
    _sql(con, trace, "DELETE FROM appraisal WHERE manifest_id = ? AND pass = ?", [manifest_id, pass_n])
    _sql(con, trace, "DELETE FROM appraisal_runs WHERE manifest_id = ? AND pass = ?", [manifest_id, pass_n])
    con.executemany(
        f"INSERT INTO appraisal VALUES (?, ?, {', '.join('?' * len(COLUMNS))})",
        [(manifest_id, pass_n, *(r.get(c) for c in COLUMNS)) for r in rows],
    )
    excluded = sum(1 for r in rows if r.get("verdict") == "EXCLUDE")
    _sql(con, trace, "INSERT INTO appraisal_runs VALUES (?, ?, ?, ?, ?, ?, ?, now())",
         [manifest_id, pass_n, root, len(rows), excluded, len(rows) - excluded, seconds])
    _sql(con, trace, "COMMIT")

    settings.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    out = path_for(pass_n, manifest_id)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps({"manifest_id": manifest_id, "pass": pass_n, **r}) + "\n")
    trace("manifest", f"{len(rows)} rows -> {out}")
    return out
