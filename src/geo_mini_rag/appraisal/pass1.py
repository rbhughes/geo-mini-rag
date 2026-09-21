"""Pass 1 — exact duplicates.

Content-addressed, but stingy about reading. Files with a unique size cannot be
duplicates of anything, so they are never opened. Same-size files get a cheap
key built from the first and last `dedupe.prefilter_bytes` of each, and only
files that collide on that key are read in full for a SHA-256.

One file in each duplicate set is kept; the rest are EXCLUDE with `dup_of`
pointing at the keeper, so nothing is lost — a later pass, or a person, can
always see what was folded away and why.
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from geo_mini_rag import settings
from geo_mini_rag.appraisal import manifest
from geo_mini_rag.rag.index import DB_PATH, connect
from geo_mini_rag.rag.trace import OFF, Tracer

PASS = 1
READS = 65536  # streaming read size for the full digest


@dataclass
class Pass1Result:
    manifest_id: str
    manifest_path: Path
    rows: list[dict]
    seconds: float
    hashed: int = 0          # files read in full
    bytes_read: int = 0
    duplicates: int = 0
    bytes_duplicated: int = 0

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in self.rows:
            key = row["verdict"] if row["verdict"] == "PENDING" else f"EXCLUDE: {row.get('reason')}"
            out[key] = out.get(key, 0) + 1
        return out


def _quick_key(path: Path, size: int, prefilter: int) -> str:
    """size + head + tail, enough to separate same-size files without reading them whole."""
    h = hashlib.sha256(str(size).encode())
    with path.open("rb") as f:
        h.update(f.read(prefilter))
        if size > prefilter * 2:
            f.seek(-prefilter, 2)
            h.update(f.read(prefilter))
    return h.hexdigest()


def _full_digest(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    read = 0
    with path.open("rb") as f:
        while chunk := f.read(READS):
            h.update(chunk)
            read += len(chunk)
    return h.hexdigest(), read


def _keeper(paths: list[str]) -> str:
    """Shallowest path wins, then shortest, then alphabetical: deterministic and
    biased towards the copy that is not buried in someone's backup folder."""
    return min(paths, key=lambda p: (p.count("/"), len(p), p))


def run(
    manifest_id: str,
    *,
    db: Path = DB_PATH,
    root: str = "",
    on_row: Callable[[dict], None] = lambda r: None,
    trace: Tracer = OFF,
) -> Pass1Result:
    policy = yaml.safe_load((settings.CONFIG_DIR / "policy.yaml").read_text())
    prefilter = policy.get("dedupe", {}).get("prefilter_bytes", 1048576)

    rows = manifest.read(PASS - 1, manifest_id)
    trace("dedupe", f"{len(rows)} rows from pass {PASS - 1}; prefilter_bytes={prefilter:,}")
    started = time.monotonic()
    result = Pass1Result(manifest_id, manifest.path_for(PASS, manifest_id), rows, 0.0)

    # Only files still in play can be duplicates worth folding away.
    live = [r for r in rows if r.get("verdict") == "PENDING"]
    by_size: dict[int, list[dict]] = defaultdict(list)
    for row in live:
        by_size[int(row.get("size") or 0)].append(row)
    candidates = [group for group in by_size.values() if len(group) > 1]
    trace("dedupe", f"{len(live)} live files, {sum(len(g) for g in candidates)} share a size with another")

    digests: dict[str, list[dict]] = defaultdict(list)
    for group in candidates:
        quick: dict[str, list[dict]] = defaultdict(list)
        for row in group:
            path = settings.ROOT / row["path"]
            try:
                quick[_quick_key(path, int(row["size"]), prefilter)].append(row)
            except OSError as exc:
                trace("dedupe", f"unreadable, left alone: {row['path']}: {exc}")
        for key, same in quick.items():
            if len(same) == 1:
                continue
            trace("dedupe", f"{len(same)} files share quick key {key[:12]}; hashing in full")
            for row in same:
                try:
                    digest, read = _full_digest(settings.ROOT / row["path"])
                except OSError as exc:
                    trace("dedupe", f"unreadable, left alone: {row['path']}: {exc}")
                    continue
                row["sha256"] = digest
                result.hashed += 1
                result.bytes_read += read
                digests[digest].append(row)

    for digest, same in digests.items():
        if len(same) < 2:
            continue
        keeper = _keeper([r["path"] for r in same])
        for row in same:
            if row["path"] == keeper:
                continue
            row["verdict"] = "EXCLUDE"
            row["reason"] = "exact_duplicate"
            row["dup_of"] = keeper
            result.duplicates += 1
            result.bytes_duplicated += int(row.get("size") or 0)
            on_row(row)
        trace("dedupe", f"{len(same)} copies of {digest[:12]}, keeping {keeper}")

    result.seconds = time.monotonic() - started
    with connect(db) as con:
        result.manifest_path = manifest.write(con, manifest_id, PASS, root, rows, result.seconds, trace)
    return result
