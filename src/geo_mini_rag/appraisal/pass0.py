"""Pass 0 — inventory.

Walk the drive, record what is there, and drop what is obviously not a document
before anything expensive happens. No full file reads: each file contributes its
path, size, mtime, extension and the first `inventory.magic_bytes` bytes, which
libmagic turns into a type verdict.

Output goes to two places, because they answer different questions:
  * the `appraisal` table in DuckDB, to poke at in a SQL client
  * data/manifests/pass0_<manifest_id>.jsonl, which the next pass reads and
    which diffs cleanly between policy runs

`manifest_id` is a hash of config/policy.yaml plus a hash of the inventory, so
the same drive under a different policy produces a new, comparable manifest
rather than an in-place rebuild.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import fsspec
import yaml

from geo_mini_rag import settings
from geo_mini_rag.appraisal import manifest
from geo_mini_rag.rag.index import DB_PATH, connect
from geo_mini_rag.rag.trace import OFF, Tracer

PASS = 0

# Extensions whose content type we can check against libmagic. Only clear
# contradictions are flagged; an unknown extension is not evidence of anything.
EXPECTED_MIME = {
    ".pdf": ("application/pdf",),
    ".htm": ("text/html", "text/plain", "text/xml"),
    ".html": ("text/html", "text/plain", "text/xml"),
    ".xml": ("text/xml", "application/xml", "text/plain"),
    ".txt": ("text/plain", "text/csv", "application/csv"),
    ".csv": ("text/csv", "text/plain"),
    ".doc": ("application/msword", "application/x-ole-storage"),
    ".xls": ("application/vnd.ms-excel", "application/x-ole-storage"),
    ".ppt": ("application/vnd.ms-powerpoint", "application/x-ole-storage"),
    ".docx": ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/zip"),
    ".xlsx": ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/zip"),
    ".pptx": ("application/vnd.openxmlformats-officedocument.presentationml.presentation", "application/zip"),
    ".jpg": ("image/jpeg",),
    ".jpeg": ("image/jpeg",),
    ".png": ("image/png",),
    ".gif": ("image/gif",),
    ".tif": ("image/tiff",),
    ".tiff": ("image/tiff",),
    ".zip": ("application/zip",),
    ".gz": ("application/gzip",),
}
# libmagic on a 2KB head cannot see a zip directory or an OLE2 stream table, so
# these answers are re-checked against the whole file, which is still cheap.
GENERIC_MIME = {
    "application/octet-stream",
    "application/x-ole-storage",
    "application/zip",
    "text/plain",
    "application/x-empty",
}


@dataclass
class Row:
    path: str           # relative to the project root where possible
    part_of: str | None  # the bundle primary this file belongs to, e.g. its .shp
    size: int
    mtime: float
    ext: str
    mime: str
    description: str
    ext_mismatch: bool
    verdict: str        # EXCLUDE here, or PENDING for everything still in play
    reason: str | None = None


@dataclass
class Pass0Result:
    manifest_id: str
    manifest_path: Path
    rows: list[Row]
    seconds: float

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in self.rows:
            key = row.verdict if row.verdict == "PENDING" else f"EXCLUDE: {row.reason}"
            out[key] = out.get(key, 0) + 1
        return out


def _rel(path: str) -> str:
    p = Path(path)
    return str(p.relative_to(settings.ROOT)) if p.is_relative_to(settings.ROOT) else str(p)


def _magic(path: str, head: bytes) -> tuple[str, str]:
    import magic

    try:
        mime = magic.from_buffer(head, mime=True)
        description = magic.from_buffer(head)
        if mime in GENERIC_MIME:
            mime = magic.from_file(path, mime=True)
            description = magic.from_file(path)
    except Exception as exc:  # noqa: BLE001 - a file we cannot type is still inventory
        return "application/octet-stream", f"libmagic failed: {exc}"
    return mime, description


def _bundle_key(name: str, sidecars: set[str]) -> tuple[str, str] | None:
    """(stem, sidecar extension) when this filename looks like a bundle sidecar.

    Handles doubled suffixes such as TensleepStructure.shp.xml, where the
    sidecar extension is ".shp.xml" and the stem is "TensleepStructure".
    """
    lowered = name.lower()
    for sidecar in sorted(sidecars, key=len, reverse=True):
        if lowered.endswith(sidecar):
            return name[: -len(sidecar)], sidecar
    return None


def _link_bundles(rows: list[Row], policy: dict, trace: Tracer = OFF) -> None:
    """Point each sidecar at its primary: same directory, same stem, primary present."""
    config = policy.get("bundles") or {}
    bundles = {k.lower(): [s.lower() for s in v] for k, v in (config.get("formats") or {}).items()}
    exclude = config.get("exclude_sidecars", False)
    if not bundles:
        return
    sidecar_to_primary = {s: primary for primary, sides in bundles.items() for s in sides}
    primaries = {}
    for row in rows:
        if row.ext in bundles:
            directory, _, name = row.path.rpartition("/")
            primaries[(directory, name[: -len(row.ext)].lower())] = row.path
    linked = 0
    for row in rows:
        directory, _, name = row.path.rpartition("/")
        found = _bundle_key(name, set(sidecar_to_primary))
        if not found:
            continue
        stem, sidecar = found
        primary = primaries.get((directory, stem.lower()))
        if primary and primary != row.path:
            row.part_of = primary
            if exclude and row.verdict == "PENDING":
                row.verdict = "EXCLUDE"
            if not row.reason or row.verdict == "EXCLUDE":
                row.reason = f"bundle_sidecar: {sidecar_to_primary[sidecar]}"
            linked += 1
    trace("bundle", f"{linked} sidecars linked to {len(primaries)} primaries")


def _junk_reason(name: str, size: int, policy: dict) -> str | None:
    inventory = policy.get("inventory", {})
    if name in inventory.get("junk_names", []):
        return "junk_name"
    for prefix in inventory.get("junk_prefixes", []):
        if name.startswith(prefix):
            return "junk_prefix"
    if size == 0:
        return "zero_bytes"
    return None


def inventory(
    root: str,
    policy: dict,
    *,
    limit: int | None = None,
    on_row: Callable[[Row], None] = lambda r: None,
    trace: Tracer = OFF,
) -> list[Row]:
    head_bytes = policy.get("inventory", {}).get("magic_bytes", 2048)
    not_documents = {e.lower() for e in policy.get("not_documents", {}).get("extensions", [])}
    special = {
        ext.lower()
        for exts in policy.get("special_classes", {}).values()
        for ext in exts
    }
    fs, base = fsspec.core.url_to_fs(root)
    trace("walk", f"{type(fs).__name__} walking {base}")
    listing = fs.find(base, detail=True)
    trace("walk", f"{len(listing)} files found; magic_bytes={head_bytes}")

    rows: list[Row] = []
    for n, (key, info) in enumerate(sorted(listing.items())):
        if limit is not None and n >= limit:
            trace("walk", f"--limit {limit} reached")
            break
        name = key.rsplit("/", 1)[-1]
        size = int(info.get("size") or 0)
        ext = Path(name).suffix.lower()
        reason = _junk_reason(name, size, policy)
        if reason:
            row = Row(_rel(key), None, size, float(info.get("mtime") or 0), ext, "", "", False,
                      "EXCLUDE", reason)
            rows.append(row)
            on_row(row)
            continue
        with fs.open(key, "rb") as f:
            head = f.read(head_bytes)
        mime, description = _magic(key, head)
        expected = EXPECTED_MIME.get(ext)
        mismatch = bool(expected) and mime not in expected
        verdict, reason = "PENDING", None
        if ext in not_documents:
            verdict, reason = "EXCLUDE", f"not_a_document: {ext}"
        elif ext in special:
            reason = f"special_class: {ext}"
        elif mismatch:
            # Recorded, not excluded: a document saved under the wrong extension
            # is still a document. Pass 5 decides what to do about it.
            reason = f"ext_mismatch: {ext} but {mime}"
        row = Row(_rel(key), None, size, float(info.get("mtime") or 0), ext, mime, description,
                  mismatch, verdict, reason)
        rows.append(row)
        on_row(row)
    _link_bundles(rows, policy, trace)
    return rows


def manifest_id(policy_text: str, rows: list[Row]) -> str:
    """Hash of the policy plus the inventory, so a policy change makes a new manifest."""
    inv = hashlib.sha256()
    for row in sorted(rows, key=lambda r: r.path):
        inv.update(f"{row.path}\0{row.size}\0{row.mtime}\n".encode())
    return hashlib.sha256(policy_text.encode() + inv.hexdigest().encode()).hexdigest()[:12]


def write(con: duckdb.DuckDBPyConnection, mid: str, root: str, rows: list[Row],
          seconds: float, trace: Tracer = OFF) -> Path:
    """Write this pass to DuckDB and to its JSONL manifest."""
    return manifest.write(con, mid, PASS, root, [asdict(r) for r in rows], seconds, trace)


def run(
    root: str,
    *,
    db: Path = DB_PATH,
    limit: int | None = None,
    on_row: Callable[[Row], None] = lambda r: None,
    trace: Tracer = OFF,
) -> Pass0Result:
    policy_path = settings.CONFIG_DIR / "policy.yaml"
    policy_text = policy_path.read_text()
    policy = yaml.safe_load(policy_text)
    trace("config", f"policy={policy_path} root={root} db={db}")

    started = time.monotonic()
    rows = inventory(root, policy, limit=limit, on_row=on_row, trace=trace)
    seconds = time.monotonic() - started
    mid = manifest_id(policy_text, rows)
    trace("manifest", f"manifest_id={mid} from policy + {len(rows)} inventory rows")

    with connect(db) as con:
        manifest = write(con, mid, root, rows, seconds, trace)
    return Pass0Result(mid, manifest, rows, seconds)
