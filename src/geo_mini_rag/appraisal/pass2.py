"""Pass 2 — version families.

The FINAL_v3_REALLY_FINAL.doc problem. Still no content reads: filenames alone
are enough to spot that report.doc, report_v2.doc, report_v2 (1).doc and
report_v2_final.doc are four takes on one document.

Members are grouped and tagged with a `family_id`; nothing is excluded here.
Choosing which member survives needs signals this pass does not have — text
yield comes from the probe in pass 3 — so pass 5 picks the keeper using
`version_family.keeper`.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from geo_mini_rag import settings
from geo_mini_rag.appraisal import manifest
from geo_mini_rag.rag.index import DB_PATH, connect
from geo_mini_rag.rag.trace import OFF, Tracer

PASS = 2
MAX_ROUNDS = 8  # stem normalization is applied until stable, but never forever


@dataclass
class Pass2Result:
    manifest_id: str
    manifest_path: Path
    rows: list[dict]
    seconds: float
    families: int = 0
    grouped: int = 0                       # files that belong to a family
    largest: list[tuple[str, int]] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in self.rows:
            key = row["verdict"] if row["verdict"] == "PENDING" else f"EXCLUDE: {row.get('reason')}"
            out[key] = out.get(key, 0) + 1
        return out


def normalize_stem(stem: str, patterns: list[re.Pattern]) -> str:
    """Strip version, copy and date markers until nothing more comes off.

    A stem that is nothing but digits is left alone: on an E&P drive those are
    API numbers and UWIs, not versions of each other.
    """
    current = stem.strip()
    if current.isdigit():
        return current.lower()
    for _ in range(MAX_ROUNDS):
        before = current
        for pattern in patterns:
            current = pattern.sub("", current).strip()
        if current == before:
            break
    return current.strip(" _-.").lower()


def compile_patterns(policy: dict) -> list[re.Pattern]:
    raw = policy.get("version_family", {}).get("strip_patterns", [])
    return [re.compile(p, re.IGNORECASE) for p in raw]


def family_key(path: str, policy: dict, patterns: list[re.Pattern]) -> tuple[str, str, str]:
    """(scope key, normalized stem, extension) — rows sharing this are one family."""
    vf = policy.get("version_family", {})
    p = Path(path)
    stem = normalize_stem(p.stem, patterns)
    ext = p.suffix.lower() if vf.get("match_extension", True) else ""
    scope = str(p.parent) if vf.get("scope", "directory") == "directory" else ""
    return scope, stem, ext


def run(
    manifest_id: str,
    *,
    db: Path = DB_PATH,
    root: str = "",
    on_family: Callable[[str, list[dict]], None] = lambda fid, members: None,
    trace: Tracer = OFF,
) -> Pass2Result:
    policy = yaml.safe_load((settings.CONFIG_DIR / "policy.yaml").read_text())
    patterns = compile_patterns(policy)
    vf = policy.get("version_family", {})
    trace("family", f"scope={vf.get('scope', 'directory')} match_extension={vf.get('match_extension', True)} "
                    f"{len(patterns)} strip patterns")

    rows = manifest.read(PASS - 1, manifest_id)
    started = time.monotonic()
    live = [r for r in rows if r.get("verdict") == "PENDING"]

    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in live:
        groups[family_key(row["path"], policy, patterns)].append(row)

    result = Pass2Result(manifest_id, manifest.path_for(PASS, manifest_id), rows, 0.0)
    for n, (key, members) in enumerate(sorted((k, v) for k, v in groups.items() if len(v) > 1)):
        family_id = f"f{n:05d}"
        for row in sorted(members, key=lambda r: r["path"]):
            row["family_id"] = family_id
            row["family_size"] = len(members)
        result.families += 1
        result.grouped += len(members)
        result.largest.append((f"{key[1]}{key[2]}", len(members)))
        trace("family", f"{family_id}: {len(members)} versions of {key[1]}{key[2]} in {key[0]}")
        on_family(family_id, members)
    result.largest.sort(key=lambda kv: -kv[1])
    result.largest = result.largest[:10]

    result.seconds = time.monotonic() - started
    with connect(db) as con:
        result.manifest_path = manifest.write(con, manifest_id, PASS, root, rows, result.seconds, trace)
    return result
