"""Shapefile retrieval scored against a set of right answers.

The two questions a person actually asks of a pile of map layers:

    discovery    "do I have spatial data for roads in Fort Collins?"
    containment  "which shapefile has Mulberry Street?"

Neither is a statistic. An earlier version of this file asked "which layer has
53 features?", which scored 0% and which nobody asks; the realistic direction,
"how many features does Teapot_Wells have", already returns rank 1. Those
questions measured the generator, not the system.

Ground truth is mechanical. For discovery, the layer carrying that title or
keyword. For containment, every layer whose text actually holds the value --
checked against the index, so a value in three layers has three right answers.
"""

import collections
import pathlib
import re

import duckdb

from geo_mini_rag.rag.search import search
from geo_mini_rag.settings import load_rag_config

cfg = load_rag_config()
db = pathlib.Path("data/index/rag.duckdb")
con = duckdb.connect(str(db), read_only=True)

# Discovery: the subject a layer declares about itself. Grouped by value, so a
# keyword three layers share is one question with three right answers.
discovery = []
for key, ask in (
    ("title", "do I have spatial data for {}?"),
    ("keywords", "is there a shapefile covering {}?"),
):
    holders_by_value = collections.defaultdict(set)
    for value, path in con.execute(
        """SELECT m.value, d.path FROM doc_meta m JOIN documents d USING (doc_id)
           WHERE d.kind = 'shapefile' AND m.key = ?""", [key]).fetchall():
        if value.upper().startswith("REQUIRED") or len(value) < 6:
            continue          # ESRI boilerplate that survived as a title
        holders_by_value[value].add(path)
    for value, want in holders_by_value.items():
        subject = value if key == "title" else value.split(",")[0].strip()
        discovery.append((ask.format(subject), want, "discovery"))

# Containment: a value that appears in the attribute table of a few layers.
containment = []
seen_values = set()
rows = con.execute(
    """SELECT d.path, ch.text FROM chunks ch JOIN documents d USING (doc_id)
       WHERE d.kind = 'shapefile' AND ch.text LIKE '%Feature group%'""").fetchall()
holders = collections.defaultdict(set)
for path, text in rows:
    for line in text.split("\n")[2:]:
        for field in line.split("; "):
            if ": " not in field:
                continue
            value = field.split(": ", 1)[1].strip()
            # A name a person could repeat: letters, a few words, no quotes or
            # brackets. Survey markers -- "#6 REBAR DOWN 0.7' IN BOX" -- and
            # footage calls like "160 FSL" are descriptions, not names.
            if not (6 <= len(value) <= 34) or not re.search(r"[A-Za-z]{3}", value):
                continue
            if re.search(r"[\"'\[\]()]|^\d+\s+F[SNEW]L$", value):
                continue
            holders[value].add(path)

for value, layers in holders.items():
    if 1 <= len(layers) <= 3 and value.lower() not in seen_values:
        seen_values.add(value.lower())
        containment.append((f"which shapefile has {value}?", layers, "containment"))
containment.sort(key=lambda q: q[0])
containment = containment[::max(1, len(containment) // 20)][:20]

questions = discovery + containment
print(f"{len(questions)} questions: {len(discovery)} discovery, {len(containment)} containment\n")
print(f"{'question':60} {'set':>4} {'hit@1':>6} {'rec@10':>7}")
top1, recalls = 0, []
for q, want, kind in questions:
    hits, _ = search(q, 10, db, cfg=cfg)
    paths = [h.path for h in hits]
    ok = bool(paths) and paths[0] in want
    recall = len({p for p in paths if p in want}) / min(len(want), 10)
    top1 += ok
    recalls.append(recall)
    print(f"{q[:60]:60} {len(want):4} {'yes' if ok else 'no':>6} {recall:7.0%}")
n = len(questions)
print(f"\ntop hit is a correct layer: {top1}/{n} = {top1/n:.0%}")
print(f"mean recall@10 over the answer set: {sum(recalls)/n:.0%}")
