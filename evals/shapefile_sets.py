"""Shapefile retrieval scored against a set of right answers.

The same shape as evals/segy_sets.py: ground truth comes from what the reader
lifted off the .prj, the .shp header and the .dbf, so a layer whose projection
reads NAD 1927 UTM Zone 13N belongs in the answer set for a question about it.
Measures retrieval and takes extraction as given.
"""
import collections
import pathlib

import duckdb

from geo_mini_rag.rag.search import search
from geo_mini_rag.settings import load_rag_config

cfg = load_rag_config()
db = pathlib.Path("data/index/rag.duckdb")
con = duckdb.connect(str(db), read_only=True)

ASKS = {
    "crs_name": "which map layers are in {}?",
    "crs_datum": "which map layers use the {} datum?",
    "geometry_type": "which map layers hold {} geometry?",
    "feature_count": "which map layer has {} features?",
    "keywords": "which map layer covers {}?",
    "title": "which map layer is titled {}?",
}
rows = con.execute(
    """SELECT m.key, m.value, d.path FROM doc_meta m JOIN documents d USING (doc_id)
       WHERE d.kind = 'shapefile' AND m.key IN ('crs_name','crs_datum','geometry_type',
                                                'feature_count','keywords','title')""").fetchall()
sets = collections.defaultdict(set)
for key, value, path in rows:
    sets[(key, value)].add(path)

questions = [(ASKS[k].format(v), files, k) for (k, v), files in sets.items() if len(files) >= 2]
questions.sort(key=lambda q: (q[2], -len(q[1])))

print(f"{len(questions)} questions, each with several right answers\n")
print(f"{'question':58} {'set':>4} {'hit@1':>6} {'rec@10':>7}")
top1, recalls = 0, []
for q, want, kind in questions:
    hits, _ = search(q, 10, db, cfg=cfg)
    paths = [h.path for h in hits]
    ok = bool(paths) and paths[0] in want
    recall = len({p for p in paths if p in want}) / min(len(want), 10)
    top1 += ok
    recalls.append(recall)
    print(f"{q[:58]:58} {len(want):4} {'yes' if ok else 'no':>6} {recall:7.0%}")
n = len(questions)
print(f"\ntop hit is a correct file: {top1}/{n} = {top1/n:.0%}")
print(f"mean recall@10 over the answer set: {sum(recalls)/n:.0%}")
