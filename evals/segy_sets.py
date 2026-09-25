"""SEG-Y retrieval scored against a set of right answers, not one.

A seismic archive is many files per survey, so "which lines did Lithoprobe
shoot" has 22 correct answers. Ground truth comes from the headers themselves,
the same self-labelling trick the LAS test uses: a file whose card says
CLIENT: LITHOPROBE belongs in the answer set for a question about Lithoprobe.
That measures retrieval, taking extraction as given.
"""
import collections, pathlib
import duckdb
from geo_mini_rag.rag.search import search
from geo_mini_rag.settings import load_rag_config

cfg = load_rag_config()
db = pathlib.Path("data/index/rag.duckdb")
con = duckdb.connect(str(db), read_only=True)

rows = con.execute("""
    SELECT m.key, m.value, d.path FROM doc_meta m JOIN documents d USING (doc_id)
    WHERE d.kind = 'segy' AND m.key IN ('client','area','line','shot_by','processed_by')
""").fetchall()
sets = collections.defaultdict(set)
for key, value, path in rows:
    sets[(key, value)].add(path)

ASKS = {
    "client": "which seismic surveys were shot for {}?",
    "area": "which seismic lines cover {}?",
    "shot_by": "which seismic data was acquired by {}?",
    "processed_by": "which seismic lines were processed by {}?",
    "line": "where is seismic line {}?",
}

questions = [(ASKS[k].format(v), files, k) for (k, v), files in sets.items() if len(files) >= 2]
questions.sort(key=lambda q: -len(q[1]))

print(f"{len(questions)} questions, each with several right answers\n")
print(f"{'question':58} {'set':>4} {'hit@1':>6} {'rec@10':>7}")
top1 = 0
recalls = []
for q, want, kind in questions:
    hits, _ = search(q, 10, db, cfg=cfg)
    paths = [h.path for h in hits]
    first_ok = bool(paths) and paths[0] in want
    found = len({p for p in paths if p in want})
    recall = found / min(len(want), 10)
    top1 += first_ok
    recalls.append(recall)
    print(f"{q[:58]:58} {len(want):4} {'yes' if first_ok else 'no':>6} {recall:7.0%}")
n = len(questions)
print(f"\ntop hit is a correct file: {top1}/{n} = {top1/n:.0%}")
print(f"mean recall@10 over the answer set: {sum(recalls)/n:.0%}")
