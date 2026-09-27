"""Group membership: a coordinate system, a state, a county.

These name a set of documents rather than describing one, and the rarity boost
cannot deliver them. 21 layers out of 2,520 earn about 0.06, which will not
lift them past 68,000 chunks. An exact match narrows the candidates instead.

Both arms are measured here, ranked against filtered, because the filtered
number alone says nothing. Ground truth is the index: the documents that carry
the value are the documents the question is asking for.

The filter keeps a document that is silent on the key. Only documents with a
parsed well number carry api_state, and excluding the rest cost the shapefile
set seven points -- a question naming Wyoming lost every road and benchmark
layer in the corpus. What a filter removes is the document that contradicts.
"""

import duckdb
from _score import DB, report

from geo_mini_rag.rag import search as search_mod

con = duckdb.connect(str(DB), read_only=True)

ASKED = (
    ("crs_name", "which layers are in {}?", 2),
    ("crs_datum", "which layers use the {} datum?", 2),
    ("api_state", "which wells are in {}?", 2),
    ("api_county", "which wells are in {} County?", 2),
)

questions = []
for key, ask, least in ASKED:
    rows = con.execute(
        """SELECT m.value, count(DISTINCT m.doc_id) AS docs
           FROM doc_meta m WHERE m.key = ? GROUP BY 1 HAVING docs >= ?
           ORDER BY docs DESC LIMIT 6""", [key, least]).fetchall()
    for value, _ in rows:
        want = {p for p, in con.execute(
            """SELECT DISTINCT d.path FROM doc_meta m JOIN documents d USING (doc_id)
               WHERE m.key = ? AND m.value = ?""", [key, value]).fetchall()}
        questions.append((ask.format(value), want, key))

print(f"{len(questions)} questions over {len({k for _, _, k in questions})} group keys\n")

for arm, keys in (("ranked", frozenset()), ("filtered", search_mod.GROUP_KEYS)):
    search_mod.GROUP_KEYS = keys
    print(f"--- {arm} ---")
    report(questions)
    print()
