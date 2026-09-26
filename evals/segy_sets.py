"""SEG-Y retrieval scored against a set of right answers, not one.

A seismic archive is many files per survey, so "which lines did Lithoprobe
shoot" has 22 correct answers. Ground truth comes from the headers themselves,
the same self-labelling trick the LAS test uses: a file whose card says
CLIENT: LITHOPROBE belongs in the answer set for a question about Lithoprobe.
That measures retrieval, taking extraction as given.
"""
import collections

import duckdb
from _score import DB, report

con = duckdb.connect(str(DB), read_only=True)

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

# The questions a loader actually gets: what was this recorded at, how long is
# a trace, how many samples. These come from the binary header, so the answer
# set is every file recorded that way.
numeric = con.execute("""
    SELECT m.key, m.value, d.path FROM doc_meta m JOIN documents d USING (doc_id)
    WHERE d.kind = 'segy'
      AND m.key IN ('sample_interval_ms','sample_interval_us','samples_per_trace',
                    'trace_length_ms','measurement_system','sample_format')
""").fetchall()
by_value = collections.defaultdict(set)
for key, value, path in numeric:
    by_value[(key, value)].add(path)

NUMERIC_ASKS = {
    "sample_interval_ms": "which seismic files were recorded at a {} ms sample interval?",
    "sample_interval_us": "which seismic data has a sample interval of {} microseconds?",
    "samples_per_trace": "which seismic files have {} samples per trace?",
    "trace_length_ms": "which seismic lines have a trace length of {} ms?",
    "measurement_system": "which seismic surveys were recorded in {}?",
    "sample_format": "which seismic files are stored as {}?",
}
for (key, value), files in by_value.items():
    if len(files) >= 2:
        shown = value.removesuffix(".0")
        questions.append((NUMERIC_ASKS[key].format(shown), files, key))

questions.sort(key=lambda q: (q[2], -len(q[1])))

print(f"{len(questions)} questions, each with several right answers\n")
report(questions)
