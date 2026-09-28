"""Spreadsheet retrieval, scored against a set of right answers.

Written because the 103-question corpus set has no spreadsheet in it at all --
it predates the reader -- so those 69 workbooks had been measured only for the
harm they might do to other questions, never for whether one can be found when
it is wanted.

Ground truth is mechanical, the way the SEG-Y and shapefile sets build theirs:
every question comes from a value the index actually holds, and its answer set
is every document holding that value, whatever format it is in. Nothing here is
written by hand or by a model, so there is no question this set can ask that the
pipeline is not genuinely expected to answer.

Four kinds, and the last two are a controlled pair:

    column       "which file has a column called Unit or Lease Name?"
                 The rule a spreadsheet gets is that a numeric or date column
                 gives up its header and nothing else. These questions are what
                 keeping the header buys.

    containment  "which file lists SALT CREEK?"
                 A value from a textual column, the thing a person actually
                 remembers about a sheet.

    well         "which spreadsheet has well 2506988?" and again as
                 "*2506988*". The workbook writes seven digits, the state code
                 dropped, so a bare number is not treated as an identifier at
                 all: at that length it could be anything, and the wildcard is
                 how a fragment is declared to be one. Both forms are asked, so
                 the boundary is a number rather than a footnote.

    format       the containment questions again, with the word "spreadsheet"
                 in them. Nothing in a chunk records which format it came from,
                 so this measures whether naming the format helps, hurts, or
                 does nothing. The two rows are comparable by construction.
"""

import collections
import re

import duckdb
from _score import DB, report

con = duckdb.connect(str(DB), read_only=True)

# A value a person could repeat: letters, a few words, no punctuation soup.
NAMELIKE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .,&/'-]{5,33}$")


def holders(key: str, value: str) -> set[str]:
    """Every document carrying this value, in any format."""
    return {p for p, in con.execute(
        """SELECT DISTINCT d.path FROM doc_meta m JOIN documents d USING (doc_id)
           WHERE m.key = ? AND m.value = ?""", [key, value]).fetchall()}


# --- column headers, which is what the numeric rule pays for -----------------
columns = []
for value, docs in con.execute(
        """SELECT m.value, count(DISTINCT m.doc_id) n
           FROM doc_meta m JOIN documents d USING (doc_id)
           WHERE d.kind = 'xls' AND m.key = 'field'
           GROUP BY 1 HAVING n <= 3 ORDER BY 1""").fetchall():
    if NAMELIKE.match(value) and not value.lower().startswith("column "):
        columns.append((f"which file has a column called {value}?",
                        holders("field", value), "column"))

# --- values from the textual columns -----------------------------------------
by_value = collections.defaultdict(set)
for key, value, path in con.execute(
        """SELECT m.key, m.value, d.path FROM doc_meta m JOIN documents d USING (doc_id)
           WHERE d.kind = 'xls' AND m.key NOT IN ('field', 'api', 'api_state', 'api_county')
        """).fetchall():
    if NAMELIKE.match(value):
        by_value[(key, value)].add(path)

containment, formatted = [], []
for (key, value), paths in sorted(by_value.items()):
    if len(paths) > 3:
        continue                      # too common to be asking about one file
    want = holders(key, value)
    if not 1 <= len(want) <= 6:
        continue
    containment.append((f"which file lists {value}?", want, "containment"))
    formatted.append((f"which spreadsheet lists {value}?", want, "format"))

# Keep the pair the same size and shape, and small enough to read.
containment, formatted = containment[::3][:24], formatted[::3][:24]

# --- well numbers, bare and starred ------------------------------------------
bare, starred = [], []
for (value,) in con.execute(
        """SELECT DISTINCT m.value FROM doc_meta m JOIN documents d USING (doc_id)
           WHERE d.kind = 'xls' AND m.key = 'api' ORDER BY m.value""").fetchall()[::80][:12]:
    # A fragment matches any longer number that runs on from it, so the answer
    # set is every document holding a number this one starts or ends.
    want = {p for p, in con.execute(
        """SELECT DISTINCT d.path FROM doc_meta m JOIN documents d USING (doc_id)
           WHERE m.key = 'api' AND m.value LIKE '%' || ?""", [value]).fetchall()}
    if want:
        bare.append((f"which spreadsheet has well {value}?", want, "well, bare"))
        starred.append((f"which spreadsheet has well *{value}*?", want, "well, starred"))

questions = columns + containment + formatted + bare + starred
print(f"{len(questions)} questions: {len(columns)} column, {len(containment)} containment, "
      f"{len(formatted)} format-qualified, {len(bare)} bare and {len(starred)} starred "
      f"well number\n")
report(questions)
