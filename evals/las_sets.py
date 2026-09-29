"""Well-log retrieval, scored against a set of right answers.

The LAS reader carries the project's central claim -- that an identifier has to
be looked up rather than ranked -- and until now it was the only domain reader
with no set of its own. Its 28 questions lived in corpus.jsonl, drafted by a
model, which put the flagship result on the softest evidence in the repository:
a 95% interval of about 13 points.

Built the way the SEG-Y set is built. Every question comes from a header field
the reader actually extracted, and the answer set is every document carrying
that value, so nothing is written by hand or by a model and no question can be
asked that the pipeline is not expected to answer.

The six kinds are what someone with a pile of logs actually wants to know: who
ran them, who logged them, where they are, when they were run, what curves are
in them, and the number of one named well.

The year questions score zero, and that is the set doing its job. A year names
a set of documents the way a state or a projection does -- log_year 1977 is on
169 of 1,632 -- so the rarity boost spreads evenly across all of them and
discriminates nothing. Filtering is the answer everywhere else this shape
appears, and adding log_year to GROUP_KEYS does take these questions from 0% to
100%. It is not done, because projection names contain years: "which shapefiles
are in NAD 1983 HARN StatePlane Colorado North" then matches log_year 1983 as
well, the two filters are ANDed, and the group set falls from 53% to 21% while
the hand-written subset loses 0.046 of its MRR. The question type is left
failing rather than paid for at that price.

Company names are asked by their root and answered by containment. FENIX &
SCISSON is written 23 ways across 461 documents -- FENIX & SCISSON INC, FENIX
AND SCISSON, FENIX & SCISSION, FENIX $ SCISSON -- so a question about one
spelling would score the other 22 as misses. Asking "which logs were run for
FENIX & SCISSON?" and accepting every log whose operator contains that name is
not normalisation, which the README declines to do; it is what the question
plainly means. The root is the shortest spelling in its family, so no question
is asked about a name that is only part of another company's.
"""

import collections

import duckdb
from _score import DB, report

con = duckdb.connect(str(DB), read_only=True)

# Sampled with a fixed stride rather than at random, so the set is the same on
# every run and a change in a number is a change in the system.
STRIDE = {"operator": 7, "service_company": 3, "field": 5, "log_year": 4,
          "curve_description": 23, "well": 101}
WANTED = 12


def values(key: str, least: int = 2, most: int | None = None) -> list[tuple[str, int]]:
    rows = con.execute(
        """SELECT m.value, count(DISTINCT m.doc_id) n
           FROM doc_meta m JOIN documents d USING (doc_id)
           WHERE d.kind = 'las' AND m.key = ? GROUP BY 1 HAVING n >= ?
           ORDER BY 1""", [key, least]).fetchall()
    return [(v, n) for v, n in rows if most is None or n <= most]


def holders(key: str, value: str) -> set[str]:
    """Every document carrying this value, in any format."""
    return {p for p, in con.execute(
        """SELECT DISTINCT d.path FROM doc_meta m JOIN documents d USING (doc_id)
           WHERE m.key = ? AND m.value = ?""", [key, value]).fetchall()}


def roots(key: str) -> list[tuple[str, int]]:
    """The shortest spelling of each company, with every spelling's documents."""
    all_values = [v for v, _ in values(key, least=1)]
    out = []
    for value, _ in values(key, least=2):
        if len(value) < 5:
            continue
        shorter = [o for o in all_values
                   if o != value and o.upper() in value.upper() and len(o) >= 5]
        if shorter:
            continue                       # a longer spelling of something else
        family = {p for p, in con.execute(
            """SELECT DISTINCT d.path FROM doc_meta m JOIN documents d USING (doc_id)
               WHERE m.key = ? AND upper(m.value) LIKE '%' || upper(?) || '%'""",
            [key, value]).fetchall()}
        if 2 <= len(family) <= 500:
            out.append((value, len(family)))
    return out


def family_of(key: str, value: str) -> set[str]:
    """Every document whose value for this key contains the name asked about."""
    return {p for p, in con.execute(
        """SELECT DISTINCT d.path FROM doc_meta m JOIN documents d USING (doc_id)
           WHERE m.key = ? AND upper(m.value) LIKE '%' || upper(?) || '%'""",
        [key, value]).fetchall()}


questions: list[tuple[str, set[str], str]] = []


def add(rows, ask, label, key):
    for value, _ in rows[::STRIDE[key]][:WANTED]:
        want = holders(key, value)
        if want:
            questions.append((ask.format(value), want, label))


for name, (ask, label, key) in (
        ("operator", ("which logs were run for {}?", "operator", "operator")),
        ("service_company", ("which wells did {} log?", "service company", "service_company"))):
    for value, _ in roots(name)[::STRIDE[key]][:WANTED]:
        questions.append((ask.format(value), family_of(name, value), label))
add(values("field", most=300), "which logs are from the {} field?", "field", "field")
add(values("log_year", most=300), "which wells were logged in {}?", "year", "log_year")

# A curve is what a petrophysicist searches a pile of logs for.
curves = [(v, n) for v, n in values("curve_description", most=400)
          if 6 <= len(v) <= 34 and v.replace(" ", "").isalpha()]
add(curves, "which logs have a {} curve?", "curve", "curve_description")

# One well, named. The identifier questions the whole project is about.
wells = [(v, n) for v, n in values("well", least=1, most=1) if 6 <= len(v) <= 40]
for value, _ in wells[::STRIDE["well"]][:WANTED]:
    want = holders("well", value)
    if want:
        questions.append((f"what is the API number for well {value}?", want, "well by name"))

counts = collections.Counter(label for _, _, label in questions)
print(f"{len(questions)} questions: " + ", ".join(f"{v} {k}" for k, v in counts.most_common()) + "\n")
report(questions)
