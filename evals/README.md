# evals

Question sets with known answers, one JSONL per set, each question tagged with
the document and page that answers it. Run outputs go to `evals/runs/`, which
is gitignored.

The models are fixed (see `config/rag.yaml`), so a run varies exactly one thing
about the corpus or the pipeline and reports what it did to the answers:

- **corpus condition** — every file under the root vs the appraisal INCLUDE list, plus the
  sub-conditions that matter on their own: near-duplicates collapsed or not,
  boilerplate stripped or not, held scans included after text recovery or not.
- **ingest settings** — chunk size and overlap, the extraction caps, how LAS
  files and other structured E&P text are summarized rather than chunked.
- **retrieval** — `top_k`, and whether exact-token search (well names, API
  numbers, depths) is mixed in with vector search.

What each run records: the manifest id (policy hash + inventory hash), the
embedding model and dimensions, chunk settings, retrieval settings, the chat
model, and the cost. Metrics: whether the answering chunk was retrieved at all,
whether the answer is right, whether its citation supports it, whether
unanswerable questions are refused, and dollars per question.

There is no public E&P question set, so ours is hand-built over the files we ingest,
including questions that cannot be answered from it.
