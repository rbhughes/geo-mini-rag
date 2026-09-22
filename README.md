# geo-mini-rag

Appraise a directory of mixed oil and gas documents, then run retrieval-augmented
generation (RAG) over what survives.

**Scope.** This project is about appraisal, ingest and embedding for E&P
documents. The models are fixed and cheap on purpose: `inclusionai/ling-3.0-flash`
answers, `openai/text-embedding-3-small` embeds, both set in `config/rag.yaml`.
Comparing models is not the experiment; deciding what belongs in the index is.

## Setup

```sh
brew install libmagic  # or: apt install libmagic1 — file type detection
uv sync
cp .env.example .env   # add OPENROUTER_API_KEY
uv run geo-mini-rag models   # free: the two configured models, with live prices
uv run geo-mini-rag ping     # one tiny paid call to confirm the key
```

## RAG over a directory

```sh
uv run geo-mini-rag ingest            # extract and chunk data/raw, embed via OpenRouter (paid, cents)
uv run geo-mini-rag stats             # what got indexed, and why the rest did not
uv run geo-mini-rag search "question" # retrieval only (one tiny embedding call)
uv run geo-mini-rag ask "question"    # answer with citations (paid)
```

Everything the pipeline uses lives in `config/rag.yaml`: both models, the
chunk size and overlap, the extraction caps and the retrieval depth. `ask -m
<id>` and `ingest -e <id>` override a model for a single run, which is for
spot checks, not for sweeps.

To watch every step on a small sample, use a separate index file:

```sh
uv run geo-mini-rag ingest --root data/tiny --db data/index/tiny.duckdb --trace
uv run geo-mini-rag ask "question" --db data/index/tiny.duckdb
```

`--trace-chars 0` prints extracted text and chunks in full. `data/tiny/` holds
one random file per extension, copied out of `data/raw`. It sits outside
`data/raw` so an `ingest` of that root does not pick it up.

Sample material: GovDocs1 `thread0` (991 mixed files from .gov sites) sits in
`data/raw/govdocs1_thread0/`, from
`https://digitalcorpora.s3.amazonaws.com/corpora/files/govdocs1/threads/thread0.zip`.

## Layout

```
config/rag.yaml        models, chunking, extraction caps, retrieval depth
config/policy.yaml     appraisal policy; its hash is part of each manifest id
data/raw/              documents to ingest, or set GEO_DOCS_ROOT (gitignored)
data/manifests/        JSONL written by each appraisal pass (gitignored)
data/ocr/              OCR'd copies of held scans, from text recovery (gitignored)
data/index/            DuckDB chunks and embeddings (gitignored)
data/tiny/             one file per extension, for tracing a small ingest (gitignored)
evals/                 question sets with known answers
src/geo_mini_rag/      cli, settings, openrouter client, appraisal/, rag/
```

## Third-party notices

File type detection uses [libmagic](https://www.darwinsys.com/file/), the
library behind `file(1)`, through
[python-magic](https://github.com/ahupp/python-magic) (MIT). libmagic is
Copyright (c) Ian F. Darwin 1986-1995 and Christos Zoulas 2003, released
under its own BSD-style licence; see the `COPYING` file shipped with it.

Sample corpus: [GovDocs1](https://digitalcorpora.org/corpora/file-corpora/files/)
(Garfinkel et al., "Bringing Science to Digital Forensics with Standardized
Forensic Corpora", DFRWS 2009), freely redistributable for research.
