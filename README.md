# geo-mini-rag

A small RAG pipeline built for cheap models, and a demonstration of what it
takes to read oil and gas file formats into one.

Two things it is for:

1. **A simple pipeline.** Walk a folder, read what can be read, chunk it, embed
   it, answer questions with citations. One DuckDB file holds everything. Both
   models are fixed and cheap on purpose — `openai/text-embedding-3-small`
   embeds, `inclusionai/ling-3.0-flash` answers. Comparing models is not the
   point.
2. **E&P formats as first-class citizens.** A well log, a seismic volume and a
   map layer are not text files, and treating them as text gets you nothing or
   nonsense. Each has a reader that pulls out the part a person would search
   for, and leaves the numbers alone.

## Setup

```sh
brew install libmagic     # or: apt install libmagic1
uv sync
cp .env.example .env      # add OPENROUTER_API_KEY
uv run geo-mini-rag models   # free: the two configured models, with live prices
```

## Use

```sh
uv run geo-mini-rag ingest             # read data/raw, embed via OpenRouter (paid, cents)
uv run geo-mini-rag stats              # what got indexed, and why the rest did not
uv run geo-mini-rag search "question"  # retrieval only, one tiny embedding call
uv run geo-mini-rag ask "question"     # answer with citations (paid)
uv run geo-mini-rag meta               # what metadata the index holds
```

Re-running `ingest` costs nothing for files that have not changed: it compares
size and mtime, and hashes content so a second copy of a file is recorded as a
duplicate instead of indexed twice. `--rebuild` starts over.

To watch every step on one folder:

```sh
uv run geo-mini-rag ingest --root data/subset --db data/index/subset.duckdb --trace
```

## What the E&P readers do

| format | what is indexed | what is not |
|---|---|---|
| **LAS** well logs | the header as a sentence, plus one fact per curve, depth range, well, field, operator | the log curves themselves |
| **SEG-Y** seismic | the EBCDIC textual header, decoded, plus sample rate and trace geometry | the traces |
| **SEG-P1** positioning | header labels, line names, shotpoint range, point count | the coordinates |
| **ESRI shapefile** | title and abstract from `.shp.xml`, CRS from `.prj`, extent from `.shp`, and the `.dbf` attributes worth searching | the geometry |

`.dbf` attributes are sorted by measurement rather than by name, because field
names are a vendor's abbreviations and there is no list to check them against.
A field whose values repeat is a category and becomes a filterable fact; one
whose values are nearly all distinct names individual things and goes into the
text; one that is 90% the same value describes the layer, not the row; one that
is bare digits is dropped, because retrieval is text and a number that names
nothing cannot be searched for.

**API well numbers** are found in any document, not just the ones with headers
— a completion report, a loader log, a scanned permit. They are validated
against `src/geo_mini_rag/ep/data/api_codes.csv`, this project's table of state
and county codes, and a label (`API`, `UWI`) must precede the digits. Measured
on 992 documents that have nothing to do with wells: 139,629 bare 10/12/14-digit
runs, of which 4,607 carry a plausible state and county and would pass on
structure alone, and none survive the label rule.

## Retrieval

Cosine similarity, with three corrections that matter for a collection like
this one:

- **An identifier in the question is looked up, not ranked.** `well 4902511080`
  scores 0.324 against the log that carries it and 0.729 against a page of
  unrelated digits. Looked up, it is exact: recall@1 goes from 36% to 92%.
- **Metadata the question names lifts the documents that carry it**, weighted by
  rarity, so a well name held by one document outweighs `state=WYOMING` held by
  1,375. `--where key=value` filters instead, including on numbers
  (`--where depth_max>5000`).
- **No document may take more than `per_document` of the answer.** A map layer
  of 2,111 wells is 500 chunks that read alike, and without this it held every
  place in the top ten for any question about wells.

`geo-mini-rag eval evals/subset.jsonl` scores retrieval against questions with
known answers: recall@k and MRR, no model judging the output.

## Layout

```
config/rag.yaml          every setting there is
src/geo_mini_rag/
  cli.py                 the commands
  openrouter.py          the one HTTP client
  rag/
    ingest.py            walk, dedupe, parse, chunk, embed
    store.py             the DuckDB file and its tables
    search.py            ranking
    parse.py             route a file to a reader
    extract.py           pdf, docx, html, text
    chunk.py             combine elements up to a size
    ocr.py               scanned PDFs
  ep/                    las, segy, segp1, shapefile, api_number
data/raw/                documents to ingest (gitignored)
evals/                   question sets with known answers
```

## Third-party notices

File type detection uses [libmagic](https://www.darwinsys.com/file/) through
[python-magic](https://github.com/ahupp/python-magic) (MIT). libmagic is
Copyright (c) Ian F. Darwin 1986-1995 and Christos Zoulas 2003, under its own
BSD-style licence.

Sample corpus: [GovDocs1](https://digitalcorpora.org/corpora/file-corpora/files/)
(Garfinkel et al., DFRWS 2009), freely redistributable for research.
