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

To see what the pipeline reads out of one file, without a key or a database:

```sh
python -m geo_mini_rag.ep.inspect data/raw/las/some.las
python -m geo_mini_rag.ep.inspect --facts data/raw/gis/layer.shp
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

**Coordinate systems are filtered, not ranked.** A question that spells out
"NAD 1927 UTM Zone 13N" is asking for those layers, and the rarity boost cannot
deliver them: 21 layers out of 2,520 documents earn about 0.06, which will not
lift them past 68,000 chunks. An exact match on a projection or datum narrows
the candidates instead.

| the 10 questions naming a projection or datum | top hit right | recall@10 |
|---|---|---|
| ranked | 20% | 25% |
| filtered | **100%** | **91%** |

`evals/shapefile_sets.py` asks the two questions a person actually puts to a
pile of map layers — *do I have spatial data for X?* and *which shapefile has
Y?* — and scores **60% on the top hit, 83% recall@10** over 87 of them. An
earlier version asked "which layer has 53 features?", scored 0%, and was
measuring the question generator rather than the system: the realistic
direction, "how many features does Teapot_Wells have", returns rank 1.

A company name is tidied only as far as its trailing legal suffix, because
ENCANA appeared as three spellings across five files and split every answer
set. Real company-name normalisation — ampersands against "and", abbreviations,
subsidiaries, former names — is a larger job and deliberately out of scope.

`.dbf` attributes are sorted by measurement rather than by name, because field
names are a vendor's abbreviations and there is no list to check them against.
A field whose values repeat is a category and becomes a filterable fact; one
whose values are nearly all distinct names individual things and goes into the
text; one that is 90% the same value describes the layer, not the row; one that
is bare digits is dropped, because retrieval is text and a number that names
nothing cannot be searched for.

### Seismic, measured

Without a reader a SEG-Y file is `application/octet-stream`: libmagic cannot
name it, so it is dropped with a reason and nothing in it is searchable. With
one, the same 54 files yield 119 passages and 741 facts, every stored number
inside the range its field allows.

Ground truth comes from the headers themselves, the way the LAS test uses the
API each log declares: a file whose card reads `CLIENT: LITHOPROBE` belongs in
the answer set for a question about Lithoprobe. A seismic archive is many files
per survey, so these questions have several right answers and are scored
against the set.

47 questions, covering who shot and processed a survey, where it is, and what
a data loader needs to know: sample interval, samples per trace, trace length,
units, sample format.

| question | files | top hit right | recall@10 |
|---|---|---|---|
| which surveys were shot for Lithoprobe? | 23 | yes | 100% |
| which lines cover Abitibi-Grenville '93? | 23 | yes | 100% |
| which files are stored as 4-byte IBM floating point? | 52 | yes | 100% |
| which surveys were recorded in meters? | 27 | yes | 100% |
| where is seismic line 93D? | 4 | yes | 100% |
| which files have 4500 samples per trace? | 8 | yes | 100% |
| which lines have a trace length of 18000 ms? | 8 | yes | 100% |
| which files were recorded at a 4 ms sample interval? | 24 | yes | 100% |
| which files have 2000 samples per trace? | 7 | yes | 100% |
| **47 questions** | | **98%** | **99%** |

A seismic line named in a question is **looked up, not ranked**, the same way an
API number is. `93D` and `53` are short identifiers, which is what embeddings
are worst at:

| | top hit right | recall@10 |
|---|---|---|
| 8 line questions, ranked | 0% | 56% |
| 8 line questions, looked up | **100%** | **100%** |
| 11 trace-length questions, ranked | 0% | 42% |
| 11 trace-length questions, looked up | **100%** | **96%** |
| 8 sample-interval questions, ranked | 38% | 53% |
| 8 sample-interval questions, looked up | **100%** | **100%** |
| 12 sample-count questions, ranked | 75% | 93% |
| 12 sample-count questions, looked up | **100%** | **100%** |

A trace length is derived — interval times sample count — so the number appears
nowhere in the header text and nothing in the index resembles the question.
Ranking cannot reach it; a lookup answers it exactly. The unit has to be in the
question, so "18000 barrels" is not read as a trace length.

A sample interval is stored twice, in microseconds and milliseconds, so a
question in either unit becomes both and matches whichever way the file was
asked about.

Sample counts are the weakest case for a lookup and still worth it: unlike a
derived quantity, 2000 is real text in the header, so ranking already found the
right files most of the time and put the wrong one first a quarter of the time.

One question still misses, and it does not matter: "recorded in feet" has 22
equally correct answers and returns a different one first.

**Well numbers in a map layer.** A `.dbf` may or may not say which column holds
them, so there are two ways in. By name, for a column called API or UWI: its
values are kept exactly as the file writes them, because `2500153` is Natrona
025 and well 00153 with the state code missing and it is not the reader's job
to guess the rest. By content, for a column that says nothing: if most of a
column validates against the code table it is a column of well numbers,
whatever it is called. GeoGraphix layers here keep them under `DataId` and
`WellID`, and across every other numeric column in the corpus — `TypeId`,
`ObjectID`, `ParentCode`, `ASR_ID` — not one value validates.

The whole column is the evidence, which is why this needs no label where prose
does. 6,908 well numbers over four layers, none of them searchable before.

**A partial well number matches by its tail.** The same well is written at
different lengths by different systems: 1,148 of these wells appear as seven
digits in a map layer and ten in a log. `*2510867` finds both, and
`--where api=*2510867` filters on it.

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

`geo-mini-rag eval` scores retrieval against questions with known answers:
recall@k and MRR, with no model judging the output.

`evals/corpus.jsonl` holds 103 of them, one per document, spread across
formats rather than across the corpus — two thirds of the files are well logs,
and a proportional sample would only measure the LAS reader. Where it lands
today:

| format | n | recall@1 | recall@5 | MRR |
|---|---|---|---|---|
| LAS | 28 | 82% | 93% | 0.851 |
| PDF | 22 | 50% | 77% | 0.636 |
| plain text | 22 | 50% | 59% | 0.545 |
| HTML | 18 | 44% | 72% | 0.569 |
| shapefile | 10 | 60% | 90% | 0.703 |
| **all** | **103** | **58%** | **77%** | **0.663** |

Read recall@5 first: `top_k` is 6, so it is what the model actually sees.

Sample size is the reason this set exists. At 22 questions the 95% interval on
a recall figure is about +/- 16 points, so any change worth less than that was
indistinguishable from luck. At 103 it is about +/- 8, which is still wide
enough to be careful with.

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
