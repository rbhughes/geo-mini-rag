"""Appraisal runs before any extraction or embedding and writes a manifest.

Passes, each reading the previous pass's JSONL in data/manifests/:
  0 inventory      fsspec walk + libmagic type verdict, extensions for E&P formats;
                   drop junk and non-documents
  1 exact dupes    sha256 with size + head/tail prefilter
  2 version family filename-stem normalization, no content reads
  3 text probe     bounded sample: PROSE, TABULAR, SCANNED, MIXED, NOT_TEXT
  4 near dupes     MinHash over probe text; flag boilerplate shingles
  5 decide         apply config/policy.yaml; INCLUDE, EXCLUDE, or HOLD per file
"""
