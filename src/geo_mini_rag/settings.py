"""Project paths, environment, and configuration."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
MANIFEST_DIR = DATA_DIR / "manifests"   # one JSONL per appraisal pass
OCR_DIR = DATA_DIR / "ocr"              # OCR'd copies of held scans (text recovery)
INDEX_DIR = DATA_DIR / "index"          # DuckDB file with chunks + embeddings
EVALS_DIR = ROOT / "evals"

load_dotenv(ROOT / ".env")


def docs_root() -> str:
    """Root of the document drive; any fsspec URL or local path."""
    raw = os.environ.get("GEO_DOCS_ROOT", "data/raw")
    if "://" in raw or Path(raw).is_absolute():
        return raw
    return str(ROOT / raw)


def load_rag_config() -> dict:
    return yaml.safe_load((CONFIG_DIR / "rag.yaml").read_text())


def chat_model(override: str | None = None) -> str:
    """The one chat model, from config/rag.yaml, unless a command overrides it."""
    return override or load_rag_config()["answer"]["model"]


def embed_model(override: str | None = None) -> str:
    return override or load_rag_config()["embed"]["model"]
