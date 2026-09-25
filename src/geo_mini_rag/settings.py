"""Project paths, environment, and configuration."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
OCR_DIR = DATA_DIR / "ocr"              # searchable copies of scanned PDFs
INDEX_DIR = DATA_DIR / "index"          # DuckDB file with chunks + embeddings
EVALS_DIR = ROOT / "evals"

load_dotenv(ROOT / ".env")


def docs_root() -> str:
    """Folder of documents to ingest. Set GEO_DOCS_ROOT to point it elsewhere."""
    raw = os.environ.get("GEO_DOCS_ROOT", "data/raw")
    return raw if Path(raw).is_absolute() else str(ROOT / raw)


def load_rag_config() -> dict:
    return yaml.safe_load((CONFIG_DIR / "rag.yaml").read_text())


def chat_model(override: str | None = None) -> str:
    """The one chat model, from config/rag.yaml, unless a command overrides it."""
    return override or load_rag_config()["answer"]["model"]


def embed_model(override: str | None = None) -> str:
    return override or load_rag_config()["embed"]["model"]
