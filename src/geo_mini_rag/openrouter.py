"""Minimal OpenRouter chat client.

Plain HTTP against the OpenAI-compatible endpoint. Every call asks OpenRouter
for real usage accounting, so ingest and evaluation can report dollar cost.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from geo_mini_rag.errors import UserError

BASE_URL = "https://openrouter.ai/api/v1"
RETRY_STATUS = {429, 500, 502, 503, 504}


@dataclass
class ChatResult:
    model: str
    text: str
    usage: dict[str, Any] = field(default_factory=dict)  # prompt_tokens, completion_tokens, cost
    provider: str | None = None
    latency_s: float = 0.0
    attempts: int = 1


def _headers() -> dict[str, str]:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise UserError("set OPENROUTER_API_KEY in .env (see .env.example)")
    return {
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": os.environ.get("OPENROUTER_APP_URL", "https://purr.io"),
        "X-Title": os.environ.get("OPENROUTER_APP_NAME", "geo-mini-rag"),
    }


def _post(path: str, body: dict[str, Any], parse, *, timeout_s: float, attempts: int, label: str):
    """POST with retries. `parse(data, attempt, latency)` builds the result; a KeyError,
    IndexError, or ValueError from it (e.g. an error body sent with HTTP 200) is retried."""
    started = time.monotonic()
    last: Exception | None = None
    with httpx.Client(base_url=BASE_URL, headers=_headers(), timeout=timeout_s) as client:
        for attempt in range(1, attempts + 1):
            try:
                r = client.post(path, json=body)
                if r.status_code in RETRY_STATUS:
                    wait = float(r.headers.get("Retry-After", 2**attempt))
                    last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                    time.sleep(min(wait, 30))
                    continue
                r.raise_for_status()
                return parse(r.json(), attempt, time.monotonic() - started)
            except (httpx.TransportError, KeyError, IndexError, ValueError) as exc:
                last = exc
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"{label}: gave up after {attempts} attempts: {last}")


def chat(
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 800,
    temperature: float = 0.0,
    timeout_s: float = 120.0,
    attempts: int = 4,
) -> ChatResult:
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "usage": {"include": True},
    }

    def parse(data, attempt, latency):
        return ChatResult(
            model=model,
            text=data["choices"][0]["message"].get("content") or "",
            usage=data.get("usage") or {},
            provider=data.get("provider"),
            latency_s=latency,
            attempts=attempt,
        )

    return _post("/chat/completions", body, parse, timeout_s=timeout_s, attempts=attempts, label=model)


@dataclass
class EmbedResult:
    model: str                  # the id we asked OpenRouter for
    served_model: str | None    # the name the upstream provider reported back
    vectors: list[list[float]]  # same order as the inputs
    usage: dict[str, Any] = field(default_factory=dict)  # prompt_tokens, cost
    provider: str | None = None
    latency_s: float = 0.0
    attempts: int = 1


def embed(model: str, inputs: list[str], *, timeout_s: float = 120.0, attempts: int = 4) -> EmbedResult:
    """Embed a batch of strings. The caller chooses the batch size."""
    body = {"model": model, "input": inputs, "encoding_format": "float", "usage": {"include": True}}

    def parse(data, attempt, latency):
        rows = sorted(data["data"], key=lambda d: d["index"])
        if len(rows) != len(inputs):
            raise ValueError(f"asked for {len(inputs)} embeddings, got {len(rows)}")
        return EmbedResult(
            model=model,
            served_model=data.get("model"),
            vectors=[row["embedding"] for row in rows],
            usage=data.get("usage") or {},
            provider=data.get("provider"),
            latency_s=latency,
            attempts=attempt,
        )

    return _post("/embeddings", body, parse, timeout_s=timeout_s, attempts=attempts, label=model)


def list_embedding_catalog() -> list[dict[str, Any]]:
    """OpenRouter's embedding models. Needs no API key."""
    r = httpx.get(f"{BASE_URL}/embeddings/models", timeout=30)
    r.raise_for_status()
    return r.json()["data"]


def list_catalog() -> list[dict[str, Any]]:
    """OpenRouter's public model catalog. Needs no API key."""
    r = httpx.get(f"{BASE_URL}/models", timeout=30)
    r.raise_for_status()
    return r.json()["data"]
