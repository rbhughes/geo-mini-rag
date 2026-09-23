"""Retrieve, build a grounded prompt, and ask an OpenRouter model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from geo_mini_rag import openrouter
from geo_mini_rag.rag.index import DB_PATH, Hit, search

SYSTEM = (
    "You answer questions using only the numbered sources provided. "
    "Cite sources inline like [1] or [2][3]. If the sources do not contain the answer, "
    "say so plainly instead of guessing."
)


@dataclass
class Answer:
    text: str
    hits: list[Hit]
    result: openrouter.ChatResult
    query_embedding: openrouter.EmbedResult

    @property
    def cost(self) -> float:
        return float(self.result.usage.get("cost") or 0) + float(self.query_embedding.usage.get("cost") or 0)


def build_messages(question: str, hits: list[Hit]) -> list[dict[str, str]]:
    blocks = []
    for h in hits:
        where = h.path + (f", page {h.page}" if h.page else "")
        blocks.append(f"[{h.rank}] ({where})\n{h.text}")
    context = "\n\n---\n\n".join(blocks)
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Sources:\n\n{context}\n\nQuestion: {question}"},
    ]


def ask(question: str, *, model: str, k: int, max_tokens: int, db: Path = DB_PATH,
        where: dict[str, str] | None = None) -> Answer:
    hits, qres = search(question, k, db, where=where)
    result = openrouter.chat(model, build_messages(question, hits), max_tokens=max_tokens)
    return Answer(result.text, hits, result, qres)
