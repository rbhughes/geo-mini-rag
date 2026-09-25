"""Score retrieval against a question set with known answers.

Deliberately retrieval-only: it asks whether the chunk that answers the
question was retrieved at all, which is what the E&P work changes. No chat
model is involved, so a run is cheap, deterministic and free of the question
of how well a small model reasons.

A question set is JSONL, one object per line:

    {"q": "...",                  the question
     "expect_path": "...",        substring of the path that should be hit
     "expect_page": 2,            optional page number
     "expect_text": "one-eighth", optional substring the chunk must contain
     "note": "..."}               optional, for humans
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from geo_mini_rag.errors import UserError
from geo_mini_rag.rag.search import search
from geo_mini_rag.rag.store import DB_PATH

DEPTHS = (1, 3, 5, 10)


@dataclass
class Question:
    q: str
    expect_path: str | None = None
    expect_page: int | None = None
    expect_text: str | None = None
    note: str = ""


@dataclass
class Result:
    question: Question
    rank: int | None          # 1-based rank of the first hit that satisfies it
    top_path: str
    cost: float = 0.0


@dataclass
class Report:
    results: list[Result] = field(default_factory=list)
    cost: float = 0.0

    @property
    def n(self) -> int:
        return len(self.results)

    def recall_at(self, k: int) -> float:
        if not self.results:
            return 0.0
        hits = sum(1 for r in self.results if r.rank is not None and r.rank <= k)
        return hits / len(self.results)

    @property
    def mrr(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 / r.rank for r in self.results if r.rank) / len(self.results)


def load(path: Path) -> list[Question]:
    if not path.exists():
        raise UserError(f"no question set at {path}")
    questions = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise UserError(f"{path}:{n} is not valid JSON: {exc}") from exc
        if "q" not in row:
            raise UserError(f"{path}:{n} has no 'q' field")
        questions.append(Question(**row))
    if not questions:
        raise UserError(f"{path} has no questions")
    return questions


def satisfies(hit, question: Question) -> bool:
    if question.expect_path and question.expect_path not in hit.path:
        return False
    if question.expect_page is not None and hit.page != question.expect_page:
        return False
    return not (question.expect_text and question.expect_text.lower() not in hit.text.lower())


def run(questions: list[Question], *, db: Path = DB_PATH, k: int = max(DEPTHS),
        on_result=lambda r: None) -> Report:
    report = Report()
    for question in questions:
        hits, qres = search(question.q, k, db)
        cost = float(qres.usage.get("cost") or 0)
        rank = next((h.rank for h in hits if satisfies(h, question)), None)
        result = Result(question, rank, hits[0].path if hits else "", cost)
        report.results.append(result)
        report.cost += cost
        on_result(result)
    return report
