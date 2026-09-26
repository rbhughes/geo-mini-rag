"""Scoring for a question with several right answers.

A well log question has one correct document. A question about an archive
rarely does: "which surveys did Lithoprobe shoot" has 23 right answers and
"which layers are in UTM zone 13" has 21, so scoring them against one file
counts every other correct hit as a miss. Both set-based evals score the same
way and print the same table, so it lives here rather than twice.
"""

from __future__ import annotations

import pathlib

from geo_mini_rag.rag.search import search
from geo_mini_rag.settings import load_rag_config

DB = pathlib.Path("data/index/rag.duckdb")


def report(questions: list[tuple[str, set[str], str]], *, depth: int = 10,
           db: pathlib.Path = DB) -> None:
    """Print one row per question, then the totals.

    `questions` is (question, the paths that answer it, a label for grouping).
    """
    cfg = load_rag_config()
    heading = f'rec@{depth}'
    print(f"{'question':60} {'set':>4} {'hit@1':>6} {heading:>7}")
    top1, recalls = 0, []
    for question, want, _ in questions:
        hits, _ = search(question, depth, db, cfg=cfg)
        paths = [hit.path for hit in hits]
        first_is_right = bool(paths) and paths[0] in want
        recall = len({p for p in paths if p in want}) / min(len(want), depth)
        top1 += first_is_right
        recalls.append(recall)
        print(f"{question[:60]:60} {len(want):4} "
              f"{'yes' if first_is_right else 'no':>6} {recall:7.0%}")
    n = len(questions) or 1
    print(f"\ntop hit is a correct file: {top1}/{len(questions)} = {top1 / n:.0%}")
    print(f"mean recall@{depth} over the answer set: {sum(recalls) / n:.0%}")
