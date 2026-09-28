"""A column of values, and what measurement says it is for.

Written for `.dbf` attribute tables and reused for spreadsheets, because the
problem is the same in both: a vendor's column names are abbreviations with no
list to check them against, so what a column is for has to be read off its
values rather than its header. `TypeId`, `ASR_ID`, `WELL_NUMBE` and `Sheet1!C`
tell you nothing; how often their values repeat tells you everything.

    repeats                 a category, worth filtering on
    nearly all distinct     names individual things, belongs in the text
    long on average         prose, belongs in the text
    one value dominates     describes the table, not the row
    bare digits             dropped: retrieval is text, and a number that
                            names nothing cannot be searched for
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property

# Defaults; a handler's own section in config/rag.yaml overrides them.
DEFAULTS = {
    "categorical_max_ratio": 0.2,   # each value recurs at least five times
    "dominant_max_share": 0.9,      # a value this common describes the table, not the row
    "prose_min_length": 40,         # average characters, above which it is text not a label
}

# A value with no letters in it: 1401, 08031, 0002498.
BARE_NUMBER = re.compile(r"^[\d.,+-]+$")


@dataclass
class Field:
    name: str
    kind: str                       # text | number | date | boolean | memo
    length: int = 0
    values: list[str] = field(default_factory=list)

    @cached_property
    def filled(self) -> list[str]:
        """Values that say something. A field of blanks is not a field."""
        return [v.strip() for v in self.values if v.strip()]

    @cached_property
    def distinct(self) -> int:
        return len(set(self.filled))

    @cached_property
    def dominant_share(self) -> float:
        """How much of the field one value covers: 1.0 when every row agrees."""
        return max(Counter(self.filled).values()) / len(self.filled) if self.filled else 0.0

    def role(self, limits: dict) -> str:
        """empty | categorical | identifier | prose | number — measured, not guessed.

        Repetition is what makes a category: COMPANY holds 70 operators across
        2,111 wells, so every value recurs about thirty times and the field is
        worth filtering on. WELL_NUMBE holds 1,564 values in the same rows and
        names individual things, so it belongs in the text instead.
        """
        if not self.filled:
            return "empty"
        if self.kind == "number":
            return "number"
        average_length = sum(len(v) for v in self.filled) / len(self.filled)
        if average_length >= limits["prose_min_length"]:
            return "prose"
        if self.distinct / len(self.filled) <= limits["categorical_max_ratio"]:
            return "categorical"
        return "identifier"

    def numbers(self) -> list[float]:
        out = []
        for value in self.filled:
            try:
                out.append(float(value))
            except ValueError:
                continue
        return out
