"""Well identifiers as document facts, whatever format the document was.

Handlers own formats; this owns a thing that appears in all of them. A
completion report, a loader log, a scanned permit and a LAS header all name
wells, and the name is what people search by and what embeddings are worst at,
so it belongs in the facts that filtering and the IDF boost work on.

Two detectors feed it: `api_number` for US API well numbers, `uwi_ca` for
Canadian UWIs. Each decides on its own evidence and neither infers a missing
part of an identifier.

    api            ten digits, one per well: 4902511080
    api_state      from the code table: WY
    api_county     from the code table: Natrona
    uwi            sixteen characters: 100041108204W600
    well_location  a legal description with no UWI around it: 041108204W6
"""

from __future__ import annotations

from geo_mini_rag.ep import api_number, uwi_ca

DEFAULTS = {**api_number.DEFAULTS, **uwi_ca.DEFAULTS}


def _merge(metadata: dict, key: str, values: list[str], canonical=lambda v: v) -> list[str]:
    """What the document already said about this key, plus what was found, once each."""
    existing = metadata.get(key, [])
    existing = list(existing) if isinstance(existing, (list, tuple, set)) else [existing]
    out: list[str] = []
    for value in [*(str(v).strip() for v in existing), *values]:
        value = canonical(value)
        if value and value not in out:
            out.append(value)
    return out


def enrich(metadata: dict, segments: list[tuple[int | None, str]], limits: dict | None = None
           ) -> tuple[dict, list[str]]:
    """Well identifier facts for one document. Returns (facts, notes).

    A handler that already lifted an identifier out of a header keeps it: the
    sets are merged, so a LAS whose header declares one number and whose body
    lists twenty ends up with twenty-one. One well is one fact in one shape --
    a header writing 490251108000 and a report writing 49-025-11080 are the
    same well, stored as the ten digits that identify it.
    """
    limits = {**DEFAULTS, **(limits or {})}
    text = "\n".join(part for _, part in segments if part)

    american = api_number.find(text, limits)
    canadian = uwi_ca.find(text, limits)
    notes: list[str] = []
    cap = limits["max_per_document"]
    if cap and len(american) > cap:
        notes.append(f"api: kept {cap:,} of {len(american):,} identifiers")
        american = american[:cap]
    if cap and len(canadian) > cap:
        notes.append(f"uwi: kept {cap:,} of {len(canadian):,} identifiers")
        canadian = canadian[:cap]

    facts: dict[str, list[str]] = {}
    if american:
        facts["api"] = _merge(metadata, "api", [w.api for w in american], api_number.ten)
        facts["api_state"] = _merge(metadata, "api_state", sorted({w.state for w in american}))
        facts["api_county"] = _merge(
            metadata, "api_county", sorted({c for w in american for c in w.counties})
        )
    if complete := [w.uwi for w in canadian if w.uwi]:
        facts["uwi"] = _merge(metadata, "uwi", complete, api_number.ten)
    elif american and metadata.get("uwi"):
        # A header that wrote the API number under UWI meant the same well.
        facts["uwi"] = _merge(metadata, "uwi", [], api_number.ten)
    if located := [w.location for w in canadian if not w.uwi]:
        facts["well_location"] = _merge(metadata, "well_location", located)
    return facts, notes


def in_question(question: str) -> list[tuple[str, str]]:
    """(key, value) identifier facts a question names, for looking up rather than guessing.

    An identifier is what embeddings are worst at: `well 4902511080` scores 0.324
    against the log that carries it and 0.729 against a page of unrelated digits,
    so no boost small enough to be safe can rescue it. Looked up instead, it is
    exact. No label is required here, because a question is not a document: the
    code table alone decides.
    """
    found = [("api", well.api) for well in api_number.find(question, require_label=False)]
    for well in uwi_ca.find(question, require_label=False):
        found.append(("uwi", well.uwi) if well.uwi else ("well_location", well.location))
    return found
