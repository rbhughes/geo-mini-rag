"""Command line entry point: `uv run geo-mini-rag --help`."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from geo_mini_rag import openrouter, settings

app = typer.Typer(
    no_args_is_help=True,
    help="A small RAG pipeline that reads E&P file formats.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
console = Console()


@app.command()
def models(
    embeddings: bool = typer.Option(False, "--embeddings", help="List every OpenRouter embedding model instead."),
) -> None:
    """Show the configured models with current OpenRouter prices. Free: no API key used."""
    cfg = settings.load_rag_config()
    if embeddings:
        rows = sorted(openrouter.list_embedding_catalog(), key=lambda m: float(m.get("pricing", {}).get("prompt") or 0))
        table = Table("embedding model", "in $/M", "context")
        for m in rows:
            mark = " (configured)" if m["id"] == cfg["embed"]["model"] else ""
            price = float(m.get("pricing", {}).get("prompt") or 0) * 1e6
            table.add_row(m["id"] + mark, f"{price:.4f}", str(m.get("context_length", "")))
        console.print(table)
        return
    # chat and embedding models live in separate OpenRouter catalogs
    catalog = {m["id"]: m for m in openrouter.list_catalog()}
    embed_catalog = {m["id"]: m for m in openrouter.list_embedding_catalog()}
    table = Table("role", "model", "in $/M", "out $/M", "context")
    for role, model_id, source in (
        ("answer", cfg["answer"]["model"], catalog),
        ("embed", cfg["embed"]["model"], embed_catalog),
    ):
        entry = source.get(model_id) or {}
        p = entry.get("pricing", {})
        table.add_row(
            role,
            model_id if entry else f"{model_id} (missing from catalog)",
            f"{float(p.get('prompt', 0)) * 1e6:.3f}",
            f"{float(p.get('completion', 0)) * 1e6:.3f}",
            str(entry.get("context_length", "")),
        )
    console.print(table)
    console.print("[dim]Both are set in config/rag.yaml. `models --embeddings` lists embedding alternatives.[/]")


MODEL_HELP = "Chat model for this one call; any OpenRouter id. Defaults to config/rag.yaml."
DB_OPTION = typer.Option(None, "--db", help="DuckDB index file. Defaults to data/index/rag.duckdb.")
WHERE_OPTION = typer.Option(
    None, "--where", "-w",
    help="Filter on document metadata: key=value, or key>value on a numeric fact. "
         "Repeatable; `geo-mini-rag meta` lists the keys.",
)


def _where(pairs: list[str] | None) -> list[tuple[str, str, str]]:
    """key=value, or a comparison on a numeric fact: depth_max>5000, log_year>=1990."""
    from geo_mini_rag.rag.search import COMPARISONS

    clauses: list[tuple[str, str, str]] = []
    for pair in pairs or []:
        for op in COMPARISONS:
            key, sep, value = pair.partition(op)
            if sep and key.strip() and value.strip():
                clauses.append((key.strip(), op, value.strip()))
                break
        else:
            raise typer.BadParameter(
                f"expected key=value or key>value, got {pair!r}", param_hint="--where"
            )
    return clauses


def _db(path: Path | None) -> Path:
    from geo_mini_rag.rag.store import DB_PATH

    return path or DB_PATH


@app.command()
def ping(model: str = typer.Option(None, "--model", "-m", help=MODEL_HELP)) -> None:
    """Send one tiny paid request to confirm the API key works."""
    res = openrouter.chat(settings.chat_model(model), [{"role": "user", "content": "Reply with the word ok."}], max_tokens=20)
    console.print(f"{res.model}: {res.text!r}  cost=${res.usage.get('cost', 0)}  {res.latency_s:.1f}s")


@app.command()
def ingest(
    root: str = typer.Option(None, help="Folder to index; defaults to GEO_DOCS_ROOT."),
    rebuild: bool = typer.Option(False, help="Drop the index and start over."),
    limit: int = typer.Option(None, help="Stop after this many files (for quick trials)."),
    embed_model: str = typer.Option(None, "--embed-model", "-e", help="OpenRouter embedding model; defaults to config/rag.yaml."),
    db: Path = DB_OPTION,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Print one line per file."),
    trace: bool = typer.Option(False, "--trace", help="Print every step: sniffing, extraction, chunks, embedding calls, SQL."),
    trace_chars: int = typer.Option(160, "--trace-chars", help="Text sample length in trace output; 0 prints text in full."),
) -> None:
    """Extract and chunk every file under the root, embed via OpenRouter, store in DuckDB. Paid."""
    from collections import Counter

    from rich.progress import Progress
    from rich.text import Text

    from geo_mini_rag.rag import ingest as ingest_mod
    from geo_mini_rag.rag.trace import OFF, Tracer

    db = _db(db)
    counts: Counter[str] = Counter()
    spent = 0.0
    root = root or settings.docs_root()
    # A live progress bar fights with line-by-line trace output, so tracing turns it off.
    with Progress(console=console, transient=True, disable=trace) as progress:
        task = progress.add_task("ingesting", total=None)

        def emit(stage: str, message: str) -> None:
            if stage == "file":
                console.rule(Text(message, style="bold"), align="left")
                return
            line = Text(f"  {stage:>7}  ", style=STAGE_STYLES.get(stage, "cyan"))
            line.append(message)
            console.print(line, soft_wrap=True)

        def on_event(e: ingest_mod.IngestEvent) -> None:
            nonlocal spent
            counts[e.status] += 1
            spent += e.cost
            progress.update(task, advance=1, description=f"{dict(counts)} ${spent:.4f}")
            if trace or verbose or e.status == "error":
                console.print(Text(f"{e.status:>9}  ", style="bold") + Text(f"{e.path}  {e.detail}"))

        tracer = Tracer(emit, trace_chars) if trace else OFF
        ingest_mod.ingest(root, db=db, rebuild=rebuild, limit=limit, embed_model=embed_model,
                     on_event=on_event, trace=tracer)
    console.print(f"{dict(counts)}  embedding cost this run: ${spent:.4f}")
    stats(db)


STAGE_STYLES = {"sql": "magenta", "embed": "green", "chunk": "yellow", "skip": "red", "error": "bold red", "ocr": "blue", "partition": "blue", "handler": "blue"}


@app.command()
def stats(db: Path = DB_OPTION) -> None:
    """Summarize what the index holds and why files were skipped."""
    from geo_mini_rag.rag import store

    db = _db(db)
    s = store.stats(db)
    table = Table("status", "kind", "files", "chunks", "embed tokens", "embed $", title=f"{db} ({s['embed_model']})")
    for status, kind, files, chunks, tokens, cost in s["by_status"]:
        table.add_row(status, kind, str(files), str(chunks or 0), str(tokens or 0), f"{cost or 0:.4f}")
    console.print(table)
    if s["reasons"]:
        reasons = Table("not indexed because", "files")
        for reason, n in s["reasons"]:
            reasons.add_row(reason, str(n))
        console.print(reasons)
    meta = Table("index meta", "value")
    for key, value in sorted(s["meta"].items()):
        meta.add_row(key, value)
    console.print(meta)


@app.command()
def meta(
    key: str = typer.Argument(None, help="Show the values for one key instead of the summary."),
    limit: int = typer.Option(25, help="Values to list."),
    db: Path = DB_OPTION,
) -> None:
    """What document metadata the index holds, and what can be filtered on. Free."""
    from geo_mini_rag.rag import store

    db = _db(db)
    with store.connect(db, read_only=True) as con:
        if key:
            rows = con.execute(
                "SELECT value, count(*) FROM doc_meta WHERE key = ? GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT ?",
                [key, limit],
            ).fetchall()
            if not rows:
                console.print(f"[red]no metadata key {key!r} in {db}[/]")
                raise typer.Exit(1)
            table = Table(key, "documents")
            for value, n in rows:
                table.add_row(value, str(n))
            console.print(table)
            span = con.execute(
                "SELECT min(num_value), max(num_value), count(num_value) FROM doc_meta WHERE key = ?",
                [key],
            ).fetchone()
            if span and span[2]:
                console.print(f"[dim]numeric on {span[2]} documents: {span[0]:g} to {span[1]:g} "
                              f"— filter with --where {key}'>'{span[0]:g}[/]")
            return
        keys = store.metadata_keys(con)
    if not keys:
        console.print(f"[dim]{db} has no document metadata: no handler has extracted any.[/]")
        return
    table = Table("key", "documents", "distinct values", title=f"metadata in {db.name}")
    for name, docs, distinct in keys:
        table.add_row(name, str(docs), str(distinct))
    console.print(table)
    console.print("[dim]filter with: search \"question\" --where key=value[/]")


@app.command("eval")
def eval_(
    question_set: str = typer.Argument("evals/subset.jsonl", help="JSONL question set."),
    db: Path = DB_OPTION,
    k: int = typer.Option(10, "-k", help="Retrieval depth to score to."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show every question, not just misses."),
) -> None:
    """Score retrieval against a question set with known answers. Cheap: one embedding call per question."""
    from geo_mini_rag.rag import evaluate

    questions = evaluate.load(Path(question_set))
    rows: list[tuple[str, str, str]] = []

    def on_result(r: evaluate.Result) -> None:
        rank = str(r.rank) if r.rank else "MISS"
        if verbose or r.rank is None:
            rows.append((rank, r.question.q, r.question.expect_path or r.question.expect_text or ""))

    report = evaluate.run(questions, db=_db(db), k=k, on_result=on_result)
    if rows:
        table = Table("rank", "question", "expected", title="misses" if not verbose else "questions")
        for rank, q, expected in rows:
            table.add_row(rank, q, expected)
        console.print(table)
    summary = Table("metric", "value", title=f"{question_set} against {_db(db).name} ({report.n} questions)")
    for depth in evaluate.DEPTHS:
        if depth <= k:
            summary.add_row(f"recall@{depth}", f"{report.recall_at(depth):.0%}")
    summary.add_row("MRR", f"{report.mrr:.3f}")
    summary.add_row("cost", f"${report.cost:.5f}")
    console.print(summary)


def _print_hits(hits, full: bool) -> None:
    show_matched = any(getattr(h, "matched", "") for h in hits)
    columns = ["#", "score", "source"] + (["metadata match"] if show_matched else []) + ["text"]
    table = Table(*columns)
    for h in hits:
        source = h.path + (f" p{h.page}" if h.page else "")
        text = h.text if full else h.text[:160].replace("\n", " ") + "…"
        row = [str(h.rank), f"{h.score:.3f}", source]
        if show_matched:
            row.append(getattr(h, "matched", ""))
        table.add_row(*row, text)
    console.print(table)


@app.command()
def search(
    question: str,
    k: int = typer.Option(None, "-k", help="Chunks to return; defaults to config/rag.yaml."),
    full: bool = typer.Option(False, help="Show whole chunks."),
    where: list[str] = WHERE_OPTION,
    db: Path = DB_OPTION,
) -> None:
    """Retrieval only, no LLM. Costs one tiny embedding call."""
    from geo_mini_rag.rag import search as search_mod

    cfg = settings.load_rag_config()
    hits, _ = search_mod.search(question, k or cfg["retrieve"]["top_k"], _db(db),
                           where=_where(where), cfg=cfg)
    _print_hits(hits, full)


@app.command()
def ask(
    question: str,
    model: str = typer.Option(None, "--model", "-m", help=MODEL_HELP),
    k: int = typer.Option(None, "-k", help="Chunks to retrieve; defaults to config/rag.yaml."),
    show_context: bool = typer.Option(False, help="Print the retrieved chunks too."),
    where: list[str] = WHERE_OPTION,
    db: Path = DB_OPTION,
) -> None:
    """Retrieve chunks and answer with an OpenRouter chat model. Paid."""
    from rich.markdown import Markdown

    from geo_mini_rag.rag import answer

    cfg = settings.load_rag_config()
    model_id = settings.chat_model(model)
    a = answer.ask(question, model=model_id, k=k or cfg["retrieve"]["top_k"], max_tokens=cfg["answer"]["max_tokens"],
                   db=_db(db), where=_where(where))
    console.print(Markdown(a.text or "_(empty response)_"))
    console.print()
    if show_context:
        _print_hits(a.hits, full=True)
    else:
        for h in a.hits:
            console.print(f"[dim][{h.rank}] {h.score:.3f} {h.path}{f' p{h.page}' if h.page else ''}[/]")
    u = a.result.usage
    console.print(
        f"[dim]{model_id} · {u.get('prompt_tokens', '?')} in / {u.get('completion_tokens', '?')} out"
        f" · ${a.cost:.6f} incl. query embedding · {a.result.latency_s:.1f}s[/]"
    )



@app.command("help")
def help_(ctx: typer.Context, command: str = typer.Argument(None, help="Show full help for this command.")) -> None:
    """List all commands, or show one command's options."""
    from typer.core import (
        TyperGroup,  # typer 0.27 vendors click, so import the group from typer
    )

    parent = ctx.parent or ctx
    group = parent.command
    if not isinstance(group, TyperGroup):   # only reachable if `help` stops being a subcommand
        raise typer.BadParameter("no command group to list", param_hint="command")
    if command:
        sub = group.get_command(parent, command)
        if sub is None:
            raise typer.BadParameter(f"no command {command!r}; run `geo-mini-rag help`", param_hint="command")
        with typer.Context(sub, info_name=command, parent=parent) as sub_ctx:
            console.print(sub.get_help(sub_ctx))
        return
    table = Table("command", "what it does", box=None, show_header=False, pad_edge=False)
    for name in group.list_commands(parent):
        cmd = group.get_command(parent, name)
        if cmd is None:
            continue
        # first docstring line, which keeps the Free/Paid note that short help would cut
        summary = (cmd.help or "").strip().splitlines()[0]
        table.add_row(f"[bold]{name}[/]", summary)
        flags = " ".join(
            "/".join(p.opts) + (f" <{p.name}>" if p.type.name != "boolean" else "")
            for p in cmd.params
            if p.name != "help" and p.opts and p.opts[0].startswith("-")
        )
        args = " ".join(f"<{p.name}>" for p in cmd.params if p.opts and not p.opts[0].startswith("-"))
        if flags or args:
            table.add_row("", f"[dim]{args + ' ' if args else ''}{flags}[/]")
    console.print("Usage: geo-mini-rag COMMAND [OPTIONS]\n")
    console.print(table)
    console.print("\nRun `geo-mini-rag help COMMAND` or `geo-mini-rag COMMAND -h` for options.")


if __name__ == "__main__":
    app()
