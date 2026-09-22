"""geo-mini-rag: appraise a directory of E&P documents, then run RAG over what survives."""


def main() -> None:
    import sys

    from geo_mini_rag.cli import app, console
    from geo_mini_rag.errors import UserError

    try:
        app()
    except UserError as exc:
        console.print(f"[red]Error:[/] {exc}")
        sys.exit(1)
